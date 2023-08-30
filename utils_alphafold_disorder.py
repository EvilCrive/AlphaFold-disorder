from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.SeqUtils import seq1
from Bio.PDB import DSSP
import numpy as np
import warnings
import pandas as pd
import argparse
import logging.config
import sys
import csv
from pathlib import Path, PurePath
import tempfile
import gzip
import shutil
import os
import foldcomp
from io import StringIO

_r_helix = (np.deg2rad(89-12), np.deg2rad(89+12))
_a_helix = (np.deg2rad(50-20), np.deg2rad(50+20))
_d2_helix = ((5.5-0.5), (5.5+0.5)) # Not used in the algorithm description
_d3_helix = ((5.3-0.5), (5.3+0.5))
_d4_helix = ((6.4-0.6), (6.4+0.6))

_r_strand = (np.deg2rad(124-14), np.deg2rad(124+14))
_a_strand = (np.deg2rad(-180), np.deg2rad(-125), np.deg2rad(145), np.deg2rad(180))
_d2_strand = ((6.7-0.6), (6.7+0.6))
_d3_strand = ((9.9-0.9), (9.9+0.9))
_d4_strand = ((12.4-1.1), (12.4+1.1))

def get_res_start(structure, move_hhem_to_end = False) :
    atom_array = []
    hhems = []
    for model in structure :
        for chain in model :
            for residue in chain :
                for atom in residue :
                    if not move_hhem_to_end :
                        atom_array.append(atom)
                    else :
                        if atom.full_id[3][0] == 'H_HEM' :
                            hhems.append(atom)
                        else :
                            atom_array.append(atom)
    if move_hhem_to_end :
        atom_array = atom_array + hhems
    
    res_array = [res for m in structure for ch in m for res in ch]
    chain_array = np.array([atom.parent.parent.id for atom in atom_array])
    resid_array = np.array([atom.parent.id[1] for atom in atom_array])
    resname_array = np.array([atom.parent.resname for atom in atom_array])
    resStart_mask = (
        (chain_array[1:] != chain_array[:-1]) |
        (resname_array[1:] != resname_array[:-1]) |
        (resid_array[1:] != resid_array[:-1])
    )
    # with the mask, we would find the last atom of the residue, so we do a +1
    res_starts_id = np.where(resStart_mask)[0]+1
    # we append at the start the residue index 0, the start
    return np.append([0], res_starts_id), atom_array

def get_ca_coord(res_starts_id, atom_array) :
    ca_coord = np.full((len(res_starts_id), 3), np.nan, dtype=np.float32)
    ca_indices = np.where(np.array([atom.name for atom in atom_array]) == "CA")[0]
    # actual CA coord assignment
    ca_coord[np.searchsorted(res_starts_id, ca_indices,"right")-1] = \
        [atom.coord for atom in atom_array if atom.name=="CA"]
   
    diff1 = np.diff(np.array([atom.parent.id[1] for atom in atom_array]))
    disc_ids = np.where((diff1!=0) & (diff1 != 1))[0] + 1
    disc_ids_res = np.searchsorted(res_starts_id, disc_ids, "right") - 1
    
    ca_coord = np.insert(
        ca_coord, disc_ids_res, 
        np.full((len(disc_ids_res), 3), np.nan),
        axis = 0
    )
    no_virtualmask = np.insert(np.ones(len(res_starts_id), dtype=bool), disc_ids_res, False)

    return ca_coord, no_virtualmask

def distance_atoms(v1, v2) :
    diff = None
    if v1.shape <= v2.shape :
        diff = v2 - v1
    else :
        diff = -(v1 - v2)
    return np.sqrt((diff*diff).sum(axis=-1))

def angle(v1, v2, v3) :
    diff12 = v2-v1 if v1.shape<=v2.shape else -(v1-v2)
    diff32 = v2-v3 if v3.shape<=v2.shape else -(v3-v2)
    diff12 /= np.linalg.norm(diff12, axis=-1)[..., np.newaxis]
    diff32 /= np.linalg.norm(diff32, axis=-1)[..., np.newaxis]
    return np.arccos((diff12*diff32).sum(axis=-1))

def dihedral(v1,v2,v3,v4) :
    diff12 = v2-v1 if v1.shape<=v2.shape else -(v1-v2)
    diff23 = v3-v2 if v2.shape<=v3.shape else -(v2-v3)
    diff34 = v4-v3 if v3.shape<=v4.shape else -(v3-v4)
    diff12 /= np.linalg.norm(diff12, axis=-1)[..., np.newaxis]
    diff23 /= np.linalg.norm(diff23, axis=-1)[..., np.newaxis]
    diff34 /= np.linalg.norm(diff34, axis=-1)[..., np.newaxis]
    n1 = np.cross(diff12, diff23)
    n2 = np.cross(diff23, diff34)
    x = (n1*n2).sum(axis=-1)
    y = (np.cross(n1,n2)*diff23).sum(axis=-1)
    return np.arctan2(y,x)

def mask_consecutive(mask, number) :
    # Return a new mask for the input mask: true if that element is in a consecutive region 
    #   (if it has 'number' times true values in the input mask)
    counts = np.zeros(len(mask) - (number-1), dtype=int)
    consecutive_mask = np.zeros(len(mask), dtype=bool)
    for i in range(number) :
        #slice
        counts[mask[i:i+len(counts)]] += 1
    consecutive_seed = (counts == number)
    for i in range(number) :
        consecutive_mask[i:i+len(consecutive_seed)] |= consecutive_seed
    return consecutive_mask

def extend_region(base_mask, extension_mask) :
    # regin_change_mask is False when region changes
    region_change_mask = np.diff(np.append([False], base_mask))
    left_end_mask = (region_change_mask) & (base_mask)
    # Remove first element and append False as last element
    left_end_mask = np.append(left_end_mask[1:], [False])
    right_end_mask = (region_change_mask) & (~base_mask)
    return ((left_end_mask | right_end_mask) & (extension_mask)) | (base_mask)

def mask_regions_with_low_contacts(coord, cand_mask, min_contacts, min_distance, max_distance) :
    ca_coords = coord[~np.isnan(coord).any(axis=-1)]
    potential_ca_coords = coord[cand_mask & ~np.isnan(coord).any(axis=-1)]
    adjacency_matrix = np.zeros((potential_ca_coords.shape[0], ca_coords.shape[0]), dtype=int)
    for i in range(potential_ca_coords.shape[0]) :
        for j in range(ca_coords.shape[0]) :
            distance = distance_atoms(potential_ca_coords[i], ca_coords[j])
            if (i!=j) & (distance < max_distance) :
                adjacency_matrix[i,j] = 1
    result_indices = [[j for j in range(ca_coords.shape[0]) if adjacency_matrix[i,j]==1] for i in range(potential_ca_coords.shape[0])]
    contacts = np.zeros(len(coord), dtype=int)
    for i, atom_i in enumerate(np.where(cand_mask)[0]) :
        indices = result_indices[i]
        n_contacts_i = distance_atoms(coord[atom_i], ca_coords[indices]) > min_distance
        contacts[atom_i] = np.count_nonzero(n_contacts_i)
    region_change_idx = np.where(np.diff(np.append([False], cand_mask)))[0]
    region_change_idx = np.append(region_change_idx, [len(coord)])
    output_mask = np.zeros(len(cand_mask), dtype=bool)
    for i in range(len(region_change_idx) -1) :
        start = region_change_idx[i]
        stop = region_change_idx[i+1]
        tot_contacts = np.sum(contacts[start:stop])
        if tot_contacts >= min_contacts :
            output_mask[start:stop] = True
    return output_mask

def calc_dist_angles(ca_coord) : 
    length = len(ca_coord)

    d2i = np.full(length, np.nan)
    d3i = np.full(length, np.nan)
    d4i = np.full(length, np.nan)
    ri = np.full(length, np.nan)
    ai = np.full(length, np.nan)

    d2i[1:length-1] = distance_atoms(ca_coord[0:length-2], ca_coord[2:length])
    d3i[1:length-2] = distance_atoms(ca_coord[0:length-3], ca_coord[3:length])
    d4i[1:length-3] = distance_atoms(ca_coord[0:length-4], ca_coord[4:length])

    ri[1:length-1] = angle(
        ca_coord[0:length-2],
        ca_coord[1:length-1],
        ca_coord[2:length]
    )

    ai[1:length-2] = dihedral(
        ca_coord[0:length-3],
        ca_coord[1:length-2],
        ca_coord[2:length-1],
        ca_coord[3:length]
    )
    return length, [d2i, d3i, d4i, ri, ai]

def calc_struct_mask(ca_coord, length, distances, angles, short_contacts = True) :
    d2i = distances[0]
    d3i = distances[1]
    d4i = distances[2]
    ri = angles[0]
    ai = angles[1]
    
    relaxed_helix = (
        (d3i >= _d3_helix[0]) & (d3i <= _d3_helix[1])
    ) | (
        (ri >= _r_helix[0]) & (ri <= _r_helix[1])
    )

    strict_helix = (
        (d3i >= _d3_helix[0]) & (d3i <= _d3_helix[1]) & (d4i >= _d4_helix[0]) & (d4i <= _d4_helix[1])
    ) | (
        (ri >= _r_helix[0]) & (ri <= _r_helix[1]) & (ai >= _a_helix[0]) & (ai <= _a_helix[1])
        )
    
    helix_mask = mask_consecutive(strict_helix, 4)
    helix_mask = extend_region(helix_mask, relaxed_helix)
    
    relaxed_strand = (d3i >= _d3_strand[0]) & (d3i <= _d3_strand[1])

    strict_strand = (
            (d2i >= _d2_strand[0]) & (d2i <= _d2_strand[1]) & 
            (d3i >= _d3_strand[0]) & (d3i <= _d3_strand[1]) & 
            (d4i >= _d4_strand[0]) & (d4i <= _d4_strand[1])
        ) | (
            (ri >= _r_strand[0]) & (ri <= _r_strand[1]) & 
            (
                (
                    (ai >= _a_strand[0]) & (ai <= _a_strand[1])
                ) | (
                    (ai >= _a_strand[2])) & (ai <= _a_strand[3]
                )
            ) 
        )
    
    strand_mask = mask_consecutive(strict_strand, 3)
    
    if short_contacts :
        short_strand_mask = mask_regions_with_low_contacts(
            ca_coord, mask_consecutive(strict_strand,3), min_contacts=5, min_distance=4.2, max_distance=5.2
        )
        strand_mask = strand_mask | short_strand_mask
        
    strand_mask = extend_region(strand_mask , relaxed_strand)
    
    return [helix_mask, strand_mask]

def finalize_sse(ca_coord, length, novirtual_mask, helix_mask, strand_mask) :
    sse = np.full(length, "C", dtype="U1") #c
    sse[helix_mask] = "H"  #a
    sse[strand_mask] = "E" #b
    sse[np.isnan(ca_coord).any(axis=-1)] = ""
    return sse[novirtual_mask]

def get_sse_psea(structure, add_short_contacts = True, move_end_hhem = True) :
    res_start_id, atom_array = get_res_start(structure, move_end_hhem)
    ca_coord, novirtual_mask = get_ca_coord(res_start_id, atom_array)
    length, [d2i, d3i, d4i, ri, ai] = calc_dist_angles(ca_coord)
    helix_mask, strand_mask = calc_struct_mask(ca_coord, length, [d2i, d3i, d4i], [ri, ai], add_short_contacts)
    
    return finalize_sse(ca_coord, length, novirtual_mask, helix_mask, strand_mask)

def compare_sses(sse, dssp) :
    sse_dssp = [dssp[i][2] for i in dssp]
         
    counter1 = 0
    counter2 = 0
    countersize = 0
    for (i,j) in zip(sse, sse_dssp) :
        if j in ['H','G','I'] : 
            j = 'a'
        elif j in ['B', 'E'] : 
            j = 'b'
        else : 
            j = 'c'
        if i == '': i = 'c'
 
        if i == j :
            counter1 += 1
        if i in ['a','b'] : i = 'p'
        if j in ['a','b'] : j = 'p'
        if i == j :
            counter2 += 1
        #else :
            #print(i,j)
        countersize+=1
    counter1 = counter1/countersize
    counter2 = counter2/countersize
    if counter1 == counter2 :
        return round(counter1,4)
    else :
        return [counter1/countersize, counter2/countersize]
