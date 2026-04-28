#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Core modules
import os
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)
import pickle

# Importing and setting up the FLAP environment
import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')

import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name = fn)
wd = flap.config.get_all_section('Module NSTX_GPI').get('Working directory', './')

import numpy as np
from scipy.signal import correlate2d
from scipy.optimize import linear_sum_assignment

from flap_nstx.tools import (
    fringe_jump_correction,
    MetricArray,
    StructureDataset, TrackedPlasmaStructure
)

# Constants for the calculation
coeff_r = np.asarray([3.75, 0,    1402.8097])/1000. 
coeff_z = np.asarray([0,    3.75, 70.544312])/1000.  

def track_structures(dataset=None,
                     time_range=None,
                     tracking='weighted',
                     tracking_assignment='max_score',
                     max_gap=3,
                     matrix_weight=None,
                     smooth_contours=None,
                     prev_str_weighting='area',
                     calculate_rough_diff_velocities=False,
                     differential_keys=None,
                     weighting='area',
                     maxing='',
                     nocalc=False,
                     recalc_tracking=False,
                     score_threshold=0.7,
                     comment=None,
                     test=False,
                     ):
    
    """
    Main loop for tracking identified plasma structures across consecutive frames.
    """

    # Fall through to calculation if no data was found, recalc is True, or load failed
    print("\nCalculating structure tracking.")
        
    highest_label = 0
    n_frames = len(dataset.frames)
    sample_time = dataset.frame_times[1] - dataset.frame_times[0]
    
    for i_frames in range(1, n_frames):
        structures_1 = dataset.frames[i_frames - 1]
        structures_2 = dataset.frames[i_frames]

        # Case A: Mass Death
        if structures_1 and not structures_2:
            for s1 in structures_1:
                s1.born, s1.died = False, True
                if s1.label is None:
                    highest_label += 1
                    s1.label = highest_label

        # Case B: Mass Birth
        elif not structures_1 and structures_2:
            for s2 in structures_2:
                highest_label += 1
                s2.label = highest_label
                s2.born, s2.died = True, False

        # Case C: Standard Tracking
        elif structures_1 and structures_2:
            for s1 in structures_1:
                if s1.label is None:
                    highest_label += 1
                    s1.label = highest_label
                    s1.born, s1.died = True, False

            # 1. Build overlap matrix
            str_overlap_matrix = _calculate_str_overlap_matrix(structures_1, structures_2, tracking, 
                                                               matrix_weight, tracking_assignment, test)
                        
            # 2. Process standard 1-to-1 links and Merges
            structures_1, structures_2, highest_label = _process_structure_merging(
                structures_1, structures_2, str_overlap_matrix, 
                gap=1, highest_label=highest_label)

            # 3. Process Splits
            structures_1, structures_2, highest_label = _process_structure_splitting(
                structures_1, structures_2, str_overlap_matrix, 
                highest_label=highest_label)
                                
        # 4. Gap Recovery 
        if i_frames >= 2 and structures_2:
            max_search = min(max_gap, i_frames)
            
            for ind_gap in range(2, max_search + 1):
                n_born = sum(1 for s in structures_2 if s.born)
                
                if n_born > 0:
                    structures_before_gap = dataset.frames[i_frames - ind_gap]
                    if structures_before_gap:
                        overlap_gap_matrix = _calculate_str_overlap_matrix(
                            structures_before_gap, structures_2, tracking, matrix_weight, tracking_assignment, test)
                        
                        structures_before_gap, structures_2, highest_label = _process_structure_merging(
                            structures_before_gap, structures_2, overlap_gap_matrix, 
                            gap=ind_gap, highest_label=highest_label)

    tracked_dataset = StructureDataset(mode='tracked', exp_id=dataset.exp_id)
    tracked_dataset.frame_times = dataset.frame_times
    tracked_dataset.frames = dataset.frames 
    tracked_blobs = {}
    
    for i_frames, structures in enumerate(dataset.frames):
        current_time = dataset.frame_times[i_frames]
        for s in structures:
            label = s.label
            if label not in tracked_blobs:
                tracked_blobs[label] = TrackedPlasmaStructure(label=label, start_time=current_time)
                # Explicitly initialize a list to hold the raw footprints
                tracked_blobs[label].structures = []
            
            # 1. Let the native FLAP engine handle the time and parameter math
            tracked_blobs[label].add_step(s, current_time)
            
            # 2. Explicitly save the physical footprint so the HDF5 serializer can find it
            tracked_blobs[label].structures.append(s)

    for label, blob in tracked_blobs.items():
        # Append directly to prevent label-based None padding!
        tracked_dataset.tracked_structures.append(blob)

    return tracked_dataset


def _process_structure_merging(structures_1, structures_2, str_overlap_matrix, 
                               gap=1, highest_label=None):
    
    for j_str2, s2 in enumerate(structures_2):
        parent_indices = np.where(str_overlap_matrix[:, j_str2] == 1)[0]
        num_parents = len(parent_indices)
        
        if num_parents == 0 and gap == 1:
            highest_label += 1
            s2.label = highest_label
            s2.born = True

        elif num_parents == 1:
            p_idx = parent_indices[0]
            s1 = structures_1[p_idx]
            
            if gap == 1 and np.sum(str_overlap_matrix[p_idx, :]) == 1:
                s2.label = s1.label
                s2 = correct_structure_angle(structure_1=s1, structure_2=s2)

            elif gap > 1 and s2.born and s1.died:
                if np.sum(str_overlap_matrix[p_idx, :]) == 1:
                    s2.label = s1.label
                    s2 = correct_structure_angle(structure_1=s1, structure_2=s2)
                    s2.born, s1.died = False, False

        elif num_parents > 1 and gap == 1:
            if np.sum(str_overlap_matrix[parent_indices, :]) == num_parents:
                dominant_p_idx = max(parent_indices, key=lambda idx: structures_1[idx].intensity)
                dom_s1 = structures_1[dominant_p_idx]
                
                s2.label = dom_s1.label
                s2 = correct_structure_angle(structure_1=dom_s1, structure_2=s2)
                
                for p_idx in parent_indices:
                    s1 = structures_1[p_idx]
                    s2.parents.append(s1.label)
                    s1.children.append(s2.label)
                    s1.merges = True
            else:
                dominant_p_idx = max(parent_indices, key=lambda idx: structures_1[idx].intensity)
                dom_s1 = structures_1[dominant_p_idx]

                all_child_indices = set()
                for p_idx in parent_indices:
                    for c_idx in np.where(str_overlap_matrix[p_idx, :] == 1)[0]:
                        all_child_indices.add(c_idx)
                        
                dominant_c_idx = max(all_child_indices, key=lambda idx: structures_2[idx].intensity)
                
                for p_idx in parent_indices:
                    s1 = structures_1[p_idx]
                    s1.splits = np.sum(str_overlap_matrix[p_idx, :]) > 1
                    s1.merges = True
                    
                    for c_idx in np.where(str_overlap_matrix[p_idx, :] == 1)[0]:
                        child = structures_2[c_idx]
                        
                        if c_idx == dominant_c_idx and p_idx == dominant_p_idx:
                            child.label = dom_s1.label
                            child = correct_structure_angle(structure_1=dom_s1, structure_2=child)
                        elif child.label is None:
                            highest_label += 1
                            child.label = highest_label
                            
                        if dom_s1.label not in child.parents:
                            child.parents.append(dom_s1.label)
                        if child.label not in s1.children:
                            s1.children.append(child.label)

    return structures_1, structures_2, highest_label


def _process_structure_splitting(structures_1, structures_2, str_overlap_matrix, 
                                 highest_label=None):
    
    for j_str1, s1 in enumerate(structures_1):
        overlaps = str_overlap_matrix[j_str1, :]
        num_overlaps = np.sum(overlaps)

        if num_overlaps == 0:
            s1.died = True

        elif num_overlaps > 1:
            child_indices = np.where(overlaps == 1)[0]

            if np.sum(str_overlap_matrix[:, child_indices]) == num_overlaps:
                dominant_idx = max(child_indices, key=lambda idx: structures_2[idx].intensity)

                for idx in child_indices:
                    s2 = structures_2[idx]
                    
                    if idx == dominant_idx:
                        s2.label = s1.label
                        s2 = correct_structure_angle(structure_1=s1, structure_2=s2)
                    else:
                        highest_label += 1
                        s2.label = highest_label

                    s2.parents.append(s1.label)
                    s1.children.append(s2.label)

                s1.splits = True
                
    return structures_1, structures_2, highest_label


def _calculate_str_overlap_matrix(structures_1, structures_2, tracking=None, 
                                  matrix_weight=None, tracking_assignment=None, test=False):
    n_str1, n_str2 = len(structures_1), len(structures_2)
    str_overlap_matrix = np.zeros((n_str1, n_str2))

    if n_str1 == 0 or n_str2 == 0:
        return str_overlap_matrix

    if tracking == 'overlap':
        for j_str1, s1 in enumerate(structures_1):
            for j_str2, s2 in enumerate(structures_2):
                if s2.shapely_polygon.intersects(s1.shapely_polygon):
                    str_overlap_matrix[j_str1, j_str2] = 1.0

    elif tracking == 'weighted':
        score_matrix = calculate_score_matrix(structures_1, structures_2, matrix_weight, coeff_r, coeff_z)

        if tracking_assignment == 'hungarian':
            row_indices, col_indices = linear_sum_assignment(score_matrix, maximize=True)
            str_overlap_matrix[row_indices, col_indices] = 1.0

        elif tracking_assignment == 'max_score':
            for ind_row in range(n_str1):
                row_scores = score_matrix[ind_row, :]
                if np.max(row_scores) > 0:
                    best_match_idx = np.argmax(row_scores)
                    str_overlap_matrix[ind_row, best_match_idx] = 1.0
    else:
        raise ValueError(f"Tracking method '{tracking}' is unavailable.")
        
    return str_overlap_matrix


def calculate_score_matrix(structures_1, structures_2, matrix_weight, coeff_r=None, coeff_z=None):
    n_str1, n_str2 = len(structures_1), len(structures_2)
    score_matrix = np.zeros((n_str1, n_str2))

    weight_iou = matrix_weight.get('iou', 0)
    weight_cccf = matrix_weight.get('cccf', 0)

    for j_str2, s2 in enumerate(structures_2):
        for j_str1, s1 in enumerate(structures_1):
            try:
                if s1.x_data_pix is None and coeff_r is not None and coeff_z is not None:
                    s1.x_data_pix = np.round((s1.x_data - coeff_r[2]) / coeff_r[0]).astype(int)
                    s1.y_data_pix = np.round((s1.y_data - coeff_z[2]) / coeff_z[1]).astype(int)

                if s2.x_data_pix is None and coeff_r is not None and coeff_z is not None:
                    s2.x_data_pix = np.round((s2.x_data - coeff_r[2]) / coeff_r[0]).astype(int)
                    s2.y_data_pix = np.round((s2.y_data - coeff_z[2]) / coeff_z[1]).astype(int)

                if s2.shapely_polygon.intersects(s1.shapely_polygon):
                    score = 0.0
                    
                    if weight_iou > 0:
                        intersection_area = s1.shapely_polygon.intersection(s2.shapely_polygon).area
                        union_area = s1.shapely_polygon.union(s2.shapely_polygon).area
                        score += (intersection_area / union_area) * weight_iou

                    if weight_cccf > 0:
                        x_min = min(s1.x_data_pix.min(), s2.x_data_pix.min())
                        x_max = max(s1.x_data_pix.max(), s2.x_data_pix.max())
                        y_min = min(s1.y_data_pix.min(), s2.y_data_pix.min())
                        y_max = max(s1.y_data_pix.max(), s2.y_data_pix.max())

                        str1_matrix = np.zeros((x_max - x_min + 1, y_max - y_min + 1))
                        str2_matrix = np.zeros((x_max - x_min + 1, y_max - y_min + 1))

                        str1_matrix[s1.x_data_pix - x_min, s1.y_data_pix - y_min] = s1.data
                        str2_matrix[s2.x_data_pix - x_min, s2.y_data_pix - y_min] = s2.data

                        str1_matrix -= np.mean(str1_matrix)
                        str2_matrix -= np.mean(str2_matrix)
                        
                        ccf_matrix = correlate2d(str1_matrix, str2_matrix)
                        norm_factor = np.sqrt(np.sum(str1_matrix**2) * np.sum(str2_matrix**2))
                        
                        if norm_factor > 0:
                            cccf_matrix = ccf_matrix / norm_factor
                            max_cccf = np.max(cccf_matrix)
                            if max_cccf > 1.0001:
                                raise ValueError(f'Cross-correlation exceeded 1: {max_cccf}')
                            score += max_cccf * weight_cccf

                    score_matrix[j_str1, j_str2] = score
            except Exception as e:
                print(f"Exception at score calculation: {e}")
                
    return score_matrix



def remove_orphans(dataset, test, min_structure_lifetime):
    label_counts = Counter()
    for structures in dataset.frames:
        if structures is not None:
            for s in structures:
                if s.label is not None:
                    label_counts[s.label] += 1

    if test: 
        print(f"Label frequencies: {label_counts}")

    label_map = {}
    new_index = 0
    for label, count in label_counts.items():
        if count >= min_structure_lifetime:
            label_map[label] = new_index
            new_index += 1

    for i_frames, structures in enumerate(dataset.frames):
        if structures is not None:
            valid_structures = []
            for s in structures:
                if s.label in label_map:
                    s.label = label_map[s.label]
                    valid_structures.append(s)
            dataset.frames[i_frames] = valid_structures

    return dataset


def calculate_differential_structure_keys(dataset):
    """
    Vectorized calculation of time-differential properties for all tracked structures.
    Safely handles both native MetricArrays and HDF5-loaded numpy arrays.
    """
    import numpy as np
    from flap_nstx.tools import MetricArray  # Ensure this is imported if not globally available
    
    # Map the desired dict_label to the base property it differentiates
    diff_map = {
        'Velocity radial COG': 'Center of gravity radial',
        'Velocity poloidal COG': 'Center of gravity poloidal',
        'Velocity radial centroid': 'Centroid radial',
        'Velocity poloidal centroid': 'Centroid poloidal',
        'Angular velocity ALI': 'Angle ALI',
        'Convexity diff': 'Convexity',
        'Solidity diff': 'Solidity',
        'Roundness diff': 'Roundness',
        'Total curvature diff': 'Total curvature',
        'Total bending energy diff': 'Total bending energy',
        'Area diff': 'Area',
        'Size radial diff': 'Size radial',
        'Size poloidal diff': 'Size poloidal',
        'Axes length minor diff': 'Axes length minor',
        'Axes length major diff': 'Axes length major',
        'Velocity radial position fit': 'Position radial fit',
        'Velocity poloidal position fit': 'Position poloidal fit',
        'Angular velocity angle fit': 'Angle fit',
        'Elongation fit diff': 'Elongation fit',
    }

    for struct in dataset.tracked_structures:
        if not struct or len(struct.time) < 2:
            continue
            
        rp = struct.regular_parameters
        dp = struct.differential_parameters
        
        # Get raw time differences
        dt_arr = np.diff(struct.time)
        
        # 1. Calculate standard derivatives (Velocity, Growth rates, etc.)
        for diff_key, reg_key in diff_map.items():
            if reg_key in rp:
                # Safely extract the raw numpy array
                reg_obj = rp[reg_key]
                reg_val = reg_obj.value if hasattr(reg_obj, 'value') else reg_obj
                
                # Perform the math using raw numpy arrays
                delta = reg_val[1:] - reg_val[:-1]
                rate_val = delta / dt_arr
                
                # Derive plot_label dynamically
                if hasattr(reg_obj, 'plot_label') and reg_obj.plot_label:
                    base_label = reg_obj.plot_label.replace('$', '')
                    derived_plot_label = f"$\\partial {base_label} / \\partial t$"
                else:
                    derived_plot_label = diff_key
                    
                # Derive unit dynamically (e.g., 'm' -> 'm/s')
                if hasattr(reg_obj, 'unit') and reg_obj.unit and reg_obj.unit != '-':
                    derived_unit = f"{reg_obj.unit}/s"
                else:
                    derived_unit = "1/s"
                
                # Wrap it back into a MetricArray (Satisfying all required arguments!)
                rate = MetricArray(value=rate_val, 
                                   dict_label=diff_key, 
                                   plot_label=derived_plot_label, 
                                   unit=derived_unit)
                dp[diff_key] = rate

        # 2. Calculate Expansion Fractions (Area ratios)
        if 'Area' in rp:
            area_obj = rp['Area']
            area_val = area_obj.value if hasattr(area_obj, 'value') else area_obj
            res_val = (area_val[1:] / area_val[:-1]) ** 0.5
            
            # Dimensionless ratio, so unit is '-'
            res = MetricArray(value=res_val, 
                              dict_label='Expansion fraction area', 
                              plot_label='$f_{E,area}$', 
                              unit='-')
            dp['Expansion fraction area'] = res
            
        if 'Axes length minor fit' in rp and 'Axes length major fit' in rp:
            minor_obj = rp['Axes length minor fit']
            minor_val = minor_obj.value if hasattr(minor_obj, 'value') else minor_obj
            
            major_obj = rp['Axes length major fit']
            major_val = major_obj.value if hasattr(major_obj, 'value') else major_obj
            
            area_fit = minor_val * major_val
            res_val = (area_fit[1:] / area_fit[:-1]) ** 0.5
            
            res = MetricArray(value=res_val, 
                              dict_label='Expansion fraction axes fit', 
                              plot_label='$f_{E,ellipse}$', 
                              unit='-')
            dp['Expansion fraction axes fit'] = res
            
    return dataset

def calculate_differential_structure_keys_old(dataset):
    """
    Vectorized calculation of time-differential properties for all tracked structures.
    Utilizes the built-in math operators of MetricArray to automatically derive
    units and LaTeX labels (e.g., Area / dt -> m^2/s).
    """
    # Map the desired dict_label to the base property it differentiates
    diff_map = {
        'Velocity radial COG': 'Center of gravity radial',
        'Velocity poloidal COG': 'Center of gravity poloidal',
        'Velocity radial centroid': 'Centroid radial',
        'Velocity poloidal centroid': 'Centroid poloidal',
        'Angular velocity ALI': 'Angle ALI',
        'Convexity diff': 'Convexity',
        'Solidity diff': 'Solidity',
        'Roundness diff': 'Roundness',
        'Total curvature diff': 'Total curvature',
        'Total bending energy diff': 'Total bending energy',
        'Area diff': 'Area',
        'Size radial diff': 'Size radial',
        'Size poloidal diff': 'Size poloidal',
        'Axes length minor diff': 'Axes length minor',
        'Axes length major diff': 'Axes length major',
        'Velocity radial position fit': 'Position radial fit',
        'Velocity poloidal position fit': 'Position poloidal fit',
        'Angular velocity angle fit': 'Angle fit',
        'Elongation fit diff': 'Elongation fit',
    }

    for struct in dataset.tracked_structures:
        if not struct or len(struct.time) < 2:
            continue
            
        rp = struct.regular_parameters
        dp = struct.differential_parameters
        
        # Dynamically create the dt array using the MetricArray engine
        dt_arr = np.diff(struct.time)
        dt_metric = MetricArray(value=dt_arr, dict_label='Sample time', plot_label=r'\Delta t', unit='s')
        
        # 1. Calculate standard derivatives (Velocity, Growth rates, etc.)
        for diff_key, reg_key in diff_map.items():
            if reg_key in rp:
                # V E C T O R I Z E D   M A T H !
                # This automatically triggers MetricArray.__sub__ and __truediv__
                delta = rp[reg_key][1:] - rp[reg_key][:-1]
                rate = delta / dt_metric
                
                # Keep the legacy dictionary key so plotting loops still find it
                rate.dict_label = diff_key
                
                # We optionally clean up the plot label for velocities so they don't get too long
                if 'Velocity' in diff_key or 'Angular' in diff_key:
                    base_label = rp[reg_key].plot_label.replace('$','')
                    rate.plot_label = f"$\\partial {base_label} / \\partial t$"
                    
                dp[diff_key] = rate

        # 2. Calculate Expansion Fractions (Area ratios)
        if 'Area' in rp:
            # Triggers MetricArray.__truediv__ and __pow__
            res = (rp['Area'][1:] / rp['Area'][:-1]) ** 0.5
            res.dict_label = 'Expansion fraction area'
            res.plot_label = '$f_{E,area}$'
            dp['Expansion fraction area'] = res
            
        if 'Axes length minor fit' in rp and 'Axes length major fit' in rp:
            area_fit = rp['Axes length minor fit'] * rp['Axes length major fit']
            res = (area_fit[1:] / area_fit[:-1]) ** 0.5
            res.dict_label = 'Expansion fraction axes fit'
            res.plot_label = '$f_{E,ellipse}$'
            dp['Expansion fraction axes fit'] = res
            
    return dataset


def correct_structure_angle(structure_1, structure_2):
    """
    Corrects unphysical angle wrapping (fringe jumps) between consecutive frames
    using raw floats.
    """
    if structure_1 is None or structure_2 is None:
        raise ValueError('Both structure_1 and structure_2 need to be defined.')

    keys_to_correct = ['Angle fit', 'Angle ALI']

    for key in keys_to_correct:
        if key in structure_1.regular_parameters and key in structure_2.regular_parameters:
            
            # BUG FIX: Extract raw floats natively, no .value!
            v1 = structure_1.regular_parameters[key]
            v2 = structure_2.regular_parameters[key]
            
            if not (np.isnan(v1) or np.isnan(v2)):
                corrected = fringe_jump_correction(np.asarray([v1, v2]), tolerance=0.5)
                
                # Directly update the raw float inside the dictionary!
                structure_2.regular_parameters[key] = corrected[1]
                
                # Sync the internal fit float so downstream calculations remain accurate
                if key == 'Angle fit':
                    structure_2._angle = corrected[1]
                    
    return structure_2