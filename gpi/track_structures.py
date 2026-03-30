#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:42:31 2026

@author: mlampert
"""

#Core modules
import os
import copy
import cv2
from collections import Counter

import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')

from flap_nstx.tools import fringe_jump_correction

import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name = fn)
wd = flap.config.get_all_section('Module NSTX_GPI')['Working directory']

import numpy as np
import pickle
#Plot settings for publications

from shapely.ops import unary_union
from scipy.signal import correlate2d
from scipy.optimize import linear_sum_assignment

#Constants for the calculation
#Using the spatial calibration to find the actual velocities.
coeff_r = np.asarray([3.75, 0,    1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
coeff_z = np.asarray([0,    3.75, 70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm

def track_structures(frame_properties=None,
                     exp_id=None,
                     time_range=None,
                     tracking='weighted',
                     tracking_assignment='max_score',
                     max_gap=3,
                     matrix_weight=None,
                     smooth_contours=None,
                     fit_shape='ellipse',
                     prev_str_weighting='area',
                     calculate_rough_diff_velocities=False,
                     differential_keys=None,
                     weighting='area',
                     maxing='',
                     remove_orphans=True,
                     min_structure_lifetime=20,
                     nocalc=False,
                     recalc_tracking=False,
                     comment=None,
                     test=False,
                     ):
    """
    Main loop for tracking identified plasma structures across consecutive frames.

    This function sequentially processes all frames in `frame_properties`. It 
    determines structure continuity (birth, propagation, splitting, merging, death), 
    assigns unique tracking labels, optionally bridges tracking gaps, and calculates 
    differential kinematics. It can cache and reload results via pickle files.

    Args:
        frame_properties (dict): The master dictionary containing the frame data.
        exp_id (int): The experiment shot number.
        time_range (list): The time range [start, end] of the dataset.
        tracking (str): Tracking method ('overlap' or 'weighted').
        tracking_assignment (str): Assignment algorithm ('hungarian' or 'max_score').
        max_gap (int): Maximum number of frames a structure can "disappear" for 
            and still be recovered. Defaults to 3.
        matrix_weight (dict): Dictionary of weights for 'weighted' tracking.
        smooth_contours (int): Contour smoothing parameter for the filename comment.
        fit_shape (str): The geometric shape used for the fit parameters.
        prev_str_weighting (str): Weighting scheme for calculating differentials.
        calculate_rough_diff_velocities (bool): Toggles calculation of differential 
            velocities across frame gaps. Defaults to False.
        weighting (str): Weighting method for aggregating frame properties.
        maxing (str): Strategy to define the "maximum" structure.
        remove_orphans (bool): Toggles removal of short-lived structures.
        min_structure_lifetime (int): Minimum frames a structure must exist to survive.
        nocalc (bool): If True, attempts to load a cached pickle file.
        recalc_tracking (bool): Forces recalculation even if cache exists.
        comment (str): Initial string comment to append to the filename.

    Returns:
        dict: The updated `frame_properties` dictionary.
    """
    
    wd = flap.config.get_all_section('Module NSTX_GPI').get('Working directory', './')

    # --- 1. Filename & Caching Setup ---
    comment = comment or ""
    comment += f"_{tracking}_{tracking_assignment}_sm{smooth_contours}"
    if remove_orphans:
        comment += f"_LT{min_structure_lifetime}"

    pickle_filename = flap_nstx.tools.filename(exp_id=exp_id,
                                               working_directory=os.path.join(wd, 'processed_data'),
                                               time_range=time_range,
                                               purpose='tracking',
                                               comment=comment,
                                               extension='pickle')

    if not differential_keys and 'derived' in frame_properties:
        differential_keys = list(frame_properties['derived'].keys())

    # Check cache
    if nocalc and os.path.exists(pickle_filename):
        try:
            with open(pickle_filename, 'rb') as f:
                # Test load to verify integrity
                pickle.load(f)
        except Exception:
            print(f"Failed to load {pickle_filename}. Recalculating.")
            nocalc = False
    else:
        if nocalc: print(f"File {pickle_filename} does not exist. Recalculating.")
        nocalc = False

    # --- 2. Main Tracking Loop ---
    if not nocalc or recalc_tracking:
        print("\n\nCalculating structure tracking.")
        
        highest_label = 0
        n_frames = len(frame_properties['structures'])
        sample_time = frame_properties['Time'][1] - frame_properties['Time'][0]
        
        for i_frames in range(1, n_frames):
            structures_1 = frame_properties['structures'][i_frames - 1]
            structures_2 = frame_properties['structures'][i_frames]

            # Case A: Previous frame had structures, current frame is empty (Mass Death)
            if structures_1 and not structures_2:
                for s1 in structures_1:
                    s1['Label'] = highest_label + 1
                    s1['Born'], s1['Died'] = False, True
                    highest_label += 1

            # Case B: Previous frame empty, current frame has structures (Mass Birth)
            elif not structures_1 and structures_2:
                for s2 in structures_2:
                    s2['Label'] = highest_label + 1
                    s2['Born'], s2['Died'] = True, False
                    highest_label += 1

            # Case C: Both frames have structures (Standard Tracking)
            elif structures_1 and structures_2:
                
                # Catch any unassigned labels in the previous frame
                for s1 in structures_1:
                    if s1.get('Label') is None:
                        highest_label += 1
                        s1['Label'] = highest_label
                        s1['Born'], s1['Died'] = True, False

                # 1. Build overlap matrix
                str_overlap_matrix = _calculate_str_overlap_matrix(structures_1, structures_2, 
                                                                   tracking, matrix_weight, 
                                                                   tracking_assignment, test)

                # 2. Process standard 1-to-1 links and Merges
                structures_1, structures_2, highest_label = _process_structure_merging(structures_1, structures_2, 
                                                                                       str_overlap_matrix, 
                                                                                       gap=1, 
                                                                                       sample_time=sample_time, 
                                                                                       fit_shape=fit_shape, 
                                                                                       highest_label=highest_label)

                # 3. Process Splits
                structures_1, structures_2, highest_label = _process_structure_splitting(structures_1, structures_2, 
                                                                                         str_overlap_matrix, 
                                                                                         sample_time=sample_time, 
                                                                                         fit_shape=fit_shape, 
                                                                                         highest_label=highest_label)

                # 4. Gap Recovery (Handling missed segmentations)
                if i_frames > max_gap:
                    for ind_gap in range(2, max_gap + 1):
                        
                        # Only look backward if new structures appeared out of nowhere
                        n_born = sum(1 for s in structures_2 if s.get('Born'))
                        if n_born > 0:
                            structures_before_gap = frame_properties['structures'][i_frames - ind_gap]
                            if structures_before_gap:
                                overlap_gap_matrix = _calculate_str_overlap_matrix(structures_before_gap, structures_2, 
                                                                                   tracking, 
                                                                                   matrix_weight, 
                                                                                   tracking_assignment, 
                                                                                   test)
                                
                                structures_before_gap, structures_2, highest_label = _process_structure_merging(structures_before_gap, 
                                                                                                                structures_2, 
                                                                                                                overlap_gap_matrix, 
                                                                                                                gap=ind_gap, 
                                                                                                                sample_time=sample_time * ind_gap, 
                                                                                                                fit_shape=fit_shape, 
                                                                                                                highest_label=highest_label)

                # 5. Calculate global frame-level differentials
                if calculate_rough_diff_velocities:
                    n_str1_overlap = np.zeros(len(structures_1))
                    frame_properties = _calculate_rough_differential_velocities(frame_properties, 
                                                                                structures_1, structures_2, 
                                                                                fit_shape, 
                                                                                sample_time,
                                                                                prev_str_weighting, 
                                                                                differential_keys, 
                                                                                n_str1_overlap, 
                                                                                i_frames, 
                                                                                weighting, 
                                                                                maxing)
                else:
                    for key in differential_keys:
                        frame_properties['derived'][key]['avg'][i_frames] = np.nan
                        frame_properties['derived'][key]['max'][i_frames] = np.nan

        # --- 3. Post-Processing & Save ---
        if remove_orphans:
            frame_properties = _remove_orphans(frame_properties, n_frames, test, min_structure_lifetime)

        with open(pickle_filename, 'wb') as f:
            pickle.dump(frame_properties, f)

    else:
        print('--- Loading tracking data from the pickle file ---')
        with open(pickle_filename, 'rb') as f:
            frame_properties = pickle.load(f)

    return frame_properties


def _calculate_str_overlap_matrix(structures_1,
                                  structures_2,
                                  tracking=None,
                                  matrix_weight=None,
                                  tracking_assignment=None,
                                  test=False):
    """
    Constructs a binary matrix representing tracking overlap between two frames.

    This function determines which structures in the previous frame (structures_1) 
    map to which structures in the current frame (structures_2). It supports 
    pure geometric overlap or weighted tracking (using IoU/CCCF scores combined 
    with Hungarian or max-score assignment algorithms).

    Example of `str_overlap_matrix` (Rows = structures_1, Cols = structures_2):
        |0, 0, 0, 1| -> lives (propagates to structure 3)
        |1, 0, 1, 1| -> splits into three (structures 0, 2, 3)
        |0, 0, 1, 0| -> lives (propagates to structure 2)
        |0, 0, 0, 0| -> dies
         ^  ^  ^  ^
         |  |  |  |--- merges into
         |  |  |------ merges into
         |  |--------- is born
         |------------ split into

    Args:
        structures_1 (list of dict): Structures from the previous frame.
        structures_2 (list of dict): Structures from the current frame.
        tracking (str, optional): Method for tracking ('overlap' or 'weighted').
        matrix_weight (dict, optional): Weights for the score matrix if using 
            'weighted' tracking.
        tracking_assignment (str, optional): Algorithm for resolving assignments 
            ('hungarian' or 'max_score').
        test (bool, optional): If True, prints diagnostic matrices. Defaults to False.

    Raises:
        ValueError: If an unknown tracking or tracking_assignment method is provided.

    Returns:
        np.ndarray: A 2D binary matrix of shape (len(structures_1), len(structures_2)).
    """
    
    n_str1 = len(structures_1)
    n_str2 = len(structures_2)
    
    str_overlap_matrix = np.zeros((n_str1, n_str2))

    # Short-circuit if either frame has no structures
    if n_str1 == 0 or n_str2 == 0:
        return str_overlap_matrix

    # --- Method 1: Pure Geometric Overlap ---
    if tracking == 'overlap':
        for j_str1, s1 in enumerate(structures_1):
            path1 = s1['Half path']
            for j_str2, s2 in enumerate(structures_2):
                path2 = s2['Half path']
                
                if path2.intersects_path(path1) or path2.contains_path(path1):
                    str_overlap_matrix[j_str1, j_str2] = 1.0

    # --- Method 2: Weighted Tracking (IoU & CCCF Scores) ---
    elif tracking == 'weighted':
        score_matrix = calculate_score_matrix(structures_1, structures_2, matrix_weight)

        if tracking_assignment == 'hungarian':
            # Hungarian algorithm finds the global optimal 1-to-1 assignment
            row_indices, col_indices = linear_sum_assignment(score_matrix, maximize=True)
            str_overlap_matrix[row_indices, col_indices] = 1.0

        elif tracking_assignment == 'max_score':
            # Greedy assignment: each parent picks its highest-scoring child
            for ind_row in range(n_str1):
                row_scores = score_matrix[ind_row, :]
                
                # Only assign if a valid, positive score exists
                if np.max(row_scores) > 0:
                    best_match_idx = np.argmax(row_scores)
                    str_overlap_matrix[ind_row, best_match_idx] = 1.0

            if test:
                print("Score Matrix:\n", score_matrix)
                print("Overlap Matrix:\n", str_overlap_matrix, "\n")

        else:
            raise ValueError(f"Tracking assignment '{tracking_assignment}' is not available.")

    else:
        raise ValueError(f"Tracking method '{tracking}' is unavailable.")
        
    return str_overlap_matrix

def calculate_score_matrix(structures_1, 
                           structures_2, 
                           matrix_weight, 
                           coeff_r=None, 
                           coeff_z=None):
    """
    Calculates an overlap and similarity score matrix between two sets of structures.

    This function compares structures from consecutive frames to determine tracking 
    correspondence. The final score is a weighted combination of their spatial 
    Intersection over Union (IoU) and the maximum of their 2D Cross-Correlation 
    Coefficient Function (CCCF).

    Args:
        structures_1 (list of dict): Structures from the previous frame.
        structures_2 (list of dict): Structures from the current frame.
        matrix_weight (dict): Weights for the scoring metrics. Expected keys 
            are 'iou' and 'cccf' (e.g., {'iou': 1.0, 'cccf': 0.0}).
        coeff_r (np.ndarray, optional): Radial coordinate calibration coefficients, 
            used as a fallback if pixel data is missing. Defaults to None.
        coeff_z (np.ndarray, optional): Poloidal coordinate calibration coefficients, 
            used as a fallback if pixel data is missing. Defaults to None.

    Returns:
        np.ndarray: A 2D matrix of shape (len(structures_1), len(structures_2)) 
        containing the calculated similarity scores.
    """
    n_str1 = len(structures_1)
    n_str2 = len(structures_2)
    
    score_matrix = np.zeros((n_str1, n_str2))

    weight_iou = matrix_weight.get('iou', 0)
    weight_cccf = matrix_weight.get('cccf', 0)

    for j_str2, s2 in enumerate(structures_2):
        poly2_obj = s2['Polygon']
        str2_polygon = poly2_obj.shapely_polygon
        
        for j_str1, s1 in enumerate(structures_1):
            poly1_obj = s1['Polygon']
            str1_polygon = poly1_obj.shapely_polygon
            
            try:
                # --- 1. Fix missing pixel data (using fallback coefficients) ---
                if poly1_obj.x_data_pix is None and coeff_r is not None and coeff_z is not None:
                    poly1_obj.x_data_pix = np.round((poly1_obj.x_data - coeff_r[2]) / coeff_r[0]).astype(int)
                    poly1_obj.y_data_pix = np.round((poly1_obj.y_data - coeff_z[2]) / coeff_z[1]).astype(int)

                if poly2_obj.x_data_pix is None and coeff_r is not None and coeff_z is not None:
                    poly2_obj.x_data_pix = np.round((poly2_obj.x_data - coeff_r[2]) / coeff_r[0]).astype(int)
                    poly2_obj.y_data_pix = np.round((poly2_obj.y_data - coeff_z[2]) / coeff_z[1]).astype(int)

                # --- 2. Calculate Similarity Scores ---
                if str2_polygon.intersects(str1_polygon):
                    score = 0.0
                    
                    # Metric A: Intersection over Union (IoU)
                    if weight_iou > 0:
                        intersection_area = str1_polygon.intersection(str2_polygon).area
                        union_area = str1_polygon.union(str2_polygon).area
                        score += (intersection_area / union_area) * weight_iou

                    # Metric B: Cross-Correlation Coefficient Function (CCCF)
                    if weight_cccf > 0:
                        # Establish a joint bounding box
                        x_min = min(poly1_obj.x_data_pix.min(), poly2_obj.x_data_pix.min())
                        x_max = max(poly1_obj.x_data_pix.max(), poly2_obj.x_data_pix.max())
                        y_min = min(poly1_obj.y_data_pix.min(), poly2_obj.y_data_pix.min())
                        y_max = max(poly1_obj.y_data_pix.max(), poly2_obj.y_data_pix.max())

                        # Initialize local matrices
                        str1_matrix = np.zeros((x_max - x_min + 1, y_max - y_min + 1))
                        str2_matrix = np.zeros((x_max - x_min + 1, y_max - y_min + 1))

                        # Vectorized matrix population
                        str1_matrix[poly1_obj.x_data_pix - x_min, poly1_obj.y_data_pix - y_min] = poly1_obj.data
                        str2_matrix[poly2_obj.x_data_pix - x_min, poly2_obj.y_data_pix - y_min] = poly2_obj.data

                        # Mean centering
                        str1_matrix -= np.mean(str1_matrix)
                        str2_matrix -= np.mean(str2_matrix)
                        
                        ccf_matrix = correlate2d(str1_matrix, str2_matrix)
                        norm_factor = np.sqrt(np.sum(str1_matrix**2) * np.sum(str2_matrix**2))
                        
                        # Apply normalization and evaluate the maximum correlation
                        if norm_factor > 0:
                            cccf_matrix = ccf_matrix / norm_factor
                            max_cccf = np.max(cccf_matrix)
                            
                            if max_cccf > 1.0001:  # Slight tolerance for float precision
                                raise ValueError(f'Cross-correlation matrix value exceeded 1: {max_cccf}')
                                
                            score += max_cccf * weight_cccf

                    score_matrix[j_str1, j_str2] = score
                else:
                    score_matrix[j_str1, j_str2] = 0.0

            except Exception as e:
                print(f"Exception at score calculation (str1={j_str1}, str2={j_str2}): {e}")
                score_matrix[j_str1, j_str2] = 0.0
                
    return score_matrix

def _process_structure_merging(structures_1, 
                               structures_2,
                               str_overlap_matrix,
                               gap=1,
                               sample_time=2.5e-6,
                               fit_shape='Ellipse',
                               highest_label=None):
    """
    Processes tracking logic for structure birth, 1-to-1 propagation, and merging.

    This function iterates through structures in the current frame (structures_2).
    It handles cases where a structure appears from nowhere (birth), propagates 
    normally, recovers from a tracking gap, or forms from the merging of multiple 
    structures from the previous frame. It also handles the complex edge case 
    where structures merge and split simultaneously.

    Args:
        structures_1 (list of dict): Structures from the previous frame (N-1).
        structures_2 (list of dict): Structures from the current frame (N).
        str_overlap_matrix (np.ndarray): 2D boolean/integer matrix where rows map 
            to structures_1 and columns map to structures_2.
        gap (int, optional): The frame gap being evaluated. Defaults to 1.
        sample_time (float, optional): Time delta between frames. Defaults to 2e-6.
        fit_shape (str, optional): Shape used for geometric fitting. Defaults to 'Ellipse'.
        highest_label (int): The current maximum integer used for tracking labels.

    Returns:
        tuple: (structures_1, structures_2, highest_label) updated with merge logic.
    """
    
    for j_str2, s2 in enumerate(structures_2):
        # Find which structures in frame 1 overlap with this structure in frame 2
        parent_indices = np.where(str_overlap_matrix[:, j_str2] == 1)[0]
        num_parents = len(parent_indices)
        
        # --- 1. Structure Birth (No overlap) ---
        if num_parents == 0 and gap == 1:
            highest_label += 1
            s2['Label'] = highest_label
            s2['Born'] = True

        # --- 2. Standard 1-to-1 Propagation ---
        elif num_parents == 1:
            p_idx = parent_indices[0]
            s1 = structures_1[p_idx]
            
            # Normal propagation
            if gap == 1 and np.sum(str_overlap_matrix[p_idx, :]) == 1:
                s2['Label'] = s1['Label']
                s2 = correct_structure_angle(structure_2=s2, structure_1=s1)
                s2 = calculate_differential_keys(structure_2=s2, structure_1=s1,
                                                 sample_time=sample_time, fit_shape=fit_shape)

            # Gap Recovery: Re-link an orphaned structure
            elif gap > 1 and s2.get('Born', False) and s1.get('Died', False):
                if np.sum(str_overlap_matrix[p_idx, :]) == 1:
                    s2['Label'] = s1['Label']
                    s2 = correct_structure_angle(structure_2=s2, structure_1=s1)
                    s2 = calculate_differential_keys(structure_2=s2, structure_1=s1,
                                                     sample_time=sample_time * gap, fit_shape=fit_shape)
                    
                    s2['Born'] = False
                    s1['Died'] = False

        # --- 3. Structure Merging ---
        elif num_parents > 1 and gap == 1:
            
            # 3a. Pure Merge (The parents ONLY overlap with this one child)
            if np.sum(str_overlap_matrix[parent_indices, :]) == num_parents:
                
                # The dominant parent passes its identity to the child
                dominant_p_idx = max(parent_indices, key=lambda idx: structures_1[idx]['Intensity'])
                dom_s1 = structures_1[dominant_p_idx]
                
                s2['Label'] = dom_s1['Label']
                s2 = correct_structure_angle(structure_2=s2, structure_1=dom_s1)
                s2 = calculate_differential_keys(structure_2=s2, structure_1=dom_s1,
                                                 sample_time=sample_time, fit_shape=fit_shape)
                
                for p_idx in parent_indices:
                    s1 = structures_1[p_idx]
                    s2['Parent'].append(s1['Label'])
                    s1['Child'].append(s2['Label'])
                    s1['Merges'] = True

            # 3b. Complex Event: Simultaneous Merge and Split
            else:
                # Find dominant parent among the cluster
                dominant_p_idx = max(parent_indices, key=lambda idx: structures_1[idx]['Intensity'])
                dom_s1 = structures_1[dominant_p_idx]

                # Find dominant child across ALL children connected to these parents
                all_child_indices = set()
                for p_idx in parent_indices:
                    for c_idx in np.where(str_overlap_matrix[p_idx, :] == 1)[0]:
                        all_child_indices.add(c_idx)
                        
                dominant_c_idx = max(all_child_indices, key=lambda idx: structures_2[idx]['Intensity'])
                
                for p_idx in parent_indices:
                    s1 = structures_1[p_idx]
                    s1['Splits'] = np.sum(str_overlap_matrix[p_idx, :]) > 1
                    s1['Merges'] = True
                    
                    children_of_p = np.where(str_overlap_matrix[p_idx, :] == 1)[0]
                    for c_idx in children_of_p:
                        child = structures_2[c_idx]
                        
                        # Only link the primary dominant parent to the primary dominant child
                        if c_idx == dominant_c_idx and p_idx == dominant_p_idx:
                            child['Label'] = dom_s1['Label']
                            child = correct_structure_angle(structure_2=child, structure_1=dom_s1)
                            child = calculate_differential_keys(structure_2=child, structure_1=dom_s1,
                                                                sample_time=sample_time, fit_shape=fit_shape)
                        # Assign new labels to the non-dominant children if they lack one
                        elif child.get('Label') is None:
                            highest_label += 1
                            child['Label'] = highest_label
                            
                        # Safely document the lineage (using append to avoid the overwrite bug)
                        if dom_s1['Label'] not in child['Parent']:
                            child['Parent'].append(dom_s1['Label'])
                        if child['Label'] not in s1['Child']:
                            s1['Child'].append(child['Label'])

    return structures_1, structures_2, highest_label

def _process_structure_splitting(structures_1, 
                                 structures_2,
                                 str_overlap_matrix,
                                 sample_time=2.5e-6,
                                 fit_shape='Ellipse',
                                 highest_label=None):
    """
    Processes tracking logic when a single structure splits into multiple structures.

    This function iterates through structures in the previous frame (structures_1). 
    If a structure overlaps with multiple distinct structures in the current frame 
    (structures_2), it is flagged as a split. The dominant child (highest intensity) 
    inherits the parent's tracking label and calculates differential kinematics. 
    The other children are assigned new, unique tracking labels.

    Args:
        structures_1 (list of dict): Structures from the previous frame (N-1).
        structures_2 (list of dict): Structures from the current frame (N).
        str_overlap_matrix (np.ndarray): 2D boolean/integer matrix where rows map 
            to structures_1 and columns map to structures_2. A value of 1 indicates overlap.
        sample_time (float, optional): Time delta between frames. Defaults to 2e-6.
        fit_shape (str, optional): Shape used for geometric fitting. Defaults to 'Ellipse'.
        highest_label (int): The current maximum integer used for tracking labels.

    Returns:
        tuple: (structures_1, structures_2, highest_label) updated with split logic.
    """
    
    for j_str1, s1 in enumerate(structures_1):
        # Extract 1D array of overlaps for this specific parent structure
        overlaps = str_overlap_matrix[j_str1, :]
        num_overlaps = np.sum(overlaps)

        # Structure dies (no overlaps in the next frame)
        if num_overlaps == 0:
            s1['Died'] = True

        # Pure 1-to-1 tracking is handled elsewhere, so we only look for splits (>1)
        elif num_overlaps > 1:
            child_indices = np.where(overlaps == 1)[0]

            # Verify this is a 'pure' split (none of these children overlap with any other parent)
            if np.sum(str_overlap_matrix[:, child_indices]) == num_overlaps:
                
                # The dominant child is the one with the maximum intensity
                dominant_idx = max(child_indices, key=lambda idx: structures_2[idx]['Intensity'])

                for idx in child_indices:
                    s2 = structures_2[idx]
                    
                    # Dominant child inherits the parent's label and calculates kinematics
                    if idx == dominant_idx:
                        s2['Label'] = s1['Label']
                        s2 = correct_structure_angle(structure_2=s2, 
                                                     structure_1=s1)
                        s2 = calculate_differential_keys(structure_2=s2,
                                                         structure_1=s1,
                                                         sample_time=sample_time,
                                                         fit_shape=fit_shape)
                    # Secondary children are born anew with new labels
                    else:
                        highest_label += 1
                        s2['Label'] = highest_label

                    # Document the family lineage
                    s2['Parent'].append(s1['Label'])
                    s1['Child'].append(s2['Label'])

                s1['Splits'] = True
            
            else:
                # This handles the complex case where structures are both merging 
                # and splitting simultaneously. This is passed down to the merging logic.
                pass
                
    return structures_1, structures_2, highest_label

def _calculate_rough_differential_velocities(frame_properties,
                                             structures_1, 
                                             structures_2, 
                                             fit_shape, 
                                             sample_time, 
                                             prev_str_weighting, 
                                             differential_keys, 
                                             n_str1_overlap,
                                             i_frames,
                                             weighting, 
                                             maxing):
    """
    Calculates differential kinematics between intersecting structures in consecutive frames.
    
    This function identifies overlapping structures between frame N-1 and frame N. 
    It computes the relative change in velocities, areas, and angles, weighting 
    the contributions if a single structure in frame N overlaps with multiple 
    structures in frame N-1.
    """
    
    n_str1 = len(structures_1)
    n_str2 = len(structures_2)
    
    if n_str1 == 0 or n_str2 == 0:
        return frame_properties

    # Pre-calculate intensities for structures_1 to fix the max_intensity bug
    intensities_1 = np.array([s.get('Intensity', 0) for s in structures_1])
    max_int_idx_1 = np.argmax(intensities_1) if n_str1 > 0 else -1

    # --- 1. Calculate Differentials per Structure ---
    for s2 in structures_2:
        # Initialize empty lists for gathering overlapping differentials
        for key in differential_keys:
            s2[key] = []

        prev_weights = []

        # Check the new frame for overlap with the old frame
        for j_str1, s1 in enumerate(structures_1):
            if s2['Half path'].intersects_path(s1['Half path']):
                
                # Assign overlap weights
                if prev_str_weighting == 'number':
                    prev_weights.append(1.0)
                elif prev_str_weighting == 'intensity':
                    prev_weights.append(s1['Intensity'])
                elif prev_str_weighting == 'area':
                    prev_weights.append(s1['Area'])
                elif prev_str_weighting == 'max_intensity':
                    prev_weights.append(1.0 if j_str1 == max_int_idx_1 else 0.0)

                # Aliases for clean math
                p1, p2 = s1['Polygon'], s2['Polygon']
                f1, f2 = s1[fit_shape], s2[fit_shape]

                # Linear Velocities
                s2['Velocity radial COG'].append((p2.center_of_gravity[0] - p1.center_of_gravity[0]) / sample_time)
                s2['Velocity poloidal COG'].append((p2.center_of_gravity[1] - p1.center_of_gravity[1]) / sample_time)
                
                s2['Velocity radial centroid'].append((p2.centroid[0] - p1.centroid[0]) / sample_time)
                s2['Velocity poloidal centroid'].append((p2.centroid[1] - p1.centroid[1]) / sample_time)
                
                s2['Velocity radial position'].append((f2.center[0] - f1.center[0]) / sample_time)
                s2['Velocity poloidal position'].append((f2.center[1] - f1.center[1]) / sample_time)

                # Expansion Rates
                s2['Expansion fraction area'].append(np.sqrt(p2.area / p1.area))
                axes_ratio = (f2.axes_length[0] * f2.axes_length[1]) / (f1.axes_length[0] * f1.axes_length[1])
                s2['Expansion fraction axes'].append(np.sqrt(axes_ratio))

                # Angular Velocities
                s2['Angular velocity angle'].append((f2.angle - f1.angle) / sample_time)
                s2['Angular velocity ALI'].append((p2.principal_axes_angle - p1.principal_axes_angle) / sample_time)

                n_str1_overlap[j_str1] += 1.0

        # Resolve the lists into single weighted averages for this specific s2 structure
        prev_weights = np.asarray(prev_weights)
        weight_sum = np.sum(prev_weights)
        
        if weight_sum > 0:
            prev_weights /= weight_sum

        for key in differential_keys:
            if len(s2[key]) > 0:
                s2[key] = np.sum(np.asarray(s2[key]) * prev_weights)
            else:
                s2[key] = np.nan

    # --- 2. Frame Property Aggregation ---
    # Extract properties for weighting the frame averages
    areas_2 = np.array([s['Area'] for s in structures_2])
    intensities_2 = np.array([s['Intensity'] for s in structures_2])
    
    area_sum = np.sum(areas_2)
    int_sum = np.sum(intensities_2)

    # Determine frame-level weights
    if weighting == 'number':
        frame_weights = np.ones(n_str2) / n_str2
    elif weighting == 'intensity':
        frame_weights = intensities_2 / int_sum if int_sum > 0 else np.ones(n_str2) / n_str2
    elif weighting == 'area':
        frame_weights = areas_2 / area_sum if area_sum > 0 else np.ones(n_str2) / n_str2
    else:
        frame_weights = np.ones(n_str2) / n_str2

    # Determine the "maximum" structure index
    ind_max = np.argmax(areas_2 if maxing == 'area' else intensities_2)

    # Append to the master frame properties dictionary
    for key in differential_keys:
        
        # Calculate Weighted Average
        avg_val = 0.0
        for j_str2, s2 in enumerate(structures_2):
            val = s2.get(key, np.nan)
            if not np.isnan(val):
                avg_val += val * frame_weights[j_str2]
                
        try:
            frame_properties['derived'][key]['avg'][i_frames] = avg_val
            frame_properties['derived'][key]['max'][i_frames] = structures_2[ind_max].get(key, np.nan)
        except KeyError:
            # Silently pass if the key hasn't been initialized in frame_properties['derived']
            pass

    return frame_properties

def _remove_orphans(frame_properties,
                    n_frames,
                    test,
                    min_structure_lifetime):
    """
    Removes short-lived 'orphan' structures and re-indexes remaining labels.

    This function sweeps the tracked structures across all frames to tally 
    their lifespans. Any structure trajectory that exists for fewer frames than 
    `min_structure_lifetime` is deleted. The remaining valid structures have 
    their tracking labels re-indexed to form a continuous integer sequence 
    starting from 0.

    Args:
        frame_properties (dict): The master dictionary containing all frame structures.
        n_frames (int): Total number of frames in the dataset.
        test (bool): If True, prints diagnostic tracking data.
        min_structure_lifetime (int): Minimum number of frames a structure must 
            exist to be kept.

    Returns:
        dict: The updated `frame_properties` dictionary with orphans removed 
        and labels re-indexed.
    """
    
    # --- 1. Tally label frequencies across all frames ---
    label_counts = Counter()
    for i_frames in range(n_frames):
        structures = frame_properties['structures'][i_frames]
        if structures is not None:
            for s in structures:
                label = s.get('Label')
                if label is None:
                    print(f"Warning: Label is None in frame {i_frames} (analyze_gpi_structures line 1174)")
                else:
                    label_counts[label] += 1

    if test: 
        print(f"Label frequencies: {label_counts}")

    # --- 2. Determine survivors and build a fast O(1) mapping dictionary ---
    # Example mapping: {old_label_4: new_label_0, old_label_9: new_label_1, ...}
    label_map = {}
    new_index = 0
    for label, count in label_counts.items():
        if count >= min_structure_lifetime:
            label_map[label] = new_index
            new_index += 1

    # --- 3. Filter out orphans and remap surviving labels in a single pass ---
    for i_frames in range(n_frames):
        structures = frame_properties['structures'][i_frames]
        
        if structures is not None:
            valid_structures = []
            
            for s in structures:
                old_label = s.get('Label')
                
                # If the old label is in our map, it survived! Update and keep it.
                if old_label in label_map:
                    s['Label'] = label_map[old_label]
                    valid_structures.append(s)
            
            # Replace the old list with the cleaned, valid list
            frame_properties['structures'][i_frames] = valid_structures

    return frame_properties

#Wrapper function for calculating differential key results.
def calculate_differential_keys(structure_2 = None,
                                structure_1 = None,
                                sample_time = None,
                                fit_shape = 'Ellipse',
                                ):

    """
    Calculates time-differential kinematic properties between two consecutive structures.

    This function compares a structure from the current frame (`structure_2`) to its 
    linked predecessor in the previous frame (`structure_1`). It computes radial and 
    poloidal velocities, expansion fractions (area growth), and angular velocities. 
    The calculated properties are appended directly to the `structure_2` dictionary.

    Args:
        structure_2 (dict): The structure data for the current frame.
        structure_1 (dict): The structure data for the previous frame.
        sample_time (float): The time delta (in seconds or samples) between the 
            two frames.
        fit_shape (str, optional): The geometric shape used for the fit parameters 
            (e.g., 'Ellipse' or 'Gaussian'). Defaults to 'Ellipse'.

    Raises:
        ValueError: If `structure_1`, `structure_2`, or a valid `sample_time` are missing.

    Returns:
        dict: The updated `structure_2` dictionary containing the new differential keys.
    """
    
    structure_2['Velocity radial COG'] = (structure_2['Polygon'].center_of_gravity[0]-
                                          structure_1['Polygon'].center_of_gravity[0])/sample_time
    structure_2['Velocity poloidal COG'] = (structure_2['Polygon'].center_of_gravity[1]-
                                            structure_1['Polygon'].center_of_gravity[1])/sample_time
    structure_2['Velocity radial centroid'] = (structure_2['Polygon'].centroid[0]-
                                               structure_1['Polygon'].centroid[0])/sample_time
    structure_2['Velocity poloidal centroid'] = (structure_2['Polygon'].centroid[1]-
                                                 structure_1['Polygon'].centroid[1])/sample_time
    structure_2['Velocity radial position'] = (structure_2[fit_shape].center[0]-
                                               structure_1[fit_shape].center[0])/sample_time
    structure_2['Velocity poloidal position'] = (structure_2[fit_shape].center[1]-
                                                 structure_1[fit_shape].center[1])/sample_time
    structure_2['Expansion fraction area'] = np.sqrt(structure_2['Polygon'].area/
                                                     structure_1['Polygon'].area)
    structure_2['Expansion fraction axes'] = np.sqrt(structure_2[fit_shape].axes_length[0]/
                                                     structure_1[fit_shape].axes_length[0]*
                                                     structure_2[fit_shape].axes_length[1]/
                                                     structure_1[fit_shape].axes_length[1])

    structure_2['Angular velocity angle'] = (structure_2['Angle']-
                                             structure_1['Angle'])/sample_time
    try:
        structure_1['Angle of least inertia']
    except:
        #Not fringe jump corrected
        structure_1['Angle of least inertia'] = structure_1['Polygon'].principal_axes_angle

    structure_2['Angular velocity ALI'] = (structure_2['Angle of least inertia']-
                                           structure_1['Angle of least inertia'])/sample_time
    return structure_2



def correct_structure_angle(structure_2=None,
                            structure_1=None):
    """
    Corrects unphysical angle wrapping (fringe jumps) between consecutive frames.

    When tracking structures between frames, the calculated angle (e.g., from 
    an ellipse fit or polygon inertia) might unphysically jump due to phase 
    wrapping (e.g., flipping abruptly from +pi/2 to -pi/2). This function compares 
    the current structure's angle to the previous one and unwraps it if necessary 
    to maintain a continuous trajectory.

    Args:
        structure_2 (dict): The structure data for the current frame.
        structure_1 (dict): The structure data for the previous frame.

    Raises:
        ValueError: If either structure_1 or structure_2 is not provided.

    Returns:
        dict: The updated `structure_2` dictionary with corrected angles.
    """
    if structure_1 is None or structure_2 is None:
        raise ValueError('Both structure_1 and structure_2 need to be defined.')

    keys_to_correct = ['Angle', 'Angle of least inertia']

    for key in keys_to_correct:
        
        # 1. Try to extract angles directly from the dictionary keys
        if key in structure_1 and key in structure_2:
            angle_1 = structure_1[key]
            angle_2 = structure_2[key]
            
        # 2. Fallback: Extract the angle from the Polygon object
        else:
            try:
                angle_1 = structure_1['Polygon'].principal_axes_angle
                angle_2 = structure_2['Polygon'].principal_axes_angle
            except (KeyError, AttributeError):
                # print(f'No valid Polygon data for {key} --> skipping fringe jump correction.')
                continue
                
        # 3. Skip correction if the angle is undefined (NaN)
        if np.isnan(angle_1) or np.isnan(angle_2):
            continue

        # 4. Apply fringe jump correction
        data = np.asarray([angle_1, angle_2])
        corrected_data = fringe_jump_correction(data, tolerance=0.5)
        
        # Update the current structure's angle
        structure_2[key] = corrected_data[1]

    # Note: Putting the angles between +-pi/2 is intentionally omitted here, 
    # as structures do not have a specific vector direction.

    return structure_2