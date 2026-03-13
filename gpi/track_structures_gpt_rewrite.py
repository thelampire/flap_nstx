#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:48:55 2026

@author: mlampert
"""
#Core modules
import os
import copy
import cv2

import warnings
warnings.filterwarnings("ignore")

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')

from flap_nstx.tools import fringe_jump_correction

import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

import numpy as np
import pickle
#Plot settings for publications

wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
#Constants for the calculation
#Using the spatial calibration to find the actual velocities.
coeff_r=np.asarray([3.75, 0,    1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
coeff_z=np.asarray([0,    3.75, 70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm


def track_structures(frame_properties, settings):

    n_frames = len(frame_properties['structures'])
    highest_label = 0
    sample_time = frame_properties['Time'][1] - frame_properties['Time'][0]

    for i in range(1, n_frames):

        structures_prev = frame_properties['structures'][i-1]
        structures_curr = frame_properties['structures'][i]

        if not structures_prev:
            highest_label = handle_births(structures_curr, highest_label)
            continue

        if not structures_curr:
            mark_all_dead(structures_prev)
            continue

        ensure_labels(structures_prev, highest_label)

        score_matrix = compute_score_matrix(structures_prev,
                                            structures_curr,
                                            settings)

        overlap_matrix = assign_structures(score_matrix,
                                           settings.tracking_assignment)

        highest_label = process_merges(
            structures_prev,
            structures_curr,
            overlap_matrix,
            highest_label,
            sample_time,
            settings
        )

        process_splits(
            structures_prev,
            structures_curr,
            overlap_matrix,
            highest_label,
            sample_time,
            settings
        )

        compute_differential_quantities(
            structures_prev,
            structures_curr,
            overlap_matrix,
            sample_time,
            settings
        )

        update_frame_statistics(frame_properties, i)

    remove_orphan_tracks(frame_properties, settings)

    return frame_properties

def compute_score_matrix(structures_1, structures_2, settings):

    n1 = len(structures_1)
    n2 = len(structures_2)

    score_matrix = np.zeros((n1, n2))

    for j2, str2 in enumerate(structures_2):

        poly2 = str2['Polygon'].shapely_polygon

        for j1, str1 in enumerate(structures_1):

            poly1 = str1['Polygon'].shapely_polygon

            if not poly1.intersects(poly2):
                continue

            score = 0

            score += compute_iou(poly1, poly2) * settings.matrix_weight['iou']

            score += compute_cross_correlation(str1, str2) \
                     * settings.matrix_weight['cccf']

            score_matrix[j1, j2] = score

    return score_matrix

def assign_structures(score_matrix, method):

    overlap = np.zeros_like(score_matrix)

    if method == 'hungarian':

        rows, cols = linear_sum_assignment(score_matrix, maximize=True)

        overlap[rows, cols] = 1

    elif method == 'max_score':

        for i in range(score_matrix.shape[0]):
            if np.max(score_matrix[i]) > 0:
                overlap[i, np.argmax(score_matrix[i])] = 1

    return overlap

def handle_births(structures, highest_label):

    for s in structures:
        s['Label'] = highest_label + 1
        s['Born'] = True
        s['Died'] = False
        highest_label += 1

    return highest_label

def process_merges(structures_1,
                   structures_2,
                   overlap_matrix,
                   highest_label,
                   sample_time,
                   settings):

    n_str2 = overlap_matrix.shape[1]

    for j2 in range(n_str2):

        merging = overlap_matrix[:, j2]

        if np.sum(merging) == 0:

            structures_2[j2]['Label'] = highest_label + 1
            structures_2[j2]['Born'] = True
            highest_label += 1

        elif np.sum(merging) == 1:

            idx = np.where(merging == 1)[0][0]

            structures_2[j2]['Label'] = structures_1[idx]['Label']

            correct_structure_angle(structures_2[j2],
                                    structures_1[idx])

            calculate_differential_keys(structures_2[j2],
                                        structures_1[idx],
                                        sample_time,
                                        settings.fit_shape)

        else:

            highest_label = resolve_merge(structures_1,
                                          structures_2,
                                          merging,
                                          j2,
                                          highest_label,
                                          sample_time,
                                          settings)

    return highest_label

def process_splits(structures_1,
                   structures_2,
                   overlap_matrix,
                   highest_label,
                   sample_time,
                   settings):

    n_str1 = overlap_matrix.shape[0]

    for j1 in range(n_str1):

        splits = overlap_matrix[j1]

        if np.sum(splits) == 0:

            structures_1[j1]['Died'] = True

        elif np.sum(splits) > 1:

            resolve_split(structures_1,
                          structures_2,
                          splits,
                          j1,
                          highest_label,
                          sample_time,
                          settings)
            
def remove_orphan_tracks(frame_properties, settings):

    # move your orphan code here unchanged
    pass

def resolve_merge(structures_1,
                  structures_2,
                  merging_indices,
                  j_str2,
                  highest_label,
                  sample_time,
                  settings):

    ind_merge = np.where(merging_indices == 1)[0]

    # Select the strongest parent (highest intensity)
    ind_high = ind_merge[0]

    for ind in ind_merge:
        if structures_1[ind]['Intensity'] > structures_1[ind_high]['Intensity']:
            ind_high = ind

    # Assign label from dominant parent
    structures_2[j_str2]['Label'] = structures_1[ind_high]['Label']

    structures_2[j_str2] = correct_structure_angle(
        structure_2=structures_2[j_str2],
        structure_1=structures_1[ind_high]
    )

    structures_2[j_str2] = calculate_differential_keys(
        structure_2=structures_2[j_str2],
        structure_1=structures_1[ind_high],
        sample_time=sample_time,
        fit_shape=settings.fit_shape
    )

    # Register merge relationships
    for ind in ind_merge:

        structures_2[j_str2]['Parent'].append(structures_1[ind]['Label'])

        structures_1[ind]['Merges'] = True

        structures_1[ind]['Child'].append(structures_2[j_str2]['Label'])

    return highest_label

def resolve_split(structures_1,
                  structures_2,
                  splitting_indices,
                  j_str1,
                  highest_label,
                  sample_time,
                  settings):

    ind_split = np.where(splitting_indices == 1)[0]

    # Find dominant child (largest intensity)
    ind_high = ind_split[0]

    for ind in ind_split:
        if structures_2[ind]['Intensity'] > structures_2[ind_high]['Intensity']:
            ind_high = ind

    for ind in ind_split:

        if ind == ind_high:

            structures_2[ind]['Label'] = structures_1[j_str1]['Label']

            structures_2[ind] = correct_structure_angle(
                structure_2=structures_2[ind],
                structure_1=structures_1[j_str1]
            )

            structures_2[ind] = calculate_differential_keys(
                structure_2=structures_2[ind],
                structure_1=structures_1[j_str1],
                sample_time=sample_time,
                fit_shape=settings.fit_shape
            )

        else:

            highest_label += 1

            structures_2[ind]['Label'] = highest_label

        structures_2[ind]['Parent'].append(structures_1[j_str1]['Label'])

        structures_1[j_str1]['Child'].append(structures_2[ind]['Label'])

    structures_1[j_str1]['Splits'] = True

    return highest_label