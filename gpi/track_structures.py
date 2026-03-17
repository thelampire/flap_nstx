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

def track_structures(frame_properties = None,
                     exp_id = None,
                     time_range = None,
                     
                     tracking = 'weighted',
                     tracking_assignment = 'max_score',
                     
                     max_gap = 3,                                               #allow tracking over gaps by introducing memory weights (max_gap=1 means consecutive frames.)
                     
                     matrix_weight = None,
                     
                     smooth_contours = None,
                     fit_shape = 'ellipse',
                     prev_str_weighting = 'area',
                     calculate_rough_diff_velocities = False,
                     differential_keys = None,
                     weighting = 'area',
                     maxing = '',
                     remove_orphans = True,
                     min_structure_lifetime = 20,
                     
                     
                     nocalc = False,
                     recalc_tracking = False,
                     
                     comment = None,
                     
                     test = False,
                     ):
    
    comment += '_'+tracking
    comment += '_'+tracking_assignment
    comment += '_sm'+str(smooth_contours)

    if remove_orphans:
        comment += '_LT'+str(min_structure_lifetime)

    pickle_filename_tracking = flap_nstx.tools.filename(exp_id = exp_id,
                                                        working_directory = wd+'/processed_data',
                                                        time_range = time_range,
                                                        purpose = 'tracking',
                                                        comment = comment,
                                                        extension = 'pickle')

    differential_keys = list(frame_properties['derived'].keys())

    if os.path.exists(pickle_filename_tracking) and nocalc:
        try:
            pickle.load(open(pickle_filename_tracking, 'rb'))
        except:
            print(pickle_filename_tracking)
            print('The pickle file cannot be loaded. Recalculating the results.')
            nocalc = False
            
    elif nocalc:
        print(pickle_filename_tracking)
        print('The pickle file does not exist. Recalculating the results.')
        nocalc = False

    if (not nocalc) or recalc_tracking:
        print("Calculating structure tracking.")
        #Structure tracking
        highest_label = 0
        n_frames = len(frame_properties['structures'])
    
        sample_time = frame_properties['Time'][1]-frame_properties['Time'][0]
        for i_frames in range(1,n_frames):

            structures_1 = frame_properties['structures'][i_frames-1]
            structures_2 = frame_properties['structures'][i_frames]
            
            if structures_1 and not structures_2:
                for j_str1,_ in enumerate(structures_1):
                    structures_1[j_str1]['Label'] = highest_label+1
                    structures_1[j_str1]['Born'] = False
                    structures_1[j_str1]['Died'] = True
                    highest_label +=  1
    
            elif not structures_1 and structures_2:
                for j_str2,_ in enumerate(structures_2):
                    structures_2[j_str2]['Label'] = highest_label+1
                    structures_2[j_str2]['Born'] = True
                    structures_2[j_str2]['Died'] = False
                    highest_label +=  1
    
            elif structures_1 and structures_2:
                for j_str1,_ in enumerate(structures_1):
                    if structures_1[j_str1]['Label'] is None:
                        structures_1[j_str1]['Label'] = highest_label+1
                        structures_1[j_str1]['Born'] = True
                        structures_1[j_str1]['Died'] = False
                        highest_label +=  1
    
                n_str1_overlap = np.zeros(len(structures_1))
                #Calculating the averages based on the input setting
                
                str_overlap_matrix = _calculate_str_overlap_matrix(structures_1,structures_2,
                                                                   tracking,
                                                                   matrix_weight,
                                                                   tracking_assignment,
                                                                   test)

                structures_1, structures_2, highest_label = _process_structure_merging(structures_1,structures_2,
                                                                                       str_overlap_matrix,
                                                                                       gap = 1,
                                                                                       sample_time = sample_time,
                                                                                       fit_shape = fit_shape,
                                                                                       highest_label = highest_label)
                
                structures_1, structures_2, highest_label = _process_structure_splitting(structures_1,structures_2,
                                                                                         str_overlap_matrix,
                                                                                         sample_time = sample_time,
                                                                                         fit_shape = fit_shape,
                                                                                         highest_label = highest_label)
                
                #This part handles the missed segmentation errors and tries to assign the blobs between frame pairs apart by at most max_gap:
                if i_frames > max_gap:
                    for ind_gap in range(2,max_gap+1):
                        n_born = 0
                        for cur_str_2 in structures_2:
                            if cur_str_2['Born']:
                                n_born +=  1
                        if n_born:  #There are structures that appeared out of nowhere so it might be that a structure was missed previously.
                            structures_before_gap = frame_properties['structures'][i_frames-ind_gap]
                            if structures_before_gap:
                                str_overlap_matrix = _calculate_str_overlap_matrix(structures_before_gap,structures_2,
                                                                                   tracking = tracking,
                                                                                   matrix_weight = matrix_weight,
                                                                                   tracking_assignment = tracking_assignment,
                                                                                   test = test)
                                
                                structures_1, structures_2, highest_label = _process_structure_merging(structures_before_gap,structures_2,
                                                                                                       str_overlap_matrix,
                                                                                                       ind_gap,
                                                                                                       sample_time * ind_gap,
                                                                                                       fit_shape,
                                                                                                       highest_label,)                
    
                if calculate_rough_diff_velocities:
                    frame_properties = _calculate_rough_differential_velocities(frame_properties,
                                                                                structures_1, structures_2, 
                                                                                fit_shape, 
                                                                                sample_time, 
                                                                                prev_str_weighting, 
                                                                                differential_keys, 
                                                                                n_str1_overlap,
                                                                                i_frames,
                                                                                weighting,maxing)
                    
                else:
                    for key in differential_keys:
                        frame_properties['derived'][key]['avg'][i_frames] = np.nan
                        frame_properties['derived'][key]['max'][i_frames] = np.nan

        if remove_orphans:
            frame_properties = _remove_orphans(frame_properties,
                                               n_frames,
                                               test,
                                               min_structure_lifetime)
        
        pickle.dump(frame_properties,open(pickle_filename_tracking,'wb'))
    else:
        frame_properties = pickle.load(open(pickle_filename_tracking,'rb'))


    return frame_properties


def _calculate_str_overlap_matrix(structures_1,structures_2,
                                  tracking = None,
                                  matrix_weight = None,
                                  tracking_assignment = None,
                                  test = False):
    
        
    """
    example str_overlap_matrix = |0,0,0,1| lives
                                 |1,0,1,1| splits into three
                                 |0,0,1,0| lives
                                 |0,0,0,0| dies
                                  ^ split into
                                    ^ is born
                                      ^ merges into
                                        ^ merges into

    """
    
    n_str1 = len(structures_1)
    n_str2 = len(structures_2)
    
    str_overlap_matrix = np.zeros([n_str1,n_str2])

    if tracking == 'overlap':
        for j_str2 in range(n_str2):
            for j_str1 in range(n_str1):
                #print(structures_2[j_str2]['Half path'],structures_1[j_str1]['Half path'])
                if (structures_2[j_str2]['Half path'].intersects_path(structures_1[j_str1]['Half path']) or
                    structures_2[j_str2]['Half path'].contains_path(structures_1[j_str1]['Half path'])):
                    
                    str_overlap_matrix[j_str1,j_str2] = 1

    elif tracking == 'weighted':

        score_matrix = calculate_score_matrix(structures_1, structures_2, matrix_weight)

        if tracking_assignment == 'hungarian':  #Structure tracking based on the Hungarian algorithm

            row_indices,col_indices = linear_sum_assignment(score_matrix,maximize = True)
            str_overlap_matrix[row_indices, col_indices] = 1.

        elif tracking_assignment == 'max_score':    #Structure tracking based on the maximum overlap

            for ind_row in range(len(str_overlap_matrix[:,0])):
                if np.sum(score_matrix[ind_row,:]) > 0:
                    str_overlap_matrix[ind_row,
                                       np.argmax(score_matrix[ind_row,:])] = 1

            if test:
                print(score_matrix)

                print(str_overlap_matrix)
                print(" ")

        else:
            raise ValueError('Tracking assignment is not available')

    else:
        raise ValueError('Tracking method '+tracking+' is unavailable.')
        
    return str_overlap_matrix

def calculate_score_matrix(structures_1,structures_2,
                           matrix_weight):

    n_str1 = len(structures_1)
    n_str2 = len(structures_2)
    
    score_matrix = np.zeros([n_str1,n_str2])

    for j_str2 in range(n_str2):
        str2_polygon = structures_2[j_str2]['Polygon'].shapely_polygon
        for j_str1 in range(n_str1):
            str1_polygon = structures_1[j_str1]['Polygon'].shapely_polygon
            try:
            #if True:
                #Fix for calculations where _data_pix data were not in the pickle file.
                if structures_1[j_str1]['Polygon'].x_data_pix is None:
                    structures_1[j_str1]['Polygon'].x_data_pix = (np.round((structures_1[j_str1]['Polygon'].x_data - coeff_r[2]) / coeff_r[0])).astype(int)
                    structures_1[j_str1]['Polygon'].y_data_pix = (np.round((structures_1[j_str1]['Polygon'].y_data - coeff_z[2]) / coeff_z[1])).astype(int)

                if structures_2[j_str2]['Polygon'].x_data_pix is None:
                    structures_2[j_str2]['Polygon'].x_data_pix = (np.round((structures_2[j_str2]['Polygon'].x_data - coeff_r[2]) / coeff_r[0])).astype(int)
                    structures_2[j_str2]['Polygon'].y_data_pix = (np.round((structures_2[j_str2]['Polygon'].y_data - coeff_z[2]) / coeff_z[1])).astype(int)

                if str2_polygon.intersects(str1_polygon):
                    intersection_area = str1_polygon.intersection(str2_polygon).area
                    union_area = unary_union([str1_polygon,str2_polygon]).area
                    score_matrix[j_str1,j_str2] +=  intersection_area / union_area * matrix_weight['iou']

                    x_min = np.min([np.min(structures_1[j_str1]['Polygon'].x_data_pix),
                                    np.min(structures_2[j_str2]['Polygon'].x_data_pix)])

                    x_max = np.max([np.max(structures_1[j_str1]['Polygon'].x_data_pix),
                                    np.max(structures_2[j_str2]['Polygon'].x_data_pix)])

                    y_min = np.min([np.min(structures_1[j_str1]['Polygon'].y_data_pix),
                                    np.min(structures_2[j_str2]['Polygon'].y_data_pix)])

                    y_max = np.max([np.max(structures_1[j_str1]['Polygon'].y_data_pix),
                                    np.max(structures_2[j_str2]['Polygon'].y_data_pix)])

                    str1_matrix = np.zeros([x_max-x_min+1,y_max-y_min+1])
                    str2_matrix = np.zeros([x_max-x_min+1,y_max-y_min+1])

                    for ind_str1_data in range(len(structures_1[j_str1]['Polygon'].data)):
                        x_data_pix = structures_1[j_str1]['Polygon'].x_data_pix[ind_str1_data]
                        y_data_pix = structures_1[j_str1]['Polygon'].y_data_pix[ind_str1_data]
                        str1_matrix[x_data_pix - x_min, y_data_pix - y_min] = structures_1[j_str1]['Polygon'].data[ind_str1_data]

                    for ind_str2_data in range(len(structures_2[j_str2]['Polygon'].data)):
                        x_data_pix = structures_2[j_str2]['Polygon'].x_data_pix[ind_str2_data]
                        y_data_pix = structures_2[j_str2]['Polygon'].y_data_pix[ind_str2_data]
                        str2_matrix[x_data_pix - x_min, y_data_pix - y_min] = structures_2[j_str2]['Polygon'].data[ind_str2_data]

                    str1_matrix -=  np.mean(str1_matrix)
                    str2_matrix -=  np.mean(str2_matrix)
                    ccf_matrix = correlate2d(str1_matrix,
                                             str2_matrix)

                    cccf_matrix = ccf_matrix/np.sqrt(np.sum(str1_matrix**2)*
                                                     np.sum(str2_matrix**2))

                    if np.max(cccf_matrix) > 1:
                        raise ValueError('Something went wrong, the cross-orrelation matrix has a value higher than 1.')
                    score_matrix[j_str1, j_str2] +=  np.max(cccf_matrix) * matrix_weight['cccf']

                else:
                    score_matrix[j_str1, j_str2] = 0.

            except Exception as e:
                print('Exception at line 988: '+str(e))
                score_matrix[j_str1, j_str2] = 0.
                
    return score_matrix

def _process_structure_merging(structures_1,structures_2,
                               str_overlap_matrix,
                               gap = 1,
                               sample_time = 2e-6,
                               fit_shape = 'Ellipse',
                               highest_label = None,):
    
    n_str2 = len(structures_2)
    
    for j_str2 in range(n_str2):
        merging_indices = np.squeeze(str_overlap_matrix[:, j_str2])
        
        #No overlap between the new structrure and the old ones
        if np.sum(merging_indices) == 0 and gap == 1:
            structures_2[j_str2]['Label'] = highest_label+1
            highest_label +=  1
            structures_2[j_str2]['Born'] = True


        #Structures are propagating without merging or splitting
        elif np.sum(merging_indices) == 1 and gap == 1:
            try:
                ind_str1 = np.where(merging_indices == 1)[0]
            except:
                ind_str1 = [0]
            #One and only one overlap
            if np.sum(str_overlap_matrix[ind_str1[0],:]) == 1:
                #print(ind_str1[0])
                structures_2[j_str2]['Label'] = structures_1[int(ind_str1[0])]['Label']
                structures_2[j_str2] = correct_structure_angle(structure_2 = structures_2[j_str2],
                                                               structure_1 = structures_1[int(ind_str1[0])])
                structures_2[j_str2] = calculate_differential_keys(structure_2 = structures_2[j_str2],
                                                                   structure_1 = structures_1[int(ind_str1[0])],
                                                                   sample_time = sample_time,
                                                                   fit_shape = fit_shape)

            else:
                #If splitting is happening, it's handled later.
                pass
            
        elif np.sum(merging_indices) == 1 and gap > 1:  #Only relabel those structures that were born out of nowhere if they have overlap with a structures gap number of frames becore
            if structures_2[j_str2]['Born']:
                try:
                    ind_str1 = np.where(merging_indices == 1)[0]
                except:
                    ind_str1 = [0]
                
                    #One and only one overlap
                if np.sum(str_overlap_matrix[ind_str1[0],:]) == 1:
                    if structures_1[int(ind_str1[0])]['Died']:
                        #print(ind_str1[0])
                        structures_2[j_str2]['Label'] = structures_1[int(ind_str1[0])]['Label']
                        structures_2[j_str2] = correct_structure_angle(structure_2 = structures_2[j_str2],
                                                                       structure_1 = structures_1[int(ind_str1[0])])
                        structures_2[j_str2] = calculate_differential_keys(structure_2 = structures_2[j_str2],
                                                                           structure_1 = structures_1[int(ind_str1[0])],
                                                                           sample_time = sample_time * gap,
                                                                           fit_shape = fit_shape)
                        structures_2[j_str2]['Born'] = False                          # There was overlap with a gapped frame, necessary for proper handling earlier.
                        structures_1[int(ind_str1[0])]['Died'] = False
                else:
                    #If splitting is occurring with a gapped frame, it is disregarded. This is a placeholder for the logic of the code.
                    pass
            
        #Previous structures merge
        elif np.sum(merging_indices) > 1 and gap == 1:                          #Merging is not handled for segmentation errors (when gap > 1)
            ind_merge = np.where(merging_indices == 1)
            
            
            if np.sum(str_overlap_matrix[ind_merge,:]) == np.sum(merging_indices):
                #There is merging, but there is no splitting
                
                ind_high = ind_merge[0][0]
                for ind_str1 in ind_merge[0]:
                    if structures_1[int(ind_high)]['Intensity'] < structures_1[int(ind_str1)]['Intensity']:
                        ind_high = ind_str1
                structures_2[j_str2]['Label'] = structures_1[int(ind_high)]['Label']
                structures_2[j_str2] = correct_structure_angle(structure_2 = structures_2[j_str2],
                                                              structure_1 = structures_1[int(ind_high)])
                structures_2[j_str2] = calculate_differential_keys(structure_2 = structures_2[j_str2],
                                                                 structure_1 = structures_1[int(ind_high)],
                                                                 sample_time = sample_time,
                                                                 fit_shape = fit_shape)
                for ind_str1 in ind_merge[0]:
                    structures_2[j_str2]['Parent'].append(structures_1[int(ind_str1)]['Label'])
                    structures_1[int(ind_str1)]['Merges'] = True
                    structures_1[int(ind_str1)]['Child'].append(structures_2[j_str2]['Label'])

            else:
                #This is a weird situation where merges and splits occur at the same time
                #Should be handled correctly, possibly assigning new labels to everything
                #and not track anything. The merging/splitting labels should be assigned
                #that needs further modification of the structures_segmentation.py

                ind_high1 = ind_merge[0][0]
                for ind_str1 in ind_merge[0]:
                    if structures_1[int(ind_high1)]['Intensity'] < structures_1[int(ind_str1)]['Intensity']:
                        ind_high1 = ind_str1

                ind_high2 = 0
                for ind_str1 in ind_merge[0]:
                    for ind_str2 in range(len(str_overlap_matrix[int(ind_str1), :])):
                        if (str_overlap_matrix[int(ind_str1),ind_str2] == 1 and
                            structures_2[int(ind_high2)]['Intensity'] < structures_2[int(ind_str2)]['Intensity']):
                            ind_high2 = ind_str2

                for ind_str1 in ind_merge[0]:
                    if np.sum(str_overlap_matrix[ind_str1,:]) == 1:
                        structures_1[int(ind_str1)]['Splits'] = False
                    else:
                        structures_1[int(ind_str1)]['Splits'] = True
                        ind_split = np.where(str_overlap_matrix[ind_str1,:] == 1)
                        for ind_str2 in ind_split[0]:
                            if int(ind_str2) !=  int(ind_high2):
                                structures_2[int(ind_str2)]['Label'] = highest_label+1
                                highest_label +=  1
                                structures_2[int(ind_str2)]['Parent'] = structures_1[int(ind_high1)]['Label']
                            else:
                                structures_2[int(ind_high2)]['Label'] = structures_1[int(ind_high1)]['Label']
                                structures_2[int(ind_high2)] = correct_structure_angle(structure_2 = structures_2[j_str2],
                                                                              structure_1 = structures_1[int(ind_str1[0])])
                                structures_2[int(ind_high2)] = calculate_differential_keys(structure_2 = structures_2[int(ind_high2)],
                                                                                           structure_1 = structures_1[int(ind_high1)],
                                                                                           sample_time = sample_time,
                                                                                           fit_shape = fit_shape)

                                structures_2[int(ind_high2)]['Parent'] = structures_1[int(ind_high1)]['Label']
                            structures_1[int(ind_str1)]['Child'].append(structures_2[ind_str2]['Label'])
                    structures_1[int(ind_str1)]['Merges'] = True


                #print('Splitting and merging is occurring at the same time at frame #'+str(i_frames)+', t = '+str(frame_properties['Time'][i_frames]*1e3)+'ms')
                if False:
                    print(str_overlap_matrix[ind_merge,:], merging_indices)

                    print('str1 label ',[structures_1[i]['Label'] for i in range(len(structures_1))])
                    print('str2 label ',[structures_2[i]['Label'] for i in range(len(structures_2))])
                    print('str1 child ',[structures_1[i]['Child'] for i in range(len(structures_1))])
                    print('str2 parent ',[structures_2[i]['Parent'] for i in range(len(structures_2))])
                    print('str1 splits ',[structures_1[i]['Splits'] for i in range(len(structures_1))])
                    print('str1 merges ',[structures_1[i]['Merges'] for i in range(len(structures_1))])

                    print(str_overlap_matrix)
                    
    return structures_1, structures_2, highest_label

def _process_structure_splitting(structures_1,structures_2,
                                 str_overlap_matrix,
                                 sample_time = 2e-6,
                                 fit_shape = 'Ellipse',
                                 highest_label = None,):
    n_str1 = len(structures_1)
    #print('splitting')
    for j_str1 in range(n_str1):
        splitting_indices = np.squeeze(str_overlap_matrix[j_str1, :])

        #Structure dies
        if np.sum(splitting_indices) == 0:
            structures_1[j_str1]['Died'] = True

        #There is one and only one overlap, taken care of previously, here for completeness
        elif np.sum(splitting_indices) == 1:
            pass

        #Previous structures are splitting into more new structures
        elif np.sum(splitting_indices) > 1:
            ind_split = np.where(splitting_indices == 1)
            if np.sum(str_overlap_matrix[:,ind_split]) == np.sum(splitting_indices):
                ind_high = ind_split[0][0]
                for ind_str2 in ind_split[0]:
                    if structures_2[int(ind_high)]['Intensity'] < structures_2[int(ind_str2)]['Intensity']:
                        ind_high = ind_str2
                for ind_str2 in ind_split[0]:
                    if structures_2[int(ind_str2)]['Label'] is not None:
                        print([structures_1[i]['Label'] for i in range(len(structures_1))])
                        print([structures_2[i]['Label'] for i in range(len(structures_2))])
                        print('str_overlap_matrix')
                        print(str_overlap_matrix)
                        print('splitting_indices',splitting_indices)

                        print('ind_split', ind_split)
                        print("2", structures_2[ind_str2]['Label'])

                    if ind_str2 == ind_high:
                        structures_2[ind_str2]['Label'] = structures_1[j_str1]['Label']
                        structures_2[ind_str2] = correct_structure_angle(structure_2 = structures_2[ind_str2],
                                                                       structure_1 = structures_1[j_str1])
                        structures_2[ind_str2] = calculate_differential_keys(structure_2 = structures_2[ind_str2],
                                                                            structure_1 = structures_1[j_str1],
                                                                            sample_time = sample_time,
                                                                            fit_shape = fit_shape)
                    else:
                        structures_2[ind_str2]['Label'] = highest_label+1
                        highest_label +=  1
                        print('HL: ',highest_label)
                    structures_2[ind_str2]['Parent'].append(structures_1[j_str1]['Label'])
                    structures_1[j_str1]['Child'].append(structures_2[ind_str2]['Label'])
                structures_1[j_str1]['Splits'] = True
            else:
                #This was handled in the previous case at the end of merging.
                pass
            
    return structures_1, structures_2, highest_label

def _calculate_rough_differential_velocities(frame_properties,
                                             structures_1, structures_2, 
                                             fit_shape, 
                                             sample_time, 
                                             prev_str_weighting, 
                                             differential_keys, 
                                             n_str1_overlap,
                                             i_frames,
                                             weighting,maxing):
    
    n_str1 = len(structures_1)
    n_str2 = len(structures_2)
    
    intensities = np.zeros(n_str2)
    areas = np.zeros(n_str2)
    for j_str2 in range(n_str2):
        #Average size calculation based on the number of structures
        areas[j_str2] = structures_2[j_str2]['Area']
        intensities[j_str2] = structures_2[j_str2]['Intensity']
        
    for j_str2 in range(n_str2):
        prev_str_weight = []
        for new_key in differential_keys:
            structures_2[j_str2][new_key] = []

        #Check the new frame if it has overlap with the old frame
        for j_str1 in range(n_str1):
            if structures_2[j_str2]['Half path'].intersects_path(structures_1[j_str1]['Half path']):
                if prev_str_weighting == 'number':
                    prev_str_weight.append(1.)
                elif prev_str_weighting == 'intensity':
                    prev_str_weight.append(structures_1[j_str1]['Intensity'])
                elif prev_str_weighting == 'area':
                    prev_str_weight.append(structures_1[j_str1]['Area'])
                elif prev_str_weighting == 'max_intensity':
                    if np.argmax(intensities) == j_str1:
                        prev_str_weight.append(1.)
                    else:
                        prev_str_weight.append(0.)

                structures_2[j_str2]['Velocity radial COG'].append((structures_2[j_str2]['Polygon'].center_of_gravity[0]-
                                                                   structures_1[j_str1]['Polygon'].center_of_gravity[0])/sample_time)
                structures_2[j_str2]['Velocity poloidal COG'].append((structures_2[j_str2]['Polygon'].center_of_gravity[1]-
                                                                     structures_1[j_str1]['Polygon'].center_of_gravity[1])/sample_time)

                structures_2[j_str2]['Velocity radial centroid'].append((structures_2[j_str2]['Polygon'].centroid[0]-
                                                                        structures_1[j_str1]['Polygon'].centroid[0])/sample_time)
                structures_2[j_str2]['Velocity poloidal centroid'].append((structures_2[j_str2]['Polygon'].centroid[1]-
                                                                          structures_1[j_str1]['Polygon'].centroid[1])/sample_time)

                structures_2[j_str2]['Velocity radial position'].append((structures_2[j_str2][fit_shape].center[0]-
                                                                        structures_1[j_str1][fit_shape].center[0])/sample_time)
                structures_2[j_str2]['Velocity poloidal position'].append((structures_2[j_str2][fit_shape].center[1]-
                                                                          structures_1[j_str1][fit_shape].center[1])/sample_time)

                structures_2[j_str2]['Expansion fraction area'].append(np.sqrt(structures_2[j_str2]['Polygon'].area/
                                                                               structures_1[j_str1]['Polygon'].area))
                structures_2[j_str2]['Expansion fraction axes'].append(np.sqrt(structures_2[j_str2][fit_shape].axes_length[0]/
                                                                               structures_1[j_str1][fit_shape].axes_length[0]*
                                                                               structures_2[j_str2][fit_shape].axes_length[1]/
                                                                               structures_1[j_str1][fit_shape].axes_length[1]))

                structures_2[j_str2]['Angular velocity angle'].append((structures_2[j_str2][fit_shape].angle-
                                                                      structures_1[j_str1][fit_shape].angle)/sample_time)
                structures_2[j_str2]['Angular velocity ALI'].append((structures_2[j_str2]['Polygon'].principal_axes_angle-
                                                                    structures_1[j_str1]['Polygon'].principal_axes_angle)/sample_time)

                n_str1_overlap[j_str1]+= 1.

        prev_str_weight = np.asarray(prev_str_weight)
    
        if not np.sum(prev_str_weight) == 0:
            prev_str_weight /=  np.sum(prev_str_weight)

        #structures_2[j_str2]['Label'] = np.mean(structures_2[j_str2]['Label'])
        for key in differential_keys:
            structures_2[j_str2][key] = np.sum(np.asarray(structures_2[j_str2][key])*prev_str_weight)
    
        """
        Frame property filling up
        """
        n_str2 = len(structures_2)
        areas = np.zeros(n_str2)
        intensities = np.zeros(n_str2)

        for j_str2 in range(n_str2):
            #Average size calculation based on the number of structures
            areas[j_str2] = structures_2[j_str2]['Area']
            intensities[j_str2] = structures_2[j_str2]['Intensity']

        areas /=  np.sum(areas)
        intensities /=  np.sum(intensities)
        #Calculating the averages based on the input setting
        if weighting == 'number':
            weight = np.zeros(n_str2)
            weight[:] = 1./n_str2
        elif weighting == 'intensity':
            weight = intensities/np.sum(intensities)
        elif weighting == 'area':
            weight = areas/np.sum(areas)

        if maxing == 'area':
            ind_max = np.argmax(areas)
        elif maxing == 'intensity':
            ind_max = np.argmax(intensities)

        for key in differential_keys:
            for j_str2 in range(len(structures_2)):
                try:
                    frame_properties['derived'][key]['avg'][i_frames] +=  structures_2[j_str2][key]*weight[j_str2]
                except:
                    pass
            try:
                frame_properties['derived'][key]['max'][i_frames] = structures_2[ind_max][key]
            except:
                frame_properties['derived'][key]['max'][i_frames] = np.nan

    
    return frame_properties



def _remove_orphans(frame_properties,
                    n_frames,
                    test,
                    min_structure_lifetime):
    
    all_labels = []
    for i_frames in range(0, n_frames):
        if frame_properties['structures'][i_frames] is not None:
            for ind_str in range(len(frame_properties['structures'][i_frames])):
                label = frame_properties['structures'][i_frames][ind_str]['Label']
                if label is None:
                    print(frame_properties['structures'][i_frames][ind_str])
                    print('Label is None. analyze_gpi_structures line 1174')
            curr_structures = frame_properties['structures'][i_frames]

            curr_labels = [curr_structures[ind_str]['Label'] for ind_str in range(len(curr_structures))]
            all_labels = np.append(all_labels,curr_labels)

    labels_to_drop = []
    labels_to_keep = []
    if test: print(all_labels)
    for label in range(int(np.max(all_labels)+1)):
        if np.sum(all_labels == label) < min_structure_lifetime:
            labels_to_drop.append(label)
        else:
            labels_to_keep.append(label)

    for i_frames in range(0, n_frames):
        if frame_properties['structures'][i_frames] is not None:
            curr_structures = frame_properties['structures'][i_frames]
            for ind_str in range(len(curr_structures)-1,-1,-1):
                if curr_structures[ind_str]['Label'] in labels_to_drop:
                    curr_structures.pop(ind_str)
    labels_to_keep = np.asarray(labels_to_keep)


    new_labels = np.arange(len(labels_to_keep)+1)
    for i_frames in range(n_frames):
        if frame_properties['structures'][i_frames] is not None:
            for ind_struct in range(len(frame_properties['structures'][i_frames])):
                ind = np.where(labels_to_keep == frame_properties['structures'][i_frames][ind_struct]['Label'])
                if len(new_labels[ind]) !=  0 and int(ind[0]) !=  -1:
                    frame_properties['structures'][i_frames][ind_struct]['Label'] = int(new_labels[ind])
    
    return frame_properties

#Wrapper function for calculating differential key results.
def calculate_differential_keys(structure_2 = None,
                                structure_1 = None,
                                sample_time = None,
                                fit_shape = 'Ellipse',
                                ):

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



def correct_structure_angle(structure_2 = None,
                            structure_1 = None,
                            ):
    
    """
    Performs fringe jump correction between two frames
    """
    
    if structure_1 is None or structure_2 is None:
        raise ValueError('Both structure_1 and structure_2 need to be defined.')

    for key in ['Angle',
                'Angle of least inertia'
                ]:
        if key in structure_2.keys() and key in structure_1.keys():
            data = np.asarray([structure_1[key],
                               structure_2[key]])

            structure_2[key] = fringe_jump_correction(data,tolerance = 0.5)[1]
        else:
            #Only invoked when Angle of least inertia is not present
            try:
                data = np.asarray([structure_1['Polygon'].principal_axes_angle,
                                   structure_2['Polygon'].principal_axes_angle])

                structure_2[key] = fringe_jump_correction(data,tolerance = 0.5)[1]
            except:
                print('No ALI data --> no fringe jump correction')
                pass
        #Putting the angles between +-pi/2, structures don't have a direction, no need for angles outside the range
        
        # if np.min(structure_2[key]) < -np.pi/2:
        #     structure_2[key] +=  np.pi
            
        # if np.min(structure_2[key]) > np.pi/2:
        #     structure_2[key] -=  np.pi
            
    return structure_2

