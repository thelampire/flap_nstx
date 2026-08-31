#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 16:41:06 2025

@author: mlampert
"""
#Core modules
import os

import copy
import time as time_mod
import pickle
import warnings
warnings.filterwarnings("ignore")

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')

from flap_nstx.gpi import analyze_gpi_structures
from flap_nstx.thomson import get_fit_nstx_thomson_profiles
from flap_nstx.tools import get_flux_coord, read_equilibrium_data, get_equilibrium_slice

import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas

#Plot settings for publications
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/plots'


def read_all_blob_data(time_range_around_peak=[-5e-3, 15e-3], 
                       nocalc=False,
                       recalc_tracking=False,
                       min_structure_lifetime=20,
                       str_finding_method='watershed',
                       fix_angle_for_correlation=False,
                       read_mean_results=False, 
                       averaging='shot',
                       average='avg', 
                       replicate_histogram2=False,
                       replicate_old_read_blob_data=False,
                       read_l_mode_only=False,
                       read_h_mode_only=False,
                       filtered_blob_db=False,
                       condition_key=None,
                       condition_range=None):
    
    if read_mean_results:
        averaging = 'shot'
    
    pickle_filename = f"{wd}/processed_data/blob_database_shot_by_shot_blob_{str_finding_method}_{averaging}_avg__{average}_calc__"
    
    if read_l_mode_only: pickle_filename += 'L_mode'
    if read_h_mode_only: pickle_filename += 'H_mode'
    
    if averaging == 'conditional' and condition_key is not None and condition_range is not None:
        pickle_filename += f"_cond_{condition_key}_{condition_range[0]}_{condition_range[1]}"
        
    pickle_filename += '.pickle'
            
    if read_l_mode_only:
        blob_database = read_blob_lh_mode_database_file(l_mode=True, filtered_blob_db=filtered_blob_db, time_range_around_peak=time_range_around_peak)
    elif read_h_mode_only:
        blob_database = read_blob_lh_mode_database_file(h_mode=True, filtered_blob_db=filtered_blob_db, time_range_around_peak=time_range_around_peak)
    else:
        if not isinstance(time_range_around_peak, (int, float)):
            print('time_range_around_peak needs to be a single number if l_mode or h_mode reading is not set. Setting it to 5e-3')
            time_range_around_peak = 5e-3
        blob_database = read_blob_database_file(time_range_around_peak=time_range_around_peak)
        
    ncalc = len(blob_database['shot'])

    full_blob_db_data = {}
    full_blob_db_error = {}

    if not os.path.exists(pickle_filename) or not nocalc:
        n_str = 0
        start_time = time_mod.time()
        keys_initialized = False
        analyzed_keys = []
        failed_shots = {'shot': [], 'index': []}
        
        for ind in range(ncalc):
            shot = blob_database['shot'][ind]
            if shot in [137651, 139435, 139434]:
                failed_shots['shot'].append(shot)
                failed_shots['index'].append(ind)
                continue
            
            if isinstance(blob_database['time'][ind], (list, np.ndarray)):
                time_range = blob_database['time'][ind]
            else:
                time_range = [blob_database['time'][ind] - time_range_around_peak,
                              blob_database['time'][ind] + time_range_around_peak]

            blob_results = read_blob_data(shot, time_range, nocalc=True, recalc_tracking=recalc_tracking,
                                          min_structure_lifetime=min_structure_lifetime, str_finding_method=str_finding_method)
                                          
            if blob_results is None or blob_results.mode != 'tracked' or not blob_results.tracked_structures: 
                failed_shots['shot'].append(shot)
                failed_shots['index'].append(ind)
                continue
        
            #Read the EFIT equilibrium once per shot and slice it once for the
            #time the structures of this shot are evaluated at.
            shot_equilibrium = read_equilibrium_data(shot=shot)
            shot_equilibrium_slice = get_equilibrium_slice(equilibrium=shot_equilibrium,
                                                           time=np.mean(blob_database['time'][ind]),
                                                           shot=shot)
        
            if not keys_initialized:
                first_struct = blob_results.tracked_structures[0]
                analyzed_keys = list(first_struct.regular_parameters.keys()) + list(first_struct.differential_parameters.keys())
                
                new_keys_to_add = ['Normalized flux coordinate', 'Poloidal angle', 'Lifetime', 
                                   'Poloidal angular velocity', 'Normalized flux coordinate velocity']
                for nk in new_keys_to_add:
                    if nk not in analyzed_keys:
                        analyzed_keys.append(nk)
                
                full_blob_db_data = {key: [] for key in analyzed_keys}
                full_blob_db_error = {key: [] for key in analyzed_keys}
                keys_initialized = True

            curr_shot_data = {key: [] for key in analyzed_keys}
            curr_shot_error = {key: [] for key in analyzed_keys}
            
            flap.delete_data_object('*')
            
            for structure in blob_results.tracked_structures: 
                
                # ===============================================================
                # 1. PRE-CALCULATE CORRECTED FLUX COORDS
                # ===============================================================
                raw_psi_norm, raw_theta_arc = None, None
                
                flux_dependent_keys = ['Normalized flux coordinate', 'Poloidal angle', 
                                       'Poloidal angular velocity', 'Normalized flux coordinate velocity']
                needs_flux = any(k in analyzed_keys for k in flux_dependent_keys) or (condition_key in flux_dependent_keys)
                
                if needs_flux:
                    try:
                       
                        # Step 3: Call the flux function with the perfectly corrected coordinates
                        raw_psi_norm, raw_theta_arc = get_flux_coord(
                            shot=shot,
                            time=np.mean(blob_database['time'][ind]),
                            R_target=structure.regular_parameters['Centroid radial'].value,
                            z_target=structure.regular_parameters['Centroid poloidal'].value,
                            equilibrium_slice=shot_equilibrium_slice
                        )
                    except Exception as e:
                        print(f"Exception in read_data_for_analyze_blob_database.py at L158: {e}")
                        raw_psi_norm = np.full(len(structure.regular_parameters['Intensity'].value), np.nan)
                        raw_theta_arc = np.full(len(structure.regular_parameters['Intensity'].value), np.nan)

                # ===============================================================
                # 2. EVENT TRIGGER DETECTION (Find t0)
                # ===============================================================
                event_idx = 0
                if averaging == 'conditional' and condition_key is not None and condition_range is not None:
                    cond_arr = None
                    if condition_key == 'Normalized flux coordinate': cond_arr = raw_psi_norm
                    elif condition_key == 'Poloidal angle': cond_arr = raw_theta_arc
                    # np.diff is used the same way as for the other differential
                    # keys, hence these arrays are one datapoint shorter
                    elif condition_key == 'Poloidal angular velocity': 
                        cond_arr = ((np.diff(raw_theta_arc) + np.pi) % (2 * np.pi) - np.pi) / 2.5e-6
                    elif condition_key == 'Normalized flux coordinate velocity': 
                        cond_arr = np.diff(raw_psi_norm) / 2.5e-6
                    elif condition_key == 'Lifetime': cond_arr = np.arange(len(structure.regular_parameters['Intensity'].value)) * 2.5e-6
                    elif condition_key in structure.regular_parameters: cond_arr = structure.regular_parameters[condition_key].value
                    elif condition_key in structure.differential_parameters: cond_arr = structure.differential_parameters[condition_key].value
                    
                    if cond_arr is not None:
                        valid_indices = np.where((cond_arr >= condition_range[0]) & (cond_arr <= condition_range[1]))[0]
                        if len(valid_indices) > 0:
                            event_idx = valid_indices[0] 
                        else:
                            continue
                    else:
                        continue

                # ===============================================================
                
                n_str += 1
                for key in analyzed_keys:
                    
                    is_differential = False
                    if key == 'Axes length minor fit': new_key = 'Axes length major fit'
                    elif key == 'Axes length major fit': new_key = 'Axes length minor fit'
                    else: new_key = key

                    if key == 'Normalized flux coordinate': 
                        raw_data = raw_psi_norm
                    elif key == 'Poloidal angle': 
                        raw_data = raw_theta_arc
                    
                    # ===============================================================
                    # Velocity Calculations using np.diff (dt = 2.5e-6)
                    # ===============================================================
                    elif key == 'Poloidal angular velocity':
                        # Wrap to [-pi, pi] safely
                        raw_data = ((np.diff(raw_theta_arc) + np.pi) % (2 * np.pi) - np.pi) / 2.5e-6
                        is_differential = True # np.diff returns length N-1
                    elif key == 'Normalized flux coordinate velocity':
                        raw_data = np.diff(raw_psi_norm) / 2.5e-6
                        is_differential = True # np.diff returns length N-1
                    # ===============================================================
                    
                    elif key == 'Lifetime': 
                        raw_data = np.arange(len(structure.regular_parameters['Intensity'].value)) * 2.5e-6
                    elif key in structure.regular_parameters: 
                        raw_data = structure.regular_parameters[new_key].value
                    elif key in structure.differential_parameters:
                        raw_data = structure.differential_parameters[new_key].value
                        is_differential = True
                    else: continue 
                        
                    if replicate_old_read_blob_data:
                        str_key_data = np.append(raw_data, raw_data[-1]) if is_differential else raw_data
                    elif replicate_histogram2:
                        str_key_data = raw_data if is_differential else raw_data[1:]
                    else:  
                        str_key_data = raw_data
                        
                    str_key_data = np.asarray(str_key_data)
                    
                    if averaging == 'no':
                        curr_shot_data[new_key] = np.append(curr_shot_data[new_key], np.ravel(str_key_data))
                        
                    elif averaging in ['shot', 'blob']:
                        str_key_data = str_key_data[~np.isnan(str_key_data)]
                        if len(str_key_data) == 0: continue
                        if averaging == 'shot': curr_shot_error[new_key].append(np.sqrt(np.var(str_key_data)))
                                
                        if average == 'avg': curr_shot_data[new_key].append(np.mean(str_key_data))
                        elif average == 'std': curr_shot_data[new_key].append(np.sqrt(np.var(str_key_data)))
                        elif average == 'max': curr_shot_data[new_key].append(np.max(str_key_data))
                        
                    elif averaging == 'conditional':
                        curr_shot_data[new_key].append((event_idx, str_key_data))
                        
            if averaging != 'conditional':
                for key in analyzed_keys:
                    try: curr_shot_data[key] = np.concatenate(curr_shot_data[key])
                    except: pass
                
            for key in full_blob_db_data.keys():
                if len(curr_shot_data[key]) == 0: 
                    if averaging == 'shot':
                        full_blob_db_data[key].append(np.nan)
                        full_blob_db_error[key].append(np.nan)
                    elif averaging in ['no', 'blob']:
                        full_blob_db_data[key].append({'shot': shot, 'data': np.nan})
                else:
                    if averaging == 'shot':
                        full_blob_db_data[key].append(np.mean(curr_shot_data[key]))
                        full_blob_db_error[key].append(np.mean(curr_shot_error[key]) / np.sqrt(len(curr_shot_error[key])))
                    elif averaging in ['no', 'blob']:
                        full_blob_db_data[key].append({'shot': shot, 'data': curr_shot_data[key]})
                    elif averaging == 'conditional':
                        full_blob_db_data[key].extend(curr_shot_data[key])
            
            elapsed_time = time_mod.time() - start_time
            avg_time_per_shot = elapsed_time / (ind + 1)
            remaining_time = avg_time_per_shot * (ncalc - ind - 1)
            print(f'\rRemaining time: {int(remaining_time // 3600)}h {int((remaining_time % 3600) // 60):02}min {int(remaining_time % 60):02}sec', end="", flush=True)
            
        print(f'\nTotal number of structures: {n_str}')

        # ===============================================================
        # 3. ALIGN & PAD UNEQUAL ARRAYS FOR CONDITIONAL AVERAGING
        # ===============================================================
        if averaging == 'conditional':
            for key in analyzed_keys:
                list_of_tuples = full_blob_db_data[key]
                if len(list_of_tuples) == 0:
                    full_blob_db_data[key] = {'mean': [], 'error': [], 'count': [], 'relative_frames': [], 'raw_matrix': []}
                    continue
                    
                max_pre = max(evt_idx for evt_idx, arr in list_of_tuples)
                max_post = max(len(arr) - 1 - evt_idx for evt_idx, arr in list_of_tuples)
                
                total_len = max_pre + max_post + 1
                padded_matrix = np.full((len(list_of_tuples), total_len), np.nan)
                
                for i, (evt_idx, arr) in enumerate(list_of_tuples):
                    start_idx = max_pre - evt_idx
                    end_idx = start_idx + len(arr)
                    padded_matrix[i, start_idx:end_idx] = arr
                    
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    count = np.sum(~np.isnan(padded_matrix), axis=0)
                    median_arr = np.nanmedian(padded_matrix, axis=0)
                    p10 = np.nanpercentile(padded_matrix, 10, axis=0)
                    p90 = np.nanpercentile(padded_matrix, 90, axis=0)
                
                err_lower = median_arr - p10
                err_upper = p90 - median_arr
                err_arr = np.vstack([err_lower, err_upper])
                err_arr[:, count == 0] = 0.0
                
                relative_frames = np.arange(-max_pre, max_post + 1)
                
                full_blob_db_data[key] = {
                    'mean': median_arr,
                    'error': err_arr,
                    'count': count,
                    'relative_frames': relative_frames,
                    'raw_matrix': padded_matrix  
                }
        # ===============================================================
        
        if len(failed_shots['shot']) > 0 and averaging != 'conditional':
            for ind, shot in enumerate(failed_shots['shot']):
                for key in full_blob_db_data.keys():
                    if averaging == 'shot':
                        full_blob_db_data[key].insert(ind, np.nan)
                        full_blob_db_error[key].insert(ind, np.nan)
                    else:
                        full_blob_db_data[key].append({'shot': shot, 'data': np.nan})
        
        with open(pickle_filename, 'wb') as f:
            pickle.dump(full_blob_db_data, f)
    else:
        with open(pickle_filename, 'rb') as f:
            full_blob_db_data = pickle.load(f)

    return full_blob_db_data

def read_all_blob_data_old(time_range_around_peak=[-5e-3,15e-3], 
                       nocalc=False,
                       recalc_tracking=False,
                       min_structure_lifetime=20,
                       str_finding_method='watershed',
                       fix_angle_for_correlation=False,
                       read_mean_results=False, 
                       averaging='shot',
                       average='avg', 
                       replicate_histogram2=False,
                       replicate_old_read_blob_data=False,
                       read_l_mode_only=False,
                       read_h_mode_only=False,
                       filtered_blob_db=False):
    
    """
    Reads, processes, and aggregates Gas Puff Imaging (GPI) blob tracking data across a database of experimental shots.

    This function loads blob database files (optionally filtered by L-mode or H-mode), extracts tracked 
    structure parameters (both regular and differential), and computes derived quantities such as the 
    normalized flux coordinate, poloidal angle, and lifetime. The resulting parameter distributions are 
    aggregated based on the chosen averaging method (e.g., shot-by-shot or blob-by-blob) and cached to a 
    Pickle file to accelerate future execution.

    Args:
        time_range_around_peak (list or float, optional): The time window relative to the peak time 
            to extract data. Defaults to [-5e-3, 15e-3]. If L/H mode specific reading is not set, 
            this is forced to a scalar (5e-3).
        nocalc (bool, optional): If True, attempts to load the processed data from a cached Pickle 
            file instead of recalculating. Defaults to False.
        recalc_tracking (bool, optional): If True, forces the underlying blob tracker to recalculate 
            structure trajectories instead of using cached tracking results. Defaults to False.
        min_structure_lifetime (int, optional): Minimum required lifespan (in frames) for a tracked 
            structure to be included in the analysis. Defaults to 20.
        str_finding_method (str, optional): The segmentation algorithm used to identify blobs 
            (e.g., 'watershed', 'contour'). Defaults to 'watershed'.
        fix_angle_for_correlation (bool, optional): Legacy parameter for angle normalization. Defaults to False.
        read_mean_results (bool, optional): If True, forces `averaging` to 'shot'. Defaults to False.
        averaging (str, optional): The data aggregation level. Options include 'shot' (averages all blobs 
            in a shot), 'blob' (retains blob-by-blob arrays), or 'no' (flattens all raw data points). 
            Defaults to 'shot'.
        average (str, optional): The statistical moment to extract when aggregating. Options are 'avg' 
            (mean), 'std' (standard deviation), or 'max' (maximum). Defaults to 'avg'.
        replicate_histogram2 (bool, optional): Compatibility flag. If True, shifts differential data arrays 
            by dropping the first element to match older pipeline logic. Defaults to False.
        replicate_old_read_blob_data (bool, optional): Compatibility flag. If True, duplicates the last 
            element of differential arrays to match older pipeline logic. Defaults to False.
        read_l_mode_only (bool, optional): If True, filters the database to analyze only L-mode shots. 
            Defaults to False.
        read_h_mode_only (bool, optional): If True, filters the database to analyze only H-mode shots. 
            Defaults to False.
        filtered_blob_db (bool, optional): If True, reads from a specifically pre-filtered sub-database 
            of shots rather than the global list. Defaults to False.

    Returns:
        dict: A dictionary containing the aggregated blob parameters. 
            - If `averaging == 'shot'`, values are lists of scalar floats (one per shot).
            - If `averaging == 'blob'` or `'no'`, values are lists of dictionaries containing the `shot` 
              number and the un-averaged `data` array.
            Missing or failed shots are represented with `np.nan`.

    Notes:
        - The function dynamically extracts parameter keys from the first valid tracked structure it finds, 
          ensuring future additions to the tracking code are automatically captured.
        - Data is automatically saved to `<working_directory>/processed_data/` using a dynamically generated 
          filename based on the input flags.
    """
    
    if read_mean_results:
        averaging='shot'
    
    pickle_filename = f"{wd}/processed_data/blob_database_shot_by_shot_blob_{str_finding_method}_{averaging}_avg__{average}_calc__"
        
    if read_l_mode_only: pickle_filename += 'L_mode'
    if read_h_mode_only: pickle_filename += 'H_mode'
    pickle_filename += '.pickle'
    
    difference_method = 'pre'
            
    if read_l_mode_only:
        blob_database = read_blob_lh_mode_database_file(l_mode=True,
                                                        filtered_blob_db=filtered_blob_db,
                                                        time_range_around_peak=time_range_around_peak)
    elif read_h_mode_only:
        blob_database = read_blob_lh_mode_database_file(h_mode=True,
                                                        filtered_blob_db=filtered_blob_db,
                                                        time_range_around_peak=time_range_around_peak)
    else:
        if not isinstance(time_range_around_peak, (int, float)):
            print('time_range_around_peak needs to be a single number if l_mode or h_mode reading is not set. Setting it to 5e-3')
        time_range_around_peak=5e-3
        blob_database = read_blob_database_file(time_range_around_peak=time_range_around_peak)
        
    ncalc = len(blob_database['shot'])

    # Leave these empty; we will build them dynamically!
    full_blob_db_data = {}
    full_blob_db_error = {}

    if not os.path.exists(pickle_filename) or not nocalc:
        n_str = 0
        start_time = time_mod.time()
        keys_initialized = False
        analyzed_keys = []
        failed_shots={'shot':[],'index':[]}
        for ind in range(ncalc):
            shot = blob_database['shot'][ind]
            if shot in [137651, 139435, 139434]:
                failed_shots['shot'].append(shot)
                failed_shots['index'].append(ind)
                continue
            
            if isinstance(blob_database['time'][ind], (list, np.ndarray)):
                time_range = blob_database['time'][ind]
            else:
                time_range = [blob_database['time'][ind] - time_range_around_peak,
                              blob_database['time'][ind] + time_range_around_peak]

            blob_results = read_blob_data(shot,
                                          time_range,
                                          nocalc=True,
                                          recalc_tracking=recalc_tracking,
                                          min_structure_lifetime=min_structure_lifetime,
                                          str_finding_method=str_finding_method)
                                          
            # Skip invalid shots or shots with no structures
            if blob_results is None or blob_results.mode != 'tracked' or not blob_results.tracked_structures: 
                failed_shots['shot'].append(shot)
                failed_shots['index'].append(ind)
                continue
        
            # --- DYNAMIC DICTIONARY GENERATION ---
            if not keys_initialized:
                first_struct = blob_results.tracked_structures[0]
                
                # Combine keys from both dictionaries dynamically
                analyzed_keys = list(first_struct.regular_parameters.keys()) + list(first_struct.differential_parameters.keys())
                if 'Lifetime' not in analyzed_keys:
                    analyzed_keys += ['Normalized flux coordinate', 'Poloidal angle', 'Lifetime']                    
                    
                full_blob_db_data = {key: [] for key in analyzed_keys}
                full_blob_db_error = {key: [] for key in analyzed_keys}
                keys_initialized = True

            # Initialize the current shot's tracking dicts
            curr_shot_data = {key: [] for key in analyzed_keys}
            curr_shot_error = {key: [] for key in analyzed_keys}
            
            flap.delete_data_object('*')
            
            #Read the EFIT equilibrium once per shot instead of once per structure.
            shot_equilibrium = read_equilibrium_data(shot=shot)
            shot_equilibrium_slice = get_equilibrium_slice(equilibrium=shot_equilibrium,
                                                           time=np.mean(blob_database['time'][ind]),
                                                           shot=shot)

            # --- OOP EXTRACTION --- 
            for structure in blob_results.tracked_structures: 
                n_str += 1
                for key in analyzed_keys:
                    
                    is_differential = False
                    #Fit axes are accidentally interchanged
                    if key == 'Axes length minor fit':
                        new_key = 'Axes length major fit'
                    elif key == 'Axes length major fit':
                        new_key = 'Axes length minor fit'
                    else:
                        new_key=key
                    #Manually added new parameters because recalculating everything takes forever
                    #Eventually these need to be added to the database.
                    if key == 'Normalized flux coordinate' or key == 'Poloidal angle':
                        if key == 'Normalized flux coordinate':
                            try:
                                norm_flux_failed=False
                                psi_norm_target, theta_arc_target = get_flux_coord(shot=shot,
                                                                                   time=np.mean(blob_database['time'][ind]),
                                                                                   R_target=structure.regular_parameters['Centroid radial'].value,
                                                                                   z_target=structure.regular_parameters['Centroid poloidal'].value,
                                                                                   equilibrium_slice=shot_equilibrium_slice)
                                raw_data = psi_norm_target
                            except Exception as e:
                                print(f'Exception occurred at read_data_for_analyze_blob_database.py at line 157: {e}')
                                raw_data = copy.deepcopy(structure.regular_parameters['Intensity'].value)
                                raw_data[:]=np.nan
                                norm_flux_failed=True
                        else:
                            if norm_flux_failed:
                                raw_data = copy.deepcopy(structure.regular_parameters['Intensity'].value)
                                raw_data[:]=np.nan
                            else:
                                raw_data = theta_arc_target
                            
                    elif key == 'Lifetime':
                        raw_data = np.arange(len(structure.regular_parameters['Intensity'].value))*2.5e-6
                    # Safely extract the raw NumPy array and determine its type natively
                    elif key in structure.regular_parameters:
                        raw_data = structure.regular_parameters[new_key].value
                    elif key in structure.differential_parameters:
                        raw_data = structure.differential_parameters[new_key].value
                        is_differential = True
                    else:
                        continue 
                        
                    # Apply legacy shift logic using the dynamic `is_differential` flag
                    if replicate_old_read_blob_data:
                        if is_differential:
                            str_key_data = np.append(raw_data, raw_data[-1])  
                        else:
                            str_key_data = raw_data
                    elif replicate_histogram2:
                        if is_differential:
                            str_key_data = raw_data
                        else:
                            str_key_data = raw_data[1:]
                    else:  
                        str_key_data = raw_data
                        
                    str_key_data = np.asarray(str_key_data)
                    
                    
                    
                    if averaging == 'no':
                        curr_shot_data[new_key]=np.append(curr_shot_data[new_key],np.ravel(str_key_data))
                        
                    elif averaging in ['shot', 'blob']:
                        str_key_data = str_key_data[~np.isnan(str_key_data)]
                        if len(str_key_data) == 0: continue
                        
                        if averaging == 'shot':
                            curr_shot_error[new_key].append(np.sqrt(np.var(str_key_data)))
                                
                        if average == 'avg':
                            curr_shot_data[new_key].append(np.mean(str_key_data))
                        elif average == 'std':
                            curr_shot_data[new_key].append(np.sqrt(np.var(str_key_data)))
                        elif average == 'max':
                            curr_shot_data[new_key].append(np.max(str_key_data))
                    elif averaging == 'conditional':
                        curr_shot_data[new_key].append(str_key_data)
                        
            if averaging != 'conditional':
                for key in analyzed_keys:
                    try:
                        curr_shot_data[key]=np.concatenate(curr_shot_data[key])
                    except:
                        pass
                
            for key in full_blob_db_data.keys():
                if len(curr_shot_data[key]) == 0: 
                    print(f"{shot} dropped from the calculation")
                    if averaging == 'shot':
                        full_blob_db_data[key].append(np.nan)
                        full_blob_db_error[key].append(np.nan)
                    
                    elif averaging in ['no', 'blob']:
                        full_blob_db_data[key].append({'shot': shot, 'data': np.nan})
                    
                    elif averaging == 'conditional':
                        pass
                else:
                    if averaging == 'shot':
                        full_blob_db_data[key].append(np.mean(curr_shot_data[key]))
                        full_blob_db_error[key].append(np.mean(curr_shot_error[key]) / np.sqrt(len(curr_shot_error[key])))
                    elif averaging in ['no', 'blob']:
                        full_blob_db_data[key].append({'shot': shot, 'data': curr_shot_data[key]})
                    elif averaging == 'conditional':
                        pass                    
            
            elapsed_time = time_mod.time() - start_time
            avg_time_per_shot = elapsed_time / (ind + 1)
            remaining_time = avg_time_per_shot * (ncalc - ind - 1)

            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            seconds = int(remaining_time % 60)

            print(f'\rRemaining time from the calculation: {hours}h {minutes:02}min {seconds:02}sec', end="", flush=True)
            
        print(f'\nTotal number of structures: {n_str}')
        
        if len(failed_shots['shot']) > 0:
            for ind, shot in enumerate(failed_shots['shot']):
                for key in full_blob_db_data.keys():
                    if averaging == 'shot':
                        full_blob_db_data[key].insert(ind, np.nan)
                        full_blob_db_error[key].insert(ind, np.nan)
                    else:
                        full_blob_db_data[key].append({'shot': shot, 'data': np.nan})
        
        with open(pickle_filename, 'wb') as f:
            pickle.dump(full_blob_db_data, f)
    else:
        with open(pickle_filename, 'rb') as f:
            full_blob_db_data = pickle.load(f)

    return full_blob_db_data


def read_blob_data(shot,
                   time_range,
                   calculate_only=False,
                   nocalc=True,
                   min_structure_lifetime=20,
                   recalc_tracking=False,
                   str_finding_method='watershed',
                   max_gap=1,
                   verbose=False,
                   pdf=False,
                   plot=False,
                   ):
    """
    Wrapper function for executing `analyze_gpi_structures` with pre-configured defaults.

    This function simplifies the execution of the main GPI structure analysis routine 
    by exposing only the essential configuration parameters. It pre-sets the majority 
    of the underlying mathematical, tracking, and plotting flags to values optimized 
    for processing standard Gas Puff Imaging "blob" data.

    Args:
        shot (float or int): The experiment or shot number to analyze.
        time_range (list or tuple): The [start, end] time range for the calculation.
        calculate_only (bool, optional): If True, suppresses the returning of the 
            results dictionary and skips any remaining plotting instructions. 
            Defaults to False.
        nocalc (bool, optional): Switch for not recalculating already existing 
            results. If True, attempts to load data from an existing `.pickle` file. 
            Defaults to True.
        min_structure_lifetime (int, optional): Minimum required lifetime in frames 
            for a structure to be kept. Identified structures living shorter than 
            this will be dropped. Defaults to 20 (approx. 50us).
        recalc_tracking (bool, optional): If True, recalculates the tracking matrices 
            and paths without recalculating the underlying image segmentation. 
            Defaults to False.
        str_finding_method (str, optional): The mathematical method used for structure 
            identification and segmentation (either 'watershed' or 'contour'). 
            Defaults to 'watershed'.
        max_gap (int, optional): The maximum number of "missing" frames allowed 
            when tracking a structure to bridge segmentation errors. Defaults to 1.
        verbose (bool, optional): If False, suppresses standard output messages, 
            which is useful for keeping the console clean during long batch calculations. 
            Defaults to False.

    Returns:
        dict or None: 
            - If `calculate_only` is False and the analysis succeeds, returns a 
              dictionary containing the structured data (`{'data', 'time', 'structures'}`).
            - If `calculate_only` is True, or if an exception is caught during 
              execution, returns None.
    """
    
    try:
    # if True:
        blob_results = analyze_gpi_structures(exp_id=shot,
                                              time_range=time_range,
                                              normalize='simple',
                                              str_finding_method=str_finding_method,
                                              threshold_bg_multiplier=2.,
                                              ellipse_method='linalg',
                                              fit_shape='ellipse',
                                              smooth_contours=5,
                                              
                                              tracking='weighted',
                                              matrix_weight={'iou':1,'cccf':0},
                                              ignore_side_structures=True,
                                              remove_orphans=True,
                                              max_gap=max_gap,
                                              min_structure_lifetime=min_structure_lifetime,
                                              tracking_assignment='max_score',      #Method of assigning the correspondence, 'hungarian' or 'max_score'
                                              score_threshold=0.7,
                                              
                                              nocalc=nocalc,
                                              recalc_tracking=recalc_tracking,
                                              structure_pixel_calc=False,
                                              #fix_structure_angles=True,       #deprecated
                                              
                                              test_structures=False,
                                              return_results=not calculate_only,
                                              calculate_only=calculate_only,
                                              plot=plot,
                                              plot_str_by_str=True,
                                              plot_scatter=True,
                                              plot_tracking=True,
                                              calculate_rough_diff_velocities=False,
                                              plot_for_publication=True,
                                              pdf=pdf,
                                              structure_pdf_save=False,
                                              structure_video_save=False,
                                              
                                              
                                              test=False,
                                              verbose=verbose,
                                              )
        if not calculate_only:
            return blob_results

    except Exception as e:
       print('Exception in read_data_for_analyze_blob_database.py line 345.')
       print(e)
       print(f"Couldn't calculate {shot}, at {time_range} seconds.")
       
    if not calculate_only:
        return None


def read_blob_database_file(time_range_around_peak=5e-3,
                            blob_db_file='/Users/mlampert/work/NSTX_workspace/db/2010.csv',
                            elm_db_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv',
                            nofilter=False
                            ):
    """
    Reads a blob database and optionally filters it against an ELM database.

    (generated by Gemini)

    This function imports a CSV containing experimental blob data, selecting only 
    entries with a specific quality assessment flag (column index 2 == 0). If 
    filtering is enabled, it cross-references these shots with an ELM database. 
    If a blob's peak time falls within the start and end times of ELMs for that 
    shot, the blob's timestamp is shifted outside the ELM window (padded by 
    `time_range_around_peak`). Finally, it filters out older shots (<= 138127).

    Args:
        time_range_around_peak (float, optional): The time padding (in seconds) 
            Defaults to 5e-3.
        blob_db_file (str, optional): Absolute filepath to the CSV containing 
            the blob database. Defaults to 
            '/Users/mlampert/work/NSTX_workspace/db/2010.csv'.
        elm_db_file (str, optional): Absolute filepath to the CSV containing 
            the ELM database. Defaults to 
            '/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'.
        nofilter (bool, optional): If False, applies ELM overlap shifting and 
            removes specific shots (<= 138127). If True, skips the ELM overlap 
            filtering entirely. Defaults to False.

    Returns:
        dict: A dictionary containing the filtered database:
            - 'shot' (numpy.ndarray): Array of valid shot numbers.
            - 'time' (numpy.ndarray): Array of adjusted peak times (in seconds).
    """

    if isinstance(time_range_around_peak, (list, np.ndarray)):
        time_range_around_peak=np.mean(np.abs(time_range_around_peak))

    # Read blob database
    database = np.asarray(pandas.read_csv(blob_db_file))
    
    # Extract the actual 1D array of indices from the np.where tuple
    ind_shots = np.where(database[:, 2] == 0)[0] 
    
    blob_shots = database[ind_shots, 0]
    peak_times = database[ind_shots, 1] / 1000.  # Convert to seconds
    
    blob_database = {'shot': blob_shots,
                     'time': peak_times}

    if not nofilter:
        # Read ELM database
        db = pandas.read_csv(elm_db_file, index_col=0)
        elm_shots = np.asarray(db)[:, 1]
        elm_times = np.asarray(db)[:, 3]
        
        # Create a boolean mask to safely flag which blobs to drop without messing up the loop index
        keep_mask = np.ones(len(blob_database['shot']), dtype=bool)

        for i, shot_blob in enumerate(blob_database['shot']):
            ind_overlap = np.where(elm_shots == shot_blob)[0]
            original_blob_time = blob_database['time'][i]
            
            if len(ind_overlap) > 0:
                current_elm_times = elm_times[ind_overlap]
                min_time = np.min(current_elm_times)
                max_time = np.max(current_elm_times)
                
                # If the blob falls exactly within the ELM boundaries
                if min_time < original_blob_time < max_time:
                    if abs(original_blob_time - min_time) < abs(original_blob_time - max_time):
                        blob_database['time'][i] = min_time - 2 * time_range_around_peak
                    else:
                        blob_database['time'][i] = max_time + 2 * time_range_around_peak
                        
                # Check if the new time is shifted by more than 50ms from the original time
                if abs(blob_database['time'][i] - original_blob_time) > 50e-3:
                    keep_mask[i] = False  # Flag for removal instead of popping

        # Apply the mask to drop invalid blobs all at once
        blob_database['shot'] = blob_database['shot'][keep_mask]
        blob_database['time'] = blob_database['time'][keep_mask]

        # Final filter to keep only newer shots
        ind_newer = np.where(blob_database['shot'] > 138127)[0]
        blob_database['shot'] = blob_database['shot'][ind_newer]
        blob_database['time'] = blob_database['time'][ind_newer]

    return blob_database


def read_blob_lh_mode_database_file(l_mode=False,
                                    h_mode=False,
                                    filtered_blob_db=True, #filter the database to shots read by read_blob_database
                                    time_range_around_peak=None,
                                    filter_lh_transition=False,
                                    filter_elms=False,
                                    ):
    """
    Reads and filters a shot database specifically for L-mode or H-mode experiments.

    This function loads a comprehensive database ('db/2010_all.csv') and filters 
    it based on the 'comments by Ricky' column to isolate either L-mode or H-mode 
    shots. 
    
    The function operates in two primary modes:
    1. Filtered (`filtered_blob_db=True`): It cross-references the isolated L/H-mode 
       shots against the standard blob database (loaded via `read_blob_database_file`), 
       inheriting its strict ELM and quality filtering automatically.
    2. Unfiltered (`filtered_blob_db=False`): It applies its own filtering logic 
       directly to the '2010_all.csv' data, keeping only shots with a 'movie rating' 
       of 'A' or 'A+'. It can also optionally drop shots containing L-H transitions 
       or overlapping ELMs within the specified time window.

    Args:
        l_mode (bool, optional): If True, isolates shots marked as 'L-mode'. 
            Defaults to False.
        h_mode (bool, optional): If True, isolates shots marked as 'H-mode'. 
            Defaults to False.
        filtered_blob_db (bool, optional): If True, intersects the L/H mode shots 
            with the highly curated `read_blob_database_file` output. If False, 
            relies on manual movie ratings and custom time boundaries. Defaults to True.
        time_range_around_peak (float or list, optional): The time window to process 
            around a peak event. Used for shifting time ranges away from ELMs.
            - If float: The range is `[peak - range, peak + range]`.
            - If list: The range is `[peak - range[0], peak + range[1]]`.
            Defaults to None.
        filter_lh_transition (bool, optional): If True (and `filtered_blob_db=False`), 
            drops any shot that contains recorded L-H or H-L transition times. 
            Defaults to False.

    Raises:
        ValueError: If neither `l_mode` nor `h_mode` is set to True.

    Returns:
        dict: A dictionary containing the filtered database:
            - 'shot' (list or numpy.ndarray): Valid shot numbers.
            - 'time' (list or numpy.ndarray): Corresponding peak times (if 
              `time_range_around_peak` is None) OR a 2D array of [start, end] 
              time ranges (if `time_range_around_peak` is provided).
    """
    
    all_db_file = '/Users/mlampert/work/NSTX_workspace/db/2010_all.csv'
    
    if l_mode:
        find_string = 'L-mode'
    elif h_mode:
        find_string = 'H-mode'
    else:
        raise ValueError('Either l_mode or h_mode needs to be set.')
    
    all_database = pandas.read_csv(all_db_file)
    all_database['peak GPI time (ms)'] /= 1e3
    
    # Cast comments to string in case of NaNs in the CSV, then find indices
    comments = all_database['comments by Ricky'].astype(str)
    lh_mode_shot_inds = np.array([ind for ind, item in enumerate(comments) if find_string in item])
    lh_mode_shots = all_database['shot'].values[lh_mode_shot_inds]
    
    if filtered_blob_db:
        # Cross-reference with the previously built database
        if isinstance(time_range_around_peak, (list,np.ndarray)):
            time_range_around_peak = np.mean(np.abs(time_range_around_peak))
        blob_db = read_blob_database_file(time_range_around_peak=time_range_around_peak if time_range_around_peak is not None else 5e-3)
        blob_shots = blob_db['shot']
        blob_times = blob_db['time']
            
        # Keep only the shots that exist in BOTH databases
        valid_mask = np.isin(blob_shots, lh_mode_shots)
        lh_mode_shots_in_blob_db = blob_shots[valid_mask]
        lh_mode_times_in_blob_db = blob_times[valid_mask]
        
        database = {'shot': lh_mode_shots_in_blob_db,
                    'time': lh_mode_times_in_blob_db}
        
    else:
        # Apply custom filtering
        # 1. Filter by movie rating ('A' or 'A+')
        movie_ratings = all_database['movie rating'].values[lh_mode_shot_inds]
        rating_mask = np.isin(movie_ratings, ['A', 'A+'])
        lh_mode_shot_inds = lh_mode_shot_inds[rating_mask]

        if isinstance(time_range_around_peak, (int, float)):
            time_range_around_peak = [-time_range_around_peak, time_range_around_peak]

        if time_range_around_peak is not None:
            
            # 2. Filter L-H and H-L transitions
            if filter_lh_transition:
                lh_times = all_database['L-H time'].values[lh_mode_shot_inds]
                hl_times = all_database['H-L time'].values[lh_mode_shot_inds]
                
                # Keep only rows where BOTH transition times are NaN (no transition occurred)
                transition_mask = np.isnan(lh_times) & np.isnan(hl_times)
                lh_mode_shot_inds = lh_mode_shot_inds[transition_mask]

            # Construct final database using only the surviving indices
            final_shots = all_database['shot'].values[lh_mode_shot_inds]
            final_peaks = all_database['peak GPI time (ms)'].values[lh_mode_shot_inds]
            
            database = {
                'shot': final_shots,
                'time': np.array([final_peaks + time_range_around_peak[0], 
                                  final_peaks + time_range_around_peak[1]]).T
            }

        else:   
            database = {
                'shot': all_database['shot'].values[lh_mode_shot_inds],
                'time': all_database['peak GPI time (ms)'].values[lh_mode_shot_inds]
            }
        
    return database

def read_all_plasma_data(time_range_around_peak=[-5e-3,15e-3],
                         nocalc=False,
                         read_l_mode_only=False,
                         read_h_mode_only=False,
                         calculate_parameters_in_sol=False,
                         ):
    """
    Retrieves and aggregates bulk plasma parameters across a database of experimental shots.

    This function reads the specified blob database (either full, L-mode only, or 
    H-mode only) to determine the valid shots and their corresponding peak times. 
    It then fetches the core or Scrape-Off Layer (SOL) plasma parameters for each 
    shot at that specific time using the `read_plasma_data` routine. The aggregated 
    results are cached in a `.pickle` file to speed up subsequent executions.

    Args:
        time_range_around_peak (float, optional): The time padding (in seconds) 
            used to shift the analysis time away from overlapping ELMs when loading 
            the initial blob databases. Defaults to 5e-3.
        nocalc (bool, optional): If True, attempts to skip the data fetching loop 
            and directly load the aggregated plasma parameters from a cached 
            `.pickle` file. Defaults to False.
        read_l_mode_only (bool, optional): If True, restricts the data fetching 
            to shots identified as L-mode. Defaults to False.
        read_h_mode_only (bool, optional): If True, restricts the data fetching 
            to shots identified as H-mode. Defaults to False.
        calculate_parameters_in_sol (bool, optional): If True, fetches the plasma 
            parameters specifically from the Scrape-Off Layer (SOL) rather than 
            the core plasma. Appends '_sol' to the cache filename. Defaults to False.

    Returns:
        dict: A dictionary containing the aggregated plasma parameters across all 
            processed shots. The keys correspond to the specific plasma parameters 
            returned by `read_plasma_data` (e.g., electron density, temperature), 
            and the values are 1D NumPy arrays containing the values for each shot.
    """
    
    
    
    pickle_filename_plasma_l_mode=wd+'/processed_data/plasma_vs_blob_plasma_data_l_mode'
    pickle_filename_plasma_h_mode=wd+'/processed_data/plasma_vs_blob_plasma_data_h_mode'
    
    if read_l_mode_only:
        pickle_filename=pickle_filename_plasma_l_mode
    elif read_h_mode_only:
        pickle_filename=pickle_filename_plasma_h_mode
    else:
        pickle_filename=wd+'/processed_data/plasma_vs_blob_plasma_data_full'
    if calculate_parameters_in_sol:
        pickle_filename+='_sol'
        
    pickle_filename+='.pickle'
    
    # pickle_filename=wd+'/processed_data/blob_database_shot_by_shot_plasma.pickle'
    if read_l_mode_only:
        blob_database=read_blob_lh_mode_database_file(l_mode=True,
                                                      filtered_blob_db=False,
                                                      time_range_around_peak=time_range_around_peak)
    elif read_h_mode_only:
        blob_database=read_blob_lh_mode_database_file(h_mode=True,
                                                      filtered_blob_db=False,
                                                      time_range_around_peak=time_range_around_peak)
    else:
        blob_database=read_blob_database_file(time_range_around_peak=time_range_around_peak,)

    if (not os.path.exists(pickle_filename) or not nocalc):
        # or read_l_mode_only or read_h_mode_only or calculate_parameters_in_sol):
        
    # if True:
        ncalc=len(blob_database['shot'])
        curr_plasma_data=read_plasma_data(exp_id=blob_database['shot'][0],
                                          time=np.mean(blob_database['time'][0]),
                                          calculate_parameters_in_sol=calculate_parameters_in_sol)
        full_plasma_data={}
        for key in curr_plasma_data:
            full_plasma_data[key]=[]

        for ind in range(ncalc):
            blob_time=np.mean(blob_database['time'][ind])
            shot=blob_database['shot'][ind]
            try:
                curr_plasma_data=read_plasma_data(exp_id=shot,
                                                  time=blob_time,
                                                  calculate_parameters_in_sol=calculate_parameters_in_sol)
                for key in curr_plasma_data.keys():
                    full_plasma_data[key]=np.append(full_plasma_data[key],
                                                    curr_plasma_data[key])
            except Exception as e:
                print(f'Exception in read_all_plasma_data line 659: {e}')
                for key in full_plasma_data.keys():
                    full_plasma_data[key]=np.append(full_plasma_data[key],
                                                    np.nan)
                
        
        pickle.dump(full_plasma_data,open(pickle_filename,'wb'))
    else:
        full_plasma_data=pickle.load(open(pickle_filename,'rb'))
        
    return full_plasma_data


def read_plasma_parameters_for_table_in_paper(database=None,
                                              print_ranges=False,
                                              calculate_for_lh_study=False):
    """
    Extracts key global plasma parameters across a database of shots for summary tables.

    This function iterates through a given database of experimental shots and their 
    corresponding times. It can operate on a single provided database, or automatically 
    fetch and process separate L-mode and H-mode databases. 

    Args:
        database (dict, optional): A dictionary containing 'shot' and 'time' arrays. 
            If None, it automatically loads the default blob database via 
            `read_blob_database_file()`. Defaults to None.
        print_ranges (bool, optional): If True, calculates and prints the minimum 
            and maximum values for each of the extracted parameters to the console 
            after the loop completes. Defaults to False.
        calculate_for_lh_study (bool, optional): If True, fetches and processes 
            both the L-mode and H-mode databases instead of the default one. 
            Defaults to False.

    Returns:
        tuple or dict: 
            - If `calculate_for_lh_study` is False: Returns a tuple containing six lists 
              (collisionality, q95, greenwald, current, btoroidal, density).
            - If `calculate_for_lh_study` is True: Returns a dictionary where keys are 
              'L-Mode' and 'H-Mode', and values are the tuples of the six lists.
    """
    
    # --- 1. Database Setup ---
    databases_to_process = {}
    
    if calculate_for_lh_study:
        # Assuming your L/H reader supports these kwargs as seen in your other scripts
        databases_to_process['L-Mode'] = read_blob_lh_mode_database_file(l_mode=True, filter_lh_transition=True)
        databases_to_process['H-Mode'] = read_blob_lh_mode_database_file(h_mode=True, filter_lh_transition=True)
    else:
        if database is None:
            databases_to_process['Standard'] = read_blob_database_file()
        else:
            databases_to_process['Standard'] = database

    results = {}
    
    # [Assuming wd is defined globally in your script]
    pdf_pages_density = PdfPages(wd+'/plots/blob_database_density_fits.pdf')
    pdf_pages_temperature = PdfPages(wd+'/plots/blob_database_temperature_fits.pdf')

    # --- 2. Process Each Database ---
    for db_name, db in databases_to_process.items():
        if calculate_for_lh_study:
            print(f"\nProcessing {db_name} Database...")
            
        density = []
        current = []
        btoroidal = []
        greenwald = []
        collisionality = []
        q95 = []

        for ind_shot in range(len(db['shot'])):
            print(f"\r{ind_shot/len(db['shot'])*100:.1f} % done from the calculation.", flush=True, end='')
            
            # Handle cases where time might be an array/window instead of a scalar
            if isinstance(db['time'][ind_shot], (list, np.ndarray)):
                time_curr = np.mean(db['time'][ind_shot])
            else:
                time_curr = db['time'][ind_shot]
                
            shot = db['shot'][ind_shot]
            start_time = time_mod.time()

            try:
                plasma_parameters = read_plasma_data(exp_id=shot, time=time_curr,
                                                     pdf_pages_density=pdf_pages_density,
                                                     pdf_pages_temperature=pdf_pages_temperature)

                greenwald.append(plasma_parameters['Greenwald fraction'])
                density.append(plasma_parameters['Line integrated density'])
                q95.append(plasma_parameters['q95'])
                current.append(plasma_parameters['Current'])
                btoroidal.append(plasma_parameters['Toroidal field'])
                collisionality.append(plasma_parameters['Collisionality'])
                
            except Exception as e:
                # print(f"\nFailed to read plasma data for shot {shot}: {e}")
                continue

            # print('Finished in: ',time_mod.time()-start_time,'s')

        # --- 3. Print Ranges ---
        if print_ranges and len(collisionality) > 0:
            print(f'\n\n--- {db_name} Parameter Ranges ---')
            print(f'Collisionality range: {min(collisionality):.3e} to {max(collisionality):.3e}')
            print(f'Density range:        {min(density):.3e} to {max(density):.3e}')
            print(f'Greenwald range:      {min(greenwald):.3f} to {max(greenwald):.3f}')
            print(f'BT range:             {min(btoroidal):.3f} to {max(btoroidal):.3f}')
            print(f'Current range:        {min(current):.3f} to {max(current):.3f}')
            print(f'q95 range:            {min(q95):.3f} to {max(q95):.3f}')
            print('-' * 40)

        # Store results for this specific database
        results[db_name] = (collisionality, q95, greenwald, current, btoroidal, density)

    pdf_pages_density.close()
    pdf_pages_temperature.close()

    # --- 4. Return Output ---
    # Return the simple tuple if running in standard mode to prevent breaking older scripts
    if not calculate_for_lh_study:
        return results['Standard']
    
    # Return the dictionary of both modes for the LH study
    return results

def read_plasma_data(exp_id=None,
                     time=None,
                     pdf_pages_density=None,
                     pdf_pages_temperature=None,
                     calculate_parameters_in_sol=False,
                     temperature_threshold=5e-3
                     ):
    """
    Retrieves and calculates fundamental plasma parameters for a specific shot and time.

    This function pulls electron density, temperature, and pressure profiles from 
    Thomson scattering data, fits them to extract core and Scrape-Off Layer (SOL) 
    characteristics, and queries the MDSplus database for magnetic geometries and 
    global parameters. Finally, it derives key physics quantities such as collisionality, 
    Larmor radii, plasma frequencies, and the Greenwald fraction.

    Args:
        exp_id (int, optional): The experimental shot number to analyze. Defaults to None.
        time (float, optional): The time step (in seconds) to slice the data. Defaults to None.
        pdf_pages_density (PdfPages, optional): A matplotlib PdfPages object to save 
            density profile fit plots. Defaults to None.
        pdf_pages_temperature (PdfPages, optional): A matplotlib PdfPages object to save 
            temperature and pressure profile fit plots. Defaults to None.
        calculate_parameters_in_sol (bool, optional): If True, uses the SOL average values 
            for density and temperature instead of the peak profile values. Note that 
            uncertainty is generally higher in the SOL. Defaults to False.
        temperature_threshold (float, optional): The minimum allowed SOL temperature 
            in keV (default 5e-3, or 5 eV). Temperatures below this imply a detached 
            plasma; if encountered, the value is forced to NaN. Defaults to 5e-3.

    Returns:
        dict: A comprehensive dictionary containing raw profile characteristics, MDSplus 
            geometric data, and derived fundamental plasma physics parameters.
    """

    # --- 1. Thomson Profile Fetching & Fitting ---
    # Bundle the repeated arguments into a dictionary
    thomson_kwargs = {
        'exp_id': exp_id,
        'spline_data': True,
        'modified_tanh': False,
        'outboard_only': False,
        'device_coordinates': True,
        'radial_range': [1.3, 1.55],
        'plot_time_vec': time
    }

    ne_params = get_fit_nstx_thomson_profiles(density=True, pdf_object=pdf_pages_density, **thomson_kwargs)
    te_params = get_fit_nstx_thomson_profiles(temperature=True, pdf_object=pdf_pages_temperature, **thomson_kwargs)
    pe_params = get_fit_nstx_thomson_profiles(pressure=True, pdf_object=pdf_pages_temperature, **thomson_kwargs)

    ind = np.argmin(np.abs(ne_params['time_vec'] - time))

    # --- 2. Determine Core vs. SOL parameters ---
    if calculate_parameters_in_sol:
        n_e = ne_params['SOL avg'][ind]
        T_e = te_params['SOL avg'][ind] 
        if T_e < temperature_threshold: 
            T_e = np.nan
    else:
        n_e = ne_params['Value at max'][ind]
        T_e = te_params['Value at max'][ind] # In keV
    
    T_i = T_e # TODO: should be read from CHERS
    R_pressure_max_grad = pe_params['Position r'][ind]

    # --- 3. Internal Helper for MDSPlus Data ---
    
    def _fetch_mds(node, r_target=None):
        """Safely fetches and slices MDSPlus data, returning np.nan on failure."""
        try:
            d = flap.get_data('NSTX_MDSPlus', name=node, exp_id=exp_id).slice_data(slicing={'Time': time})
            if r_target is not None:
                d = d.slice_data(slicing={'Device R': r_target})
            return np.mean(d.data)
        except Exception as e:
            print(f"Failed to read {node} for shot {exp_id}: {e}")
            return np.nan

    # Fetch scalar MDSPlus values
    current       = _fetch_mds(r'\EFIT02::\IPMEAS')
    b_toroidal    = _fetch_mds(r'\EFIT02::\BT0')
    minor_radius  = _fetch_mds(r'\EFIT02::\AMINOR')
    R_separatrix  = _fetch_mds(r'\EFIT02::\RMIDOUT') - 0.02
    q95           = _fetch_mds(r'\EFIT02::\Q95')
    lower_triang  = _fetch_mds(r'\EFIT02::\TRIBOT')
    upper_triang  = _fetch_mds(r'\EFIT02::\TRITOP')
    elongation    = _fetch_mds(r'\EFIT02::\KAPPA')
    inner_gap     = _fetch_mds(r'\EFIT02::\GAPIN')
    outer_gap     = _fetch_mds(r'\EFIT02::\GAPOUT')
    cdens_95      = _fetch_mds(r'\EFIT02::\J95N')
    cdens_99      = _fetch_mds(r'\EFIT02::\J99N')
    q_boundary    = _fetch_mds(r'\EFIT02::\QL')
    R_mag_axis    = _fetch_mds(r'\EFIT02::\RMAXIS')
    z_outer_sp    = _fetch_mds(r'\EFIT02::\ZVSOUT')
    R_outer_sp    = _fetch_mds(r'\EFIT02::\RVSOUT')
    R_lower_xpt   = _fetch_mds(r'\EFIT02::\RXPT1')
    z_lower_xpt   = _fetch_mds(r'\EFIT02::\ZXPT1')

    # Fetch spatial MDSPlus values at max pressure gradient
    B_tor_grad = _fetch_mds(r'\EFIT02::\BTZ0', r_target=R_pressure_max_grad)
    B_pol_grad = _fetch_mds(r'\EFIT02::\BZZ0', r_target=R_pressure_max_grad)
    B_rad_grad = _fetch_mds(r'\EFIT02::\BRZ0', r_target=R_pressure_max_grad)
    magnetic_field = np.sqrt(B_tor_grad**2 + B_pol_grad**2 + B_rad_grad**2)

    # Fetch spatial MDSPlus values at separatrix for connection length
    B_tor_sep = _fetch_mds(r'\EFIT02::\BTZ0', r_target=R_separatrix)
    B_pol_sep = _fetch_mds(r'\EFIT02::\BZZ0', r_target=R_separatrix)

    # --- 4. Line Integrated Density ---
    try:
        d_ne = flap.get_data('NSTX_THOMSON', exp_id=exp_id, name='', object_name='THOMSON_DATA', 
                             options={'pressure':False, 'temperature':False, 'density':True, 
                                      'spline_data':False, 'add_flux_coordinates':False, 'force_mdsplus':False})
        ind_lid = np.argmin(np.abs(d_ne.coordinate('Time')[0][1,:] - time))
        norm_factor = np.max(d_ne.coordinate('Device R')[0][:,:], axis=0) - np.min(d_ne.coordinate('Device R')[0][:,:], axis=0)
        density = (np.trapz(d_ne.data[:,:], d_ne.coordinate('Device R')[0][:,:], axis=0) / norm_factor)[ind_lid]
    except Exception as e:
        print(f"Failed to read LID for shot {exp_id}: {e}")
        density = np.nan

    # --- 5. Physics Derived Parameters ---
    gamma = 1.0
    Z = 1.0
    ln_LAMBDA = 17
    m_i = 2.014 * 1.66e-27       # Deuterium mass (kg)
    m_e = 9.1093835e-31          # Electron mass (kg)
    q_e = 1.6e-19                # Elementary charge (C)
    k_B = 1.38e-23               # Boltzmann constant
    epsilon_0 = 8.854e-12        # Vacuum permittivity

    omega_pe = np.sqrt(n_e * q_e**2 / m_e / epsilon_0)
    omega_pi = np.sqrt(n_e * q_e**2 / m_i / epsilon_0)
    
    # Sound speed (convert T_e from keV to eV internally via 1e3)
    c_s = np.sqrt(gamma * Z * k_B * T_e * 1e3 * 11606 / m_i) 

    rho_e = 2.384e-6 * np.sqrt(T_e * 1e3) / magnetic_field
    rho_i = 1.019e-4 * np.sqrt(T_i * 1e3) / magnetic_field
    rho_s = rho_i

    ei_collision_rate = 2.9e-12 * n_e * Z**2 * ln_LAMBDA / ((T_e * 1e3)**(3/2))

    inverse_aspect = minor_radius / R_separatrix
    collisionality = ei_collision_rate * q95 * R_separatrix / c_s / inverse_aspect**1.5
    greenwald_fraction = density / (current / (np.pi * minor_radius**2) * 1e14)

    # Connection Length Calculation
    try:
        connection_length = np.sqrt((np.abs(z_outer_sp) * np.sqrt(B_pol_sep**2 + B_tor_sep**2) / B_pol_sep)**2 + 
                                    ((R_separatrix - R_lower_xpt) + (R_outer_sp - R_lower_xpt))**2)
        if connection_length < 1.5: 
            connection_length = np.nan
    except Exception as e:
        print(f"Failed to calculate connection length for shot {exp_id}: {e}")
        connection_length = np.nan

    omega_gyro_electron = q_e * np.abs(B_tor_grad) / m_e
    omega_gyro_ion = q_e * np.abs(B_tor_grad) / m_i
    
    collisionality_dimless = ei_collision_rate * connection_length / rho_s / omega_gyro_electron

    # --- 6. Compile Return Dictionary ---
    return {
        'Line integrated density': density,
        'Current': current,
        'Greenwald fraction': greenwald_fraction,
        'Toroidal field': b_toroidal,
        'Collision rate ei': ei_collision_rate,
        
        'Collisionality': collisionality,
        'Collisionality dimensionless': collisionality_dimless,
        'Connection length': connection_length,
        'q95': q95,
        'q boundary': q_boundary,
        
        'Magnetic field toroidal': B_tor_sep,
        'Magnetic field poloidal': B_pol_sep,
        'Magnetic field radial': B_rad_grad, 
        'Magnetic field absolute': magnetic_field,
        
        'Outer strike point R': R_outer_sp,
        'Outer strike point z': z_outer_sp,
        'Lower x point R': R_lower_xpt,
        'Lower x point z': z_lower_xpt,
        
        'Sound speed': c_s,
        'Plasma frequency': omega_pe,
        'Plasma frequency electron': omega_pe,
        'Plasma frequency ion': omega_pi,
        
        'Plasma elongation': elongation,
        'Plasma triangularity upper': upper_triang,
        'Plasma triangularity lower': lower_triang,
        'Plasma triangularity': (upper_triang + lower_triang) / 2,
        
        'Minor radius': minor_radius,
        'Magnetic axis radius': R_mag_axis,
        'Pedestal radius': R_separatrix,
        'Larmor radius': rho_e,
        'Larmor radius electron': rho_e,
        'Larmor radius ion': rho_i,
        'Larmor radius sound': rho_s,
        'Larmor frequency electron': omega_gyro_electron,
        'Larmor frequency ion': omega_gyro_ion,
        
        'Inner gap': inner_gap,
        'Outer gap': outer_gap,
        'Current density at 95': cdens_95,
        'Current density at 99': cdens_99,
        
        'Density at max': ne_params['Value at max'][ind],
        'Density pedestal height': ne_params['Height'][ind],
        'Density SOL offset': ne_params['SOL offset'][ind],
        'Density pedestal position': ne_params['Position'][ind],
        'Density pedestal width': ne_params['Width'][ind],
        'Density max gradient': ne_params['Max gradient'][ind],
        'Density SOL': n_e,
        
        'Temperature at max': te_params['Value at max'][ind],
        'Temperature pedestal height': te_params['Height'][ind],
        'Temperature SOL offset': te_params['SOL offset'][ind],
        'Temperature pedestal position': te_params['Position'][ind],
        'Temperature pedestal width': te_params['Width'][ind],
        'Temperature max gradient': te_params['Max gradient'][ind],
        'Temperature SOL': T_e,
        
        'Pressure at max': pe_params['Value at max'][ind],
        'Pressure pedestal height': pe_params['Height'][ind],
        'Pressure SOL offset': pe_params['SOL offset'][ind],
        'Pressure pedestal position': pe_params['Position'][ind],
        'Pressure pedestal width': pe_params['Width'][ind],
        'Pressure max gradient': pe_params['Max gradient'][ind],   
        'Pressure SOL': n_e * T_e,
    }

def read_plasma_data_old(exp_id=None,
                     time=None,
                     pdf_pages_density=None,
                     pdf_pages_temperature=None,
                     calculate_parameters_in_sol=False,
                     temperature_threshold=5e-3 #5eV threshold for SOL temperature, below plasma would be detached which it isn't
                     ):
    """
    Reads all important plasma parameters from the MDSplus database.

    Parameters
    ----------
    exp_id : integer, optional
        Shot number. The default is None.
    time : float, optional
        Time of the plasma parameter to be calculated. The default is None.
    pdf_pages_density : PdfPages matplotlib object, optional
        PDFpages matplotlib object for plotting the density profiles. The default is None.
    pdf_pages_temperature : PdfPages matplotlib object, optional
        PdfPages matplotlib object for plotting the density profiles. The default is None.
    calculate_parameters_in_sol : boolean, optional
        Switch for calculating parameters in the SOL. Here, uncertainty is high. The default is False.
    temperature_threshold : TYPE, optional
        #5eV threshold for SOL temperature, below plasma would be detached which it isn't. The default is 5e-3.

    Returns
    -------
    dict
        Dictionary with the plasma parameters.

    """

    #THESE READ THE ENTIRE SHOT"S PROFILES AND FIT THEM
    ne_params=get_fit_nstx_thomson_profiles(exp_id=exp_id,
                                            density=True,
                                            spline_data=True,
                                            modified_tanh=False,
                                            outboard_only=False,

                                            #flux_coordinates=True,
                                            # flux_range=[0.7,1.1],
                                            device_coordinates=True,
                                            radial_range=[1.3,1.55],
                                            pdf_object=pdf_pages_density,
                                            plot_time_vec=time
                                            )

    te_params=get_fit_nstx_thomson_profiles(exp_id=exp_id,
                                            temperature=True,
                                            spline_data=True,
                                            modified_tanh=False,
                                            outboard_only=False,

                                            #flux_coordinates=True,
                                            # flux_range=[0.7,1.1],
                                            device_coordinates=True,
                                            radial_range=[1.3,1.55],
                                            pdf_object=pdf_pages_temperature,
                                            plot_time_vec=time
                                            )
    
    pe_params=get_fit_nstx_thomson_profiles(exp_id=exp_id,
                                            pressure=True,
                                            spline_data=True,
                                            modified_tanh=False,
                                            outboard_only=False,

                                            #flux_coordinates=True,
                                            # flux_range=[0.7,1.1],
                                            device_coordinates=True,
                                            radial_range=[1.3,1.55],
                                            pdf_object=pdf_pages_temperature,
                                            plot_time_vec=time
                                            )
    
    ind=np.argmin(np.abs(ne_params['time_vec']-time))
    
    if calculate_parameters_in_sol:
        n_e=ne_params['SOL avg'][ind]
        T_e=te_params['SOL avg'][ind] #to convert from keV to Kelvins
        if T_e < temperature_threshold: T_e=np.nan
    else:
        n_e=ne_params['Value at max'][ind]
        T_e=te_params['Value at max'][ind] #This is in keV
    
    T_i=T_e                                                                     #TODO: should be read from CHERS
    
    R_pressure_max_grad=pe_params['Position r'][ind]
    
    """Line integrated density"""
    try:
        d_ne=flap.get_data('NSTX_THOMSON',
                        exp_id=exp_id,
                        name='',
                        object_name='THOMSON_DATA',
                        options={'pressure':False,
                                 'temperature':False,
                                 'density':True,
                                 'spline_data':False,
                                 'add_flux_coordinates':False,
                                 'force_mdsplus':False})

        ind=np.argmin(np.abs(d_ne.coordinate('Time')[0][1,:]-time))

        #goodind=np.where(np.logical_and(d_ne.coordinate('Flux r')[0][:,elm_index] < 1.0, d_ne.coordinate('Flux r')[0][:,elm_index] > 0))
        # dR = (d_ne.coordinate('Device R')[0][:,:]-
        #       np.insert(d_ne.coordinate('Device R')[0][0:-1,:],0,0,axis=0))
        norm_factor=(np.max(d_ne.coordinate('Device R')[0][:,:],axis=0)-
                     np.min(d_ne.coordinate('Device R')[0][:,:],axis=0))

        density=(np.trapz(d_ne.data[:,:],
                          d_ne.coordinate('Device R')[0][:,:],
                          axis=0)/norm_factor)[ind]
        #LID=np.sum(((d_ne.data[:,:])[:,:])*dR,axis=0)/np.sum(dR)
    except Exception as e:
        print(e)
        print('Failed to read LID for shot ',exp_id)
        density=np.nan

    """Plasma current"""
    try:
        current=np.mean(flap.get_data('NSTX_MDSPlus',
                                      name=r'\EFIT02::\IPMEAS',
                                      exp_id=exp_id,
                                      ).slice_data(slicing={'Time':time}).data)
    except Exception as e:
        print(e)
        print('Failed to read current for shot ',exp_id)
        current=np.nan

    """Toroidal field"""
    try:
        b_toroidal=np.mean(flap.get_data('NSTX_MDSPlus',
                                      name=r'\EFIT02::\BT0',
                                      exp_id=exp_id,
                                      ).slice_data(slicing={'Time':time}).data)
    except Exception as e:
        print(e)
        print('Failed to read Bt for shot ',exp_id)
        b_toroidal=np.nan

    """Minor radius"""
    try:
        minor_radius=flap.get_data('NSTX_MDSPlus',
                              name=r'\EFIT02::\AMINOR',
                              exp_id=exp_id,
                              ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read current for shot ',exp_id)
        minor_radius=np.nan

    """Pedestal radius"""
    try:
        R_separatrix=flap.get_data('NSTX_MDSPlus',
                                   name=r'\EFIT02::\RMIDOUT',
                                   exp_id=exp_id,
                                   ).slice_data(slicing={'Time':time}).data-0.02
    except Exception as e:
        print(e)
        print('Failed to read RMIDOUT for shot ',exp_id)
        R_separatrix=np.nan

    """Safety factor"""
    try:
        q95=flap.get_data('NSTX_MDSPlus',
                         name=r'\EFIT02::\Q95',
                         exp_id=exp_id,
                         ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read Q95 for shot ',exp_id)
        q95=np.nan

    """Lower triangularity"""
    try:
        lower_triang=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\TRIBOT',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read TRIBOT for shot ',exp_id)
        lower_triang=np.nan

    """Upper triangularity"""
    try:
        upper_triang=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\TRITOP',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read TRITOP for shot ',exp_id)
        upper_triang=np.nan

    """Elongation"""
    try:
        elongation=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\KAPPA',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read KAPPA for shot ',exp_id)
        elongation=np.nan

    """Inner gap"""
    try:
        inner_gap=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\GAPIN',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read GAPIN for shot ',exp_id)
        inner_gap=np.nan

    """Outer gap"""
    try:
        outer_gap=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\GAPOUT',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read GAPOUT for shot ',exp_id)
        outer_gap=np.nan

    """Current density at psi_norm=0.95"""
    try:
        cdens_95=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\J95N',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read J95N for shot ',exp_id)
        cdens_95=np.nan

    """Current density at psi_norm=0.99"""
    try:
        cdens_99=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\J99N',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read J99N for shot ',exp_id)
        cdens_99=np.nan

    """Toroidal field"""
    try:
    # if True:
        #These read the magnetic field on the mid-plane
        B_tor=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\BTZ0',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).slice_data(slicing={'Device R':R_pressure_max_grad}).data
    except Exception as e:
        print(e)
        print('Failed to read B_tor for shot ',exp_id)
        B_tor=np.nan
        
    try:
        B_pol=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\BZZ0',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).slice_data(slicing={'Device R':R_pressure_max_grad}).data
    except Exception as e:
        print(e)
        print('Failed to read poloidal field for shot ',exp_id)
        B_pol=np.nan
        
    try:    
        B_rad=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\BRZ0',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).slice_data(slicing={'Device R':R_pressure_max_grad}).data
        


    except Exception as e:
        print(e)
        print('Failed to read radial magnetic field for shot ',exp_id)
        B_rad=np.nan
        
    magnetic_field = np.sqrt(B_tor**2 + B_pol**2 + B_rad**2)
        
    """Safety factor at boundary"""
    
    try:
        q_boundary=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\QL',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read QL for shot ',exp_id)
        q_boundary=np.nan
    
    """Radial position of the magnetic axis"""
    
    try:
        R_mag_axis=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\RMAXIS',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read RMAXIS for shot ',exp_id)
        R_mag_axis=np.nan
    
    #\FPOL	

    gamma=3/3.
    Z=1.
    m_i=2.014*1.66e-27                                               # Deuterium mass
    m_e=9.1093835e-31
    q_e=1.6e-19

    ln_LAMBDA=17

    Z=1.
    k_B=1.38e-23                                                      #Boltzmann constant

    mu0=4*np.pi*1e-7
    epsilon_0=8.854e-12
    
    """Electron plasma frequency"""
    omega_pe=np.sqrt(n_e*q_e**2/m_e/epsilon_0)

    """Ion plasma srequency"""
    omega_pi=np.sqrt(n_e*q_e**2/m_i/epsilon_0)

    """Sound speed"""
    c_s=np.sqrt(gamma*Z*k_B*T_e*1e3*11606/m_i) #=v_te
    
    """Larmor radii"""
    rho_e=2.384e-6*np.sqrt(T_e*1e3)/magnetic_field                           #Electron Larmor radius, Not rewriting because other codes might crash
    rho_i=1.019e-4*np.sqrt(T_i*1e3)/magnetic_field                           #Ion Larmor radius
    
    rho_s=rho_i
    
    n_i=n_e

    #ei_collision_rate=(n_i * Z**2 * q_e**4 * ln_LAMBDA)/(T_e**(3/2)*np.sqrt(m_e)*epsilon_0**2*16*np.pi**2)
    ei_collision_rate=2.9e-12*n_i*Z**2*ln_LAMBDA/((T_e*1e3)**(3/2))
    
    """Collisionality"""
    inverse_aspect=minor_radius/R_separatrix
    collisionality=ei_collision_rate*q95*R_separatrix/c_s/inverse_aspect**1.5

    """Greenwald fraction"""
    greenwald_fraction=density/(current.copy()/(np.pi*minor_radius**2)*1e14) #From line integrated density
    
    
    """Connection length"""
    try:
    # if True:
        #These read the magnetic field on the mid-plane
        B_tor=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\BTZ0',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).slice_data(slicing={'Device R':R_separatrix}).data
        
        B_pol=flap.get_data('NSTX_MDSPlus',
                             name=r'\EFIT02::\BZZ0',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).slice_data(slicing={'Device R':R_separatrix}).data
        
        z_outer_strike_point=flap.get_data('NSTX_MDSPlus',
                                           name='r\EFIT02::\ZVSOUT',
                                           exp_id=exp_id,
                                           ).slice_data(slicing={'Time':time}).data
        
        R_outer_strike_point=flap.get_data('NSTX_MDSPlus',
                                           name=r'\EFIT02::\RVSOUT',
                                           exp_id=exp_id,
                                           ).slice_data(slicing={'Time':time}).data
        
        R_lower_x_point=flap.get_data('NSTX_MDSPlus',
                                      name=r'\EFIT02::\RXPT1',
                                      exp_id=exp_id,
                                      ).slice_data(slicing={'Time':time}).data
        
        z_lower_x_point=flap.get_data('NSTX_MDSPlus',
                                      name=r'\EFIT02::\ZXPT1',
                                      exp_id=exp_id,
                                      ).slice_data(slicing={'Time':time}).data
        
        connection_length=np.sqrt((np.abs(z_outer_strike_point)*np.sqrt(B_pol**2+B_tor**2)/B_pol)**2 + 
                                  ((R_separatrix-R_lower_x_point) + (R_outer_strike_point-R_lower_x_point))**2)
        
        if connection_length < 1.5: connection_length = np.nan
    except Exception as e:
        print(e)
        print('Failed to read J99N for shot ',exp_id)
        connection_length=np.nan
    
    """Electron gyrofrequency"""
    omega_gyro_electron=q_e*np.abs(B_tor)/m_e
    omega_gyro_ion=q_e*np.abs(B_tor)/m_i
    
    """Dimensionless collisionality"""
    #collisionality_dimless=1.7e-14*ne_params['Value at max'][ind]*connection_length/te_params['Value at max'][ind]**2 
    collisionality_dimless = ei_collision_rate*connection_length/rho_s/omega_gyro_electron
    

    return {'Line integrated density':density,
            
            'Current':current,
            'Greenwald fraction':greenwald_fraction,
            'Toroidal field':b_toroidal,
            'Collision rate ei':ei_collision_rate,
            
            'Collisionality':collisionality,
            'Collisionality dimensionless':collisionality_dimless,
            
            'Connection length':connection_length,
            'q95':q95,
            'q boundary':q_boundary,
            
            'Magnetic field toroidal':B_tor,                                    #at the separatrix
            'Magnetic field poloidal':B_pol,
            'Magnetic field radial':B_rad,
            'Magnetic field absolute':magnetic_field,                           #at the separatrix
            
            'Outer strike point R':R_outer_strike_point,
            'Outer strike point z':z_outer_strike_point,
            'Lower x point R':R_lower_x_point,
            'Lower x point z':z_lower_x_point,
            
            'Sound speed':c_s,
            'Plasma frequency':omega_pe,
            'Plasma frequency electron':omega_pe,
            'Plasma frequency ion':omega_pi,
            
            'Plasma elongation':elongation,
            'Plasma triangularity upper':upper_triang,
            'Plasma triangularity lower':lower_triang,
            'Plasma triangularity':(upper_triang+lower_triang)/2,
            
            'Minor radius':minor_radius,
            'Magnetic axis radius':R_mag_axis,
            'Pedestal radius':R_separatrix,
            'Larmor radius':rho_e,
            'Larmor radius electron':rho_e,
            'Larmor radius ion':rho_i,
            'Larmor radius sound':rho_s,
            'Larmor frequency electron':omega_gyro_electron,
            'Larmor frequency ion':omega_gyro_ion,
            
            'Inner gap':inner_gap,
            'Outer gap':outer_gap,
            'Current density at 95':cdens_95,
            'Current density at 99':cdens_99,
            
            'Density at max':ne_params['Value at max'][ind],
            'Density pedestal height': ne_params['Height'][ind],
            'Density SOL offset':ne_params['SOL offset'][ind],
            'Density pedestal position':ne_params['Position'][ind],
            'Density pedestal width':ne_params['Width'][ind],
            'Density max gradient':ne_params['Max gradient'][ind],
            'Density SOL':n_e,
            
            'Temperature at max':te_params['Value at max'][ind],
            'Temperature pedestal height': te_params['Height'][ind],
            'Temperature SOL offset':te_params['SOL offset'][ind],
            'Temperature pedestal position':te_params['Position'][ind],
            'Temperature pedestal width':te_params['Width'][ind],
            'Temperature max gradient':te_params['Max gradient'][ind],
            'Temperature SOL':T_e,
            
            'Pressure at max':pe_params['Value at max'][ind],
            'Pressure pedestal height': pe_params['Height'][ind],
            'Pressure SOL offset':pe_params['SOL offset'][ind],
            'Pressure pedestal position':pe_params['Position'][ind],
            'Pressure pedestal width':pe_params['Width'][ind],
            'Pressure max gradient':pe_params['Max gradient'][ind],   
            'Pressure SOL':n_e*T_e,
            
            }

def return_interesting_key_pairs(with_plasma_frequency=False):
    """
    Returns the interesting key paramaters for plotting data for publications.

    Parameters
    ----------
    with_plasma_frequency : boolean, optional
        Return the interesting key pairs including plasma frequency. The default is False.

    Returns
    -------
    interesting_key_pairs : array
        Returns an array with interesting key pairs for plotting. One can set whether the key pairs with plasma frequency are returned.
    units :                 dictionary
        Returns the axis labels for the interesting key pairs as [label, unit, multiplier for data to get it into "unit" units].

    """
    
    if not with_plasma_frequency:
        interesting_key_pairs=np.asarray([['Axes length minor fit','Line integrated density'], #
                                          ['Angle ALI','Line integrated density'],
                                          ['Angle ALI','Sound speed'],
                                          #['Angle of least inertia','Plasma frequency'],
                                          ['Angle ALI','Pressure at max'],
                                          ['Velocity poloidal centroid','Temperature pedestal width'],
                                          ['Velocity poloidal centroid','Collisionality'],
                                          #['Velocity poloidal centroid','Plasma frequency'],
                                          ['Velocity poloidal centroid','Density at max'],
                                          
                                          ['Angular velocity ALI','Line integrated density'],
                                          ['Angular velocity ALI','Collisionality'],
                                          
                                          #['Angular velocity ALI','Plasma frequency'],
                                          ])
    else:
        interesting_key_pairs=np.asarray([['Axes length minor fit','Line integrated density'], #
                                          ['Angle ALI','Line integrated density'],
                                          ['Angle ALI','Sound speed'],
                                          ['Angle ALI','Plasma frequency'],
                                          ['Angle ALI','Pressure at max'],
                                          ['Velocity poloidal centroid','Temperature pedestal width'],
                                          ['Velocity poloidal centroid','Collisionality'],
                                          ['Velocity poloidal centroid','Plasma frequency'],
                                          ['Velocity poloidal centroid','Density at max'],
                                          
                                          ['Angular velocity ALI','Line integrated density'],
                                          ['Angular velocity ALI','Collisionality'],
                                          
                                          ['Angular velocity ALI','Plasma frequency'],
                                          ])
    
    #Name, unit, multiplier
    units={'Axes length minor fit':['$b_{ellipse}$','mm', 1e3],
           'Angle ALI':['$\\theta_{blob}$','rad', 1],
           'Angular velocity ALI':['$\omega_{blob}$','krad/s',1e-3],
           'Velocity poloidal centroid':['$v_{pol}$','km/s', 1e-3],
           'Line integrated density':['$n_{e,LID}$','$10^{19}\\ m^{-3}$',1e-19],
           'Pressure at max':['$p_{e,max\,\\nabla p}$','kPa',1],
           'Density at max':['$n_{e, max\,\\nabla p}$','$10^{19}\\ m^{-3}$', 1e-19],
           'Temperature pedestal width':['$\\Delta_{T_e,ped}$','mm',1e3],
           'Sound speed':['$c_{s,max\,\\nabla p}$','km/s', 1e-3],
           'Plasma frequency':['$\omega_{p,e,max\,\\nabla p}$','GHz', 1e-9],
           'Collisionality':['$\\nu_{ei,max\,\\nabla p}$','-',1],
           'Connection length':['$\\L_{||}$','m',1],
           'Collisionality dimensionless':['$\\Lambda$','-',1],
           }
    
    return (interesting_key_pairs,units)

def read_blob_elm_database(time_range_around_peak=5e-3,
                           blob_db_file='/Users/mlampert/work/NSTX_workspace/db/2010.csv',
                           elm_db_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good_ne.csv',
                           nofilter=False
                           ):
    """
    Obsolete.

    Parameters
    ----------
    time_range_around_peak : TYPE, optional
        DESCRIPTION. The default is 5e-3.
    blob_db_file : TYPE, optional
        DESCRIPTION. The default is '/Users/mlampert/work/NSTX_workspace/db/2010.csv'.
    elm_db_file : TYPE, optional
        DESCRIPTION. The default is '/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good_ne.csv'.
    nofilter : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    database : TYPE
        DESCRIPTION.

    """

    database=np.asarray(pandas.read_csv(blob_db_file))
    ind_shots = np.where(database[:,2] == 0)[0]
    blob_shots = database[ind_shots, 0]
    peak_times = database[ind_shots, 1] / 1000.
    blob_database={'shot':blob_shots,
                   'time':peak_times}

    db=pandas.read_csv(elm_db_file, index_col=0)
    elm_shots=np.asarray(db)[:,1]
    elm_times=np.asarray(db)[:,3]
    elm_database={'shot':elm_shots,
                  'time':elm_times}
    database={}
    for key in ['shot','time']:
        database[key]=np.append(blob_database[key],elm_database[key])

    return database
