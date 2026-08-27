#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:52:23 2023

@author: mlampert
"""
#Core modules
import os
import time as time_mod
import pickle
import copy

from string import ascii_lowercase as alc

import warnings
warnings.filterwarnings("ignore")

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')

from flap_nstx.analysis import read_all_blob_data, read_blob_data, read_all_plasma_data
from flap_nstx.analysis import read_blob_database_file, read_blob_lh_mode_database_file
from flap_nstx.analysis import return_interesting_key_pairs

from flap_nstx.tools import plot_pearson_matrix, calculate_corr_acceptance_levels
from flap_nstx.tools import correlation, mutual_information, get_flux_coord
from flap_nstx.tools import read_equilibrium_data, get_equilibrium_slice
from flap_nstx.tools import filename as nstx_filename
from flap_nstx.gpi import calculate_flux_structure_keys

import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas
import ppscore as pps
import seaborn as sns

import scipy
from scipy.stats import linregress

#Plot settings for publications
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/plots'

FLUX_STRUCTURE_KEYS = ['Normalized flux coordinate',
                       'Poloidal angle',
                       'Lifetime',
                       'Poloidal angular velocity',
                       'Normalized flux coordinate velocity']


def _add_flux_parameters_to_saved_file(shot,
                                       time_range,
                                       str_finding_method='watershed',
                                       normalize='simple',
                                       remove_interlaced_structures=True,
                                       theta_method='geometric',
                                       overwrite=False,
                                       ):
    """
    Adds the flux coordinate based keys to an already calculated structure file.

    The structure finding and the tracking are NOT repeated, the tracked dataset is
    restored from the file written by `analyze_gpi_structures` and only the keys
    listed in `FLUX_STRUCTURE_KEYS` are calculated (via
    `calculate_flux_structure_keys`) and saved back into the same file(s).

    Args:
        shot (int): Shot number.
        time_range (list): The [start, end] time range the file was calculated for.
        str_finding_method (str, optional): Segmentation method used for the saved
            file ('watershed' or 'contour'). Defaults to 'watershed'.
        normalize (str, optional): Normalization used for the saved file, needed for
            reconstructing the filename. Defaults to 'simple'.
        remove_interlaced_structures (bool, optional): Interlace setting used for the
            saved file, needed for reconstructing the filename. Defaults to True.
        theta_method (str, optional): Poloidal angle definition, 'geometric' or
            'arclength'. Defaults to 'geometric'.
        overwrite (bool, optional): Recalculate the flux keys even if they are already
            present in the file. Defaults to False.

    Returns:
        bool: True if the file was (re)written, False otherwise.
    """

    comment = ''
    if normalize is not None:
        comment += normalize
    if remove_interlaced_structures:
        comment += '_nointer'
    comment += '_' + str_finding_method

    base_filename = nstx_filename(exp_id=shot,
                                  working_directory=wd + '/processed_data',
                                  time_range=time_range,
                                  purpose='structure char',
                                  comment=comment)

    pickle_filename = base_filename + '.pickle'
    hdf5_filename = base_filename + '.h5'

    if not os.path.exists(pickle_filename):
        print(f'  {pickle_filename} does not exist, nothing to extend.')
        return False

    try:
        with open(pickle_filename, 'rb') as f:
            tracked_dataset = pickle.load(f)
    except Exception as e:
        print(f'  Could not load {pickle_filename}: {e}')
        return False

    if getattr(tracked_dataset, 'mode', None) != 'tracked' or not tracked_dataset.tracked_structures:
        print(f'  {pickle_filename} does not contain tracked structures, skipping.')
        return False

    if not overwrite:
        first_struct = tracked_dataset.tracked_structures[0]
        existing_keys = (list(first_struct.regular_parameters.keys()) +
                         list(first_struct.differential_parameters.keys()))
        if all(key in existing_keys for key in FLUX_STRUCTURE_KEYS):
            print('  Flux parameters are already available in the file, skipping.')
            return False

    try:
        tracked_dataset = calculate_flux_structure_keys(tracked_dataset,
                                                        exp_id=shot,
                                                        theta_method=theta_method)
    except Exception as e:
        print(f'  Could not calculate the flux parameters for #{shot}: {e}')
        return False

    with open(pickle_filename, 'wb') as f:
        pickle.dump(tracked_dataset, f)
    print(f'  Flux parameters saved into {pickle_filename}')

    if os.path.exists(hdf5_filename):
        try:
            tracked_dataset.save_hdf5(hdf5_filename)
            print(f'  Flux parameters saved into {hdf5_filename}')
        except Exception as e:
            print(f'  Could not update {hdf5_filename}: {e}')

    return True


def calculate_all_blob_results(time_range_around_peak=[-5e-3,15e-3],
                               str_finding_method='watershed',
                               plot=False,
                               pdf=False,
                               nocalc=False,
                               recalc_tracking=False,
                               test=False,
                               calculate_for_lh_study=False,
                               calculate_l_mode_only=False,
                               calculate_h_mode_only=False,
                               download_data_only=False,
                               shot_range=None,  # <-- NEW PARAMETER
                               add_flux_parameters_only=False,
                               overwrite_flux_parameters=False,
                               theta_method='geometric',
                               ):
    
    """
    Executes batch processing of structure tracking across experimental databases.

    This function automates the blob tracking analysis (`read_blob_data`) over 
    entire sets of experimental shots. It can operate in two modes:
    1. Standard Mode: Processes a default database of blob shots.
    2. L/H Study Mode: Specifically processes curated L-mode and H-mode databases, 
       with options to only download the raw MDSplus data without running the 
       heavy tracking calculations.

    Args:
        time_range_around_peak (float or list, optional): The time padding (in seconds) 
            around the peak signal to define the analysis window. Defaults to [-5e-3, 15e-3].
        min_structure_lifetime (int, optional): Minimum frame lifetime for a 
            structure to be retained in the analysis. Defaults to 20.
        str_finding_method (str, optional): Algorithm used for image segmentation 
            ('watershed' or 'contour'). Defaults to 'watershed'.
        plot (bool, optional): Passed to `read_blob_data` to toggle plotting. 
            Defaults to False.
        pdf (bool, optional): Passed to `read_blob_data` to toggle PDF saving. 
            Defaults to False.
        nocalc (bool, optional): If True, skips calculation if a cached result 
            already exists. Defaults to False.
        recalc_tracking (bool, optional): If True, forces tracking recalculation 
            without recalculating the segmentation. Defaults to False.
        test (bool, optional): Unused test flag. Defaults to False.
        calculate_for_lh_study (bool, optional): If True, fetches and processes 
            the specific L-mode and H-mode databases instead of the default one. 
            Defaults to False.
        download_data_only (bool, optional): If True (and `calculate_for_lh_study` 
            is True), only downloads the raw 'NSTX_GPI' data via FLAP and skips 
            the structure tracking analysis. Defaults to False.
        shot_range (list or tuple, optional): Specify [min_shot, max_shot] to only 
            process a specific subset of shots. Defaults to None (processes all).
        add_flux_parameters_only (bool, optional): If True, no structure finding or 
            tracking is performed. The already existing result files of each shot are 
            loaded and only the flux coordinate based keys ('Normalized flux 
            coordinate', 'Poloidal angle', 'Lifetime', 'Poloidal angular velocity', 
            'Normalized flux coordinate velocity') are calculated and written back 
            into the very same files. Defaults to False.
        overwrite_flux_parameters (bool, optional): If True, the flux coordinate based 
            keys are recalculated even if they are already present in the saved file. 
            Only used when `add_flux_parameters_only` is True. Defaults to False.
        theta_method (str, optional): Poloidal angle definition. 'geometric' (default)
            is defined over the whole GPI field of view, 'arclength' is only valid on
            closed flux surfaces.
            
    Returns:
        None: Executes batch processing and saves results/data to disk.
    """
                
    # --- 1. Database Setup ---
    databases_to_process = []
    
    if not calculate_for_lh_study:
        # Standard database
        databases_to_process.append(
            read_blob_database_file(time_range_around_peak=time_range_around_peak)
        )
    else:
        # L/H Mode databases
        common_kwargs = {
            'filter_lh_transition': True,
            'filter_elms': False,
            'filtered_blob_db': False,
            'time_range_around_peak': time_range_around_peak
        }
        if calculate_l_mode_only:
            databases_to_process.append(read_blob_lh_mode_database_file(l_mode=True, **common_kwargs))
        if calculate_h_mode_only:
            databases_to_process.append(read_blob_lh_mode_database_file(h_mode=True, **common_kwargs))
        else:
            databases_to_process.append(read_blob_lh_mode_database_file(l_mode=True, **common_kwargs))
            databases_to_process.append(read_blob_lh_mode_database_file(h_mode=True, **common_kwargs))
    # --- 2. Batch Processing ---
    # Calculate total shots for accurate ETA tracking
    total_shots = sum(len(db['shot']) for db in databases_to_process)
    shots_processed = 0
    total_elapsed_time = 0.0

    for db in databases_to_process:
        for ind, shot in enumerate(db['shot']):
            
            # Filter older shots in LH study mode
            if calculate_for_lh_study and shot < 138113:
                total_shots -= 1 # Adjust total so ETA doesn't break
                continue
                
            # ===============================================================
            # NEW: Filter based on the requested shot range
            # ===============================================================
            if shot_range is not None and not (shot_range[0] <= shot <= shot_range[1]):
                total_shots -= 1 # Adjust total so ETA doesn't break
                continue
            # ===============================================================
                
            start_time = time_mod.time()

            # Time parsing logic
            if not calculate_for_lh_study:
                blob_time = db['time'][ind]
                
                # BUG FIX: Safely handle if the user passed a single float vs a list/tuple!
                if isinstance(time_range_around_peak, (list, np.ndarray, tuple)):
                    time_range = [blob_time + time_range_around_peak[0], 
                                  blob_time + time_range_around_peak[1]]
                else:
                    time_range = [blob_time - time_range_around_peak, 
                                  blob_time + time_range_around_peak]
            else:
                avg_time = np.mean(db['time'][ind,:])
                multiplier = 1e-3 if avg_time > 10 else 1.0  # Guard against ms vs s
                time_range = [db['time'][ind, 0] * multiplier, 
                              db['time'][ind, 1] * multiplier]

            # Execution logic
            if add_flux_parameters_only:
                print(f'Adding flux parameters to shot #{int(shot)} for window {time_range}...')
                _add_flux_parameters_to_saved_file(int(shot),
                                                   time_range,
                                                   str_finding_method=str_finding_method,
                                                   theta_method=theta_method,
                                                   overwrite=overwrite_flux_parameters)
            elif calculate_for_lh_study and download_data_only:
                print(f'Downloading shot #{int(shot)}...')
                try:
                    flap.get_data('NSTX_GPI', exp_id=int(shot), name='', object_name='GPI')
                except Exception as e:
                    print(f'Failed to download shot #{int(shot)}: {e}')
            else:
                print(f'Calculating shot #{int(shot)} for window {time_range}...')
                read_blob_data(int(shot),
                               time_range,
                               nocalc=nocalc,
                               pdf=pdf,
                               plot=plot,
                               recalc_tracking=recalc_tracking,
                               str_finding_method=str_finding_method,
                               max_gap=2 if calculate_for_lh_study else 1,
                               calculate_only=True)

            # Cleanup
            flap.delete_data_object('*')

            # --- 3. ETA Tracking ---
            shots_processed += 1
            execution_time = time_mod.time() - start_time
            total_elapsed_time += execution_time
            
            # Cumulative moving average for a stable ETA
            avg_time_per_shot = execution_time
            shots_remaining = total_shots - shots_processed
            remaining_hours = (avg_time_per_shot * shots_remaining) / 3600.
            
            print(f'Shot took {execution_time:.1f}s. Estimated time remaining: {remaining_hours:.2f} hours.\n')


def calculate_blob_parameter_histograms(time_range_around_peak=5e-3,
                                        n_bins=51,
                                        pdf=False,
                                        pdf_filename=None,
                                        plot=True,
                                        plot_for_publication=False,
                                        save_data_into_txt=False,
                                        calc_mean_distribution=False,
                                        nocalc=True,
                                        recalc_tracking=False,
                                        min_structure_lifetime=20,
                                        str_finding_method='watershed',
                                        analyze_h_mode_only=False,
                                        analyze_l_mode_only=False,
                                        filtered_blob_db=False, 
                                        plot_LH_diff=False,
                                        save_data_for_publication=False,
                                        ):
    import matplotlib
    
    if pdf:
        matplotlib.use('agg')
    else:
        matplotlib.use('qt5agg')

    wd = flap.config.get_all_section('Module NSTX_GPI')['Working directory']

    # --- 1. Filename Setup ---
    mean_str = 'mean' if calc_mean_distribution else 'nomean'
    
    if pdf_filename is None:
        pdf_filename = f"{wd}/plots/blob_database_parameter_histograms_{mean_str}_{str_finding_method}.pdf"

    pickle_filename = f"{wd}/processed_data/blob_database_full_data_{mean_str}_{str_finding_method}"
    if analyze_l_mode_only: pickle_filename += '_l_mode'
    elif analyze_h_mode_only: pickle_filename += '_h_mode'
    pickle_filename += '.pickle'
    print(pickle_filename)
    # --- 2. Data Extraction & Aggregation ---
    if not os.path.exists(pickle_filename) or not nocalc:
        
        # --- 3. Database Loading ---
        if not analyze_h_mode_only and not analyze_l_mode_only:
            blob_database = read_blob_database_file(time_range_around_peak=time_range_around_peak)
        else:
            blob_database = read_blob_lh_mode_database_file(h_mode=analyze_h_mode_only,
                                                            l_mode=analyze_l_mode_only,
                                                            time_range_around_peak=time_range_around_peak,
                                                            filtered_blob_db=filtered_blob_db)   
                                                            
        if isinstance(time_range_around_peak, (float, int)): 
            time_range_around_peak = [time_range_around_peak, time_range_around_peak]
            
        ncalc = len(blob_database['shot'])
        
        # Will be initialized dynamically on the first successful shot
        full_data = {} 
        analyzed_keys = []
        n_str = 0
        keys_initialized = False

        for ind in range(ncalc):
            start_time = time_mod.time()
            shot=int(blob_database['shot'][ind])
            if analyze_h_mode_only or analyze_l_mode_only:
                _time_range = list(blob_database['time'][ind])
            else:
                blob_time = blob_database['time'][ind]
                _time_range = [blob_time - time_range_around_peak[0], 
                               blob_time + time_range_around_peak[1]]
            try:
                # Assumes read_blob_data returns the new StructureDataset object
                blob_results = read_blob_data(shot,
                                              _time_range,
                                              nocalc=True,
                                              recalc_tracking=recalc_tracking,
                                              min_structure_lifetime=min_structure_lifetime,
                                              str_finding_method=str_finding_method)
            except Exception as e:
                print(f'\nException in read_blob_data Line 110: {e}')
                continue

            flap.delete_data_object('*')
            
            # Skip invalid shots or shots with no structures
            if blob_results is None or blob_results.mode != 'tracked' or not blob_results.tracked_structures: 
                continue

            # --- DYNAMIC DICTIONARY INITIALIZATION ---
            if not keys_initialized:
                first_struct = blob_results.tracked_structures[0]
                analyzed_keys = list(first_struct.regular_parameters.keys()) + list(first_struct.differential_parameters.keys())
                if 'Lifetime' not in analyzed_keys:
                    analyzed_keys += ['Normalized flux coordinate', 'Poloidal angle', 'Lifetime']   
                full_data = {key: [] for key in analyzed_keys}
                keys_initialized = True

            #Read the EFIT equilibrium once per shot instead of once per structure.
            shot_equilibrium = read_equilibrium_data(shot=shot)
            shot_equilibrium_slice = get_equilibrium_slice(equilibrium=shot_equilibrium,
                                                           time=np.mean(blob_database['time'][ind]),
                                                           shot=shot)

            # --- OOP DATA EXTRACTION ---
            for structure in blob_results.tracked_structures:
                n_str += 1
                for key in analyzed_keys:
                    
                    # Safely extract the data and dynamically check if it's differential
                    if key == 'Normalized flux coordinate' or key == 'Poloidal angle':
                        if key == 'Normalized flux coordinate':
                            try:
                                norm_flux_failed=False
                                psi_norm_target, theta_arc_target = get_flux_coord(shot=shot,
                                                                                   time=np.mean(blob_database['time'][ind]),
                                                                                   R_target=structure.regular_parameters['Centroid radial'].value,
                                                                                   z_target=structure.regular_parameters['Centroid poloidal'].value,
                                                                                   equilibrium_slice=shot_equilibrium_slice)
                                theta_arc_target = (theta_arc_target + np.pi) % (2 * np.pi) - np.pi
                                raw_data = psi_norm_target
                            except Exception as e:
                                print(f'Exception occurred at read_data_for_analyze_blob_database.py at line 157: {e}')
                                raw_data = copy.deepcopy(structure.regular_parameters['Centroid radial'].value)
                                raw_data[:]=np.nan
                                print(shot,np.mean(blob_database['time'][ind]))
                                norm_flux_failed=True
                        else:
                            if norm_flux_failed:
                                raw_data = copy.deepcopy(structure.regular_parameters['Centroid radial'].value)
                                raw_data[:]=np.nan
                            else:
                                raw_data = theta_arc_target
                            
                    elif key == 'Lifetime':
                        raw_data = np.arange(len(structure.regular_parameters['Centroid radial'].value))*2.5e-6
                    elif key in structure.regular_parameters:
                        raw_data = structure.regular_parameters[key].value
                        is_differential = False
                    elif key in structure.differential_parameters:
                        raw_data = structure.differential_parameters[key].value
                        is_differential = True
                    else:
                        continue 
                        
                    try:
                        if is_differential:
                            full_data[key].extend(raw_data)
                        else:
                            full_data[key].extend(raw_data[1:]) # Drop the first point to match lengths
                    except Exception as e:
                        print(f'Exception appending data for {key}: {e}')

            remaining_time = (time_mod.time() - start_time) * (ncalc - ind - 1)
            hours, rem = divmod(remaining_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f'\rRemaining time: {int(hours)}h {int(minutes):02d}min {int(seconds):02d}sec', end='', flush=True)

        print(f'\nTotal structures analyzed: {n_str}')
        
        # Convert lists to arrays for analysis
        for key in full_data:
            full_data[key] = np.array(full_data[key], dtype=float)
            
        with open(pickle_filename, 'wb') as f:
            pickle.dump(full_data, f)
    else:
        with open(pickle_filename, 'rb') as f:
            full_data = pickle.load(f)
            
            
    #return full_data #This is here for some reason. Might cause trouble with other codes.
    
    
    # --- 4. Plotting & Export ---
    if plot:
        ranges = {
                'Axes length minor': [0.0, 0.1],
                'Axes length minor diff': [-10e3,10e3],
                'Axes length major': [0.0, 0.2],
                'Axes length major diff': [-25e3,25e3],
                'Size radial': [0., 0.25],
                'Size radial diff': [-30e3,30e3],
                'Size poloidal': [0., 0.15],
                'Size poloidal diff': [-25e3, 25e3],
                'Intensity':[0,2000],
                'Signed area':[-0.01, 0.01],
                'Angle ALI': [-np.pi,np.pi],
                'Curvature':[0, 1000],
                'Angle fit': [-np.pi,np.pi],
                
                'Position radial fit': [1.4, 1.7], 
                'Position poloidal fit': [0., 0.35],
                'Velocity radial COG': [-10e3,10e3],
                'Velocity poloidal COG': [-20e3,20e3],
                'Velocity radial centroid': [-10e3,10e3],
                'Velocity poloidal centroid': [-20e3,20e3],
                'Velocity radial position fit': [-5e3, 5e3], 
                'Velocity poloidal position fit': [-20e3, 20e3],
                'Expansion fraction axes fit': [0.25, 2], 
                #'Elongation fit diff': [-0.075, 0.075], 
                'Angular velocity angle fit': [-250e3, 250e3],
                'Angular velocity ALI': [-250e3, 250e3],
                'Poloidal angle':[-0.2,0.8],
                'Area': [0, 0.006], 
                'Area diff': [-1.5e3, 1.5e3], 
                'Expansion fraction area': [0.25, 2], 
                'Convexity': [0.9, 1.0],
                'Convexity diff': [-50e3,50e3],
                'Solidity': [0.5, 1.0], 
                'Solidity diff': [-100e3, 100e3], 
                #'Total curvature': [0.9, 1.0],
                'Total bending energy': [0e8, 1.5e8], 
                'Total bending energy diff': [-1e14, 1e14], 
                
                #'Convexity diff': [-0.01, 0.01],
                #'Solidity diff': [-0.25, 0.25], 
                #'Total curvature diff': [-0.05, 0.05],
                #'Total bending energy diff': [-0.3e8, 0.3e8], 
                #'Area diff': [-0.0015, 0.0015],
                }   

        discrete_step_size={'Axes length minor':0.00375, 
                            'Axes length major':0.00375,
                            'Angle envelope':np.pi/51,
                            'Size radial':0.00375, 
                            'Size poloidal':0.00375,
                            'Axes length minor diff':0.00375/2.5e-6, 
                            'Axes length major diff':0.00375/2.5e-6,
                            'Angle envelope diff':np.pi/64/2.5e-6,
                            'Size radial diff':0.00375/2.5e-6, 
                            'Size poloidal diff':0.00375/2.5e-6,
                            'Lifetime':2.5e-6,
                            }

        if plot_for_publication:
            multiplier = {
                'Area': 1e4, 
                'Area diff': 1e4, 
                'Angle fit': 1, 
                'Angular velocity angle fit': 1e-3,
                'Roundness': 1, 
                'Roundness diff': 1e3, 
                'Total curvature': 1, 
                'Total curvature diff': 1e3
            }
            xlabel = {
                'Area': ['Area', '[$\\rm cm^2$]'], 
                'Area diff': ['$\\rm\\Delta$Area', '[$\\rm cm^2$]'],
                'Angle fit': ['Angle', '[rad]'], 
                'Angular velocity angle fit': ['$\\rm\\omega$', '[krad/s]'],
                'Roundness': ['Roundness', '[a.u.]'], 
                'Roundness diff': ['$\\rm\\Delta$Roundness', '[a.u.]'],
                'Total curvature': ['Curvature', '[a.u.]'], 
                'Total curvature diff': ['$\\rm\\Delta$Curvature', '[a.u.]']
            }
            target_keys = ['Area', 
                           'Area diff', 
                           'Angle fit', 
                           'Angular velocity angle fit',
                           'Roundness', 
                           'Roundness diff', 
                           'Total curvature', 
                           'Total curvature diff']
            
            labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

            if not plot_LH_diff:
                suffix = '_L_mode' if analyze_l_mode_only else '_H_mode' if analyze_h_mode_only else ''
                pdf_page = PdfPages(f"{wd}/plots/8hist_blob_db_LT{min_structure_lifetime}_{str_finding_method}_{suffix}.pdf")
                fig, axes = plt.subplots(4, 2, figsize=(8.5/2.54, 17/2.54))

                for ind, key in enumerate(target_keys):
                    data = full_data[key][~np.isnan(full_data[key])] * multiplier[key]
                    ax = axes[ind // 2, ind % 2]
                    
                    if key == 'Angle fit' or key == 'Angle ALI':
                        data = np.mod(data, np.pi)

                    hist_range = np.asarray(ranges[key]) * multiplier[key] if key in ranges else None
                    n, bins, _ = ax.hist(data, 
                                         bins=n_bins, 
                                         weights=np.ones_like(data)/len(data), 
                                         range=hist_range)

                    if save_data_for_publication:
                        with open(f"{wd}/{labels[ind]}_db_histogram_{key}.txt", 'w+') as file1:
                            for i in range(len(n)):
                                file1.write(f"{(bins[1:] + bins[:-1])[i]/2}\t{n[i]}\n")

                    plt.locator_params(axis='y', nbins=5)
                    ax.set_xlabel(f"{xlabel[key][0]} {xlabel[key][1]}")
                    ax.set_ylabel('Relative frequency')
                    ax.set_title(f"Histogram of \n {xlabel[key][0]}")
                    ax.text(-0.4, 1.1, f"({labels[ind]})", transform=ax.transAxes, size=9)
                    if ind % 2 == 1:
                        ax.axvline(x=0, color='red')

                plt.tight_layout(pad=0.1)
                pdf_page.savefig()
                pdf_page.close()
                plt.close(fig)

            else:
                # Plot L and H mode differences
                l_file = f"{wd}/processed_data/blob_database_full_data_nomean_{str_finding_method}_l_mode.pickle"
                h_file = f"{wd}/processed_data/blob_database_full_data_nomean_{str_finding_method}_h_mode.pickle"
                
                with open(l_file, 'rb') as f: data_l_mode = pickle.load(f)
                with open(h_file, 'rb') as f: data_h_mode = pickle.load(f)

                pdf_page = PdfPages(f"{wd}/plots/8hist_blob_db_LT{min_structure_lifetime}_{str_finding_method}LH_diff.pdf")
                fig, axes = plt.subplots(4, 2, figsize=(8.5/2.54, 17/2.54))

                for ind, key in enumerate(target_keys):
                    l_data = data_l_mode[key][~np.isnan(data_l_mode[key])] * multiplier[key]
                    h_data = data_h_mode[key][~np.isnan(data_h_mode[key])] * multiplier[key]

                    if key == 'Angle fit' or key == 'Angle ALI':
                        l_data = np.mod(l_data.astype(float), np.pi)
                        h_data = np.mod(h_data.astype(float), np.pi)

                    # Output LaTeX Table string
                    values = [np.mean(l_data), np.mean(h_data),
                              np.sqrt(np.var(l_data)), np.sqrt(np.var(h_data)),
                              scipy.stats.skew(l_data), scipy.stats.skew(h_data),
                              scipy.stats.kurtosis(l_data), scipy.stats.kurtosis(h_data)
                              ]
                    formatted = " & ".join(f"{v:.3f}" for v in values)
                    print(f"{xlabel[key][0]} {xlabel[key][1]} & {formatted} \\\\")

                    ax = axes[ind // 2, ind % 2]
                    hist_range = np.asarray(ranges[key]) * multiplier[key] if key in ranges else None

                    # Loop through L and H mode for concise plotting
                    for mode_name, dataset in [('L mode', l_data), ('H mode', h_data)]:
                        if key in discrete_step_size.keys():
                            step_size = discrete_step_size[key]  
                            
                            # Find the absolute min and max of your data
                            min_val = np.floor(np.min(dataset))
                            max_val = np.ceil(np.max(dataset))
                            
                            # Generate explicit bin edges shifted by half a step
                            discrete_bins = np.arange(min_val - step_size/2, max_val + step_size, step_size)
                            if len(discrete_bins) > 101:
                                discrete_bins=discrete_bins[0:101]
                            # Pass the array to the 'bins' argument instead of an integer!
                            ax.hist(dataset, 
                                    bins=discrete_bins, 
                                    weights=np.ones_like(dataset)/len(dataset),
                                    range=hist_range, 
                                    alpha=0.5,
                                    label=mode_name)
                        else:
                            ax.hist(dataset, 
                                    bins=51, 
                                    weights=np.ones_like(dataset)/len(dataset),
                                    range=hist_range, 
                                    alpha=0.5, 
                                    label=mode_name)


                    plt.locator_params(axis='y', nbins=5)
                    ax.set_xlabel(f"{xlabel[key][0]} {xlabel[key][1]}")
                    ax.set_ylabel('Relative frequency')
                    ax.set_title(f"Histogram of \n {xlabel[key][0]}")
                    ax.text(-0.4, 1.1, f"({labels[ind]})", transform=ax.transAxes, size=9)
                    ax.legend(fontsize=5)
                    if ind % 2 == 1:
                        ax.axvline(x=0, color='red')

                plt.tight_layout(pad=0.1)
                pdf_page.savefig()
                pdf_page.close()
                plt.close(fig)

        else:
            # Standard single plots
            if pdf:
                pdf_page = PdfPages(pdf_filename)
            if not plot_LH_diff:
                # Safe fallback if analyzed_keys isn't defined (e.g. nocalc=True)
                keys_to_plot = analyzed_keys if 'analyzed_keys' in locals() and analyzed_keys else list(full_data.keys())
                
                for key in keys_to_plot:
                    clean_data = full_data[key][~np.isnan(full_data[key])]
                    try:
                        fig, ax = plt.subplots(figsize=(8.5/2.54, 8.5/2.54))
                        ax.hist(clean_data, 
                                bins=51)
                        ax.set_xlabel(f"{key} bins")
                        ax.set_ylabel('Relative frequency')
                        ax.set_title(f"Histogram of {key}")
                        if key in ranges:
                            ax.set_xlim(ranges[key])
                        
                        if pdf:
                            pdf_page.savefig()
                        plt.close(fig)
                    except Exception:
                        print(f"Failed to plot {key}")
                
                if pdf:
                    pdf_page.close()
            else:
                # Standard single plots with L-mode and H-mode overlapping for ALL keys
                l_file = f"{wd}/processed_data/blob_database_full_data_nomean_{str_finding_method}_l_mode.pickle"
                h_file = f"{wd}/processed_data/blob_database_full_data_nomean_{str_finding_method}_h_mode.pickle"
                
                try:
                    with open(l_file, 'rb') as f: data_l_mode = pickle.load(f)
                    with open(h_file, 'rb') as f: data_h_mode = pickle.load(f)
                except FileNotFoundError as e:
                    print(f"Could not load L/H mode comparison files for single plotting: {e}")
                    return full_data
    
                if pdf:
                    pdf_page = PdfPages(pdf_filename)
                
                # Safe fallback to iterate through absolutely every key available
                keys_to_plot = analyzed_keys if 'analyzed_keys' in locals() and analyzed_keys else list(data_l_mode.keys())
                
                for key in keys_to_plot:
                    if key not in data_l_mode or key not in data_h_mode:
                        continue
    
                    # Strip NaNs
                    l_data = data_l_mode[key][~np.isnan(data_l_mode[key])]
                    h_data = data_h_mode[key][~np.isnan(data_h_mode[key])]
    
                    # Ensure all angular keys are properly wrapped to [0, pi]
                    # if 'Angle' in key:
                    #     l_data = np.mod(l_data.astype(float), np.pi)
                    #     h_data = np.mod(h_data.astype(float), np.pi)
    
                    try:
                        fig, ax = plt.subplots(figsize=(8.5/2.54, 8.5/2.54))
                        hist_range = ranges[key] if key in ranges else None
    
                        # Loop through L and H mode for concise overlapping plotting
                        for mode_name, dataset, color in [('L mode', l_data, 'blue'), ('H mode', h_data, 'orange')]:
                            if key in discrete_step_size.keys():
                                step_size = discrete_step_size[key]  
                                
                                # Find the absolute min and max of your data
                                min_val = np.floor(np.min(dataset))
                                max_val = np.ceil(np.max(dataset))
                                
                                # Generate explicit bin edges shifted by half a step
                                discrete_bins = np.arange(min_val - step_size/2, max_val + step_size, step_size)
                                if len(discrete_bins) > 101:
                                    discrete_bins=discrete_bins[0:101]
                                # Pass the array to the 'bins' argument instead of an integer!
                                ax.hist(dataset, 
                                        bins=discrete_bins, 
                                        weights=np.ones_like(dataset)/len(dataset),
                                        range=hist_range, 
                                        alpha=0.5,
                                        color=color,
                                        label=mode_name)
                            else:
                                if len(dataset) > 0:
                                    ax.hist(dataset, 
                                            bins=51, 
                                            weights=np.ones_like(dataset)/len(dataset),
                                            range=hist_range, 
                                            alpha=0.5,
                                            color=color,
                                            label=mode_name)

    
                        ax.set_xlabel(f"{key} bins")
                        ax.set_ylabel('Relative frequency')
                        ax.set_title(f"Histogram of {key}")
                        ax.legend(fontsize=7)
                        
                        if key in ranges:
                            ax.set_xlim(ranges[key])
                        
                        plt.tight_layout(pad=0.1)
                        if pdf:
                            pdf_page.savefig(fig)
                        plt.close(fig)
                        
                    except Exception as e:
                        print(f"Failed to plot {key}: {e}")
                
                if pdf:
                    pdf_page.close()


    return full_data


def calculate_blob_blob_parameter_correlation_matrix(time_range_around_peak=5e-3,
                                                     threshold_corr=False,
                                                     pdf=True,
                                                     pdf_filename=None,
                                                     calc_mean_distribution=False,
                                                     plot_interesting_only=False,
                                                     recalc_tracking=False,
                                                     str_finding_method='watershed',
                                                     nocalc=True,
                                                     averaging='no',
                                                     average=['avg', 'avg'],
                                                     fix_angle_for_correlation=True,
                                                     min_structure_lifetime=20,
                                                     analyze_h_mode_only=False,
                                                     analyze_l_mode_only=False,
                                                     analyze_lh_difference=False,
                                                     save_data_for_publication=False,
                                                     ):
    import matplotlib
    
    if pdf:
        matplotlib.use('agg')
    else:
        matplotlib.use('qt5agg')

    # --- 1. Mode and Filename Setup ---
    if analyze_h_mode_only:
        plasma_mode = 'h_mode'
    elif analyze_l_mode_only:
        plasma_mode = 'l_mode'
    elif analyze_lh_difference:
        plasma_mode = 'lh_diff'
    else:
        plasma_mode = ''
        
    if pdf_filename is None:
        if averaging == 'no':
            pdf_filename = f"{wd}/plots/correlation_matrix_blob_blob_{str_finding_method}_full_{plasma_mode}.pdf"
        else:
            pdf_filename = f"{wd}/plots/correlation_matrix_blob_blob_{str_finding_method}_{averaging}_{average[0]}_{average[1]}_{plasma_mode}.pdf"

    # --- 2. Define Target Keys ---
    if plot_interesting_only:
        analyzed_keys = ['Area', 
                         'Area diff', 
                         'Axes length major fit', 
                         'Axes length minor fit',
                         'Convexity', 
                         'Elongation fit', 
                         'Position radial fit', 
                         'Roundness',
                         'Roundness diff', 
                         'Size radial fit', 
                         'Velocity radial position fit']  
        
        gpi_labels = ['Area', 
                      '$\\rm \\Delta$Area', 
                      'Major semi-axis', 
                      'Minor semi-axis',
                      'Convexity', 
                      'Elongation', 
                      '$\\rm R_{pos}$', 
                      'Roundness',
                      '$\\rm \\Delta$Roundness', 
                      '$\\rm d_{rad}$', 
                      '$\\rm v_{rad}$']
    else:
        analyzed_keys = None # Will be populated dynamically later
        gpi_labels = None

    # --- 3. Internal Math Helper ---
    def _compute_corr_matrix(data_dict, keys, averaging):
        """Helper to calculate the Pearson matrix for a given data dictionary."""
        
        # BUG FIX: Safely extract the data depending on the averaging mode!
        processed_data = {}
        for key in keys:
            if key not in data_dict:
                processed_data[key] = np.nan # Failsafe
                continue
                
            if averaging == 'no':
                # Data is a list of dictionaries [{'shot':123, 'data':[1,2,3]}, ...]
                # Flatten it into a 1D array
                if len(data_dict[key]) > 0 and isinstance(data_dict[key][0], dict):
                    try:
                        processed_data[key] = np.concatenate([shot_dict['data'] for shot_dict in data_dict[key]])
                    except Exception as e:
                        print(e)
                        print(key, data_dict[key])
                        raise ValueError
                else:
                    processed_data[key] = np.array(data_dict[key])
            else:
                # Data is already a 1D array of shot averages
                processed_data[key] = np.array(data_dict[key])

        matrix = np.zeros([len(keys), len(keys)])
        for i, key1 in enumerate(keys):
            try:
                valid1 = ~np.isnan(processed_data[key1])
            except Exception as e:
                print(f'Exception in analyze_blob_database.py line 682: {e}')
                matrix[:, i] = np.nan
                continue
                
            for j, key2 in enumerate(keys):
                try:
                    valid2 = ~np.isnan(processed_data[key2])
                    valid_mask = valid1 & valid2                    
                    d1 = np.real(processed_data[key1][valid_mask])
                    d2 = np.real(processed_data[key2][valid_mask])
                    
                    if fix_angle_for_correlation:
                        if key1 in ['Angle fit', 'Angle ALI']:
                            d1 = np.mod(d1, np.pi/2)
                        if key2 in ['Angle fit', 'Angle ALI']:
                            d2 = np.mod(d2, np.pi/2)
                    
                    d1 -= np.mean(d1)
                    d2 -= np.mean(d2)
                    
                    # Pearson correlation coefficient
                    # Add safety check for zero division!
                    denom = np.sqrt(np.sum(d1**2) * np.sum(d2**2))
                    if denom == 0:
                        matrix[j, i] = np.nan
                    else:
                        matrix[j, i] = np.sum(d1 * d2) / denom
                        
                except Exception as e:
                    print(f"Failed correlation for {key1} & {key2}: {e}")
                    matrix[j, i] = np.nan
        return matrix

    # --- 4. Data Loading and Matrix Calculation ---
    if not analyze_lh_difference:
        full_data = read_all_blob_data(time_range_around_peak = time_range_around_peak,
                                       min_structure_lifetime = min_structure_lifetime,
                                       averaging = 'shot' if calc_mean_distribution else 'no',
                                       nocalc = nocalc,
                                       recalc_tracking = recalc_tracking,
                                       str_finding_method = str_finding_method,
                                       read_l_mode_only = analyze_l_mode_only,
                                       read_h_mode_only = analyze_h_mode_only,
                                       replicate_histogram2 = True,
                                       )
        # BUG FIX: Handle the dynamic keys properly!
        plot_keys = analyzed_keys if plot_interesting_only else list(full_data.keys())
        
        correlation_matrix = _compute_corr_matrix(full_data, plot_keys, averaging)
        colormap = 'seismic'
        
    else:
        # Calculate L-mode
        full_data_l  =  read_all_blob_data(time_range_around_peak = time_range_around_peak,
                                         min_structure_lifetime = min_structure_lifetime,
                                         averaging = 'shot' if calc_mean_distribution else 'no',
                                         nocalc = nocalc, 
                                         recalc_tracking = recalc_tracking,
                                         str_finding_method = str_finding_method,
                                         read_l_mode_only = True, 
                                         replicate_histogram2 = True
                                         )   
        # Calculate H-mode
        full_data_h  =  read_all_blob_data(time_range_around_peak = time_range_around_peak,
                                         min_structure_lifetime = min_structure_lifetime,
                                         averaging = 'shot' if calc_mean_distribution else 'no',
                                         nocalc = nocalc, 
                                         recalc_tracking = recalc_tracking,
                                         str_finding_method = str_finding_method,
                                         read_h_mode_only = True, 
                                         replicate_histogram2 = True
                                         )
        
        # BUG FIX: Handle the dynamic keys properly!
        plot_keys = analyzed_keys if plot_interesting_only else list(full_data_l.keys())
        corr_l  =  _compute_corr_matrix(full_data_l, plot_keys, averaging)
        corr_h = _compute_corr_matrix(full_data_h, plot_keys, averaging)

        # Difference
        correlation_matrix = corr_h - corr_l
        colormap = 'twilight_shifted'

    # --- 5. Export and Plotting ---
    if save_data_for_publication:
        np.savetxt(f"{wd}/correlation_matrix_data.txt", correlation_matrix, delimiter='\t')

    if pdf:
        pdf_page = PdfPages(pdf_filename)

    # Use the gpi_labels if plotting the interesting subset, otherwise fallback to the raw keys
    labels_to_plot = gpi_labels if plot_interesting_only else plot_keys
    if not plot_interesting_only:
        charsize=5
    else:
        charsize=15
    plot_pearson_matrix(correlation_matrix,
                        xlabels = labels_to_plot,
                        ylabels = labels_to_plot,
                        colormap = colormap,
                        figsize = (17/2.54 / (1 + plot_interesting_only), 
                                   17/2.54 / (1 + plot_interesting_only)),
                        charsize = charsize,
                        charsize_score=charsize/1.5,
                        plot_large = not plot_interesting_only,
                        plot_colorbar = not plot_interesting_only,
                        plot_values = True,
                        )   

    if analyze_lh_difference:
        plt.tight_layout()

    if pdf:
        pdf_page.savefig()
        pdf_page.close()
        
    return correlation_matrix, plot_keys

def plot_blob_blob_parameter_trends(pdf = True,
                                    pdf_filename = None,
                                    time_range_around_peak=5e-3,
                                    plot_if_correlation_is_higher_than = None,
                                    plot_if_pps_is_higher_than = None,
                                    nocalc = True,
                                    calc_mean_distribution = True,
                                    min_structure_lifetime = 20,
                                    plot_for_publication = False,
                                    str_finding_method = 'watershed',
                                    recalc_tracking = False,
                                    averaging = 'no',
                                    analyze_l_mode_only = False,
                                    analyze_h_mode_only = False,
                                    analyze_lh_difference = False,
                                    save_data_for_publication = False,
                                    ):
    import matplotlib
    if pdf:
        matplotlib.use('agg')
    else:
        matplotlib.use('qt5agg')

    if analyze_h_mode_only:
        plasma_mode = '_h_mode'
    elif analyze_l_mode_only:
        plasma_mode = '_l_mode'
    elif analyze_lh_difference:
        plasma_mode = '_lh_diff'
    else:
        plasma_mode = ''
        
    if not plot_for_publication:
        if pdf_filename is None:
            if plot_if_correlation_is_higher_than is not None:
                pdf_filename = f"{wd}/plots/gpi_gpi_trends_corr_{plot_if_correlation_is_higher_than}_{str_finding_method}.pdf"
            else:
                pdf_filename = f"{wd}/plots/gpi_gpi_trends_{str_finding_method}{plasma_mode}.pdf"
    else:
        pdf_filename = f"{wd}/plots/gpi_gpi_trend_8plot_{str_finding_method}{plasma_mode}.pdf"

    # --- 1. Data Loading ---
    def _load_blob_data(l_mode=False, h_mode=False):
        
        mode_str = 'l_mode' if l_mode else 'h_mode' if h_mode else 'full'
        p_file = f"{wd}/processed_data/gpi_gpi_trends_{str_finding_method}_{averaging}_{mode_str}.pickle"
        
        if not os.path.exists(p_file) or not nocalc:
            data  =  read_all_blob_data(min_structure_lifetime = min_structure_lifetime,
                                        time_range_around_peak=time_range_around_peak,
                                        averaging = 'shot' if calc_mean_distribution else 'no',
                                        nocalc = True, recalc_tracking = recalc_tracking,
                                        str_finding_method = str_finding_method,
                                        read_l_mode_only = l_mode, 
                                        read_h_mode_only = h_mode,
                                        replicate_histogram2 = True)
            with open(p_file, 'wb') as f:
                pickle.dump(data, f)
            return data
            
        with open(p_file, 'rb') as f:
            return pickle.load(f)

    if analyze_lh_difference:
        full_data_l_mode = _load_blob_data(l_mode=True)
        full_data_h_mode = _load_blob_data(h_mode=True)
        full_data = full_data_l_mode # Default reference
    else:
        full_data = _load_blob_data(l_mode=analyze_l_mode_only, h_mode=analyze_h_mode_only)

    analyzed_keys = list(full_data.keys())
    
    # --- Helper to flatten dict structures ---
    def _flatten_data(data_dict):
        flat_dict = {}
        for k in data_dict.keys():
            if len(data_dict[k]) == 0: continue
            if isinstance(data_dict[k][0], dict):
                flat_dict[k] = np.concatenate([shot['data'] for shot in data_dict[k]])
            else:
                flat_dict[k] = np.array(data_dict[k])
        return flat_dict

    flat_data = _flatten_data(full_data)
    
    if analyze_lh_difference:
        flat_data_l = flat_data # Already flattened L-mode
        flat_data_h = _flatten_data(full_data_h_mode)

    # --- 3. Plotting Logic ---
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import sys
    
    discrete_step_size={'Axes length minor':0.00375, 
                        'Axes length major':0.00375,
                        'Angle envelope':np.pi/51,
                        'Size radial':0.00375, 
                        'Size poloidal':0.00375,
                        'Axes length minor diff':0.00375/2.5e-6, 
                        'Axes length major diff':0.00375/2.5e-6,
                        'Angle envelope diff':np.pi/64/2.5e-6,
                        'Size radial diff':0.00375/2.5e-6, 
                        'Size poloidal diff':0.00375/2.5e-6,
                        'Lifetime':2.5e-6,
                        }
    
    if not plot_for_publication:
        
        total_keys = len(analyzed_keys)
        #total_pairs = (total_keys * (total_keys - 1)) // 2
        total_pairs = total_keys**2
        processed_pairs = 0
        
        if pdf: pdf_page = PdfPages(pdf_filename)
        
        for i, key1 in enumerate(analyzed_keys):
            if key1 not in flat_data: 
                processed_pairs += (total_keys - i - 1)
                continue
                
            valid1 = ~np.isnan(flat_data[key1])
            
            for j, key2 in enumerate(analyzed_keys):
                #if key1 != key2 and j > i:
                if True:
                    processed_pairs += 1
                    
                    pct = (processed_pairs / total_pairs) * 100
                    sys.stdout.write(f"\rPlotting progress: {pct:.1f}% complete")
                    sys.stdout.flush()
                    
                    if key2 not in flat_data: continue
                    
                    valid_mask = valid1 & ~np.isnan(flat_data[key2])
                    d1 = flat_data[key1][valid_mask]
                    d2 = flat_data[key2][valid_mask]
                    
                    if len(d1) == 0: continue
                    
                    # Calculate Correlation for filtering (using the reference flat_data)
                    d1_4c = d1 - np.mean(d1)
                    d2_4c = d2 - np.mean(d2)
                    denom = np.sqrt(np.sum(d1_4c**2) * np.sum(d2_4c**2))
                    corr = np.sum(d1_4c * d2_4c) / denom if denom != 0 else 0
                    
                    do_plot = False
                    if plot_if_correlation_is_higher_than and abs(corr) > plot_if_correlation_is_higher_than: do_plot = True
                    elif not plot_if_correlation_is_higher_than and not plot_if_pps_is_higher_than: do_plot = True
        
                    if do_plot:
                        # Determine plotting bounds across BOTH datasets to ensure the bins align perfectly
                        if analyze_lh_difference and key1 in flat_data_h and key2 in flat_data_h:
                            valid_h = ~np.isnan(flat_data_h[key1]) & ~np.isnan(flat_data_h[key2])
                            d1_h = flat_data_h[key1][valid_h]
                            d2_h = flat_data_h[key2][valid_h]
                            
                            all_d1 = np.concatenate([d1, d1_h])
                            all_d2 = np.concatenate([d2, d2_h])
                            if key1 == 'Poloidal angle':
                                r_x=[0,np.pi/8]
                            else:
                                r_x = [np.percentile(all_d1, 1), np.percentile(all_d1, 99)]
                            if key2 == 'Poloidal angle':
                                r_y=[0,np.pi/8]
                            else:
                                r_y = [np.percentile(all_d2, 1), np.percentile(all_d2, 99)]
                        else:
                            r_x = [np.percentile(d1, 1), np.percentile(d1, 99)]
                            r_y = [np.percentile(d2, 1), np.percentile(d2, 99)]
                            d1_h, d2_h = None, None # Fallback

                        # ---------------------------------------------------------
                        # BRANCH A: Plot Difference Maps (L vs H vs Diff)
                        # ---------------------------------------------------------
                        if analyze_lh_difference and d1_h is not None and len(d1_h) > 0:
                            fig, axes = plt.subplots(1, 3, figsize=(25/2.54, 8.5/2.54))
                            
                            if key1 in discrete_step_size.keys():
                                step_size = discrete_step_size[key1]  
                                
                                # Find the absolute min and max of your data
                                min_val = np.floor(np.min(d1))
                                max_val = np.ceil(np.max(d1))
                                
                                # Generate explicit bin edges shifted by half a step
                                discrete_bins_x = np.arange(min_val - step_size/2, max_val + step_size, step_size)
                                if len(discrete_bins_x) > 101:
                                    discrete_bins_x=discrete_bins_x[0:101]
                                # Pass the array to the 'bins' argument instead of an integer!
                            else:
                                discrete_bins_x=51
                            if key2 in discrete_step_size.keys():
                                step_size = discrete_step_size[key2]  
                                                                
                                # Find the absolute min and max of your data
                                min_val = np.floor(np.min(d2))
                                max_val = np.ceil(np.max(d2))
                                
                                # Generate explicit bin edges shifted by half a step
                                discrete_bins_y = np.arange(min_val - step_size/2, max_val + step_size, step_size)
                                if len(discrete_bins_y) > 101:
                                    discrete_bins_y=discrete_bins_y[0:101]
                                # Pass the array to the 'bins' argument instead of an integer!
                            else:
                                discrete_bins_y=51
                                
                            bins=[discrete_bins_x, discrete_bins_y]
                            
                            # 1. Calculate L-mode Histogram
                            counts_l, xedges, yedges = np.histogram2d(d1, d2, bins=bins, range=[r_x, r_y])
                            img_l = (counts_l / np.sum(counts_l) * 100) if np.sum(counts_l) > 0 else counts_l
                            
                            # 2. Calculate H-mode Histogram (using exact same bins)
                            counts_h, _, _ = np.histogram2d(d1_h, d2_h, bins=[xedges, yedges])
                            img_h = (counts_h / np.sum(counts_h) * 100) if np.sum(counts_h) > 0 else counts_h
                            
                            # 3. Calculate Difference (L - H)
                            img_diff = img_l - img_h
                            
                            # --- Calculate Statistics for L-Mode and H-Mode ---
                            slope_l, intercept_l, r_value_l, _, _ = linregress(d1, d2)
                            slope_h, intercept_h, r_value_h, _, _ = linregress(d1_h, d2_h)
                            
                            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                            
                            # Plot L-Mode
                            im0 = axes[0].imshow(img_l.T, origin="lower", aspect="auto", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis')
                            axes[0].set_title(f"L-Mode: {key1} vs {key2}", fontsize=9)
                            axes[0].set_xlabel(key1)
                            axes[0].set_ylabel(key2)
                            axes[0].text(0.05, 0.95, f"$r = {r_value_l:.3f}$\n$R^2 = {r_value_l**2:.3f}$", transform=axes[0].transAxes, fontsize=8, verticalalignment='top', bbox=props)
                            best_fit_x = np.array([r_x[0], r_x[1]])
                            best_fit_y = slope_l * best_fit_x + intercept_l
                            axes[0].plot(best_fit_x, best_fit_y, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                            # Plot H-Mode
                            im1 = axes[1].imshow(img_h.T, origin="lower", aspect="auto", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis')
                            
                            axes[1].set_title("H-Mode", fontsize=9)
                            axes[1].set_xlabel(key1)
                            axes[1].text(0.05, 0.95, f"$r = {r_value_h:.3f}$\n$R^2 = {r_value_h**2:.3f}$", transform=axes[1].transAxes, fontsize=8, verticalalignment='top', bbox=props)
                            best_fit_x = np.array([r_x[0], r_x[1]])
                            best_fit_y = slope_h * best_fit_x + intercept_h
                            axes[1].plot(best_fit_x, best_fit_y, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                            
                            # Plot Difference (centered colormap)
                            vmax = np.max(np.abs(img_diff))
                            im2 = axes[2].imshow(img_diff.T, origin="lower", aspect="auto", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                            axes[2].set_title("Difference (L - H)", fontsize=9)
                            axes[2].set_xlabel(key1)
                            
                            # Add colorbars
                            for ax, im in zip(axes, [im0, im1, im2]):
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes('right', size='5%', pad=0.05)
                                fig.colorbar(im, cax=cax)

                        # ---------------------------------------------------------
                        # BRANCH B: Standard Single Histogram
                        # ---------------------------------------------------------
                        else:
                            fig, ax_hist = plt.subplots(1, 1, figsize=(17/2.54, 8.5/2.54))
                            
                            counts, xedges, yedges = np.histogram2d(d1, d2, bins=[51, 51], range=[r_x, r_y])
                            img_data = (counts / np.sum(counts)) * 100 if np.sum(counts) > 0 else counts
                            
                            im = ax_hist.imshow(img_data.T, origin="lower", aspect="auto",
                                                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                                                cmap='viridis')
                            
                            ax_hist.set_xlabel(key1)
                            ax_hist.set_ylabel(key2)
                            ax_hist.set_title(f"Density: {key1} vs {key2}", fontsize=9)

                            slope, intercept, r_value, p_value, std_err = linregress(d1, d2)
                            r_squared = r_value ** 2

                            best_fit_x = np.array([r_x[0], r_x[1]])
                            best_fit_y = slope * best_fit_x + intercept
                            
                            ax_hist.plot(best_fit_x, best_fit_y, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                            
                            # Fixed `corr` to `r_value` to avoid NameError
                            stats_text = f"$r = {r_value:.3f}$\n$R^2 = {r_squared:.3f}$"
                            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                            ax_hist.text(0.05, 0.95, stats_text, transform=ax_hist.transAxes, fontsize=8, verticalalignment='top', bbox=props)
                            
                            divider = make_axes_locatable(ax_hist)
                            cax = divider.append_axes('right', size='5%', pad=0.05)
                            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                            cbar.ax.set_title('[%]', fontsize=8)
                        
                        plt.tight_layout()
                        if pdf: pdf_page.savefig(fig)
                        plt.close(fig)
        
        print("\rPlotting progress: 100.0% complete!       ")
        if pdf: pdf_page.close()

    else:
        # Publication 2D Histograms
        publication_plots = {
            ('Area', 'Convexity'): {
                'range': [[0, 0.004], [0.92, 1.0]],
                'x_mult': 1e4,  'x_label': 'Area',                    'x_unit': '[$\\rm cm^2$]',
                'y_mult': 1,    'y_label': 'Convexity',               'y_unit': '[a.u.]'
            },
            ('Size radial fit', 'Convexity'): {
                'range': [[0.01, 0.07], [0.92, 1.0]],
                'x_mult': 1e2,  'x_label': '$\\rm d_{rad}$',          'x_unit': '[cm]',
                'y_mult': 1,    'y_label': 'Convexity',               'y_unit': '[a.u.]'
            },
            ('Elongation fit', 'Roundness'): {
                'range': [[-0.75, 0.5], [0.2, 1.0]],
                'x_mult': 1,    'x_label': 'Elongation',              'x_unit': '[a.u.]',
                'y_mult': 1,    'y_label': 'Roundness',               'y_unit': '[a.u.]'
            },
            ('Position radial fit', 'Velocity radial position fit'): {
                'range': [[1.42, 1.6], [-2e3, 2e3]],
                'x_mult': 1,    'x_label': '$\\rm R_{pos}$',          'x_unit': '[m]',
                'y_mult': 1e-3, 'y_label': '$\\rm v_{rad}$',          'y_unit': '[km/s]'
            },
            ('Axes length minor fit', 'Velocity radial position fit'): {
                'range': [[0, 0.075], [-2e3, 2e3]],
                'x_mult': 1e2,  'x_label': 'Minor semi-axis',         'x_unit': '[cm]',
                'y_mult': 1e-3, 'y_label': '$\\rm v_{rad}$',          'y_unit': '[km/s]'
            },
            ('Axes length major fit', 'Velocity radial position fit'): {
                'range': [[0.0, 0.03], [-2e3, 2e3]],
                'x_mult': 1e2,  'x_label': 'Major semi-axis',         'x_unit': '[cm]',
                'y_mult': 1e-3, 'y_label': '$\\rm v_{rad}$',          'y_unit': '[km/s]'
            },
            ('Area diff', 'Roundness diff'): {
                'range': [[-0.05e-2, 0.05e-2], [-0.15, 0.15]],
                'x_mult': 1e4,  'x_label': '$\\rm\\Delta$Area',       'x_unit': '[$\\rm cm^2$]',
                'y_mult': 1e3,  'y_label': '$\\rm\\Delta$Roundness',  'y_unit': '[a.u.]'
            },
            # BUG FIX: Added the 8th plot pair so the grid completes!
            ('Area diff', 'Total curvature diff'): {
                'range': [[-0.05e-2, 0.05e-2], [-0.05, 0.05]],
                'x_mult': 1e4,  'x_label': '$\\rm\\Delta$Area',       'x_unit': '[$\\rm cm^2$]',
                'y_mult': 1e3,  'y_label': '$\\rm\\Delta$Curvature',  'y_unit': '[a.u.]'
            }
        }

        labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        pdf_page = PdfPages(pdf_filename)
        fig, axes = plt.subplots(4, 2, figsize=(8.5/2.54, 17/2.54))
        
        # Ensure we use flattened datasets for histogramming
        if analyze_lh_difference:
            flat_l_mode = _flatten_data(full_data_l_mode)
            flat_h_mode = _flatten_data(full_data_h_mode)

        def get_2d_hist(data_dict, k1, k2, cfg):
            valid = ~np.isnan(data_dict[k1]) & ~np.isnan(data_dict[k2])
            d1 = data_dict[k1][valid] * cfg['x_mult']
            d2 = data_dict[k2][valid] * cfg['y_mult']
            
            r = [np.asarray(cfg['range'][0]) * cfg['x_mult'], 
                 np.asarray(cfg['range'][1]) * cfg['y_mult']]
            
            return np.histogram2d(d1, d2, bins=[31, 31], range=r)

        for ind, ((key1, key2), config) in enumerate(publication_plots.items()):

            counts_l, xedges, yedges = get_2d_hist(flat_l_mode if analyze_lh_difference else flat_data, key1, key2, config)
            base_img = (counts_l / np.sum(counts_l)) * 100

            if analyze_lh_difference:
                counts_h, _, _ = get_2d_hist(flat_h_mode, key1, key2, config)
                img_data = (counts_h / np.sum(counts_h)) * 100 - base_img
                v_args = {'vmin': -0.7, 'vmax': 0.7, 'cmap': 'seismic'}
            else:
                img_data = base_img
                v_args = {}

            ax = axes[ind // 2, ind % 2]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            im = ax.imshow(img_data.T, origin="lower", aspect="auto",
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], **v_args)

            if save_data_for_publication and not analyze_lh_difference:
                with open(f"{wd}/{labels[ind]}_db_2Dhistogram.txt", 'w+') as f:
                    f.write("X bins\n" + str((xedges[1:] + xedges[:-1]) / 2))
                    f.write("\nY bins\n" + str((yedges[1:] + yedges[:-1]) / 2))
                    f.write("\nCounts\n" + str(base_img.T))

            fig.colorbar(im, cax=cax, orientation='vertical')
            ax.text(-0.45, 1.1, f"({labels[ind]})", transform=ax.transAxes, size=9)
            ax.text(1.02, 1.05, '[%]', transform=ax.transAxes, size=6)
            
            ax.set_xlabel(f"{config['x_label']} {config['x_unit']}")
            ax.set_ylabel(f"{config['y_label']} {config['y_unit']}")

        plt.tight_layout(pad=0.2)
        pdf_page.savefig()
        pdf_page.close()
        plt.close(fig)

def plot_blob_blob_parameter_predictive_power_score(threshold_corr=False,
                                                    pdf=True,
                                                    nocalc=True,
                                                    calc_mean_distribution=True,
                                                    str_finding_method='watershed' # BUG FIX: Added argument
                                                    ):
    """
    Calculates and plots the Predictive Power Score (PPS) matrix for blob parameters.

    Unlike Pearson correlation which only detects linear relationships, the PPS 
    can detect non-linear relationships and asymmetries (e.g., variable A might 
    predict B better than B predicts A). This function loads the aggregated blob 
    database, filters out statistical outliers (Z-score > 3), computes the PPS 
    matrix, and saves the resulting heatmap.

    Args:
        threshold_corr (bool, optional): Unused threshold flag. Defaults to False.
        pdf (bool, optional): If True, saves the generated plot to a PDF file. 
            Defaults to True.
        nocalc (bool, optional): If True, attempts to load a previously cached 
            PPS matrix from a `.pickle` file to save computation time. 
            Defaults to True.
        calc_mean_distribution (bool, optional): If True, loads the shot-averaged 
            blob database. If False, loads the un-averaged blob database. 
            Defaults to True.

    Returns:
        None
    """
    
    if pdf:
        pdf_pages = PdfPages(wd + '/plots/predictive_power_score_blob_vs_blob.pdf')

    # --- 1. File Paths ---
    mean_str = 'mean' if calc_mean_distribution else 'nomean'
    # BUG FIX: Inserted str_finding_method into both strings so they match the rest of the suite!
    pickle_filename = f"{wd}/processed_data/blob_database_full_data_{mean_str}_{str_finding_method}.pickle"
    pickle_filename_pps = f"{wd}/processed_data/blob_blob_predictive_power_score_{str_finding_method}.pickle"
    
    if not os.path.exists(pickle_filename):
        print(f"Error: Could not find raw data cache at {pickle_filename}")
        return

    with open(pickle_filename, 'rb') as f:
        full_blob_data = pickle.load(f)

    # --- 2. Data Processing & PPS Calculation ---
    if not nocalc or not os.path.exists(pickle_filename_pps):
        
        # BUG FIX: Flatten the dictionary arrays safely!
        processed_data = {}
        for key in full_blob_data.keys():
            if len(full_blob_data[key]) == 0: continue
            
            if isinstance(full_blob_data[key][0], dict):
                processed_data[key] = np.concatenate([shot['data'] for shot in full_blob_data[key]])
            else:
                processed_data[key] = np.array(full_blob_data[key])
                
        df = pandas.DataFrame(processed_data)

        # Drop rows that are entirely NaN
        df = df.dropna(thresh=1)

        # Remove statistical outliers (keep rows where all Z-scores are < 3)
        df = df[(np.abs(scipy.stats.zscore(df, nan_policy='omit')) < 3).all(axis=1)]

        print("Calculating Predictive Power Score matrix. This may take a moment...")
        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
        
        with open(pickle_filename_pps, 'wb') as f:
            pickle.dump(matrix_df, f)
    else:
        with open(pickle_filename_pps, 'rb') as f:
            matrix_df = pickle.load(f)

    # --- 3. Plotting ---
    ppscore_matrix_prelim = np.asarray(matrix_df).T
    
    # Extract labels directly from the PPS matrix to ensure dimension matching
    xlabels = list(matrix_df.columns)

    # Assuming plot_pearson_matrix is imported
    plot_pearson_matrix(ppscore_matrix_prelim,
                        xlabels=xlabels,
                        ylabels=xlabels,
                        title='Blob vs Blob Predictive Power Score (PPS)',
                        colormap='Blues',
                        figsize=(17/2.54, 17/2.54),
                        charsize=6,
                        zrange=[0, 1.0]
                        )
                        
    if pdf:
        pdf_pages.savefig()
        pdf_pages.close()

def calculate_blob_plasma_parameter_correlation_matrix(threshold_corr=False,
                                                       threshold_multiplier=2,
                                                       pdf=True,
                                                       pdf_filename=None,
                                                       str_finding_method='watershed',
                                                       fix_angle_for_correlation=True,
                                                       averaging='shot',
                                                       average='avg',
                                                       quantity='correlation',
                                                       figsize=(17/2.54, 17/2.54),
                                                       plot_for_publication=False,
                                                       colormap='seismic',
                                                       plot_colorbar=True,
                                                       plot_full=False,
                                                       linewidth=1.5,
                                                       ticksize=1,
                                                       charsize=9,
                                                       nocalc=False,
                                                       nocalc_plasma_data=True,
                                                       nocalc_blob_data=True,
                                                       analyze_lh_difference=False,
                                                       ):
    """
    Calculates and plots the statistical relationship between blob structures and global plasma parameters.

    This function compares localized Gas Puff Imaging (GPI) blob parameters against core/global 
    MDSplus/Thomson plasma parameters. It supports calculating Pearson correlations, 
    Mutual Information, or Predictive Power Scores (PPS). The resulting heatmap 
    can be thresholded to highlight significant correlations.

    Args:
        threshold_corr (bool, optional): If True, applies a statistical threshold 
            to the correlation matrix. Defaults to False.
        threshold_multiplier (int, optional): Confidence sigma for the thresholding. Defaults to 2.
        pdf (bool, optional): If True, saves output to a PDF. Defaults to True.
        pdf_filename (str, optional): Custom output filename. Defaults to None.
        str_finding_method (str, optional): Segmentation method used. Defaults to 'watershed'.
        fix_angle_for_correlation (bool, optional): Normalizes angles modulo pi/2. Defaults to True.
        averaging (str, optional): Averaging methodology ('no', 'blob', 'shot'). Defaults to 'shot'.
        average (str, optional): Moment to use ('avg', 'std', 'max'). Defaults to 'avg'.
        quantity (str, optional): Type of analysis ('correlation', 'mutual_information', 
            'predictive_power'). Defaults to 'correlation'.
        figsize (tuple, optional): Dimensions of the generated figure. Defaults to (17/2.54, 17/2.54).
        plot_for_publication (bool, optional): Applies publication-specific formatting 
            and predefined parameter subsets. Defaults to False.
        colormap (str, optional): Matplotlib colormap string. Defaults to 'seismic'.
        plot_colorbar (bool, optional): Displays the colorbar next to the matrix. Defaults to True.
        plot_full (bool, optional): Forces plotting the full matrix instead of a subset. Defaults to False.
        linewidth (float, optional): Line thickness for matrix grid. Defaults to 1.5.
        ticksize (int, optional): Font size for ticks. Defaults to 1.
        charsize (int, optional): Font size for text inside matrix boxes. Defaults to 9.
        nocalc (bool, optional): If True, attempts to load pre-calculated PPS matrices. Defaults to False.
        nocalc_plasma_data (bool, optional): If True, loads cached plasma data. Defaults to True.
        nocalc_blob_data (bool, optional): If True, loads cached blob data. Defaults to True.

    Returns:
        None
    """
    
    plt.close('all')
    if pdf:
        import matplotlib
        matplotlib.use('agg')
        
    wd = flap.config.get_all_section('Module NSTX_GPI')['Working directory']

    # --- 1. Configurations and Labels ---
    metric_map = {
        'mutual_information': {'str': 'mutual_information', 'title': 'Blob vs plasma parameter mutual information map', 'zrange': [0, 1], 'cmap': 'Purples'},
        'correlation': {'str': 'correlation', 'title': 'Blob vs plasma parameter correlation map', 'zrange': [-1, 1], 'cmap': 'seismic'},
        'predictive_power': {'str': 'pps', 'title': 'Blob vs plasma parameter predictive power map', 'zrange': [0, 1], 'cmap': 'Blues'},
        'coefficient_of_determination': {'str': 'R2', 'title': 'Blob vs plasma parameter coefficient of determination map', 'zrange': [0, 1], 'cmap': 'Blues'},
    }
    
    cfg = metric_map.get(quantity, metric_map['correlation'])
    if not plot_for_publication: 
        colormap = cfg['cmap']

    if pdf_filename is None:
        avg_str = 'full' if averaging == 'no' else f"{averaging}_{average}"
        thres_str = f"_thres_{int(threshold_multiplier)}" if threshold_corr else "_nothres"
        full_str = "_full" if plot_full else ""
        lh_str = '_LH_diff' if analyze_lh_difference else ""
        pdf_filename = f"{wd}/plots/{cfg['str']}_matrix_gpi_plasma_{str_finding_method}_{avg_str}{thres_str}{lh_str}{full_str}.pdf"
        
    if pdf:
        pdf_page = PdfPages(pdf_filename)
        
    corr_accept = calculate_corr_acceptance_levels()
        
    if not analyze_lh_difference:
        # --- 2. Data Loading & Derived Parameter Calculation ---
        full_plasma_data = read_all_plasma_data(nocalc=nocalc_plasma_data)
        full_blob_data = read_all_blob_data(nocalc=nocalc_blob_data, 
                                            str_finding_method=str_finding_method,
                                            fix_angle_for_correlation=fix_angle_for_correlation,
                                            averaging=averaging, average=average, time_range_around_peak=5e-3)
    
        # Calculate Blob Size Dimensionless
        scale_length = (full_plasma_data['Larmor radius sound']**0.8 * full_plasma_data['Connection length']**0.4 /
                        full_plasma_data['Pedestal radius']**0.2)
        full_plasma_data['Blob size dimensionless'] = (np.sqrt(full_blob_data['Area'] / np.pi) / scale_length)**2.5
    
        # --- 3. Key Curation ---
        if plot_for_publication:
            interesting_key_pairs, units = return_interesting_key_pairs(with_plasma_frequency=True)
            gpi_labels = list(np.unique(interesting_key_pairs[:, 0]))
            plasma_labels = list(np.unique(interesting_key_pairs[:, 1])[[2, 1, 4, 6, 0, 3, 5]])
            plot_full = True
        else:
            gpi_labels = list(full_blob_data.keys())
            plasma_labels = list(full_plasma_data.keys())
            units = None
    
    
        # --- 4. Main Mathematical Loop ---
        if quantity in ['correlation', 'mutual_information']:
            
            if plot_full:
                full_blob_data.update(full_plasma_data)
                data1_dict, data2_dict = full_blob_data, full_blob_data
                label_1, label_2 = gpi_labels + plasma_labels, gpi_labels + plasma_labels
            else:
                data1_dict, data2_dict = full_blob_data, full_plasma_data
                label_1, label_2 = gpi_labels, plasma_labels
    
            correlation_matrix = np.zeros([len(label_2), len(label_1)])
    
            for ind1, key1 in enumerate(label_1):
                for ind2, key2 in enumerate(label_2):
                    
                    try:
                        # Fetch arrays safely depending on structure
                        if averaging == 'shot':
                            d1 = data1_dict[key1]
                            d2 = data2_dict[key2]
                        else:
                            # Flatten the blob-by-blob nested structures to align with scalar plasma parameters
                            d1 = np.concatenate([shot['data'] for shot in data1_dict[key1]]) if type(data1_dict[key1][0]) is dict else data1_dict[key1]
                            d2_expanded = np.concatenate([np.full(len(shot['data']), data2_dict[key2][i]) for i, shot in enumerate(data1_dict[key1])]) if type(data1_dict[key1][0]) is dict else data2_dict[key2]
                            d2 = d2_expanded
    
                        valid_mask = ~np.isnan(d1) & ~np.isnan(d2)
                        d1, d2 = d1[valid_mask], d2[valid_mask]
    
                        if len(d1) == 0 or len(d2) == 0:
                            correlation_matrix[ind2, ind1] = np.nan
                            continue
    
                        if quantity == 'correlation':
                            correlation_matrix[ind2, ind1] = correlation(d1, d2, 
                                                                         threshold_correlation=threshold_corr,
                                                                         correlation_accept=corr_accept,
                                                                         confidence_sigma=threshold_multiplier)
                        elif quantity == 'mutual_information':
                            d1 -= np.mean(d1)
                            d2 -= np.mean(d2)
                            correlation_matrix[ind2, ind1] = mutual_information(d1, d2)
                            
                    except Exception as e:
                        print(f"Calculation failed for {key1} vs {key2}: {e}")
                        correlation_matrix[ind2, ind1] = np.nan
    
        elif quantity == 'predictive_power':
            avg_str = 'full' if averaging == 'no' else f"{averaging}_{average}"
            pickle_filename_pps = f"{wd}/processed_data/blob_plasma_predictive_power_score_{avg_str}.pickle"
            
            if not nocalc or not os.path.exists(pickle_filename_pps):
                
                # BUG FIX: Construct a standard dictionary first, then cast to DataFrame!
                raw_df_dict = {}
                
                # Combine blob and plasma data into one flat dataframe
                for key in gpi_labels:
                    raw_df_dict[key] = full_blob_data[key] if averaging == 'shot' else np.concatenate([s['data'] for s in full_blob_data[key]])
                for key in plasma_labels:
                    # Assuming gpi_labels[0] has data we can use to map the lengths
                    ref_key = gpi_labels[0]
                    raw_df_dict[key] = full_plasma_data[key] if averaging == 'shot' else np.concatenate([np.full(len(s['data']), full_plasma_data[key][i]) for i, s in enumerate(full_blob_data[ref_key])])
    
                df = pandas.DataFrame(raw_df_dict)
                matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
                matrix_df = matrix_df.reindex(index=gpi_labels+plasma_labels, columns=gpi_labels+plasma_labels)
                
                with open(pickle_filename_pps, 'wb') as f:
                    pickle.dump(matrix_df, f)
            else:
                with open(pickle_filename_pps, 'rb') as f:
                    matrix_df = pickle.load(f)
    
            correlation_matrix = np.asarray(matrix_df).T
            label_1, label_2 = list(matrix_df.columns), list(matrix_df.index)
            plot_full = True
    
        # --- 5. Plotting ---
        if not plot_full:
            plot_pearson_matrix(correlation_matrix, xlabels=label_1, ylabels=label_2,
                                title=cfg['title'], colormap=colormap, zrange=cfg['zrange'],
                                figsize=figsize, charsize=charsize, linewidth=linewidth,
                                ticksize=ticksize, minor_ticksize=0.001, plot_colorbar=plot_colorbar)
        else:
            if plot_for_publication:
                # Apply Publication labels
                if units is not None:
                    gpi_labels = [units[l][0] for l in gpi_labels]
                    plasma_labels = [units[l][0] for l in plasma_labels]
                labels = gpi_labels + plasma_labels
                
                fig, ax = plt.subplots(figsize=(8.5/2.54, 8.5/2.54 * 1.2))
                plot_pearson_matrix(correlation_matrix, xlabels=labels, ylabels=labels,
                                    title=cfg['title'], colormap=colormap, zrange=cfg['zrange'],
                                    fig_ax=(fig, ax), charsize=charsize, charsize_score=charsize*2/3,
                                    ticksize=ticksize, linewidth=linewidth, minor_ticksize=0.001, 
                                    plot_colorbar=plot_colorbar)
    
                # Draw Quadrant Rectangles
                ax.add_patch(Rectangle((-0.45, -0.45), 3.9, 3.9, fill=False, edgecolor='red', lw=2, zorder=0))
                ax.add_patch(Rectangle((-0.45, 3.55), 3.9, 6.9, fill=False, edgecolor='yellow', lw=2, zorder=0))
                ax.add_patch(Rectangle((3.55, -0.45), 6.9, 3.9, fill=False, edgecolor='cyan', lw=2, zorder=0))
                ax.add_patch(Rectangle((3.55, 3.55), 6.9, 6.9, fill=False, edgecolor='magenta', lw=2, zorder=0))
            else:
                plot_pearson_matrix(correlation_matrix, xlabels=label_1, ylabels=label_2,
                                    title=cfg['title'], colormap=colormap, zrange=cfg['zrange'],
                                    figsize=figsize, charsize=3, charsize_score=2, ticksize=ticksize,
                                    linewidth=linewidth, minor_ticksize=0.001, plot_colorbar=plot_colorbar)
                
        plt.tight_layout(pad=0.1)
        
        
    else: #analyze_lh_difference=True  # lots of repetition, but easy to implement this way
        # --- 2. Dynamic Data Loading Helper ---
        def _load_data(l_mode=False, h_mode=False):
            mode_str = '_l_mode' if l_mode else '_h_mode' if h_mode else '_full'
            p_plasma = f"{wd}/processed_data/plasma_vs_blob_plasma_data{mode_str}.pickle"
            p_blob = f"{wd}/processed_data/plasma_vs_blob_blob_data{mode_str}.pickle"
            
            if not os.path.exists(p_plasma):
                plasma_data = read_all_plasma_data(nocalc=nocalc, read_l_mode_only=l_mode, read_h_mode_only=h_mode)
                with open(p_plasma, 'wb') as f: pickle.dump(plasma_data, f)
            else:
                with open(p_plasma, 'rb') as f: plasma_data = pickle.load(f)
                
            if not os.path.exists(p_blob):
            # if True:
                blob_data = read_all_blob_data(nocalc=nocalc, str_finding_method='watershed', 
                                               fix_angle_for_correlation=True, averaging='shot', average='avg',
                                               read_l_mode_only=l_mode, read_h_mode_only=h_mode)
                with open(p_blob, 'wb') as f: pickle.dump(blob_data, f)
            else:
                with open(p_blob, 'rb') as f: blob_data = pickle.load(f)
            
            for key in plasma_data.keys():
                plasma_data[key]=np.asarray(plasma_data[key])
            
            for key in blob_data.keys():
                blob_data[key]=np.asarray(blob_data[key])    
            
            # Calculate derived metrics
            scale_length = (plasma_data['Larmor radius sound']**0.8 * plasma_data['Connection length']**0.4 /
                            plasma_data['Pedestal radius']**0.2)
            plasma_data['Blob size dimensionless'] = (np.sqrt(blob_data['Area']) / np.pi / scale_length)**2.5
            
            # In-place masking of invalid connection lengths
            plasma_data['Connection length'] = np.where(plasma_data['Connection length'] < 1.5, np.nan, plasma_data['Connection length'])
            
            return plasma_data, blob_data

        # --- 3. Load Datasets ---
        plasma_l, blob_l = _load_data(l_mode=True)
        plasma_h, blob_h = _load_data(h_mode=True)
        # --- 4. Plotting Mode: Publication Scatter Grid ---
        flap_nstx.tools.set_matplotlib_for_publication(labelsize=6., linewidth=0.5, major_ticksize=2.)
        if quantity in ['correlation', 'mutual_information', 'coefficient_of_determination']:
            
            label_1, label_2 = list(blob_l.keys()), list(plasma_l.keys())
            if plot_full:
                correlation_matrix = np.zeros([len(label_1)+len(label_2), len(label_1)+len(label_2), 3])
            else:
                correlation_matrix = np.zeros([len(label_2), len(label_1), 3])
                
            for ind_mode, mode in enumerate(['L-mode', 'H-mode', 'LH Diff']):
                if ind_mode == 0:
                    if plot_full:
                        blob_l.update(plasma_l)
                        data1_dict, data2_dict = blob_l, blob_l
                        label_1, label_2 = label_1 + label_2, label_1 + label_2
                        charsize=3
                    else:
                        data1_dict, data2_dict = blob_l, plasma_l
                        
                elif ind_mode == 1:
                    if plot_full:
                        blob_h.update(plasma_h)
                        data1_dict, data2_dict = blob_h, blob_h
                    else:
                        data1_dict, data2_dict = blob_h, plasma_h
                    
                if ind_mode < 2:
                    for ind1, key1 in enumerate(label_1):
                        for ind2, key2 in enumerate(label_2):
                            
                            try:
                                # Fetch arrays safely depending on structure
                                if averaging == 'shot':
                                    d1 = data1_dict[key1]
                                    d2 = data2_dict[key2]
                                else:
                                    # Flatten the blob-by-blob nested structures to align with scalar plasma parameters
                                    d1 = np.concatenate([shot['data'] for shot in data1_dict[key1]]) if type(data1_dict[key1][0]) is dict else data1_dict[key1]
                                    d2_expanded = np.concatenate([np.full(len(shot['data']), data2_dict[key2][i]) for i, shot in enumerate(data1_dict[key1])]) if type(data1_dict[key1][0]) is dict else data2_dict[key2]
                                    d2 = d2_expanded
            
                                valid_mask = ~np.isnan(d1) & ~np.isnan(d2)
                                d1, d2 = d1[valid_mask], d2[valid_mask]
            
                                if len(d1) == 0 or len(d2) == 0:
                                    correlation_matrix[ind2, ind1, ind_mode] = np.nan
                                    continue
            
                                if quantity == 'correlation':
                                    correlation_matrix[ind2, ind1, ind_mode] = correlation(d1, d2, 
                                                                                           threshold_correlation=threshold_corr,
                                                                                           correlation_accept=corr_accept,
                                                                                           confidence_sigma=threshold_multiplier)
                                elif quantity == 'mutual_information':
                                    d1 -= np.mean(d1)
                                    d2 -= np.mean(d2)
                                    correlation_matrix[ind2, ind1, ind_mode] = mutual_information(d1, d2)
                                    
                                elif quantity == 'coefficient_of_determination':
                                    slope, intercept, r_value, p_value, std_err = linregress(d2, d1)
                                    correlation_matrix[ind2, ind1, ind_mode] = r_value ** 2
                                    
                            except Exception as e:
                                print(f"Calculation failed for {key1} vs {key2}: {e}")
                                correlation_matrix[ind2, ind1, ind_mode] = np.nan
                else:
                    correlation_matrix[:,:,ind_mode] = correlation_matrix[:,:,0] - correlation_matrix[:,:,1]
                    colormap='seismic'
                    cfg['zrange']=[-1,1]
                    
                if plot_for_publication:
                    # Apply Publication labels
                    if units is not None:
                        gpi_labels = [units[l][0] for l in gpi_labels]
                        plasma_labels = [units[l][0] for l in plasma_labels]
                    labels = gpi_labels + plasma_labels
                    
                    fig, ax = plt.subplots(figsize=(8.5/2.54, 8.5/2.54 * 1.2))
                    plot_pearson_matrix(correlation_matrix[:,:,ind_mode], xlabels=labels, ylabels=labels,
                                        title=cfg['title']+' '+mode, colormap=colormap, zrange=cfg['zrange'],
                                        fig_ax=(fig, ax), charsize=charsize, charsize_score=charsize*2/3,
                                        ticksize=ticksize, linewidth=linewidth, minor_ticksize=0.001, 
                                        plot_colorbar=plot_colorbar)
        
                    # Draw Quadrant Rectangles
                    ax.add_patch(Rectangle((-0.45, -0.45), 3.9, 3.9, fill=False, edgecolor='red', lw=2, zorder=0))
                    ax.add_patch(Rectangle((-0.45, 3.55), 3.9, 6.9, fill=False, edgecolor='yellow', lw=2, zorder=0))
                    ax.add_patch(Rectangle((3.55, -0.45), 6.9, 3.9, fill=False, edgecolor='cyan', lw=2, zorder=0))
                    ax.add_patch(Rectangle((3.55, 3.55), 6.9, 6.9, fill=False, edgecolor='magenta', lw=2, zorder=0))
                else:
                    if plot_full:
                        charsize, charsize_score, linewidth = 2, 1.5, 0.5
                    else:
                        charsize, charsize_score, linewidth = 3, 3, 1
                    plot_pearson_matrix(correlation_matrix[:,:,ind_mode], xlabels=label_1, ylabels=label_2,
                                        title=cfg['title']+' '+mode, colormap=colormap, zrange=cfg['zrange'],
                                        figsize=figsize, charsize=charsize, charsize_score=charsize_score, ticksize=0.5,
                                        linewidth=linewidth, minor_ticksize=0.001, plot_colorbar=plot_colorbar)
        
                pdf_page.savefig()
            
    if pdf:        
        pdf_page.close()
        matplotlib.use('qt5agg')

def plot_all_cross_data_matrix(analyze_lh_difference=True):
    
    averaging=['no','blob','shot']
    quantity=['correlation','mutual_information', 'coefficient_of_determination']
    method=['watershed', 'contour']
    
    for avg in averaging:
        for qty in quantity:
            for mthd in method:
                calculate_blob_plasma_parameter_correlation_matrix(nocalc=False, 
                                                                   averaging=avg, 
                                                                   average='avg', 
                                                                   str_finding_method=mthd, 
                                                                   quantity=qty, 
                                                                   plot_for_publication=False, 
                                                                   plot_full=False, 
                                                                   threshold_corr=False, 
                                                                   linewidth=1, 
                                                                   ticksize=3, 
                                                                   charsize=9, 
                                                                   analyze_lh_difference=analyze_lh_difference)
    

def plot_blob_plasma_parameter_trends(pdf_filename=None,
                                      nocalc=True,
                                      threshold_corr=False,
                                      threshold_multiplier=2,
                                      plot_for_publication=False,
                                      plot_all_trends=False,
                                      plot_2d_histogram=False,
                                      analyze_l_mode_only=False,
                                      analyze_h_mode_only=False,
                                      analyze_lh_difference=False,
                                      save_data_for_publication=False,
                                      ):
    """
    Plots correlation trends between blob properties and global plasma parameters.

    This function generates scatter plots, linear regressions, or 2D histograms 
    comparing Gas Puff Imaging (GPI) blob metrics to core plasma physics parameters. 
    It supports multiple analysis modes (L-mode, H-mode, or full data) and can 
    threshold plots to only show statistically significant correlations.

    Args:
        pdf_filename (str, optional): Custom output path for the generated PDF. 
            If None, an automatic filename is generated. Defaults to None.
        nocalc (bool, optional): If True, loads pre-calculated data from pickle files. 
            Defaults to True.
        threshold_corr (bool, optional): If True, only generates plots for parameter 
            pairs that exceed a specific correlation confidence threshold. Defaults to False.
        threshold_multiplier (int, optional): The sigma multiplier used for the 
            correlation thresholding. Defaults to 2.
        plot_for_publication (bool, optional): If True, generates a curated 3x3 grid 
            of scatter plots with linear regression lines and publication formatting. 
            Defaults to False.
        plot_2d_histogram (bool, optional): If True, generates a grid of 2D 
            histograms for the curated parameter pairs instead of scatter plots. 
            Defaults to False.
        analyze_l_mode_only (bool, optional): Restricts data to L-mode. Defaults to False.
        analyze_h_mode_only (bool, optional): Restricts data to H-mode. Defaults to False.
        analyze_lh_difference (bool, optional): Overlays L-mode and H-mode data on 
            the same plots for visual comparison. Defaults to False.
        save_data_for_publication (bool, optional): Exports the underlying plot data 
            to `.txt` files. Defaults to False.

    Returns:
        None
    """
    import matplotlib
    matplotlib.use('agg')

    # --- 1. Filename Setup ---
    if pdf_filename is None:
        base_name = 'plasma_vs_blob_2d_histrogram' if plot_2d_histogram else 'everything_vs_everything'
        str_add = ''
        if analyze_l_mode_only: str_add = '_l_mode'
        elif analyze_h_mode_only: str_add = '_h_mode'
        elif analyze_lh_difference: str_add = '_lh_diff'
        
        thres_str = f"_thres_{threshold_multiplier}" if threshold_corr else ""
        pdf_filename = f"{wd}/plots/{base_name}{thres_str}{str_add}.pdf"

    pdf_page = PdfPages(pdf_filename)

    # --- 2. Dynamic Data Loading Helper ---
    def _load_data(l_mode=False, h_mode=False):
        mode_str = '_l_mode' if l_mode else '_h_mode' if h_mode else '_full'
        p_plasma = f"{wd}/processed_data/plasma_vs_blob_plasma_data{mode_str}.pickle"
        p_blob = f"{wd}/processed_data/plasma_vs_blob_blob_data{mode_str}.pickle"
        
        if not os.path.exists(p_plasma):
            plasma_data = read_all_plasma_data(nocalc=nocalc, read_l_mode_only=l_mode, read_h_mode_only=h_mode)
            with open(p_plasma, 'wb') as f: pickle.dump(plasma_data, f)
        else:
            with open(p_plasma, 'rb') as f: plasma_data = pickle.load(f)
            
        if not os.path.exists(p_blob):
        # if True:
            blob_data = read_all_blob_data(nocalc=nocalc, str_finding_method='watershed', 
                                           fix_angle_for_correlation=True, averaging='shot', average='avg',
                                           read_l_mode_only=l_mode, read_h_mode_only=h_mode)
            with open(p_blob, 'wb') as f: pickle.dump(blob_data, f)
        else:
            with open(p_blob, 'rb') as f: blob_data = pickle.load(f)
        
        for key in plasma_data.keys():
            plasma_data[key]=np.asarray(plasma_data[key])
        
        for key in blob_data.keys():
            blob_data[key]=np.asarray(blob_data[key])    
        
        # Calculate derived metrics
        scale_length = (plasma_data['Larmor radius sound']**0.8 * plasma_data['Connection length']**0.4 /
                        plasma_data['Pedestal radius']**0.2)
        plasma_data['Blob size dimensionless'] = (np.sqrt(blob_data['Area']) / np.pi / scale_length)**2.5
        
        # In-place masking of invalid connection lengths
        plasma_data['Connection length'] = np.where(plasma_data['Connection length'] < 1.5, np.nan, plasma_data['Connection length'])
        
        return plasma_data, blob_data

    # --- 3. Load Datasets ---
    if analyze_lh_difference:
        plasma_l, blob_l = _load_data(l_mode=True)
        plasma_h, blob_h = _load_data(h_mode=True)
    else:
        full_plasma_data, full_blob_data = _load_data(l_mode=analyze_l_mode_only, h_mode=analyze_h_mode_only)
    # --- 4. Plotting Mode: Publication Scatter Grid ---
    
    if plot_for_publication and not plot_2d_histogram:
        flap_nstx.tools.set_matplotlib_for_publication(labelsize=6., linewidth=0.5, major_ticksize=2.)
        interesting_key_pairs, units = return_interesting_key_pairs()

        ncol, nrow = 3, 3
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(17/2.54, 10/2.54))
        
        data_plasma_iterate = [plasma_l, plasma_h] if analyze_lh_difference else [full_plasma_data]
        data_blob_iterate = [blob_l, blob_h] if analyze_lh_difference else [full_blob_data]
        colors = ['tab:blue', 'tab:orange'] if analyze_lh_difference else ['tab:blue']
        legend_labels = ['L-mode', 'H-mode'] if analyze_lh_difference else ['']

        for ind_full_data, (p_data, b_data) in enumerate(zip(data_plasma_iterate, data_blob_iterate)):
            
            # Apply safe boolean masking to remove outliers
            valid_mask = (p_data['Pressure at max'] <= 3.5) & (p_data['Temperature pedestal width'] >= 0.005)
            for k in p_data: p_data[k] = p_data[k][valid_mask]
            for k in b_data: b_data[k] = b_data[k][valid_mask]

            for ind, (key1, key2) in enumerate(interesting_key_pairs):
                if ind >= ncol * nrow: break # Prevent index out of bounds
                
                ax = axs[ind // ncol, ind % ncol]
                valid_data = ~np.isnan(b_data[key1]) & ~np.isnan(p_data[key2])
                
                data1 = b_data[key1][valid_data] * units[key1][2]
                data2 = p_data[key2][valid_data] * units[key2][2]

                # Standardize arrays for stats
                d1_c, d2_c = data1 - np.mean(data1), data2 - np.mean(data2)
                correlation = np.sum(d1_c * d2_c) / np.sqrt(np.sum(d1_c**2) * np.sum(d2_c**2))
                slope, intercept, r_value, p_value, std_err = linregress(data2, data1)
                r_squared = r_value ** 2

                ax.plot(data2, data1, linestyle='None', marker='o', ms=1, label=legend_labels[ind_full_data])
                sns.regplot(x=data2, y=data1, ci=68.27, ax=ax, color=colors[ind_full_data], scatter_kws={'s': 1})

                if ind_full_data == 0:
                    ax.set_xlabel(f"{units[key2][0]} [{units[key2][1]}]")
                    ax.set_ylabel(f"{units[key1][0]} [{units[key1][1]}]")
                    ax.text(-0.15, 1.02, f"({alc[ind]})", transform=ax.transAxes, size=6, va='bottom', ha='left')
                    
                    if not analyze_lh_difference:
                        pos = [0.05, 0.005] if correlation < 0 else [0.65, 0.005]
                        ax.text(pos[0], pos[1], f"$\\rho \\ =\\ {correlation:.2f}$", size=6, va='bottom', ha='left', transform=ax.transAxes)
                        ax.text(pos[0], pos[1] + 0.12, f"$R^2 \\ =\\ {r_squared:.2f}$", size=6, va='bottom', ha='left', transform=ax.transAxes)

                        if save_data_for_publication:
                            with open(f"{wd}/{alc[ind]}_blob_plasma_2dhist.txt", 'w+') as f:
                                f.write(f"{key2} data\n{data2}\n\n{key1} data\n{data1}\n\nCorrelation: {correlation}\nR^2 value: {r_squared}")

        if analyze_lh_difference:
            for ax in axs.flat: ax.legend(fontsize=6)
    
        plt.tight_layout(pad=0.1)
        pdf_page.savefig()
        
    elif plot_all_trends:
        flap_nstx.tools.set_matplotlib_for_publication(labelsize=6., linewidth=0.5, major_ticksize=2.)
        legend_labels = ['L-mode', 'H-mode']
        colors = ['tab:blue', 'tab:orange']
        corr_accept = calculate_corr_acceptance_levels()
        corr_threshold_accept=[0,0]
        for key1 in blob_l.keys():
            for key2 in plasma_l.keys():
                correlation=[0,0]
                r_squared=[0,0]
                for ind_mode in [0,1]:
                    if ind_mode == 0:
                        blob_data = blob_l
                        plasma_data = plasma_l
                    else:
                        blob_data = blob_h
                        plasma_data = plasma_h
                        
                    valid_data = ~np.isnan(blob_data[key1]) & ~np.isnan(plasma_data[key2])
                
                    data1 = blob_data[key1][valid_data]
                    data2 = plasma_data[key2][valid_data]
        
                    # Standardize arrays for stats
                    d1_c, d2_c = data1 - np.mean(data1), data2 - np.mean(data2)
                    correlation[ind_mode] = np.sum(d1_c * d2_c) / np.sqrt(np.sum(d1_c**2) * np.sum(d2_c**2))
                    slope, intercept, r_value, p_value, std_err = linregress(data2, data1)
                    r_squared[ind_mode] = r_value ** 2
                    N = np.sum(valid_data)
                    if N < 160:
                        corr_threshold_accept[ind_mode]=corr_accept['avg'][N] + threshold_multiplier * corr_accept['stddev'][N]
                    else:
                        corr_threshold_accept[ind_mode]=0.064
                                            
                                            
                                            
                if isinstance(threshold_corr,list) and len(threshold_corr) > 1:
                    condition=(threshold_corr is not None and 
                               any(abs(correlation[ind]) < max(threshold_corr) for ind in [0,1]) and 
                               any(abs(correlation[ind]) > min(threshold_corr) for ind in [0,1]) and 
                               any(abs(correlation[ind]) > corr_threshold_accept[ind] for ind in [0,1]))
                else:
                    condition=(threshold_corr is not None and 
                               any(abs(correlation[ind]) > threshold_corr for ind in [0,1]) and 
                               any(abs(correlation[ind]) > corr_threshold_accept[ind] for ind in [0,1]))
                
                if condition:
                    fig, ax = plt.subplots(figsize=(8.5/2.54, 8.5/2.54))
                    for ind_mode in [0,1]:  
                        if ind_mode == 0:
                            blob_data = blob_l
                            plasma_data = plasma_l
                        else:
                            blob_data = blob_h
                            plasma_data = plasma_h
                            
                        valid_data = ~np.isnan(blob_data[key1]) & ~np.isnan(plasma_data[key2])

                        data1 = blob_data[key1][valid_data]
                        data2 = plasma_data[key2][valid_data]
                        
                        ax.plot(data2, data1, linestyle='None', marker='o', ms=1, label=legend_labels[ind_mode], color=colors[ind_mode])
                        sns.regplot(x=data2, y=data1, 
                                    ci=68.27, ax=ax, 
                                    color=colors[ind_mode],
                                    scatter_kws={'s': 1},
                                    robust=True,
                                    )
                        pos = [0.05, 0.005] if correlation[ind_mode] < 0 else [0.65, 0.005]
                        if ind_mode == 1:
                            add_pos=0.03
                            add_text='H-mode'
                        else:
                            add_pos=0.
                            add_text='L-mode'
                            
                        ax.text(pos[0], pos[1] +          3*add_pos, 
                                f"$\\rho \\ =\\ {correlation[ind_mode]:.2f}$ | {corr_threshold_accept[ind_mode]:.2f}", 
                                size=6, 
                                va='bottom', 
                                ha='left', 
                                transform=ax.transAxes, 
                                color=colors[ind_mode])
                        
                        ax.text(pos[0], pos[1] +   0.03 + 3*add_pos,
                                f"$R^2 \\ =\\ {r_squared[ind_mode]:.2f}$", 
                                size=6, 
                                va='bottom', 
                                ha='left', 
                                transform=ax.transAxes, 
                                color=colors[ind_mode])
                        
                        ax.text(pos[0], pos[1] + 2*0.03 + 3*add_pos, 
                                add_text, 
                                size=6, 
                                va='bottom', 
                                ha='left', 
                                transform=ax.transAxes, 
                                color=colors[ind_mode])
                        
                        ax.set_xlabel(key2)
                        ax.set_ylabel(key1)
                        if ind_mode == 1:
                            print(f"{key2} - {key1} - $\\rho$= {correlation[ind_mode]:.2f} - R^2 = {r_squared[ind_mode]:.2f}")
                    plt.tight_layout(pad=0.1)
                    pdf_page.savefig()
    # --- 5. Plotting Mode: 2D Histograms ---
    elif plot_2d_histogram:
        fig, axes = plt.subplots(4, 3, figsize=(8.5/2.54, 17/2.54))
        interesting_key_pairs, units = return_interesting_key_pairs()

        for ind, (key1, key2) in enumerate(interesting_key_pairs):
            if ind >= 12: break
            
            valid_mask = ~np.isnan(full_blob_data[key1]) & ~np.isnan(full_plasma_data[key2])
            data1 = np.real(full_blob_data[key1][valid_mask])
            data2 = np.real(full_plasma_data[key2][valid_mask])

            ax = axes[ind // 3, ind % 3]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            im = ax.hist2d(data1, data2, bins=[21, 21])
            fig.colorbar(im[3], cax=cax, orientation='vertical')
            ax.set_xlabel(key1)
            ax.set_ylabel(key2)
            
        plt.tight_layout(pad=0.1)
        pdf_page.savefig()

    # --- 6. Plotting Mode: Dynamic Thresholded Scatter ---
    else:
        corr_accept = calculate_corr_acceptance_levels()
        for key1 in full_blob_data.keys():
            for key2 in ['Connection length', 'Collisionality dimensionless', 'Blob size dimensionless']:
                
                valid_mask = ~np.isnan(full_blob_data[key1]) & ~np.isnan(full_plasma_data[key2])
                data1 = full_blob_data[key1][valid_mask]
                data2 = full_plasma_data[key2][valid_mask]
                
                if len(data1) < 2: continue # Skip if arrays are too small to correlate

                d1_c, d2_c = data1 - np.mean(data1), data2 - np.mean(data2)
                correlation = np.sum(d1_c * d2_c) / np.sqrt(np.sum(d1_c**2) * np.sum(d2_c**2))

                plot_page = True
                if threshold_corr:
                    N = np.sum(valid_mask)
                    threshold = corr_accept['avg'][N] + threshold_multiplier * corr_accept['stddev'][N]
                    if np.abs(correlation) <= threshold: plot_page = False

                slope, intercept, r_value, p_value, std_err = linregress(data1, data2) if len(data1) > 2 else (0,0,0,0,0)
                if r_value**2 <= 0.2: plot_page = False

                if plot_page:
                    fig, ax = plt.subplots(figsize=(8.5/2.54, 8.5/2.54 * 1.2))
                    ax.scatter(data1, data2)
                    sns.regplot(x=data1, y=data2, ci=68.27, ax=ax, scatter_kws={'s': 1})
                    
                    pos = [0.05, 0.005] if correlation < 0 else [0.65, 0.005]
                    ax.text(pos[0], pos[1], f"$\\rho \\ =\\ {correlation:.2f}$", size=6, va='bottom', ha='left', transform=ax.transAxes)
                    ax.text(pos[0], pos[1] + 0.12, f"$R^2 \\ =\\ {r_value**2:.2f}$", size=6, va='bottom', ha='left', transform=ax.transAxes)
                    
                    ax.set_xlabel(key1)
                    ax.set_ylabel(key2)
                    ax.set_title(f"{key1} vs \n{key2}")
                    plt.tight_layout(pad=0.1)
                    pdf_page.savefig()
                    plt.close(fig)

    pdf_page.close()
    matplotlib.use('qt5agg')

def plot_blob_experiment_vs_theory_radial_velocity(pdf_filename=None,
                                                   nocalc=True):
    """
    Plots experimentally measured radial blob velocities against theoretical scalings.

    This function extracts the shot-averaged radial velocity and size of blobs from 
    the experimental tracking database. It then compares these measured velocities 
    to two fundamental theoretical scaling regimes: 
    1. The Inertial Regime scaling (~ c_s * rho_s / a)
    2. The Sheath Limited Regime scaling (~ c_s * (rho_s / a)^2)
    where c_s is the sound speed, rho_s is the Larmor radius, and a is the blob size.
    The resulting scatter plots are saved to a PDF.

    Args:
        pdf_filename (str, optional): Custom filepath for the output PDF. If None, 
            defaults to 'plots/experiment_vs_theory.pdf' in the working directory. 
            Defaults to None.
        nocalc (bool, optional): If True, attempts to load pre-calculated aggregated 
            plasma and blob data from cached pickle files. Defaults to True.

    Returns:
        None
    """
    import matplotlib
    matplotlib.use('agg')

    # --- 1. Filename Setup ---
    if pdf_filename is None:
        pdf_filename = f"{wd}/plots/experiment_vs_theory.pdf"
    elif not pdf_filename.endswith('.pdf'):
        pdf_filename += '.pdf'

    pdf_page = PdfPages(pdf_filename)

    # --- 2. Load Data ---
    full_plasma_data = read_all_plasma_data(nocalc=nocalc)
    full_blob_data = read_all_blob_data(nocalc=nocalc, 
                                        str_finding_method='watershed',
                                        fix_angle_for_correlation=True,
                                        averaging='shot',
                                        average='avg')
    
    # --- 3. Extract and Calculate Velocities ---
    # Experimental Data
    exp_vrad_pos = np.abs(full_blob_data['Velocity radial position fit'])
    
    # Theoretical Physics Inputs
    c_s = full_plasma_data['Sound speed']
    rho_s = full_plasma_data['Larmor radius sound']
    blob_size = full_blob_data['Size radial fit']
    
    # Theoretical Scalings
    vrad_inertial = c_s * (rho_s / blob_size)
    vrad_sheath_limited = c_s * (rho_s / blob_size)**2
    
    # --- 4. Plot 1: Inertial Regime ---
    fig1, ax1 = plt.subplots(figsize=(8.5/2.54, 8.5/2.54))
    ax1.scatter(exp_vrad_pos, vrad_inertial, s=5)
    
    ax1.set_xlabel('Experimental $|v_{rad}|$')
    ax1.set_ylabel('Theoretical Inertial $v_{rad}$')
    ax1.set_title('Inertial Regime Scaling')
    
    plt.tight_layout(pad=0.1)
    pdf_page.savefig(fig1)
    plt.close(fig1)
    
    # --- 5. Plot 2: Sheath Limited Regime ---
    fig2, ax2 = plt.subplots(figsize=(8.5/2.54, 8.5/2.54))
    ax2.scatter(exp_vrad_pos, vrad_sheath_limited, s=5)
    
    ax2.set_xlabel('Experimental $|v_{rad}|$')
    ax2.set_ylabel('Theoretical Sheath Limited $v_{rad}$')
    ax2.set_title('Sheath Limited Regime Scaling')
    
    plt.tight_layout(pad=0.1)
    pdf_page.savefig(fig2)
    plt.close(fig2)
    
    # --- 6. Cleanup ---
    pdf_page.close()
    matplotlib.use('qt5agg')
    
    
    
def plot_blob_regime_graph(pdf_filename=None,
                           nocalc=True,
                           save_data_for_publication=False,
                           analyze_lh_difference=True,
                           ):
    """
    Plots a regime graph of dimensionless blob size vs. dimensionless collisionality.

    This function categorizes experimental plasma blobs into different theoretical 
    regimes (RB, RX, C_I, C_S) based on their dimensionless size ($\Theta$) and 
    dimensionless collisionality ($\Lambda$). It maps the aggregated data onto a 
    log-log scatter plot with hardcoded regime boundary lines.

    Args:
        pdf_filename (str, optional): Custom filepath for the output PDF. If None, 
            defaults to 'plots/blob_regimes.pdf' in the working directory. 
            Defaults to None.
        nocalc (bool, optional): If True, attempts to load pre-calculated aggregated 
            plasma and blob data from cached pickle files. Defaults to True.
        save_data_for_publication (bool, optional): If True, exports the raw 
            $\Theta$ and $\Lambda$ arrays to a text file for external plotting. 
            Defaults to False.

    Returns:
        None
    """
    import matplotlib
    matplotlib.use('agg')

    # --- 1. Filename Setup ---
    if pdf_filename is None:
        pdf_filename = f"{wd}/plots/blob_regimes.pdf"
    elif not pdf_filename.endswith('.pdf'):
        pdf_filename += '.pdf'

    pdf_page = PdfPages(pdf_filename)
    if not analyze_lh_difference:
        # --- 2. Load Data ---
        full_plasma_data = read_all_plasma_data(nocalc=nocalc, calculate_parameters_in_sol=True)
        full_blob_data = read_all_blob_data(nocalc=nocalc, 
                                            str_finding_method='watershed',
                                            fix_angle_for_correlation=True, 
                                            averaging='shot', 
                                            average='avg')
    
        # --- 3. Derived Calculations ---
        scale_length = (full_plasma_data['Larmor radius sound']**0.8 * full_plasma_data['Connection length']**0.4 /
                        full_plasma_data['Pedestal radius']**0.2)
    
        x_data = (np.sqrt(full_blob_data['Area'] / np.pi) / scale_length)**2.5
        y_data = full_plasma_data['Collisionality dimensionless']
    
        valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    
        # --- 4. Export Data ---
        if save_data_for_publication:
            with open(f"{wd}/blob_regime_data.txt", 'w+') as f:
                f.write('Dimensionless blob size\n')
                f.write(str(x_data[valid_mask]) + '\n')
                f.write('Dimensionless collisionality\n')
                f.write(str(y_data[valid_mask]) + '\n')
    
        # --- 5. Plotting ---
        fig, ax = plt.subplots(figsize=(8.5/2.54, 8.5/1.5/2.54))
        
        ax.scatter(x_data, y_data, s=5)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$\\Theta$')
        ax.set_ylabel('$\\Lambda$')
    
        x_max = 100
        y_min = 0.01
        ax.set_xlim([0.1, x_max])
        ax.set_ylim([y_min, 10])
    
        # --- 6. Regime Boundaries ---
        # Draw separating lines using standard ax.plot([x1, x2], [y1, y2])
        ax.plot([1e-3, x_max], [1e-3, x_max], color='black', lw=1)  # p1 to p2
        ax.plot([10, x_max],   [1, 1],        color='black', lw=1)  # p3 to p4
        ax.plot([10, 10],      [0.01, 1],     color='black', lw=1)  # p5 to p6 (which is p3)
        ax.plot([0.1, 10],     [y_min, 1],    color='black', lw=1)  # p7 to p8 (which is p3)
    
        # --- 7. Regime Annotations ---
        text_arr = [
            ("RB",   0.1,  0.8),
            ("RX",   0.8,  0.8),
            ("$C_I$", 0.4,  0.15),
            ("$C_S$", 0.85, 0.05)
        ]
        
        for (text, xpos, ypos) in text_arr:
            ax.text(xpos, ypos, text, transform=ax.transAxes, size=9, 
                    verticalalignment='bottom', horizontalalignment='left')
    else:
        
        def _load_data(l_mode=False, h_mode=False):
            mode_str = '_l_mode' if l_mode else '_h_mode' if h_mode else '_full'
            p_plasma = f"{wd}/processed_data/plasma_vs_blob_plasma_data{mode_str}.pickle"
            p_blob = f"{wd}/processed_data/plasma_vs_blob_blob_data{mode_str}.pickle"
            
            if not os.path.exists(p_plasma):
                plasma_data = read_all_plasma_data(nocalc=nocalc, read_l_mode_only=l_mode, read_h_mode_only=h_mode)
                with open(p_plasma, 'wb') as f: pickle.dump(plasma_data, f)
            else:
                with open(p_plasma, 'rb') as f: plasma_data = pickle.load(f)
                
            if not os.path.exists(p_blob):
            # if True:
                blob_data = read_all_blob_data(nocalc=nocalc, str_finding_method='watershed', 
                                               fix_angle_for_correlation=True, averaging='shot', average='avg',
                                               read_l_mode_only=l_mode, read_h_mode_only=h_mode)
                with open(p_blob, 'wb') as f: pickle.dump(blob_data, f)
            else:
                with open(p_blob, 'rb') as f: blob_data = pickle.load(f)
            
            for key in plasma_data.keys():
                plasma_data[key]=np.asarray(plasma_data[key])
            
            for key in blob_data.keys():
                blob_data[key]=np.asarray(blob_data[key])    
            
            # Calculate derived metrics
            scale_length = (plasma_data['Larmor radius sound']**0.8 * plasma_data['Connection length']**0.4 /
                            plasma_data['Pedestal radius']**0.2)
            plasma_data['Blob size dimensionless'] = (np.sqrt(blob_data['Area']) / np.pi / scale_length)**2.5
            
            # In-place masking of invalid connection lengths
            plasma_data['Connection length'] = np.where(plasma_data['Connection length'] < 1.5, np.nan, plasma_data['Connection length'])
            
            return plasma_data, blob_data
        
        plasma_l, blob_l = _load_data(l_mode=True)
        plasma_h, blob_h = _load_data(h_mode=True)
        
        flap_nstx.tools.set_matplotlib_for_publication(labelsize=6., linewidth=0.5, major_ticksize=2.)
        legend_labels = ['H-mode', 'L-mode']
        colors = ['tab:blue', 'tab:orange']

        fig, ax = plt.subplots(figsize=(8.5/2.54, 8.5/1.5/2.54))

        for ind_mode in [0,1]:
            if ind_mode == 1:
                blob_data = blob_l
                plasma_data = plasma_l
            else:
                blob_data = blob_h
                plasma_data = plasma_h
            
            # --- 3. Derived Calculations ---
            scale_length = (plasma_data['Larmor radius sound']**0.8 * plasma_data['Connection length']**0.4 /
                            plasma_data['Pedestal radius']**0.2)
        
            x_data = (np.sqrt(blob_data['Area'] / np.pi) / scale_length)**2.5
            y_data = plasma_data['Collisionality dimensionless']
        
            valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)

                
            ax.scatter(x_data, y_data, s=5)
            ax.text(0.04, 0.04+ind_mode*0.06, legend_labels[ind_mode], 
                    transform=ax.transAxes, size=9, 
                    verticalalignment='bottom', horizontalalignment='left',
                    color=colors[ind_mode])
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$\\Theta$')
        ax.set_ylabel('$\\Lambda$')
    
        x_max = 100
        y_min = 0.0005
        ax.set_xlim([0.1, x_max])
        ax.set_ylim([y_min, 10])
    
        # --- 6. Regime Boundaries ---
        # Draw separating lines using standard ax.plot([x1, x2], [y1, y2])
        ax.plot([1e-3, x_max], [1e-3, x_max], color='black', lw=1)  # p1 to p2
        ax.plot([10, x_max],   [1, 1],        color='black', lw=1)  # p3 to p4
        ax.plot([10, 10],      [y_min, 1],     color='black', lw=1)  # p5 to p6 (which is p3)
        ax.plot([0.1, 10],     [1e-2, 1],    color='black', lw=1)  # p7 to p8 (which is p3)
    
        # --- 7. Regime Annotations ---
        text_arr = [
            ("RB",   0.1,  0.8),
            ("RX",   0.8,  0.8),
            ("$C_I$", 0.5,  0.1),
            ("$C_S$", 0.85, 0.1)
        ]
        
        for (text, xpos, ypos) in text_arr:
            ax.text(xpos, ypos, text, transform=ax.transAxes, size=9, 
                    verticalalignment='bottom', horizontalalignment='left')
        
                    
    plt.tight_layout(pad=0.1)
    pdf_page.savefig(fig)
    pdf_page.close()
    plt.close(fig)
    
    matplotlib.use('qt5agg')

    
def plot_well_known_parameter_dependences(pdf_filename=None,
                                          nocalc=True,
                                          save_data_for_publication=False,
                                          analyze_lh_difference=True,
                                          ):
    """
    Plots a multi-panel figure of specific, well-known blob and plasma parameter dependencies.

    This function isolates specific physical relationships (e.g., how the blob's 
    dimensionless radial velocity or geometric properties scale with connection 
    length or collisionality). It calculates required dimensionless variables 
    (like a_star, v_star, and dimensionless radial velocity) on the fly, 
    generates scatter plots with linear regressions, and computes the Pearson 
    correlation and R-squared values for each pair.

    Args:
        pdf_filename (str, optional): Custom filepath for the output PDF. If None, 
            defaults to 'plots/interesting_parameter_pairs_2.pdf'. Defaults to None.
        nocalc (bool, optional): If True, attempts to load pre-calculated aggregated 
            plasma and blob data from cached pickle files. Defaults to True.
        save_data_for_publication (bool, optional): If True, exports the raw 
            array data and statistical results for each subplot into text files. 
            Defaults to False.

    Returns:
        None
    """
    import matplotlib
    matplotlib.use('agg')

    # --- 1. Filename Setup ---
    if pdf_filename is None:
        pdf_filename = f"{wd}/plots/interesting_parameter_pairs_2.pdf"
    elif not pdf_filename.endswith('.pdf'):
        pdf_filename += '.pdf'

    pdf_page = PdfPages(pdf_filename)
    
                
    ncol, nrow = 2, 3
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(8.5/2.54, 8.5*1.5/2.54))
    
    
    interesting_key_pairs = [
        ('Angle fit',                       'Connection length'),
        ('Angular velocity ALI',            'Connection length'),
        ('Roundness',                       'Connection length'),
        ('Solidity',                        'Connection length'),
        ('Velocity radial dimensionless',   'Connection length'),
        ('Velocity radial dimensionless',   'Collisionality dimensionless')
    ]
    
    units = {
        'Angle fit':                       ['$\\theta_{blob}$', 'rad', 1],
        'Angular velocity ALI':            ['$\\omega_{blob}$', 'krad/s', 1e-3],
        'Roundness':                       ['Roundness', '', 1],
        'Solidity':                        ['Solidity', '', 1],
        'Velocity radial dimensionless':   ['$\\hat{v}$', '', 1],
        'Connection length':               ['$L_{||}$', 'm', 1],
        'Collisionality dimensionless':    ['$\\Lambda$', '', 1]
    }
    
    if not analyze_lh_difference:
        # --- 2. Load Data ---
        full_plasma_data = read_all_plasma_data(nocalc=nocalc, calculate_parameters_in_sol=True)
        full_blob_data = read_all_blob_data(nocalc=nocalc, str_finding_method='watershed',
                                            fix_angle_for_correlation=True, averaging='shot', average='avg')
        
        # --- 3. Derived Dimensionless Calculations ---
        # a_star and v_star scaling
        a_star = (full_plasma_data['Larmor radius sound']**0.8 * full_plasma_data['Connection length']**0.4 / 
                  np.abs(full_blob_data['Velocity radial position fit'])**0.2)
                  
        v_star = full_plasma_data['Sound speed'] * (a_star / full_blob_data['Position radial fit'])**0.5
        
        full_blob_data['Velocity radial dimensionless'] = full_blob_data['Velocity radial position fit'] / v_star
        full_plasma_data['Inverse A hat squared'] = 1 / (full_blob_data['Size radial fit'] / a_star)**2
        
        
        ncol, nrow = 2, 3
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(8.5/2.54, 8.5*1.5/2.54))
        
        # --- 5. Main Plotting Loop ---
        for ind, (key1, key2) in enumerate(interesting_key_pairs):
            ax = axs[ind // ncol, ind % ncol]
            
            # Mask NaNs safely
            valid_mask = ~np.isnan(full_blob_data[key1]) & ~np.isnan(full_plasma_data[key2])
            data1 = full_blob_data[key1][valid_mask] * units[key1][2]
            data2 = full_plasma_data[key2][valid_mask] * units[key2][2]
            
            # Math & Stats
            d1_c, d2_c = data1 - np.mean(data1), data2 - np.mean(data2)
            correlation = np.sum(d1_c * d2_c) / np.sqrt(np.sum(d1_c**2) * np.sum(d2_c**2))
            slope, intercept, r_value, p_value, std_err = linregress(data2, data1)
            r_squared = r_value ** 2
            
            # Scatter and Regression
            ax.plot(data2, data1, linestyle='None', marker='o', ms=1)
            sns.regplot(x=data2, y=data1, ci=68.27, ax=ax, scatter_kws={'s': 1})
            
            # Dynamic axis labels
            xlabel = f"{units[key2][0]} [{units[key2][1]}]" if units[key2][1] else units[key2][0]
            ylabel = f"{units[key1][0]} [{units[key1][1]}]" if units[key1][1] else units[key1][0]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Panel letter annotation
            ax.text(-0.4, 1.02, f"({alc[ind]})", transform=ax.transAxes, size=9, va='bottom', ha='left')
            
            # Adjust text positioning based on correlation sign or specific key
            if key1 == "Velocity radial dimensionless":
                pos_corr, pos_r2 = [0.05, 0.755], [0.05, 0.875]
            elif correlation < 0:
                pos_corr, pos_r2 = [0.05, 0.005], [0.05, 0.125]
            else:
                pos_corr, pos_r2 = [0.65, 0.005], [0.60, 0.125]
                
            ax.text(pos_corr[0], pos_corr[1], f"$\\rho \\ =\\ {correlation:.2f}$", 
                    size=6, va='bottom', ha='left', transform=ax.transAxes)
            ax.text(pos_r2[0], pos_r2[1], f"$R^2 \\ =\\ {r_squared:.2f}$", 
                    size=6, va='bottom', ha='left', transform=ax.transAxes)
            
            # Export Data
            if save_data_for_publication:
                with open(f"{wd}/{alc[ind]}_well_known_params.txt", 'w+') as f:
                    f.write(f"{key1} data:\n{data1}\n\n")
                    f.write(f"{key2} data:\n{data2}\n\n")
                    f.write(f"R2 value: {r_squared}\n")
                    f.write(f"Correlation: {correlation}")
    else:
        
        def _load_data(l_mode=False, h_mode=False):
            mode_str = '_l_mode' if l_mode else '_h_mode' if h_mode else '_full'
            p_plasma = f"{wd}/processed_data/plasma_vs_blob_plasma_data{mode_str}.pickle"
            p_blob = f"{wd}/processed_data/plasma_vs_blob_blob_data{mode_str}.pickle"
            
            if not os.path.exists(p_plasma):
                plasma_data = read_all_plasma_data(nocalc=nocalc, read_l_mode_only=l_mode, read_h_mode_only=h_mode)
                with open(p_plasma, 'wb') as f: pickle.dump(plasma_data, f)
            else:
                with open(p_plasma, 'rb') as f: plasma_data = pickle.load(f)
                
            if not os.path.exists(p_blob):
            # if True:
                blob_data = read_all_blob_data(nocalc=nocalc, str_finding_method='watershed', 
                                               fix_angle_for_correlation=True, averaging='shot', average='avg',
                                               read_l_mode_only=l_mode, read_h_mode_only=h_mode)
                with open(p_blob, 'wb') as f: pickle.dump(blob_data, f)
            else:
                with open(p_blob, 'rb') as f: blob_data = pickle.load(f)
            
            for key in plasma_data.keys():
                plasma_data[key]=np.asarray(plasma_data[key])
            
            for key in blob_data.keys():
                blob_data[key]=np.asarray(blob_data[key])    
            
            # Calculate derived metrics
            scale_length = (plasma_data['Larmor radius sound']**0.8 * plasma_data['Connection length']**0.4 /
                            plasma_data['Pedestal radius']**0.2)
            plasma_data['Blob size dimensionless'] = (np.sqrt(blob_data['Area']) / np.pi / scale_length)**2.5
            
            # In-place masking of invalid connection lengths
            plasma_data['Connection length'] = np.where(plasma_data['Connection length'] < 1.5, np.nan, plasma_data['Connection length'])
            
            return plasma_data, blob_data
        
        plasma_l, blob_l = _load_data(l_mode=True)
        plasma_h, blob_h = _load_data(h_mode=True)
        
        flap_nstx.tools.set_matplotlib_for_publication(labelsize=6., linewidth=0.5, major_ticksize=2.)
        legend_labels = ['H-mode', 'L-mode']
        colors = ['tab:blue', 'tab:orange']
    
    
        for ind_mode in [0,1]:
            if ind_mode == 1:
                full_blob_data = blob_l
                full_plasma_data = plasma_l
            else:
                full_blob_data = blob_h
                full_plasma_data = plasma_h

            a_star = (full_plasma_data['Larmor radius sound']**0.8 * full_plasma_data['Connection length']**0.4 / 
                      np.abs(full_blob_data['Velocity radial position fit'])**0.2)
                      
            v_star = full_plasma_data['Sound speed'] * (a_star / full_blob_data['Position radial fit'])**0.5
            
            full_blob_data['Velocity radial dimensionless'] = full_blob_data['Velocity radial position fit'] / v_star
            full_plasma_data['Inverse A hat squared'] = 1 / (full_blob_data['Size radial fit'] / a_star)**2
            
            # --- 5. Main Plotting Loop ---
            for ind, (key1, key2) in enumerate(interesting_key_pairs):
                ax = axs[ind // ncol, ind % ncol]
                
                # Mask NaNs safely
                valid_mask = ~np.isnan(full_blob_data[key1]) & ~np.isnan(full_plasma_data[key2])
                data1 = full_blob_data[key1][valid_mask] * units[key1][2]
                data2 = full_plasma_data[key2][valid_mask] * units[key2][2]
                
                # Math & Stats
                d1_c, d2_c = data1 - np.mean(data1), data2 - np.mean(data2)
                correlation = np.sum(d1_c * d2_c) / np.sqrt(np.sum(d1_c**2) * np.sum(d2_c**2))
                slope, intercept, r_value, p_value, std_err = linregress(data2, data1)
                r_squared = r_value ** 2
                
                # Scatter and Regression
                ax.plot(data2, data1, linestyle='None', marker='o', ms=1, color=colors[ind_mode])
                sns.regplot(x=data2, y=data1, ci=68.27, ax=ax, scatter_kws={'s': 1}, color=colors[ind_mode])
                
                # Dynamic axis labels
                xlabel = f"{units[key2][0]} [{units[key2][1]}]" if units[key2][1] else units[key2][0]
                ylabel = f"{units[key1][0]} [{units[key1][1]}]" if units[key1][1] else units[key1][0]
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                
                # Panel letter annotation
                ax.text(-0.4, 1.02, f"({alc[ind]})", transform=ax.transAxes, size=9, va='bottom', ha='left')
                
                # Adjust text positioning based on correlation sign or specific key
                if ind_mode == 0:
                    pos_corr, pos_r2 = [0.05, 0.005], [0.05, 0.125]
                else:
                    pos_corr, pos_r2 = [0.65, 0.005], [0.60, 0.125]
                    
                ax.text(pos_corr[0], pos_corr[1], f"$\\rho \\ =\\ {correlation:.2f}$", 
                        size=6, va='bottom', ha='left', transform=ax.transAxes,
                        color=colors[ind_mode])
                ax.text(pos_r2[0], pos_r2[1], f"$R^2 \\ =\\ {r_squared:.2f}$", 
                        size=6, va='bottom', ha='left', transform=ax.transAxes,
                        color=colors[ind_mode])
                if ind == 5:
                    ax.text(0.7, 0.7+ind_mode*0.1, legend_labels[ind_mode], 
                            size=6, va='bottom', ha='left', transform=ax.transAxes,
                            color=colors[ind_mode])
    plt.tight_layout(pad=0.1)
    pdf_page.savefig(fig)
    pdf_page.close()
    plt.close(fig)
    
    matplotlib.use('qt5agg')
    
import itertools
import time

def plot_conditional_evolution_matrices(dt=2.5e-6, 
                                        min_r_squared=0.3,
                                        min_points_required=5,
                                        min_count_required=25,
                                        add_corrected_angles=False, #Does not make a difference in the trends
                                        plot_spaghetti=False,    # <-- NEW PARAMETER
                                        num_spaghetti=50,       # <-- NEW PARAMETER
                                        nocalc=True):
    """
    Plots the conditionally averaged evolution of blob parameters with optional 
    "spaghetti plot" overlays of raw individual trajectories to verify trends.
    """
    
    # [Assuming wd is defined globally in your script]
    
    l_mode_data = read_all_blob_data(time_range_around_peak=[-5e-3,15e-3], 
                                     nocalc=nocalc, recalc_tracking=False, 
                                     min_structure_lifetime=10, read_l_mode_only=True, 
                                     averaging='conditional', condition_key='Lifetime', 
                                     condition_range=[-2.5e-6,2.5e-6], replicate_histogram2=True)
    
    h_mode_data = read_all_blob_data(time_range_around_peak=[-5e-3,15e-3], 
                                     nocalc=nocalc, recalc_tracking=False, 
                                     min_structure_lifetime=10, read_h_mode_only=True, 
                                     averaging='conditional', condition_key='Lifetime', 
                                     condition_range=[-2.5e-6,2.5e-6], replicate_histogram2=True)

    
    # ===============================================================
    # Angle Difference Synthesis Helper
    # ===============================================================
    def add_angle_differences(dataset):
        base_angle = 'Poloidal angle'
        targets = [('Angle fit', 'Angle diff (Poloidal + Fit)'),
                   ('Angle ALI', 'Angle diff (Poloidal + ALI)')]
        
        if base_angle not in dataset: return dataset
            
        for target_key, new_key in targets:
            if target_key in dataset:
                t_base = dataset[base_angle]['relative_frames']
                t_target = dataset[target_key]['relative_frames']
                
                if len(t_base) == 0 or len(t_target) == 0: continue
                common_t, ind_base, ind_target = np.intersect1d(t_base, t_target, return_indices=True)
                
                val_base = dataset[base_angle]['mean'][ind_base]
                val_target = dataset[target_key]['mean'][ind_target]
                
                if '-' in new_key: diff = val_base - val_target
                elif '+' in new_key: diff = val_base + val_target
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                
                err_base_raw = np.asarray(dataset[base_angle]['error'])
                err_target_raw = np.asarray(dataset[target_key]['error'])
                err_base = err_base_raw[:, ind_base] if err_base_raw.ndim == 2 else err_base_raw[ind_base]
                err_target = err_target_raw[:, ind_target] if err_target_raw.ndim == 2 else err_target_raw[ind_target]
                err_diff = np.sqrt(err_base**2 + err_target**2)
                
                count_base = dataset[base_angle].get('count', np.full_like(val_base, np.inf))[ind_base]
                count_target = dataset[target_key].get('count', np.full_like(val_target, np.inf))[ind_target]
                
                dataset[new_key] = {
                    'mean': diff, 'error': err_diff, 'count': np.minimum(count_base, count_target), 'relative_frames': common_t
                }
                
                # Transform raw matrices for spaghetti plot
                if 'raw_matrix' in dataset[base_angle] and 'raw_matrix' in dataset[target_key]:
                    raw_base = dataset[base_angle]['raw_matrix'][:, ind_base]
                    raw_target = dataset[target_key]['raw_matrix'][:, ind_target]
                    raw_diff = (raw_base - raw_target) if '-' in new_key else (raw_base + raw_target)
                    dataset[new_key]['raw_matrix'] = (raw_diff + np.pi) % (2 * np.pi) - np.pi

        return dataset
    if add_corrected_angles:
        l_mode_data = add_angle_differences(l_mode_data)
        h_mode_data = add_angle_differences(h_mode_data)
    
    
    keys_to_ignore=['Axes length major', 'Axes length minor', 'Size radial', 'Size poloidal', 
                    'Angle envelope', 'Signed area', 'Total curvature', 'Total bending energy', 
                    'Center of gravity radial', 'Center of gravity poloidal', 
                    'Position radial fit', 'Position poloidal fit', 
                    'Expansion fraction axes fit', 
                    'Velocity radial COG', 'Velocity poloidal COG', 
                    'Velocity radial position fit', 'Velocity poloidal position fit', 
                    'Total curvature diff',
                    'Total bending energy diff', 
                    'Velocity radial position diff', 'Velocity poloidal position diff']
        
    valid_keys = [k for k in l_mode_data.keys() if k in h_mode_data and k not in keys_to_ignore]
    key_pairs = list(itertools.permutations(valid_keys, 2))
    pdf_filename = wd + f'/plots/conditional_average_blob_vs_blob_thres_{min_r_squared}.pdf'
    pdf_page = PdfPages(pdf_filename)
    
    plots_generated, plots_filtered = 0, 0
    angle_diff_keys = ['Angle diff (Poloidal - Fit)','Angle diff (Poloidal + Fit)',
                       'Angle diff (Poloidal - ALI)','Angle diff (Poloidal + ALI)']

    total_pairs = len(key_pairs)
    print(f"Starting cross-parameter evolution plotting. Analyzing {total_pairs} pairs...")
    start_time = time.time()

    for i, (key_x, key_y) in enumerate(key_pairs):
        
        def extract_and_align_data(dataset):
            if key_x not in dataset or key_y not in dataset: return None
                
            t_x, t_y = dataset[key_x]['relative_frames'], dataset[key_y]['relative_frames']
            if len(t_x) == 0 or len(t_y) == 0: return None
                
            common_t, ind_x, ind_y = np.intersect1d(t_x, t_y, return_indices=True)
            x_mean, y_mean = dataset[key_x]['mean'][ind_x], dataset[key_y]['mean'][ind_y]
            
            x_count = dataset[key_x].get('count', np.full_like(x_mean, np.inf))[ind_x]
            y_count = dataset[key_y].get('count', np.full_like(y_mean, np.inf))[ind_y]
            
            x_err_raw, y_err_raw = np.asarray(dataset[key_x]['error']), np.asarray(dataset[key_y]['error'])
            x_err = x_err_raw[:, ind_x] if x_err_raw.ndim == 2 else x_err_raw[ind_x]
            y_err = y_err_raw[:, ind_y] if y_err_raw.ndim == 2 else y_err_raw[ind_y]
            
            valid_mask = (~np.isnan(x_mean) & ~np.isnan(y_mean) & 
                          (x_count >= min_count_required) & (y_count >= min_count_required))
            
            if np.sum(valid_mask) < min_points_required: return None
            
            # --- Extract raw matrices for the spaghetti plot ---
            x_raw = dataset[key_x].get('raw_matrix')
            y_raw = dataset[key_y].get('raw_matrix')
            
            if x_raw is not None and y_raw is not None:
                x_raw, y_raw = x_raw[:, ind_x], y_raw[:, ind_y]
                x_raw, y_raw = x_raw[:, valid_mask], y_raw[:, valid_mask]
                
            return {
                't': common_t[valid_mask] * dt * 1e6, 'x': x_mean[valid_mask], 'y': y_mean[valid_mask],
                'x_err': x_err[:, valid_mask] if x_err.ndim == 2 else x_err[valid_mask],
                'y_err': y_err[:, valid_mask] if y_err.ndim == 2 else y_err[valid_mask],
                'x_raw': x_raw, 'y_raw': y_raw
            }

        l_data = extract_and_align_data(l_mode_data)
        h_data = extract_and_align_data(h_mode_data)
        
        if l_data is not None or h_data is not None:
            r_L, r2_L = (np.corrcoef(l_data['x'], l_data['y'])[0, 1], np.corrcoef(l_data['x'], l_data['y'])[0, 1]**2) if l_data and np.std(l_data['x']) > 0 and np.std(l_data['y']) > 0 else (0.0, 0.0)
            r_H, r2_H = (np.corrcoef(h_data['x'], h_data['y'])[0, 1], np.corrcoef(h_data['x'], h_data['y'])[0, 1]**2) if h_data and np.std(h_data['x']) > 0 and np.std(h_data['y']) > 0 else (0.0, 0.0)
            
            if max(r2_L, r2_H) < min_r_squared:
                plots_filtered += 1
            else:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                t_min = min((np.min(d['t']) for d in [l_data, h_data] if d is not None))
                t_max = max((np.max(d['t']) for d in [l_data, h_data] if d is not None))
        
                for ax, d, mode_name, r, r2 in zip(axes, [l_data, h_data], ['L-Mode', 'H-Mode'], [r_L, r_H], [r2_L, r2_H]):
                    if d is None:
                        ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center')
                        ax.set_title(mode_name)
                        continue
                    
                    # ===============================================================
                    # NEW: Spaghetti Plot (Z-Order = 0 so it stays in the background)
                    # ===============================================================
                    if plot_spaghetti and d.get('x_raw') is not None and d.get('y_raw') is not None:
                        n_blobs = d['x_raw'].shape[0]
                        sample_size = min(num_spaghetti, n_blobs)
                        
                        # Use a fixed seed so the spaghetti plots look consistent if you rerun the script
                        np.random.seed(42)
                        sampled_indices = np.random.choice(n_blobs, sample_size, replace=False)
                        
                        for idx in sampled_indices:
                            bx = d['x_raw'][idx, :]
                            by = d['y_raw'][idx, :]
                            
                            # Ensure we don't plot lines connecting across NaN gaps
                            b_valid = ~np.isnan(bx) & ~np.isnan(by)
                            if np.sum(b_valid) > 1:
                                ax.plot(bx[b_valid], by[b_valid], color='gray', alpha=0.15, linewidth=0.8, zorder=0)
                    # ===============================================================

                    # Error bars (Z-Order = 1)
                    ax.errorbar(d['x'], d['y'], xerr=d['x_err'], yerr=d['y_err'], fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                    
                    # Scatter points (Z-Order = 2)
                    sc = ax.scatter(d['x'], d['y'], c=d['t'], cmap='coolwarm', vmin=t_min, vmax=t_max, s=40, zorder=2, edgecolor='k', linewidth=0.5)
                    
                    ax.set_title(f"{mode_name} Evolution", fontsize=12, fontweight='bold')
                    ax.set_xlabel(key_x)
                    
                    if key_x == 'Poloidal angle': ax.set_xlim([0.1,0.5])
                    if key_y == 'Poloidal angle': ax.set_ylim([0.1,0.5])
                    
                    if key_x in angle_diff_keys: ax.set_xlim([-0.2,0.7])
                    if key_y in angle_diff_keys: ax.set_ylim([-0.2,0.7])
                    
                    if key_x in ['Angle ALI','Angle fit']: ax.set_xlim([-0.4,0.1])
                    if key_y in ['Angle ALI','Angle fit']: ax.set_ylim([-0.4,0.1])
                    
                    if key_x in ['Angular velocity angle fit', 'Angular velocity ALI']: ax.set_xlim([-15e3,15e3])
                    if key_y in ['Angular velocity angle fit', 'Angular velocity ALI']: ax.set_ylim([-15e3,15e3])
                    
                    if key_x in ['Normalized flux coordinate velocity']: ax.set_xlim([-1.5e3,1.5e3])
                    if key_y in ['Normalized flux coordinate velocity']: ax.set_ylim([-1.5e3,1.5e3])
                    
                    if key_x in ['Poloidal angular velocity']: ax.set_xlim([-0.5e3,0.5e3])
                    if key_y in ['Poloidal angular velocity']: ax.set_ylim([-0.5e3,0.5e3])
                    
                    if key_x in ['Convexity diff', 'Solidity diff']: ax.set_xlim([-1.5e3,1.5e3])
                    if key_y in ['Convexity diff', 'Solidity diff']: ax.set_ylim([-1.5e3,1.5e3])
                    
                    if key_x in ['Roundness diff']: ax.set_xlim([-3e3,3e3])
                    if key_y in ['Roundness diff']: ax.set_ylim([-3e3,3e3])
                    
                    if key_x in ['Elongation fit diff']: ax.set_xlim([-5e3,5e3])
                    if key_y in ['Elongation fit diff']: ax.set_ylim([-5e3,5e3])
                        
                    if ax == axes[0]: ax.set_ylabel(key_y)
                        
                    stats_text = f"$r = {r:.3f}$\n$R^2 = {r2:.3f}$"
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                    ax.grid(True, linestyle='--', alpha=0.5)
        
                fig.subplots_adjust(right=0.9)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
                cbar = fig.colorbar(sc, cax=cbar_ax)
                cbar.set_label('Time relative to event trigger [$\mu s$]', rotation=270, labelpad=15)
                fig.suptitle(f"{key_y} vs {key_x} (Median & 10th and 90th Percentiles)", fontsize=14)
                
                pdf_page.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                plots_generated += 1

        elapsed_time = time.time() - start_time
        avg_time_per_plot = elapsed_time / (i + 1)
        remaining_time = avg_time_per_plot * (total_pairs - (i + 1))
        hours, rem = divmod(remaining_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'\rPlotting Progress: {i+1}/{total_pairs} | Remaining time: {int(hours)}h {int(minutes):02}min {int(seconds):02}sec', end="", flush=True)

    pdf_page.close()
    print("\n" + "-" * 40)
    print("Plotting Complete.")
    print(f"Generated {plots_generated} comparison plots.")
    print(f"Skipped {plots_filtered} pairs due to R² < {min_r_squared}.")
    print(f"Saved to: {pdf_filename}")