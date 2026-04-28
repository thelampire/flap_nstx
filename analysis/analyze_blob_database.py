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
from flap_nstx.tools import correlation, mutual_information

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


def calculate_all_blob_results(time_range_around_peak=[-5e-3,15e-3],
                               str_finding_method='watershed',
                               plot=False,
                               pdf=False,
                               nocalc=False,
                               recalc_tracking=False,
                               test=False,
                               calculate_for_lh_study=False,
                               download_data_only=False,
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
            if calculate_for_lh_study and download_data_only:
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
            avg_time_per_shot = total_elapsed_time / shots_processed
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
            
            if analyze_h_mode_only or analyze_l_mode_only:
                _time_range = list(blob_database['time'][ind])
            else:
                blob_time = blob_database['time'][ind]
                _time_range = [blob_time - time_range_around_peak[0], 
                               blob_time + time_range_around_peak[1]]
            try:
                # Assumes read_blob_data returns the new StructureDataset object
                blob_results = read_blob_data(int(blob_database['shot'][ind]),
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
                full_data = {key: [] for key in analyzed_keys}
                keys_initialized = True

            # --- OOP DATA EXTRACTION ---
            for structure in blob_results.tracked_structures:
                n_str += 1
                for key in analyzed_keys:
                    
                    # Safely extract the data and dynamically check if it's differential
                    if key in structure.regular_parameters:
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
    return full_data
    # --- 4. Plotting & Export ---
    if plot:
        ranges = {
            'Position radial fit': [1.4, 1.6], 
            'Position poloidal fit': [0.15, 0.35],
            'Velocity radial position fit': [-3e3, 3e3], 
            'Velocity poloidal position fit': [-10e3, 10e3],
            'Expansion fraction axes fit': [0.75, 1.25], 
            'Elongation fit diff': [-0.075, 0.075], 
            'Angular velocity angle fit': [-250e3, 250e3],
            
            'Area': [0, 0.006], 
            'Expansion fraction area': [0.75, 1.25], 
            'Convexity': [0.9, 1.0],
            'Solidity': [0.75, 1.0], 
            'Total curvature': [0.9, 1.0],
            'Total bending energy': [0e8, 1.5e8], 
            
            'Convexity diff': [-0.01, 0.01],
            'Solidity diff': [-0.25, 0.25], 
            'Total curvature diff': [-0.05, 0.05],
            'Total bending energy diff': [-0.3e8, 0.3e8], 
            'Area diff': [-0.0015, 0.0015],
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
                pdf_page = PdfPages(f"{wd}/plots/8hist_blob_db_LT{min_structure_lifetime}_{str_finding_method}{suffix}.pdf")
                fig, axes = plt.subplots(4, 2, figsize=(8.5/2.54, 17/2.54))

                for ind, key in enumerate(target_keys):
                    data = full_data[key][~np.isnan(full_data[key])] * multiplier[key]
                    ax = axes[ind // 2, ind % 2]
                    
                    if key == 'Angle fit':
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

                    if key == 'Angle fit':
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
                    processed_data[key] = np.concatenate([shot_dict['data'] for shot_dict in data_dict[key]])
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

    plot_pearson_matrix(correlation_matrix,
                        xlabels = labels_to_plot,
                        ylabels = labels_to_plot,
                        colormap = colormap,
                        figsize = (17/2.54 / (1 + plot_interesting_only), 
                                   17/2.54 / (1 + plot_interesting_only)),
                        charsize = 15,
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
    def _load_data(l_mode=False, h_mode=False):
        
        mode_str = 'l_mode' if l_mode else 'h_mode' if h_mode else 'full'
        p_file = f"{wd}/processed_data/gpi_gpi_trends_{str_finding_method}_{averaging}{mode_str}.pickle"
        
        if not os.path.exists(p_file):
            data  =  read_all_blob_data(min_structure_lifetime = min_structure_lifetime,
                                        averaging = 'shot' if calc_mean_distribution else 'no',
                                        nocalc = nocalc, recalc_tracking = recalc_tracking,
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
        full_data_l_mode = _load_data(l_mode=True)
        full_data_h_mode = _load_data(h_mode=True)
        full_data = full_data_l_mode # Default reference
    else:
        full_data = _load_data(l_mode=analyze_l_mode_only, h_mode=analyze_h_mode_only)

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

    # --- 2. Predictive Power Score (PPS) Matrix ---
    pps_pickle = f"{wd}/processed_data/blob_database_full_data_mean_pps_{str_finding_method}.pickle"
    if not nocalc or not os.path.exists(pps_pickle):
        df = pandas.DataFrame(flat_data)
        df = df.dropna(thresh=1)
        
        # Keep rows where absolute Z-score is < 3
        df = df[(np.abs(scipy.stats.zscore(df)) < 3).all(axis=1)]
        
        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
        with open(pps_pickle, 'wb') as f:
            pickle.dump(matrix_df, f)
    else:
        with open(pps_pickle, 'rb') as f:
            matrix_df = pickle.load(f)

    ppscore_matrix = np.asarray(matrix_df).T
    xlabels = list(matrix_df.columns)

    # --- 3. Plotting Logic ---
    if not plot_for_publication:
        if pdf: pdf_page = PdfPages(pdf_filename)
        for i, key1 in enumerate(analyzed_keys):
            if key1 not in flat_data: continue
            valid1 = ~np.isnan(flat_data[key1])
            
            for j, key2 in enumerate(analyzed_keys):
                if key2 not in flat_data: continue
                
                if key1 != key2 and j > i:
                    valid_mask = valid1 & ~np.isnan(flat_data[key2])
                    
                    d1 = flat_data[key1][valid_mask]
                    d2 = flat_data[key2][valid_mask]
                    
                    d1_4c = d1 - np.mean(d1)
                    d2_4c = d2 - np.mean(d2)
                    denom = np.sqrt(np.sum(d1_4c**2) * np.sum(d2_4c**2))
                    corr = np.sum(d1_4c * d2_4c) / denom if denom != 0 else 0
                    
                    pps_val = ppscore_matrix[xlabels.index(key1), xlabels.index(key2)] if key1 in xlabels and key2 in xlabels else 0

                    do_plot = False
                    if plot_if_correlation_is_higher_than and abs(corr) > plot_if_correlation_is_higher_than: do_plot = True
                    elif plot_if_pps_is_higher_than and pps_val > plot_if_pps_is_higher_than: do_plot = True
                    elif not plot_if_correlation_is_higher_than and not plot_if_pps_is_higher_than: do_plot = True

                    if do_plot:
                        fig, ax = plt.subplots(figsize=(8.5/2.54, 8.5/2.54))
                        ax.scatter(d1, d2, s=0.5)
                        ax.set_xlabel(key1)
                        ax.set_ylabel(key2)
                        ax.set_title(f"{key1} vs {key2}")
                        if pdf: pdf_page.savefig()
                        plt.close(fig)
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
        'predictive_power': {'str': 'pps', 'title': 'Blob vs plasma parameter predictive power map', 'zrange': [0, 1], 'cmap': 'Blues'}
    }
    
    cfg = metric_map.get(quantity, metric_map['correlation'])
    if not plot_for_publication: 
        colormap = cfg['cmap']

    if pdf_filename is None:
        avg_str = 'full' if averaging == 'no' else f"{averaging}_{average}"
        thres_str = f"_thres_{int(threshold_multiplier)}" if threshold_corr else "_nothres"
        pdf_filename = f"{wd}/plots/{cfg['str']}_matrix_gpi_plasma_{str_finding_method}_{avg_str}{thres_str}.pdf"
        
    if pdf:
        pdf_page = PdfPages(pdf_filename)

    # --- 2. Data Loading & Derived Parameter Calculation ---
    full_plasma_data = read_all_plasma_data(nocalc=nocalc_plasma_data)
    full_blob_data = read_all_blob_data(nocalc=nocalc_blob_data, 
                                        str_finding_method=str_finding_method,
                                        fix_angle_for_correlation=fix_angle_for_correlation,
                                        averaging=averaging, average=average)

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

    corr_accept = calculate_corr_acceptance_levels()

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

    if pdf:
        pdf_page.savefig()
        pdf_page.close()
        matplotlib.use('qt5agg')

def plot_all_cross_data_matrix():
    
    averaging=['no','blob','shot']
    quantity=['correlation','mutual_information', 'predictive_power']
    method=['watershed', 'contour']
    
    for avg in averaging:
        for qty in quantity:
            for mthd in method:
                calculate_blob_plasma_parameter_correlation_matrix(nocalc=False, 
                                                                   averaging=avg, 
                                                                   average='avg', 
                                                                   str_finding_method=mthd, 
                                                                   quantity=qty, 
                                                                   plot_for_publication=True, 
                                                                   plot_full=True, 
                                                                   threshold_corr=False, 
                                                                   linewidth=1, 
                                                                   ticksize=3, 
                                                                   charsize=9)
    

def plot_blob_plasma_parameter_trends(pdf_filename=None,
                                      nocalc=True,
                                      threshold_corr=False,
                                      threshold_multiplier=2,
                                      plot_for_publication=False,
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
            blob_data = read_all_blob_data(nocalc=nocalc, str_finding_method='watershed', 
                                           fix_angle_for_correlation=True, averaging='shot', average='avg',
                                           read_l_mode_only=l_mode, read_h_mode_only=h_mode)
            with open(p_blob, 'wb') as f: pickle.dump(blob_data, f)
        else:
            with open(p_blob, 'rb') as f: blob_data = pickle.load(f)
            
        # Calculate derived metrics
        scale_length = (plasma_data['Larmor radius sound']**0.8 * plasma_data['Connection length']**0.4 /
                        plasma_data['Pedestal radius']**0.2)
        plasma_data['Blob size dimensionless'] = (np.sqrt(blob_data['Area'] / np.pi) / scale_length)**2.5
        
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

    # --- 2. Load Data ---
    full_plasma_data = read_all_plasma_data(nocalc=nocalc, calculate_parameters_in_sol=True)
    full_blob_data = read_all_blob_data(nocalc=nocalc, str_finding_method='watershed',
                                        fix_angle_for_correlation=True, averaging='shot', average='avg')

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

    plt.tight_layout(pad=0.1)
    pdf_page.savefig(fig)
    pdf_page.close()
    plt.close(fig)
    
    matplotlib.use('qt5agg')

    
def plot_well_known_parameter_dependences(pdf_filename=None,
                                          nocalc=True,
                                          save_data_for_publication=False,
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
    
    # --- 4. Plotting Configuration ---
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

    plt.tight_layout(pad=0.1)
    pdf_page.savefig(fig)
    pdf_page.close()
    plt.close(fig)
    
    matplotlib.use('qt5agg')