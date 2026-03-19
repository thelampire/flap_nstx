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

from flap_nstx.gpi import analyze_gpi_structures, transform_frames_to_structures, read_analyzed_keys
from flap_nstx.thomson import get_fit_nstx_thomson_profiles

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


def read_all_blob_data(time_range_around_peak=5e-3, #Reads either mean or full blob results. the original read_blob_data procedure reads one shot only
                       nocalc=False,
                       recalc_tracking=False,
                       min_structure_lifetime=20,
                       str_finding_method='watershed',
                       fix_angle_for_correlation=False,
                       read_mean_results=False, #Obsolete
                       averaging='shot',
                       average='avg', #[avg, std, max, no] returns average, returns standard deviation, returns maximum value in the shot (for read_mean_results), or for the blob (average_blob_by_blob)
                       
                       replicate_histogram2=False,
                       replicate_old_read_blob_data=False,
                       
                       read_l_mode_only=False,
                       read_h_mode_only=False,
                       ):
    
    
    """
    Reads all blob data from the databae provided by either read_blob_database or read_blob_lh_mode_database.

    Parameters
    ----------
    time_range_around_peak : TYPE, optional
        DESCRIPTION. The default is 5e-3.
    nocalc : TYPE, optional
        DESCRIPTION. The default is False.
    recalc_tracking : TYPE, optional
        DESCRIPTION. The default is False.
    min_structure_lifetime : TYPE, optional
        DESCRIPTION. The default is 20.
    str_finding_method : TYPE, optional
        DESCRIPTION. The default is 'watershed'.
    fix_angle_for_correlation : TYPE, optional
        DESCRIPTION. The default is False.
    read_mean_results : TYPE, optional
        DESCRIPTION. The default is False.
    averaging : TYPE, optional
        DESCRIPTION. The default is 'shot'.
    average : TYPE, optional
        DESCRIPTION. The default is 'avg'.
    read_l_mode_only : TYPE, optional
        DESCRIPTION. The default is False.
    read_h_mode_only : TYPE, optional
        DESCRIPTION. The default is False.
    
    replicate_histogram2 : boolean, optional
        legacy code for backwards compatibility. If one produces finite difference
        data, it will have less data points than the data itself. the _histogram2
        method solved this by leaving the first data point out essentially shifting
        the data back by one sampling time, while read_blob_data handled it by
        adding another diff data point to the end of the data. Both are wrong but
        has to be here for replicability and legacy reasons.
        
        The correct method is to use either symmetric difference and drop the
        data points of the data at the beginning and the end. Or one can use
        "pre" setting and drop only the first data point.

        The default is False
    replicate_old_read_blob_data: boolean, optional
        Added a diff data points to the end of each data creating nearest neighbor
        "interpolation". Retained for legacy resons.
        The default is False.
     : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    full_blob_db_data : TYPE
        DESCRIPTION.
        
    TODO: read_blob_lh_mode_database was modified. This code has to be modified
    to accommodate the new database that can have time ranges in it instead of
    a single time point.

    """
    
    if read_mean_results:
        averaging='shot'
        
    if averaging == 'shot':
        pickle_filename=wd+'/processed_data/blob_database_shot_by_shot_blob_'+str_finding_method+'_'+average+'.pickle'
    elif averaging == 'blob':
        pickle_filename=wd+'/processed_data/blob_database_shot_by_shot_blob_'+str_finding_method+'_'+average+'_blob_by_blob_avg_data.pickle'
    elif averaging == 'no':
        pickle_filename=wd+'/processed_data/blob_database_shot_by_shot_blob_'+str_finding_method+'_full_data.pickle'
    else:
        raise ValueError('Averaging needs to be either shot, blob or no')
    
    difference_method='pre',
    
    if replicate_histogram2:
        averaging='no'
        
            
    if read_l_mode_only:
        blob_database=read_blob_lh_mode_database_file(l_mode=True,
                                                      time_range_around_peak=time_range_around_peak)
    elif read_h_mode_only:
        blob_database=read_blob_lh_mode_database_file(h_mode=True,
                                                      time_range_around_peak=time_range_around_peak)
    else:
        blob_database=read_blob_database_file(time_range_around_peak=time_range_around_peak)
        
    analyzed_keys=read_analyzed_keys()
    
    diff_keys=['Velocity radial COG', 'Velocity poloidal COG',
               'Velocity radial centroid', 'Velocity poloidal centroid',
               'Velocity radial position', 'Velocity poloidal position',
               'Expansion fraction area', 'Expansion fraction axes',
               'Angular velocity angle', 'Angular velocity ALI'] #These are calculated at the time when the blob results are calculated
        
    additional_diff_keys=['Convexity', 'Solidity', 'Roundness', 'Total curvature',
                          'Total bending energy','Area','Elongation', 
                          'Size radial', 'Size poloidal'] # These are calculated in this method.

    ncalc=len(blob_database['shot'])

    full_blob_db_data={}

    for key in analyzed_keys:
        full_blob_db_data[key]=[]
    for key in additional_diff_keys:
        full_blob_db_data[key+' diff']=[]
    
    full_blob_db_error = copy.deepcopy(full_blob_db_data)

    curr_blob_data_ref = copy.deepcopy(full_blob_db_data)
    curr_blob_error_ref = copy.deepcopy(full_blob_db_data)

    if not os.path.exists(pickle_filename) or not nocalc or read_l_mode_only or read_h_mode_only:
        n_str = 0
        for ind in range(ncalc):
            start_time=time_mod.time()
            curr_shot_data=copy.deepcopy(curr_blob_data_ref)
            curr_shot_error=copy.deepcopy(curr_blob_error_ref)
            blob_time=blob_database['time'][ind]
            shot=blob_database['shot'][ind]

            blob_results = read_blob_data(shot,
                                          [blob_time - time_range_around_peak,
                                           blob_time + time_range_around_peak],
                                          nocalc = True,
                                          recalc_tracking = recalc_tracking,
                                          min_structure_lifetime = min_structure_lifetime,
                                          str_finding_method = str_finding_method,
                                          )
            
            flap.delete_data_object('*')
            str_by_str = transform_frames_to_structures(blob_results)
            
            for structure in str_by_str: 
                for key in analyzed_keys + additional_diff_keys:
                    # if key != 'Angle of least inertia':
                    if key in analyzed_keys:
                        if replicate_old_read_blob_data:
                            if key in diff_keys:
                                str_key_data=np.append(structure[key], structure[key][-1])  #this is to match the same number of elements in the differential data
                            else:
                                str_key_data=structure[key]
                        elif replicate_histogram2:
                            if key in diff_keys:
                                str_key_data=structure[key]
                            else:
                                str_key_data=structure[key][1:]
                        else:   #New scheme
                            pass
                        key_string_add=''
                        
                    elif key in additional_diff_keys:
                        str_key_data=[]
                        if difference_method == 'post':
                            first_slice=slice(1,len(structure[key]))
                            second_slice=slice(0,len(structure[key])-1)
                            diff_multiplier=1
                            
                        elif difference_method == 'symmetric':                      #Implemented but unused. Would need to change the way the other diff keys are calculated.
                            first_slice=slice(2,len(structure[key]))
                            second_slice=slice(0,len(structure[key])-2)
                            diff_multiplier=0.5
                        
                        str_key_data=np.real(structure[key][first_slice] - 
                                             structure[key][second_slice]) * diff_multiplier
                        if replicate_histogram2:
                            pass
                        elif replicate_old_read_blob_data:
                            str_key_data=np.append(str_key_data,str_key_data[-1])
                        else:   #New scheme
                            pass
                        
                        key_string_add=' diff'
                        
                    """
                    The idea with averaging is that if there is no averaging, just append the data to the previous.
                    If there is either shot or blob averaging, then append the blob average first no matter which
                    one is chosen. If shot average is chosen, then it's calculated as shot_averaged blob_averaged data
                    and not shot_averaged no_average data. They are different, but this way the result is not biased
                    towards long duraction structures but to the number of blobs which is more relevant.
                    """
                    
                    if averaging == 'no':
                        curr_shot_data[key + key_string_add]=np.append(curr_shot_data[key + key_string_add],
                                                                       str_key_data) 
                    elif averaging == 'shot' or averaging == 'blob':
                        str_key_data=str_key_data[~np.isnan(str_key_data)]
                        if averaging == 'shot':
                            curr_shot_error[key + key_string_add]=np.append(curr_shot_error[key + key_string_add],
                                                                            np.sqrt(np.var(str_key_data)))
                        
                        if average == 'avg':
                            curr_shot_data[key + key_string_add]=np.append(curr_shot_data[key + key_string_add],
                                                                           np.mean(str_key_data))
                        elif average == 'std':
                            curr_shot_data[key + key_string_add]=np.append(curr_shot_data[key + key_string_add],
                                                                           np.sqrt(np.var(str_key_data)))
                        elif average == 'max':
                            curr_shot_data[key + key_string_add]=np.append(curr_shot_data[key + key_string_add],
                                                                           np.max(str_key_data))
                                   
            for key in full_blob_db_data.keys():
                if averaging == 'shot':
                    full_blob_db_data[key]=np.append(full_blob_db_data[key],
                                                  np.mean(curr_shot_data[key]))
    
                    full_blob_db_error[key]=np.append(full_blob_db_error[key],
                                                      np.mean(curr_shot_error[key]) /
                                                      np.sqrt(len(curr_shot_error[key])))
                else:
                    full_blob_db_data[key]=np.append(full_blob_db_data[key],
                                                     {'shot':shot,
                                                      'data':curr_shot_data[key]})
                    
            remaining_time=(time_mod.time()-start_time)*(ncalc-ind-1)

            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            seconds = int(remaining_time % 60)

            print('\rRemaining time from the calculation: '+f"{hours}h {minutes:02}min {seconds:02}sec", end="", flush=True)
        print(f'Total number of structures: {n_str}')
        
        if not read_h_mode_only and not read_l_mode_only:
            pickle.dump(full_blob_db_data,open(pickle_filename,'wb'))
    else:
        full_blob_db_data=pickle.load(open(pickle_filename,'rb'))

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
                   ):
    """
    Wrapper function for running analyze_gpi_structures with only the essential input
    parameters.

    Parameters
    ----------
    shot : float/int
        shot number.
    time_range : list
        time range of the calculation.
    calculate_only : boolean, optional
        Calculation only, no plotting. The default is False.
    nocalc : TYPE, optional
        Switch for not recalculating already existing results. The default is True.
    min_structure_lifetime : TYPE, optional
        Minimum structures lifetime in frame number. Over this value, the identified structures will be dropped
        The default is 20 (50us).
    recalc_tracking : boolean, optional
        Recalculate tracking but not the segmentation. The default is False.
    str_finding_method : string, optional
        Method for structure tracking. Either watershed or contour. The default is 'watershed'.
    max_gap : integer, optional
        The maximum frame number to go back during tracking calculation to fix segmentation errors. The default is 1.
    verbose : TYPE, optional
        Suppresses messages. Useful for long calculations. The default is False.


    Returns
    -------
    blob_results : dictionary
        Same as from analyze_gpi_structures having {data,time,structures}.

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
                                              fix_structure_angles=True,
                                              
                                              test_structures=False,
                                              return_results=not calculate_only,
                                              
                                              plot=False,
                                              plot_str_by_str=True,
                                              plot_scatter=True,
                                              plot_tracking=True,
                                              calculate_rough_diff_velocities=False,
                                              plot_for_publication=True,
                                              pdf=False,
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

#TODO:rename 
def calculate_blob_parameter_histograms2(time_range_around_peak=5e-3,
                                        calc_mean_distribution=False,
                                        nocalc=True,
                                        recalc_tracking=False,
                                        min_structure_lifetime=20,
                                        str_finding_method='watershed',
                                        analyze_h_mode_only=False,
                                        analyze_l_mode_only=False,
                                        ):
    """
    Just for reading the data, should be merged with read data and the indices
    of differential and normal data would need to be handled properly.
    """

    if calc_mean_distribution:
        pickle_filename=wd+'/processed_data/blob_database_full_data_mean_'+str_finding_method+'.pickle'
    else:
        pickle_filename=wd+'/processed_data/blob_database_full_data_nomean_'+str_finding_method+'.pickle'

    if not analyze_h_mode_only and not analyze_l_mode_only:
        blob_database=read_blob_database_file(time_range_around_peak=time_range_around_peak)
    elif analyze_h_mode_only:
        blob_database=read_blob_lh_mode_database_file(h_mode=True,
                                                 time_range_around_peak=time_range_around_peak)
    elif analyze_l_mode_only:
        blob_database=read_blob_lh_mode_database_file(l_mode=True,
                                                      time_range_around_peak=time_range_around_peak)
        
    analyzed_keys=read_analyzed_keys()
    
    additional_diff_keys=['Convexity', 'Solidity', 'Roundness', 'Total curvature',
                          'Total bending energy','Area','Elongation']

    ncalc=len(blob_database['shot'])

    full_data={}

    for key in analyzed_keys:
        full_data[key]=[]
        
    for key in additional_diff_keys:
        full_data[key+' diff']=[]

    if not os.path.exists(pickle_filename) or not nocalc or analyze_l_mode_only or analyze_h_mode_only:
        n_str=0
        for ind in range(ncalc):
            blob_time=blob_database['time'][ind]
            start_time=time_mod.time()
            blob_results=read_blob_data(blob_database['shot'][ind],
                                        [blob_time-time_range_around_peak,
                                         blob_time+time_range_around_peak],
                                        nocalc=True,
                                        recalc_tracking=recalc_tracking,
                                        min_structure_lifetime=min_structure_lifetime,
                                        str_finding_method=str_finding_method,
                                        )

            flap.delete_data_object('*')
            str_by_str=transform_frames_to_structures(blob_results)

            for ind_str, structure in enumerate(str_by_str):
                n_str+=1
                for key in analyzed_keys:
                    if key in          ['Velocity radial COG', 'Velocity poloidal COG',
                                       'Velocity radial centroid', 'Velocity poloidal centroid',
                                       'Velocity radial position', 'Velocity poloidal position',
                                       'Expansion fraction area', 'Expansion fraction axes',
                                       'Angular velocity angle', 'Angular velocity ALI']:
                        full_data[key]=np.append(full_data[key],
                                                 structure[key])
                    else:
                        full_data[key]=np.append(full_data[key],
                                                 structure[key][1:])

                for key in additional_diff_keys:
                    diff=(np.asarray(structure[key])[1:] - np.asarray(structure[key])[0:-1])
                    full_data[key+' diff']=np.append(full_data[key+' diff'],diff)
            remaining_time=(time_mod.time()-start_time)*(ncalc-ind-1)

            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            seconds = int(remaining_time % 60)

            print('\rRemaining time from the calculation: '+f"{hours}h {minutes:02}min {seconds:02}sec", end="", flush=True)
        print(f'Total number of structures: {n_str}')
        
        if not analyze_h_mode_only and not analyze_l_mode_only:
            pickle.dump(full_data,open(pickle_filename,'wb'))
    else:
        full_data=pickle.load(open(pickle_filename,'rb'))

    return full_data

def read_blob_database_file(time_range_around_peak=5e-3,
                            blob_db_file='/Users/mlampert/work/NSTX_workspace/db/2010.csv',
                            elm_db_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv',
                            nofilter=False
                            ):
    """
    Reads the blob database from db/2010.csv Same shots as in 2010_all.csv with
    a different quality assessment (from S. Zweben). Filteres the database
    with the nofilter=False switch based on the ELM database by M. Lampert in
    db/ELM_findings_mlampert_velocity_good.csv'

    Parameters
    ----------
    time_range_around_peak : float, optional
        Finds elms in the time range around the peak and the database is filtered based on this range. The default is 5e-3.
    blob_db_file : string, optional
        Database of the blobs. The default is '/Users/mlampert/work/NSTX_workspace/db/2010.csv'.
    elm_db_file : string, optional
        Database of the ELMs. The default is '/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'.
    nofilter : boolean, optional
        Filters the database based on ELMs in elm_db_files. The default is False.

    Returns
    -------
    blob_database : dictionary
        {'shot':shot list,
         'time':time list}.

    """

    database=np.asarray(pandas.read_csv(blob_db_file))
    ind_shots=np.where(database[:,2]==0)
    blob_shots=database[ind_shots,0][0,:]
    peak_times=database[ind_shots,1][0,:]/1000.
    blob_database={'shot':blob_shots,
                   'time':peak_times}

    db=pandas.read_csv(elm_db_file, index_col=0)
    elm_shots=np.asarray(db)[:,1]
    elm_times=np.asarray(db)[:,3]
    _elm_database={'shot':elm_shots,
                        'time':elm_times}
    if not nofilter:
        for ind_blob,shot_blob in enumerate(blob_database['shot']):
            ind_overlap=np.where(_elm_database['shot'] == shot_blob)[0]
            blob_time=blob_database['time'][ind_blob]
            if len(ind_overlap) > 0:
                elm_times=_elm_database['time'][ind_overlap]
                min_time=np.min(elm_times)
                max_time=np.max(elm_times)
                if np.logical_and(blob_time < max_time,
                                  blob_time > min_time):
                    if abs(blob_time-min_time) < abs(blob_time-max_time):
                        blob_database['time'][ind_blob] = min_time - 2*time_range_around_peak
                    else:
                        blob_database['time'][ind_blob] = max_time + 2*time_range_around_peak
                if (blob_database['time'][ind_blob] > blob_time+50e3 or
                    blob_database['time'][ind_blob] < blob_time-50e3):

                    blob_database['time'].pop(ind_blob)
                    blob_database['shot'].pop(ind_blob)

        ind=np.where(blob_database['shot'] > 138127)
        blob_database['shot']=blob_database['shot'][ind]
        blob_database['time']=blob_database['time'][ind]

    return blob_database


def read_blob_lh_mode_database_file(l_mode=False,
                                    h_mode=False,
                                    filtered_blob_db=True, #filter the database to shots read by read_blob_database
                                    time_range_around_peak=None,
                                    filter_lh_transition=False,
                                    filter_elms=False,
                                    ):
    """
    Reads either the l-mode or the h-mode database from the db/2010_all.csv directory.

    Parameters
    ----------
    l_mode : boolean, optional
        Reads the L-mode database. The default is False.
    h_mode : boolean, optional
        Reads the H-mode database. The default is False.
    filtered_blob_db : boolean, optional
        Reads a filtered database based on the ELM database. It removes a lot of shots,
        but then no post ELM processing is needed. The default is True.
    time_range_around_peak : float or list, optional
        if float:. The default is None.
    filter_lh_transition : boolean, optional
        filter the database for lh transitions in the database. The default is False.
    filter_elms : boolean, optional
        filter the database for elms in the database file. The default is False.

    Raises
    ------
    ValueError
        Raised of neither l_mode or h_mode are set.

    Returns
    -------
    database : dictionary
        {shot: list of shots
         time: either list of times (for time_range_around_peak=float) or list of time ranges (for time_range_around_peak=[time_before, time_after])}

    """
    
    all_db_file='/Users/mlampert/work/NSTX_workspace/db/2010_all.csv'
    
    if l_mode:
        find_string='L-mode'
    elif h_mode:
        find_string='H-mode'
    else:
        raise ValueError('Either l_mode or h_mode needs to be set.')
    
    all_database=pandas.read_csv(all_db_file)
    all_database['peak GPI time (ms)'] /= 1e3
    lh_mode_shot_inds=[ind for ind,item in enumerate(list(all_database['comments by Ricky'])) if find_string in item]
    lh_mode_shots=all_database['shot'][np.asarray(lh_mode_shot_inds)]
    
    if filtered_blob_db:
        if time_range_around_peak is not None:
            blob_shots=read_blob_database_file(time_range_around_peak=time_range_around_peak)['shot']
            blob_times=read_blob_database_file(time_range_around_peak=time_range_around_peak)['time']
        else:
            blob_shots=read_blob_database_file()['shot']
            blob_times=read_blob_database_file()['time']
        lh_mode_shots_in_blob_db=([int(shot) for shot in blob_shots if shot in np.asarray(lh_mode_shots)])
        lh_mode_times_in_blob_db=np.asarray([time for time,shot in zip(blob_times, blob_shots) if shot in np.asarray(lh_mode_shots)])
        database={'shot':lh_mode_shots_in_blob_db,
                  'time':lh_mode_times_in_blob_db}
    else:
        for ind,movie_rating in enumerate(all_database['movie rating'][np.asarray(lh_mode_shot_inds)]):
            if movie_rating not in ['A', 'A+']:
                lh_mode_shot_inds.pop(ind)
        lh_mode_shot_inds=np.asarray(lh_mode_shot_inds)

        if type(time_range_around_peak) in [int,float]:
            time_range_around_peak=[time_range_around_peak,
                                    time_range_around_peak]

        if time_range_around_peak is not None:
                    
            database={'shot':np.asarray(all_database['shot'][lh_mode_shot_inds]),
                      'time':np.asarray([all_database['peak GPI time (ms)'][lh_mode_shot_inds]-time_range_around_peak[0],
                                         all_database['peak GPI time (ms)'][lh_mode_shot_inds]+time_range_around_peak[1]])}
        
            if filter_lh_transition:
                lh_mode_shot_inds=list(lh_mode_shot_inds)
                for ind_index, ind in enumerate(lh_mode_shot_inds):
                    if (not np.isnan(all_database['L-H time'][ind]) or
                        not np.isnan(all_database['H-L time'][ind])):
                            lh_mode_shot_inds.pop(ind_index)
                lh_mode_shot_inds=np.asarray(lh_mode_shot_inds)
                
            if filter_elms:
                for ind_index, ind in enumerate(lh_mode_shot_inds):
                    if (all_database['ELM time'][ind] > all_database['peak GPI time (ms)'][ind]-[time_range_around_peak[0]] and 
                        all_database['ELM time'][ind] < all_database['peak GPI time (ms)'][ind]+time_range_around_peak[1]):
                        lh_mode_shot_inds.pop(ind_index)
                        # if (np.abs(all_database['ELM time'][ind]-(all_database['peak signal'][ind])-time_range_around_peak)<
                        #     (all_database['peak signal'][ind])+time_range_around_peak-np.abs(all_database['ELM time'][ind])):
                        #     database['time'][0,ind_index]=all_database['ELM time'][ind]
                        # else:
                        #     database['time'][1,ind_index]=all_database['ELM time'][ind]
        else:   
            database={'shot':np.asarray(all_database['shot'][lh_mode_shot_inds]),
                      'time':np.asarray(all_database['peak GPI time (ms)'][lh_mode_shot_inds])}
        
    return database



def read_all_plasma_data(time_range_around_peak=5e-3,
                         nocalc=False,
                         read_l_mode_only=False,
                         read_h_mode_only=False,
                         calculate_parameters_in_sol=False,
                         ):
    """
    Reads all plasma data from either read_blob_lh_mode_database or read_blob_database
    shots.

    Parameters
    ----------
    time_range_around_peak : float, optional
        Filtering of the database for ELMs is based on this time range around the peak. The default is 5e-3.
    nocalc : boolean, optional
        If set, plasma parameters are read from the already saved dadtabase. The default is False.
    read_l_mode_only : boolean, optional
        Read only l-mode plasma data. The default is False.
    read_h_mode_only : boolean, optional
        Read only h-mode plasma data. The default is False.
    calculate_parameters_in_sol : boolean, optional
        Calculate parameters in the scrape-off layer. The default is False.

    Returns
    -------
    full_plasma_data : list
        List of dictionaries from read_plasma_parameters().

    TODO: read_blob_lh_mode_database was modified. This code has to be modified
    to accommodate the new database that can have time ranges in it instead of
    a single time point.

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
                                                 time_range_around_peak=time_range_around_peak)
    elif read_h_mode_only:
        blob_database=read_blob_lh_mode_database_file(h_mode=True,
                                                 time_range_around_peak=time_range_around_peak)
    else:
        blob_database=read_blob_database_file(time_range_around_peak=time_range_around_peak,)

    if (not os.path.exists(pickle_filename) or not nocalc):
        # or read_l_mode_only or read_h_mode_only or calculate_parameters_in_sol):
        
    # if True:
        ncalc=len(blob_database['shot'])

        curr_plasma_data=read_plasma_parameters(exp_id=blob_database['shot'][0],
                                                time=blob_database['time'][0],
                                                calculate_parameters_in_sol=calculate_parameters_in_sol)
        full_plasma_data={}
        for key in curr_plasma_data:
            full_plasma_data[key]=[]

        for ind in range(ncalc):
            blob_time=blob_database['time'][ind]
            shot=blob_database['shot'][ind]

            curr_plasma_data=read_plasma_parameters(exp_id=shot,
                                                    time=blob_time,
                                                    calculate_parameters_in_sol=calculate_parameters_in_sol)
            for key in curr_plasma_data.keys():
                full_plasma_data[key]=np.append(full_plasma_data[key],
                                                curr_plasma_data[key])
                
        
        pickle.dump(full_plasma_data,open(pickle_filename,'wb'))
    else:
        full_plasma_data=pickle.load(open(pickle_filename,'rb'))
        
    return full_plasma_data


def read_plasma_parameters_for_table_in_paper(database=None,
                                              print_ranges=False):
    """
    Reads the plasma parameter ranges for an overview table in the paper. Uses
    the code 
    
    read_plasma parameters()
    
    to read the parameters for each shot in the database read by read_blob_database()
    or the database provided in the input parameters.

    Parameters
    ----------
    database : TYPE, optional
        Shot database input. The default is None.
    print_ranges : TYPE, optional
        Print the calculated ranges for the plasma parameters. The default is False.

    Returns
    -------
    collisionality : list
        A list of all the average collisionalities for shots in the database.
    q95 : list
        A list of all the average q95 for shots in the database.
    greenwald : list
        A list of all the average greenwald fractions for shots in the database.
    current : list
        A list of all the average currents for shots in the database.
    btoroidal : list
        A list of all the average btoroidal for shots in the database.
    density : list
        A list of all the average density for shots in the database.

    """
    if database is None:
        database=read_blob_database_file()

    density=[]
    current=[]
    btoroidal=[]
    greenwald=[]
    collisionality=[]
    q95=[]
    pdf_pages_density=PdfPages(wd+'/plots/blob_database_density_fits.pdf')
    pdf_pages_temperature=PdfPages(wd+'/plots/blob_database_temperature_fits.pdf')

    for ind_shot in range(len(database['shot'])):
        print(f"\r{ind_shot/len(database['shot'])*100} % done from the calculation.", flush=True, end='')
        time_curr=database['time'][ind_shot]
        shot=database['shot'][ind_shot]

        start_time=time_mod.time()

        plasma_parameters=read_plasma_parameters(exp_id=shot,time=time_curr,
                                                 pdf_pages_density=pdf_pages_density,
                                                 pdf_pages_temperature=pdf_pages_temperature)

        greenwald.append(plasma_parameters['Greenwald fraction'])
        density.append(plasma_parameters['Line integrated density'])
        q95.append(plasma_parameters['q95'])
        current.append(plasma_parameters['Current'])

        btoroidal.append(plasma_parameters['Toroidal field'])
        collisionality.append(plasma_parameters['Collisionality'])
        print(str((ind_shot+1)/len(database['shot'])*100.)+'% done')
        print('Finished in: ',time_mod.time()-start_time,'s')

    if print_ranges:
        print('Collisionality range: ',min(collisionality),max(collisionality))
        print('Density range: ',min(density), max(density))
        print('Greenwald range: ',min(greenwald),max(greenwald))
        print('BT range: ',min(btoroidal),max(btoroidal))
        print('current range: ',min(current),max(current))

    pdf_pages_density.close()
    pdf_pages_temperature.close()

    return collisionality, q95, greenwald, current, btoroidal, density



def read_plasma_parameters(exp_id=None,
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
                                      name='\EFIT02::\IPMEAS',
                                      exp_id=exp_id,
                                      ).slice_data(slicing={'Time':time}).data)
    except Exception as e:
        print(e)
        print('Failed to read current for shot ',exp_id)
        current=np.nan

    """Toroidal field"""
    try:
        b_toroidal=np.mean(flap.get_data('NSTX_MDSPlus',
                                      name='\EFIT02::\BT0',
                                      exp_id=exp_id,
                                      ).slice_data(slicing={'Time':time}).data)
    except Exception as e:
        print(e)
        print('Failed to read Bt for shot ',exp_id)
        b_toroidal=np.nan

    """Minor radius"""
    try:
        minor_radius=flap.get_data('NSTX_MDSPlus',
                              name='\EFIT02::\AMINOR',
                              exp_id=exp_id,
                              ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read current for shot ',exp_id)
        minor_radius=np.nan

    """Pedestal radius"""
    try:
        R_separatrix=flap.get_data('NSTX_MDSPlus',
                                   name='\EFIT02::\RMIDOUT',
                                   exp_id=exp_id,
                                   ).slice_data(slicing={'Time':time}).data-0.02
    except Exception as e:
        print(e)
        print('Failed to read RMIDOUT for shot ',exp_id)
        R_separatrix=np.nan

    """Safety factor"""
    try:
        q95=flap.get_data('NSTX_MDSPlus',
                         name='\EFIT02::\Q95',
                         exp_id=exp_id,
                         ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read Q95 for shot ',exp_id)
        q95=np.nan

    """Lower triangularity"""
    try:
        lower_triang=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\TRIBOT',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read TRIBOT for shot ',exp_id)
        lower_triang=np.nan

    """Upper triangularity"""
    try:
        upper_triang=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\TRITOP',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read TRITOP for shot ',exp_id)
        upper_triang=np.nan

    """Elongation"""
    try:
        elongation=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\KAPPA',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read KAPPA for shot ',exp_id)
        elongation=np.nan

    """Inner gap"""
    try:
        inner_gap=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\GAPIN',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read GAPIN for shot ',exp_id)
        inner_gap=np.nan

    """Outer gap"""
    try:
        outer_gap=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\GAPOUT',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read GAPOUT for shot ',exp_id)
        outer_gap=np.nan

    """Current density at psi_norm=0.95"""
    try:
        cdens_95=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\J95N',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read J95N for shot ',exp_id)
        cdens_95=np.nan

    """Current density at psi_norm=0.99"""
    try:
        cdens_99=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\J99N',
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
                             name='\EFIT02::\BTZ0',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).slice_data(slicing={'Device R':R_pressure_max_grad}).data
    except Exception as e:
        print(e)
        print('Failed to read B_tor for shot ',exp_id)
        B_tor=np.nan
        
    try:
        B_pol=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\BZZ0',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).slice_data(slicing={'Device R':R_pressure_max_grad}).data
    except Exception as e:
        print(e)
        print('Failed to read poloidal field for shot ',exp_id)
        B_pol=np.nan
        
    try:    
        B_rad=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\BRZ0',
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
                             name='\EFIT02::\QL',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read QL for shot ',exp_id)
        q_boundary=np.nan
    
    """Radial position of the magnetic axis"""
    
    try:
        R_mag_axis=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\RMAXIS',
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
                             name='\EFIT02::\BTZ0',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).slice_data(slicing={'Device R':R_separatrix}).data
        
        B_pol=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\BZZ0',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).slice_data(slicing={'Device R':R_separatrix}).data
        
        z_outer_strike_point=flap.get_data('NSTX_MDSPlus',
                                           name='\EFIT02::\ZVSOUT',
                                           exp_id=exp_id,
                                           ).slice_data(slicing={'Time':time}).data
        
        R_outer_strike_point=flap.get_data('NSTX_MDSPlus',
                                           name='\EFIT02::\RVSOUT',
                                           exp_id=exp_id,
                                           ).slice_data(slicing={'Time':time}).data
        
        R_lower_x_point=flap.get_data('NSTX_MDSPlus',
                                      name='\EFIT02::\RXPT1',
                                      exp_id=exp_id,
                                      ).slice_data(slicing={'Time':time}).data
        
        z_lower_x_point=flap.get_data('NSTX_MDSPlus',
                                      name='\EFIT02::\ZXPT1',
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

def return_interesting(with_plasma_frequency=False):
    """
    Returns the interesting key paramaters for plotting data for publications.

    Parameters
    ----------
    with_plasma_frequency : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    interesting_key_pairs : array
        Returns an array with interesting key pairs for plotting. One can set whether the key pairs with plasma frequency are returned.
    units :                 dictionary
        Returns the axis labels for the interesting key pairs as [label, unit, multiplier for data to get it into "unit" units].

    """
    
    if not with_plasma_frequency:
        interesting_key_pairs=np.asarray([['Axes length minor','Line integrated density'], #
                                          ['Angle of least inertia','Line integrated density'],
                                          ['Angle of least inertia','Sound speed'],
                                          #['Angle of least inertia','Plasma frequency'],
                                          ['Angle of least inertia','Pressure at max'],
                                          ['Velocity poloidal centroid','Temperature pedestal width'],
                                          ['Velocity poloidal centroid','Collisionality'],
                                          #['Velocity poloidal centroid','Plasma frequency'],
                                          ['Velocity poloidal centroid','Density at max'],
                                          
                                          ['Angular velocity ALI','Line integrated density'],
                                          ['Angular velocity ALI','Collisionality'],
                                          
                                          #['Angular velocity ALI','Plasma frequency'],
                                          ])
    else:
        interesting_key_pairs=np.asarray([['Axes length minor','Line integrated density'], #
                                          ['Angle of least inertia','Line integrated density'],
                                          ['Angle of least inertia','Sound speed'],
                                          ['Angle of least inertia','Plasma frequency'],
                                          ['Angle of least inertia','Pressure at max'],
                                          ['Velocity poloidal centroid','Temperature pedestal width'],
                                          ['Velocity poloidal centroid','Collisionality'],
                                          ['Velocity poloidal centroid','Plasma frequency'],
                                          ['Velocity poloidal centroid','Density at max'],
                                          
                                          ['Angular velocity ALI','Line integrated density'],
                                          ['Angular velocity ALI','Collisionality'],
                                          
                                          ['Angular velocity ALI','Plasma frequency'],
                                          ])
    
    #Name, unit, multiplier
    units={'Axes length minor':['$b_{ellipse}$','mm', 1e3],
           'Angle of least inertia':['$\\theta_{blob}$','rad', 1],
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
    ind_shots=np.where(database[:,2]==0)
    blob_shots=database[ind_shots,0][0,:]
    peak_times=database[ind_shots,1][0,:]/1000.
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
