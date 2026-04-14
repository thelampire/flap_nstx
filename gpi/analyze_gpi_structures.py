#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:11:18 2022

@author: mlampert

@Derived from calculate_frame_by_frame_velocity.py
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

from flap_nstx.gpi import (normalize_gpi, 
                           identify_structures, track_structures, _plot_ellipses_centers)
from flap_nstx.tools import detrend_multidim, set_matplotlib_for_publication
from flap_nstx.tools import StructureDataset

import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

import numpy as np
import pickle
#Plot settings for publications

wd = flap.config.get_all_section('Module NSTX_GPI')['Working directory']



def analyze_gpi_structures(exp_id=None,                          #Shot number
                           time_range=None,                      #The time range for the calculation
                           data_object=None,                     #Input data object if available from outside (e.g. generated sythetic signal)

                           x_range=None,                       #X range for the calculation
                           y_range=None,                       #Y range for the calculation

                                               #Normalizer inputs
                           normalize='simple',                #Normalization options,
                                                                                        #None: no normalization
                                                                                        #'roundtrip': zero phase LPF IIR filter
                                                                                        #'halved': different normalzation for before and after the ELM
                                                                                        #'simple': simple low-pass filtered normalization
                           normalize_f_kernel='Elliptic',        #The kernel for filtering the gas cloud
                           normalize_f_high=1e3,                 #High pass frequency for the normalizer data

                           #Input for size pre-processing
                           str_finding_method='watershed',         # Contour or watershed based structure finding
                           ignore_side_structures=False,
                           ellipse_method='linalg',
                           fit_shape='ellipse',
                           subtraction_order=None,      #Polynomial subtraction order
                           remove_interlaced_structures=True,    #Merge the found structures which contain each other
                           
                           #Inputs for size processing
                           nlevel=51,                            #Number of contour levels for the structure size and velocity calculation.
                           filter_level=5,                       #Number of embedded paths to be identified as an individual structure
                           global_levels=False,                  #Set for having structure identification based on a global intensity level.
                           levels=None,                          #Levels of the contours for the entire dataset. If None, it equals data.min(),data.max() divided to nlevel intervals.
                           threshold_method='variance',          #variance or background for the size calculation
                           threshold_coeff=1.0,                  #Variance multiplier threshold for size determination
                           threshold_bg_range={'x':[54,65],      #For the background subtraction, ROI where the bg intensity is calculated
                                               'y':[0,79]},

                           threshold_bg_multiplier=2.,           #Background multiplier for the thresholding
                           weighting='intensity',                #Weighting of the results based on the 'number' of structures, the 'intensity' of the structures or the 'area' of the structures (options are in '')
                           maxing='intensity',                   #Return the properties of structures which have the largest "area" or "intensity"
                           prev_str_weighting='intensity',       #weighting for the differential quantities like angular and linear velocity
                           str_size_lower_thres=0.00375*4,       #Structures having sizes under this value are filtered out from the results. (Default is 4 pixels for both radial, poloidal)
                           elongation_threshold=0.1,             #Structures having major/minor_axis-1 lower than this value are set to angle=np.nan

                           tracking='weighted',                  #Tracking methods 'overlap' or 'weighted'
                           tracking_assignment='max_score',      #Method of assigning the correspondence, 'hungarian' or 'max_score'
                           max_gap=1,
                           smooth_contours=5,                    #Smooths contours with the corner cutting technique this many times.
                           remove_orphans=True,                  #Structures which "live" shorter than min_structure_lifetime
                           min_structure_lifetime=10,            #are cut out from the calculation
                           calculate_rough_diff_velocities=False,#Calculate velocities from average or maximum structures (deprecated)
                           structure_pixel_calc=False,           #Calculate and plot the structure sizes in pixels

                           score_threshold=0.7,                  #Threshold for tracking of the structures based on the weighted tracking.
                           matrix_weight={'iou':1,'cccf':0},     #Weights for the tracking matrix. iou: intersection over union, cccf cross correlation coefficient funcion
                           #Fixing incorrect calculations
                           #fix_structure_angles=False,
                           
                           #Plot options:
                           plot=True,                            #Plot the results
                           pdf=False,                            #Print the results into a PDF
                           plot_error=False,                     #Plot the errorbars of the velocity calculation based on the line fitting and its RMS error
                           error_window=4.,                      #Plot the average signal with the error bars calculated from the normalized variance.
                           overplot_average=True,
                           plot_tracking=True,                   #Plot the tracked structures with a line
                           plot_scatter=False,                   #Add scatter points  to the lineplots
                           structure_video_save=False,           #Save the video of the overplot ellipses
                           video_start_frame=0,
                           video_resolution=(1024,1024),
                           video_framerate=24,
                           
                           nocolorbar=False,
                           structure_pdf_save=False,             #Save the struture finding algorithm's plot output into a PDF (can create very large PDF's, the number of pages equals the number of frames)

                           plot_separatrix=True,
                           plot_flux_surfaces=True,

                           plot_time_range=None,                 #Plot the results in a different time range than the data is read from
                           plot_for_publication=False,           #Modify the plot sizes to single column sizes and golden ratio axis ratios
                           plot_vertical_line_at=None,
                           plot_str_by_str=False,
                           plot_watershed_steps=False,           #Plot the steps of the watershed segmentation at the sample number this is set to.
                           plot_example_structure_frames=False,  #Plot 10 example frames from the sample number this is set to.
                           plot_example_frames_results=False,    #Plot example results for one shot: Area, Angle, Elongation, Roundness
                           plot_nframe=None,
                           plot_ncol=None,
                           linewidth=None,

                            #File input/output options
                           filename=None,                        #Filename for restoring data
                           save_results=True,                    #Save the results into a .pickle file to filename+.pickle
                           nocalc=True,                          #Restore the results from the .pickle file from filename+.pickle
                           recalc_tracking=False,
                           
                            #Output options:
                           return_results=False,                 #Return the results if set.
                           return_pixel_displacement=False,
                           cache_data=True,                      #Cache the data or try to open is from cache

                            #Test options
                           test=False,                           #Test the results
                           test_structures=False,                #Test the structure size calculation
                           test_histogram=False,                 #Plot the poloidal velocity histogram

                           save_data_for_publication=False,
                           verbose=False,
                           skip_mdsplus=False,
                           ):
    
    """
    Analyzes Gas Puff Imaging (GPI) structures in plasma physics data.

    (Gemini generated docstring)

    This function performs comprehensive structure identification, size processing,
    tracking, and velocity calculation on GPI data. It handles data normalization,
    contour/watershed segmentation, tracking structures over time, and generating 
    various diagnostic plots and video outputs.

    Args:
        --- General Inputs ---
        exp_id (int, optional): Shot number for the experiment.
        time_range (list or tuple, optional): The time range [start, end] for the calculation.
        data_object (object, optional): Input data object if available from outside 
            (e.g., generated synthetic signal).
        x_range (list or tuple, optional): X-axis spatial range for the calculation.
        y_range (list or tuple, optional): Y-axis spatial range for the calculation.

        --- Normalizer Inputs ---
        normalize (str, optional): Normalization options. 
            Options include: None (no normalization), 'roundtrip' (zero phase LPF IIR filter), 
            'halved' (different normalization for before and after the ELM), or 
            'simple' (simple low-pass filtered normalization). Defaults to 'simple'.
        normalize_f_kernel (str, optional): The kernel type for filtering the gas cloud. 
            Defaults to 'Elliptic'.
        normalize_f_high (float, optional): High-pass frequency for the normalizer data. 
            Defaults to 1e3.

        --- Structure Pre-processing ---
        str_finding_method (str, optional): 'contour' or 'watershed' based structure finding. 
            Defaults to 'watershed'.
        ignore_side_structures (bool, optional): If True, ignores structures touching the edges.
        ellipse_method (str, optional): Method for fitting ellipses. Defaults to 'linalg'.
        fit_shape (str, optional): Shape to fit to the structures. Defaults to 'ellipse'.
        subtraction_order (int, optional): Polynomial subtraction order.
        remove_interlaced_structures (bool, optional): Merge found structures which contain 
            each other. Defaults to True.

        --- Structure Processing ---
        nlevel (int, optional): Number of contour levels for structure size and velocity 
            calculation. Defaults to 51.
        filter_level (int, optional): Number of embedded paths to be identified as an 
            individual structure. Defaults to 5.
        global_levels (bool, optional): Set to True for structure identification based on a 
            global intensity level. Defaults to False.
        levels (list, optional): Contour levels for the entire dataset. If None, it dynamically 
            calculates based on data min/max divided into `nlevel` intervals.
        threshold_method (str, optional): Method ('variance' or 'background') for size 
            calculation. Defaults to 'variance'.
        threshold_coeff (float, optional): Variance multiplier threshold for size determination. 
            Defaults to 1.0.
        threshold_bg_range (dict, optional): ROI where background intensity is calculated 
            for background subtraction. Defaults to {'x':[54,65], 'y':[0,79]}.
        threshold_bg_multiplier (float, optional): Background multiplier for thresholding. 
            Defaults to 2.0.
        weighting (str, optional): Weighting of results ('number', 'intensity', or 'area' 
            of structures). Defaults to 'intensity'.
        maxing (str, optional): Return properties of structures with the largest 'area' 
            or 'intensity'. Defaults to 'intensity'.
        prev_str_weighting (str, optional): Weighting for differential quantities like angular 
            and linear velocity. Defaults to 'intensity'.
        str_size_lower_thres (float, optional): Minimum size threshold; structures below this 
            are filtered out. Defaults to 0.015 (4 pixels for radial/poloidal).
        elongation_threshold (float, optional): Structures with major/minor_axis-1 lower than 
            this have angle set to np.nan. Defaults to 0.1.

        --- Tracking ---
        tracking (str, optional): Tracking method ('overlap' or 'weighted'). Defaults to 'weighted'.
        tracking_assignment (str, optional): Correspondence assignment method ('hungarian' 
            or 'max_score'). Defaults to 'max_score'.
        max_gap (int, optional): Maximum frame gap allowed for tracking a single structure. 
            Defaults to 1.
        smooth_contours (int, optional): Number of times to smooth contours using the corner 
            cutting technique. Defaults to 5.
        remove_orphans (bool, optional): Remove structures living shorter than 
            `min_structure_lifetime`. Defaults to True.
        min_structure_lifetime (int, optional): Minimum frames a structure must exist to be kept. 
            Defaults to 10.
        calculate_rough_diff_velocities (bool, optional): (Deprecated) Calculate velocities 
            from average or maximum structures. Defaults to False.
        structure_pixel_calc (bool, optional): Calculate/plot structure sizes in pixels. 
            Defaults to False.
        score_threshold (float, optional): Threshold for weighted tracking. Defaults to 0.7.
        matrix_weight (dict, optional): Tracking matrix weights for IoU (intersection over union) 
            and CCCF (cross-correlation coefficient function). Defaults to {'iou': 1, 'cccf': 0}.
        fix_structure_angles (bool, optional): Toggles fixing of incorrect angle calculations. 
            Defaults to False.

        --- Plotting Options ---
        plot (bool, optional): Toggles main results plotting. Defaults to True.
        pdf (bool, optional): Print results to a PDF. Defaults to False.
        plot_error (bool, optional): Plot velocity calculation error bars (based on line fitting 
            and RMS error). Defaults to False.
        error_window (float, optional): Plot average signal with error bars from normalized variance. 
            Defaults to 4.0.
        overplot_average (bool, optional): Toggles overplotting the average. Defaults to True.
        plot_tracking (bool, optional): Plot tracked structures with lines. Defaults to True.
        plot_scatter (bool, optional): Add scatter points to line plots. Defaults to False.
        structure_video_save (bool, optional): Save video of overplotted ellipses. Defaults to False.
        video_start_frame (int, optional): Starting frame for saved video. Defaults to 0.
        video_resolution (tuple, optional): Resolution for saved video. Defaults to (1024, 1024).
        video_framerate (int, optional): Framerate for saved video. Defaults to 24.
        nocolorbar (bool, optional): Suppress colorbars on plots. Defaults to False.
        structure_pdf_save (bool, optional): Save structural finding algorithm output to PDF. 
            Warning: can generate very large files. Defaults to False.
        plot_separatrix (bool, optional): Overplot the separatrix. Defaults to True.
        plot_flux_surfaces (bool, optional): Overplot flux surfaces. Defaults to True.
        plot_time_range (list, optional): Specific time range for plotting, if different from 
            read data.
        plot_for_publication (bool, optional): Format plots to single-column sizes and 
            golden ratio dimensions. Defaults to False.
        plot_vertical_line_at (float, optional): X-axis value to draw a vertical reference line.
        plot_str_by_str (bool, optional): Plot individual structures step-by-step.
        plot_watershed_steps (int, optional): Frame/sample number to plot watershed segmentation steps.
        plot_example_structure_frames (int, optional): Sample number to plot 10 example frames.
        plot_example_frames_results (bool, optional): Plot example Area, Angle, Elongation, 
            and Roundness for one shot. Defaults to False.
        plot_nframe (int, optional): Number of frames to plot.
        plot_ncol (int, optional): Number of columns for subplots.
        linewidth (float, optional): Line width for plot elements.

        --- File I/O & Output Options ---
        filename (str, optional): Base filename for restoring/saving data.
        save_results (bool, optional): Save results to a .pickle file (filename + .pickle). 
            Defaults to True.
        nocalc (bool, optional): Skip calculation and restore results from the .pickle file. 
            Defaults to True.
        recalc_tracking (bool, optional): Force recalculation of tracking. Defaults to False.
        return_results (bool, optional): Return the calculated results dictionary/object. 
            Defaults to False.
        return_pixel_displacement (bool, optional): Return displacement in pixels instead of 
            physical units. Defaults to False.
        cache_data (bool, optional): Cache data or attempt to read from cache. Defaults to True.
        save_data_for_publication (bool, optional): Export data formats optimized for publication. 
            Defaults to False.
        verbose (bool, optional): Enable detailed logging output. Defaults to False.
        skip_mdsplus (bool, optional): Skip reading data from the MDSplus tree. Defaults to False.

        --- Testing Options ---
        test (bool, optional): Run general tests on results. Defaults to False.
        test_structures (bool, optional): Test the structure size calculation. Defaults to False.
        test_histogram (bool, optional): Plot the poloidal velocity histogram for debugging. 
            Defaults to False.

    Returns:
        Varies based on flags. If `return_results` is True, it returns the processed 
        structure tracking and analysis data (a dict). Otherwise, 
        returns None and saves output to files/plots.
    """



    """
    SETTING UP THE FILENAME FOR DATA SAVING
    """
    
    #Input error handling
    if exp_id is None and data_object is None:
        raise ValueError('Either exp_id or data_object needs to be set for the calculation.')

    if data_object is None:
        if time_range is None and filename is None:
            raise ValueError('It takes too much time to calculate the entire shot, please set a time_range.')
        else:
            if not isinstance(time_range, (list,np.ndarray)) and filename is None:
                raise TypeError('time_range is not a list.')
            if filename is None and len(time_range) != 2:
                raise ValueError('time_range should be a list of two elements.')

    if weighting not in ['number', 'area', 'intensity']:
        raise ValueError("Weighting can only be by the 'number', 'area' or 'intensity' of the structures.")
    if maxing not in ['area', 'intensity']:
        raise ValueError("Maxing can only be by the 'area' or 'intensity' of the structures.")


    if filename is None:
        comment=''
        
        if normalize is not None:
            comment += normalize

        if remove_interlaced_structures:
            comment += '_nointer'
        comment += '_'+str_finding_method
        if data_object is not None:
            try:
                if type(data_object) == type(flap.DataObject):
                    exp_id = data_object.exp_id
                elif type(data_object) == str:
                    exp_id = flap.get_data_object_ref(data_object).exp_id
            except Exception as e:
                print('Exception in analyze_gpi_structures at line 200.')
                print(e)
                exp_id = 0

        filename = flap_nstx.tools.filename(exp_id = exp_id,
                                            working_directory = wd + '/processed_data',
                                            time_range = time_range,
                                            purpose = 'structure char',
                                            comment = comment)

        filename_was_none = True
    else:
        filename_was_none = False

    plot_results=plot

    # if plot_for_publication:
    #     set_matplotlib_for_publication(labelsize=9,
    #                                    linewidth=0.5,
    #                                    major_ticksize=2.,
    #                                    )

    fit_shape=fit_shape.capitalize()

    pickle_filename=filename+'.pickle'
    
    if not os.path.exists(pickle_filename) and nocalc:
        print(pickle_filename)
        print('The pickle file does not exist. Recalculating the results.')
        nocalc = False

    if ((not test and not test_structures) or
        (not test and not plot_results and structure_pdf_save and test_structures)):
        import matplotlib
        matplotlib.use('agg')

    if not nocalc or structure_video_save or plot_example_structure_frames:


        if structure_pdf_save:
            filename=flap_nstx.tools.filename(exp_id=exp_id,
                                              working_directory=wd+'/plots',
                                              time_range=time_range,
                                              purpose='found structures',
                                              comment=comment,
                                              extension='pdf')
            pdf_structures=PdfPages(filename)
            
        """
        # READING THE DATA
        """
        
        #Read data
        if data_object is None:
            print("\n------- Reading NSTX GPI data --------")
            if cache_data:
                try:
                    data=flap.get_data_object('GPI',exp_id=exp_id)
                except:
                    print('Data is not cached, it needs to be read.')
                    data=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                    name='',
                                    object_name='GPI')
            else:
                data=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                name='',
                                object_name='GPI')
            if x_range is None or y_range is None:
                x_range=[0, data.data.shape[1]-1]
                y_range=[0, data.data.shape[2]-1]

            slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                     'Image x':flap.Intervals(x_range[0],x_range[1]),
                     'Image y':flap.Intervals(y_range[0],y_range[1])}

            data=flap.slice_data('GPI',
                              exp_id=exp_id,
                              slicing=slicing,
                              output_name='GPI_SLICED_FULL')

        elif isinstance(data_object, str):
            if exp_id is None:
                exp_id='*'

            data=flap.get_data_object(data_object,
                                   exp_id=exp_id)
            time_range=[data.coordinate('Time')[0][0,0,0],
                        data.coordinate('Time')[0][-1,0,0]]
            exp_id=data.exp_id
            object_name='GPI_SLICED_FULL'
            flap.add_data_object(data, object_name)

            if x_range is None:
                x_range=[0, data.data.shape[1]-1]

            if y_range is None:
                y_range=[0, data.data.shape[2]-1]


        elif isinstance(data_object, flap.DataObject):
            data=copy.deepcopy(data_object)
            object_name='GPI'
            flap.add_data_object(data, object_name)

            if x_range is None:
                x_range=[0, data.data.shape[1]-1]

            if y_range is None:
                y_range=[0, data.data.shape[2]-1]

            if time_range is None:
                time_range=[data.coordinate('Time')[0][:,0,0].min(),
                            data.coordinate('Time')[0][:,0,0].max()]
        else:
            raise TypeError(f"Invalid data_object type: {type(data_object)}")

        """
        # NORMALIZATION PROCESS
        """

        if normalize is not None and data_object is None:

            slicing_for_filtering=copy.deepcopy(slicing)
            slicing_for_filtering['Time']=flap.Intervals(time_range[0]-1/normalize_f_high*10,
                                                         time_range[1]+1/normalize_f_high*10)

            slicing_time_only={'Time':flap.Intervals(time_range[0],
                                                     time_range[1])}


            flap.slice_data('GPI',
                            exp_id=exp_id,
                            slicing=slicing_for_filtering,
                            output_name='GPI_SLICED_FOR_FILTERING')

            object_name='GPI_SLICED_FOR_FILTERING'
            coefficient=normalize_gpi(object_name,
                                      exp_id=exp_id,
                                      slicing_time=slicing_time_only,
                                      normalize=normalize,
                                      normalize_f_high=normalize_f_high,
                                      normalize_f_kernel=normalize_f_kernel,
                                      normalizer_object_name='GPI_LPF_INTERVAL',
                                      output_name='GPI_GAS_CLOUD')

            data_obj=flap.get_data_object('GPI_SLICED_FULL',
                                          exp_id=exp_id)


            data_obj.data = data_obj.data/coefficient
            flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_STR_SIZE')
            object_name='GPI_SLICED_DENORM_STR_SIZE'

        if subtraction_order is not None:
            if verbose: print("*** Subtracting the trend of the data ***")
            data=detrend_multidim(object_name,
                                  exp_id=exp_id,
                                  order=subtraction_order,
                                  coordinates=['Image x',
                                               'Image y'],
                                  output_name='GPI_DETREND_STR_SIZE')

            object_name='GPI_DETREND_STR_SIZE'

        if global_levels:
            if levels is None:
                data=flap.get_data_object_ref(object_name)
                min_data=data.data.min()
                max_data=data.data.max()
                levels=np.arange(nlevel)/(nlevel-1)*(max_data-min_data)+min_data

        if threshold_method == 'variance':
            thres_obj_str_size=flap.slice_data(object_name,
                                               exp_id=exp_id,
                                               summing={'Image x':'Mean',
                                                        'Image y':'Mean'},
                                                        output_name='GPI_SLICED_TIMETRACE')
            intensity_thres_level_str_size=np.sqrt(np.var(thres_obj_str_size.data))*threshold_coeff+np.mean(thres_obj_str_size.data)

        if threshold_method == 'background_average':
            intensity_thres_level_str_size=threshold_bg_multiplier*np.mean(flap.slice_data(object_name,
                                                                                  slicing={'Image x':flap.Intervals(threshold_bg_range['x'][0],
                                                                                                                    threshold_bg_range['x'][1]),
                                                                                           'Image y':flap.Intervals(threshold_bg_range['y'][0],
                                                                                                                    threshold_bg_range['y'][1])}).data)
        """
        #     VARIABLE DEFINITION
        """
        
        #Calculate correlation between subsequent frames in the data
        #Setting the variables for the calculation
        time_dim=data.get_coordinate_object('Time').dimension_list[0]
        n_frames=data.data.shape[time_dim]
        time=data.coordinate('Time')[0][:,0,0]
        sample_time=time[1]-time[0]
        sample_0=flap.get_data_object_ref('GPI_SLICED_FULL',
                                          exp_id=exp_id).coordinate('Sample')[0][0,0,0]
        
        if plot_flux_surfaces or plot_separatrix:
            try:
                if plot_separatrix:
                    d_sep_x=flap.get_data('NSTX_MDSPlus',
                                          name=r'\EFIT02::\RBDRY',
                                          exp_id=exp_id,
                                          object_name='SEP X OBJ'
                                          )

                    d_sep_y=flap.get_data('NSTX_MDSPlus',
                                          name=r'\EFIT02::\ZBDRY',
                                          exp_id=exp_id,
                                          object_name='SEP Y OBJ'
                                          )
                else:
                    d_sep_x=None
                    d_sep_y=None

                if plot_flux_surfaces:
                    d_flux=flap.get_data('NSTX_MDSPlus',
                                         name=r'\EFIT02::\PSIRZ',
                                         exp_id=exp_id,
                                         object_name='PSI RZ OBJ'
                                         )
                else:
                    d_flux=None
            except Exception as e:
                print('Exception occurred in analyze_gpi_structures.py at line 568.')
                print(e)
                
                d_sep_x=None
                d_sep_y=None
                d_flux=None
        
        if not ((structure_video_save or plot_example_structure_frames) and nocalc):
            # coordinate_names=[data.coordinates[i].unit.name for i in range(len(data.coordinates))]
            # distance_unit='pix'
            # time_unit='sample'
            # for ind in range(len(coordinate_names)):
                # if coordinate_names[ind] == 'Time':
                #     time_unit=data.coordinates[ind].unit.unit
                # if coordinate_names[ind] == 'Device R':
                #     distance_unit=data.coordinates[ind].unit.unit

            #frame_properties=frame_properties_dict(exp_id, time, time_unit, distance_unit)
            raw_dataset = StructureDataset(mode='untracked', exp_id=exp_id)
            
            #Inicializing for frame handling
            frame=None
            structures_dict=None

            if test or test_structures or structure_pdf_save:
                fig_dpi=80
                plt.figure(figsize=(800/fig_dpi, 600/fig_dpi), dpi=fig_dpi)

            if not skip_mdsplus and data_object is None:
                elm_time=(time[-1]-time[0])/2
                try:
                    R_sep=flap.get_data('NSTX_MDSPlus',
                                        name='\EFIT02::\RBDRY',
                                        exp_id=exp_id,
                                        object_name='SEP R OBJ').slice_data(slicing={'Time':elm_time}).data
    
                    z_sep=flap.get_data('NSTX_MDSPlus',
                                        name='\EFIT02::\ZBDRY',
                                        exp_id=exp_id,
                                        object_name='SEP Z OBJ').slice_data(slicing={'Time':elm_time}).data
                    
                    

                    #Constants for the calculation
                    #Using the spatial calibration to find the actual velocities.
                    coeff_r=np.asarray([3.75, 0,    1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
                    coeff_z=np.asarray([0,    3.75, 70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
                    
                    # Originally used coordinates for reference. (Vertical, radial geometrical coordinates)
                    # coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
                    # coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    
                    # Extract the upper boundary calculation for readability
                    z_bound_upper = coeff_z[2] + 79*coeff_z[0] + 64*coeff_z[1]
                    
                    # Apply the mask
                    sep_GPI_ind = np.where((R_sep > coeff_r[2]) & (z_sep > coeff_z[2]) & (z_sep < z_bound_upper))
                    
                    sep_GPI_ind=np.asarray(sep_GPI_ind[0])
                    sep_GPI_ind=np.insert(sep_GPI_ind,0,sep_GPI_ind[0]-1)
                    sep_GPI_ind=np.insert(sep_GPI_ind,len(sep_GPI_ind),sep_GPI_ind[-1]+1)
    
                    # z_sep_GPI=z_sep[(sep_GPI_ind)]
                    # R_sep_GPI=R_sep[sep_GPI_ind]
                    # GPI_z_vert=coeff_z[0]*np.arange(80)/80*64+coeff_z[1]*np.arange(80)+coeff_z[2]
                    # R_sep_GPI_interp=np.interp(GPI_z_vert,
                    #                            np.flip(z_sep_GPI),
                    #                            np.flip(R_sep_GPI))
                    # z_sep_GPI_interp=GPI_z_vert
                
                except Exception as e:
                    print(e)
                    print('\n Could not read EFIT data. Setting separatrix data to None')
                    # z_sep_GPI=None
                    # R_sep_GPI=None
                    # R_sep_GPI_interp=None
                    # z_sep_GPI_interp=None


            for i_frames in range(n_frames):

                print(f"\r{i_frames/(n_frames-1)*100.}% done from the calculation.", end="", flush=True)

                slicing_frame={'Sample':sample_0+i_frames}

                frame=flap.slice_data(object_name,
                                      exp_id=exp_id,
                                      slicing=slicing_frame,
                                      output_name='GPI_FRAME')

                frame.data=np.asarray(frame.data, dtype='float64')

                if structure_video_save or structure_pdf_save:
                    plt.cla()
                    test_structures=True
                if plot_watershed_steps and i_frames == plot_watershed_steps:
                    plot_full=True
                    pdf_plot_watershed=PdfPages(wd+'/plots/watershed_steps.pdf')
                else:
                    plot_full=False
                    
                    
                slicing={'Time':frame.coordinate('Time')[0][0,0]}
                if d_sep_x is not None and d_sep_y is not None:
                    d_sep_x_sliced=d_sep_x.slice_data(slicing=slicing)
                    d_sep_y_sliced=d_sep_y.slice_data(slicing=slicing)

                    separatrix_data=np.zeros([d_sep_x_sliced.shape[0],2])
                    separatrix_data[:,0]=d_sep_x_sliced.data
                    separatrix_data[:,1]=d_sep_y_sliced.data
                else:
                    separatrix_data=None

                if d_flux is not None:
                    surface_data_obj=d_flux.slice_data(slicing=slicing)
                else:
                    surface_data_obj=None

                structures_dict = identify_structures(str_finding_method=str_finding_method,
                                                      data_object='GPI_FRAME',
                                                      ignore_side_structure=ignore_side_structures,
                                                      threshold_level=intensity_thres_level_str_size,
                                                      exp_id=exp_id,
                                                      filter_level=filter_level,
                                                      nlevel=nlevel,
                                                      levels=levels,
                                                      mfilter_range=5,
                                                      smooth_contours=smooth_contours,
                                                      spatial=not structure_pixel_calc,
                                                      pixel=structure_pixel_calc,
                                                      remove_interlaced_structures=remove_interlaced_structures,
                                                      ellipse_method=ellipse_method,
                                                      fit_shape=fit_shape,
                                                      str_size_lower_thres=str_size_lower_thres,
                                                      elongation_threshold=elongation_threshold,
                                                      test=test,
                                                      plot_result=test_structures,
                                                      plot_full=plot_full,
                                                      plot_full_for_publication=plot_for_publication,
                                                      video_resolution=video_resolution,
                                                      structure_video_save=structure_video_save,
                                                      plot_flux_surfaces=plot_flux_surfaces,
                                                      surface_data_obj=surface_data_obj,
                                                      plot_separatrix=plot_separatrix,
                                                      separatrix_data=separatrix_data,
                                                      save_data_for_publication=save_data_for_publication,
                                                      verbose=verbose)

                if plot_watershed_steps and i_frames == plot_watershed_steps:
                    plt.tight_layout(pad=0.1)
                    pdf_plot_watershed.savefig()
                    pdf_plot_watershed.close()
                
                current_time = frame.coordinate('Time')[0][0,0]
                raw_dataset.add_frame(structures_dict, current_time)
                
                if structure_pdf_save:
                    plt.title(str(exp_id)+' @ '+"{:.3f}".format(time[i_frames]*1e3)+'ms')
                    plt.show()
                    pdf_structures.savefig()

            if structure_pdf_save:
                pdf_structures.close()
            #Saving results into a pickle file
            # if fix_structure_angles:
            #     frame_properties=_fix_structure_angles(frame_properties)
            with open(pickle_filename, 'wb') as f:
                pickle.dump(structures_dict,f)
            if test:
                plt.close()
        else:
            print('\n\n--- Loading data from the pickle file ---')
            with open(pickle_filename, 'rb') as f:
                structures_dict=pickle.load(f)
    else:
        print('\n\n--- Loading data from the pickle file ---')
        with open(pickle_filename, 'rb') as f:
            structures_dict=pickle.load(f)
        #labels= 'label,born,died'
        
    """
        Structure tracking
    """
    
    tracked_dataset = track_structures(dataset=raw_dataset,
                                           max_gap=max_gap,
                                           time_range=time_range,
                                           tracking=tracking,
                                           tracking_assignment=tracking_assignment,
                                           matrix_weight=matrix_weight,
                                           test=test,
                                           prev_str_weighting=prev_str_weighting,
                                           calculate_rough_diff_velocities=calculate_rough_diff_velocities,
                                           weighting=weighting,
                                           maxing=maxing,
                                           remove_orphans=remove_orphans,
                                           min_structure_lifetime=min_structure_lifetime,
                                           nocalc=nocalc,
                                           recalc_tracking=recalc_tracking,
                                           smooth_contours=smooth_contours,
                                           comment=comment)

    """
    #PLOTTING THE RESULTS
    """

    import matplotlib.colors as mcolors
    colortable=list(mcolors.TABLEAU_COLORS.keys())
    n_color=len(colortable)

    if time_range is None:
        time_range=[time[0], time[-1]]

    if not filename_was_none and not time_range is None:
        sample_time=time[1]-time[0]
        if time_range[0] < time[0]-sample_time or time_range[1] > time[-1]+sample_time:
            raise ValueError('Please run the calculation again with the timerange. The pickle file doesn\'t have the desired range')

    if structure_video_save:
        _structure_video_save(sample_0=sample_0,
                              object_name=object_name,
                              exp_id=exp_id,
                              n_frames=n_frames,
                              d_sep_x=d_sep_x,
                              d_sep_y=d_sep_y,
                              d_flux=d_flux,
                              video_resolution=video_resolution,
                              levels=levels,
                              nocolorbar=nocolorbar,
                              plot_flux_surfaces=plot_flux_surfaces,
                              plot_separatrix=plot_separatrix,
                              # time=time,  <-- DELETE THIS LINE!
                              str_finding_method=str_finding_method,
                              video_framerate=video_framerate,
                              dataset=raw_dataset,
                              colortable=colortable,
                              wd=wd,
                              n_color=n_color,
                              video_start_frame=video_start_frame,
                              )

    if plot_example_structure_frames:
        _plot_example_structure_frames(exp_id=exp_id,
                                       time_range=time_range,
                                       plot_time_range=plot_time_range,
                                       sample_0=sample_0,
                                       wd=wd,
                                       object_name=object_name,
                                       plot_nframe=plot_nframe,
                                       plot_ncol=plot_ncol,
                                       levels=levels,
                                       dataset=tracked_dataset,
                                       time=time,
                                       plot_example_structure_frames=plot_example_structure_frames,
                                       plot_separatrix=plot_separatrix,
                                       separatrix_coordinates=(d_sep_x,d_sep_y),
                                       # save_data_for_publication=save_data_for_publication,
                                       )

    #Plotting the results
    if plot_results or pdf:
        _plot_results(dataset=tracked_dataset,
                      pdf=pdf,
                      plot_results=plot_results,
                      plot_time_range=plot_time_range,
                      time_range=time_range,
                      plot_str_by_str=plot_str_by_str,
                      comment=comment,
                      exp_id=exp_id,
                      plot_for_publication=plot_for_publication,
                      plot_vertical_line_at=plot_vertical_line_at,
                      overplot_average=overplot_average,
                      plot_scatter=plot_scatter,
                      plot_tracking=plot_tracking,
                      n_color=n_color,
                      colortable=colortable,
                      )

    if plot_example_frames_results:
        _plot_example_frames_results(dataset=tracked_dataset,
                                     exp_id=exp_id,
                                     time_range=time_range,
                                     plot_time_range=plot_time_range,
                                     wd=wd,
                                     n_color=n_color,
                                     colortable=colortable,
                                     pdf=pdf,
                                     save_data_for_publication=save_data_for_publication,
                                     )

    if return_results:
        return tracked_dataset


def transform_frames_to_structures(frame_properties):
    """
    Transforms structure data from a frame-centric format to a structure-centric format.

    

    This function iterates through frame-by-frame structure data, identifies all unique
    tracked structures based on their 'Label', and pivots the data so that the time 
    evolution of each individual structure's properties is grouped together.

    Note: 
        - The function relies on an external `read_analyzed_keys()` function to 
          determine which properties to extract.
        - Property values exactly equal to 0 are converted to `np.nan`.

    Args:
        frame_properties (dict): A dictionary containing the frame-by-frame data. 
            Expected to have at least the following structure:
            - 'structures': A list (frames) of lists (structures within the frame), 
              where each structure is a dictionary containing a 'Label' key (int) 
              and other property keys.
            - 'Time': A list of time values corresponding to each frame.

    Returns:
        list of dict: A list representing individual structures, where the index 
            corresponds to the structure's 'Label'. Each element is a dictionary 
            containing lists of property values over time (e.g., 'Time', 'Area', 
            'Intensity', etc., depending on `read_analyzed_keys()`). 
            Example output format:
            [
                {'Time': [0.1, 0.2], 'Intensity': [10.5, 11.2]},  # Structure Label 0
                {'Time': [0.2, 0.3], 'Intensity': [5.0, np.nan]}  # Structure Label 1
            ]
    """
    max_label = 0
    for frame_structs in frame_properties['structures']:
        if frame_structs:
            max_label = max(max_label, max((s.label if s.label is not None else 0 for s in frame_structs), default=0))                    
    struct_by_struct=[]
    for ind in range(max_label+1):
        struct_by_struct.append({'time':[],}.copy())

    # analyzed_keys=read_analyzed_keys()

    regular_keys=frame_properties['structures'][0]['Regular parameters'].keys()
    differential_keys=frame_properties['structures'][0]['Differential parameters'].keys()
    
    struct_by_struct = [
        {
            'time': [], 
            'Regular parameters': {k: [] for k in regular_keys},
            'Differential parameters': {k: [] for k in differential_keys} # Did you mean differential_keys here?
        } 
        for _ in range(max_label + 1)
    ]
    for parameters in ['Regular parameters', 'Differential parameters']:
        for i_frame, frame_structs in enumerate(frame_properties['structures']):
            if not frame_structs:
                continue
            
            current_time = frame_properties['time'][i_frame]
            
            for struct in frame_structs:
                label = struct.label
                if label is None: continue
    
                label=int(label)
                struct_by_struct[label]['time'].append(current_time)
                
                for key in struct[parameters].keys():
                    # Safely extract the value. If it's exactly 0 or missing, use NaN
                    val = struct[parameters].get(key, np.nan)
                    val_float = float(val) if isinstance(val, flap_nstx.tools.Metric) else val
                    if val_float == 0:
                        val_float = np.nan
                        
                    struct_by_struct[label][parameters][key].append(val_float)

    return struct_by_struct

def _fix_structure_angles(frame_properties):

    """
    Normalizes structure angles and recalculates the Angle of Least Inertia (ALI).

    

    This internal utility iterates through all structures in the provided frames. 
    It normalizes the primary 'Angle' to strictly fall within the range [-pi, pi].

    Args:
        frame_properties (dict): A dictionary containing frame-by-frame tracked 
            structure data. Expected to contain:
            - 'structures': A list of lists representing frames and their structures. 
              Each structure must be a dictionary with keys: 'Angle', 'Center of gravity', 
              'Data' (intensity), 'X coord', and 'Y coord'.

    Returns:
        dict: The updated `frame_properties` dictionary with corrected 'Angle' and 
    """    

    for i_frames, frame_structs in enumerate(frame_properties['structures']):
        if frame_structs:
            for struct in frame_structs:
                struct['Regular parameters']['Angle fit'] = (struct['Regular parameters']['Angle fit'] + np.pi) % (2 * np.pi) - np.pi

    return frame_properties



def _plot_results(dataset=None,
                  pdf=False,
                  plot_results=False,
                  plot_time_range=None,
                  time_range=None,
                  plot_str_by_str=False,
                  comment=None,
                  exp_id=None,
                  
                  plot_for_publication=False,
                  plot_vertical_line_at=None,
                  overplot_average=False,
                  plot_scatter=False,
                  plot_tracking=False,
                  n_color=None,
                  colortable=None,
                  ):
    
    """
    Manages the plotting and saving of structure analysis results.

    This internal function configures the matplotlib backend based on whether 
    the plots are meant to be displayed interactively or just saved. It also 
    handles time range validation, sets up figure dimensions (including 
    publication-ready sizing), manages PDF export, and delegates the rendering
    to either average or structure-by-structure plotting routines.

    Args:
        pdf (bool, optional): If True, saves the generated plots to a multipage 
            PDF file in the working directory. Defaults to False.
        plot_results (bool, optional): If True, uses the 'QT5Agg' backend for 
            interactive plotting. If False, uses the 'agg' backend for background 
            rendering. Defaults to False.
        plot_time_range (list or tuple, optional): The specific [start, end] time 
            range to plot. Must be within `time_range`. Defaults to None.
        time_range (list or tuple, optional): The original [start, end] time range 
            of the calculated data. Defaults to None.
        plot_str_by_str (bool, optional): If True, plots individual structures 
            step-by-step. If False, plots average results. Defaults to False.
        comment (str, optional): A string to append to the generated PDF filename 
            for identification. Defaults to None.
        exp_id (int or str, optional): The experiment or shot ID, used for 
            generating the file path and name. Defaults to None.
        frame_properties (dict, optional): The main data dictionary containing 
            the 'structures' and 'derived' calculation keys. Defaults to None.
        plot_for_publication (bool, optional): If True, forces the figure size to 
            single-column width (8.5 cm) and a golden ratio height for publication. 
            Defaults to False.
        plot_vertical_line_at (float, optional): X-coordinate at which to draw a 
            vertical reference line. Defaults to None.
        overplot_average (bool, optional): If True, overplots the average value 
            on the graphs. Defaults to False.
        plot_scatter (bool, optional): If True, uses scatter plots instead of or 
            in addition to lines. Defaults to False.
        plot_tracking (bool, optional): If True, plots the tracked structures 
            with lines (passed to `_plot_str_by_str`). Defaults to False.
        n_color (int, optional): The number of distinct colors to use in the 
            color table. Defaults to None.
        colortable (list or object, optional): A specific color table or colormap 
            to use for rendering structures. Defaults to None.

    Raises:
        ValueError: If `plot_time_range` falls outside the bounds of the original 
            `time_range`.

    Returns:
        None
    """
    
    #This is a bit unusual here, but necessary due to the structure size calculation based on the contours which are not plot
    if plot_results:
        import matplotlib
        matplotlib.use('QT5Agg')
        #import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('agg')
       # import matplotlib.pyplot as plt

    if plot_time_range is not None:
        if plot_time_range[0] < time_range[0] or plot_time_range[1] > time_range[1]:
            raise ValueError('The plot time range is not in the interval of the original time range.')
        time_range=plot_time_range



    #Plotting the radial velocity
    if pdf:
        if plot_str_by_str:
            comment+='_sbs'
        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                          working_directory=wd+'/plots',
                                          time_range=time_range,
                                          purpose='ccf velocity',
                                          comment=comment)

        pdf_filename=filename+'.pdf'
        pdf_pages=PdfPages(pdf_filename)

    if plot_for_publication:

        figsize=(8.5/2.54,
                 8.5/2.54/1.618*1.1)
    else:
        figsize=None
        
    if not pdf: pdf_pages=None
    

    _plot_str_by_str(dataset=dataset,
                     plot_scatter=plot_scatter,
                     figsize=figsize,
                     colortable=colortable,
                     n_color=n_color,
                     time_range=time_range,
                     plot_for_publication=plot_for_publication,
                     pdf=pdf,
                     pdf_pages=pdf_pages,
                     )

    if pdf:
       pdf_pages.close()

    if plot_for_publication:
        import matplotlib.style as pltstyle
        pltstyle.use('default')
    return


def _plot_example_structure_frames(exp_id=None,
                                   time_range=None,
                                   plot_time_range=None,
                                   sample_0=None,
                                   wd=None,
                                   object_name=None,
                                   plot_nframe=None,
                                   plot_ncol=None,
                                   levels=None,
                                   dataset=None, # Changed from frame_properties to your StructureDataset!
                                   plot_example_structure_frames=None,
                                   save_data_for_publication=False,
                                   plot_separatrix=False,
                                   separatrix_coordinates=None,
                                   linewidth=None):
    
    """
    Plots a grid of example GPI frames with overlaid tracked structures.
    """

    (d_sep_x, d_sep_y) = separatrix_coordinates
    
    if plot_time_range is not None:
        if plot_time_range[0] < time_range[0] or plot_time_range[1] > time_range[1]:
            raise ValueError('The plot time range is not in the interval of the original time range.')
        time_range = plot_time_range
    
    import matplotlib.colors as mcolors
    colortable = list(mcolors.TABLEAU_COLORS.keys())
    n_color = len(colortable)

    pdf_filenames_frames = flap_nstx.tools.filename(exp_id=exp_id,
                                                    time_range=time_range,
                                                    purpose="example_frames",
                                                    extension='.pdf',
                                                    comment='s0'+str(sample_0))

    pdf_pages = PdfPages(wd+'/plots/'+pdf_filenames_frames)
    from flap_nstx.gpi import _plot_ellipses_centers
    import scipy
    
    if plot_time_range:
        data_object = flap.get_data_object(object_name)
        sliced_data = data_object.slice_data(slicing={'Time':flap.Intervals(plot_time_range[0], plot_time_range[1])})
        frame_sample_0 = sliced_data.coordinate('Sample')[0][0,0,0] - sample_0
        sample_0 = sliced_data.coordinate('Sample')[0][0,0,0]
    else:
        frame_sample_0 = 0
        
    x_coord_name, x_unit_name = 'Device R', '[m]'
    y_coord_name, y_unit_name = 'Device z', '[m]'

    if plot_nframe is None and plot_ncol is None:
        plot_nframe = 15
        plot_ncol = 5

    fig, axes = plt.subplots(int(plot_nframe/plot_ncol), plot_ncol,
                             figsize=(8.5/2.54, 3.5*plot_nframe/plot_ncol/2.54))

    for i_frames in range(0, plot_nframe):
        ax = axes[i_frames//plot_ncol, np.mod(i_frames, plot_ncol)]

        slicing_frame = {'Sample': sample_0 + i_frames + plot_example_structure_frames}

        frame = flap.slice_data(object_name,
                                exp_id=exp_id,
                                slicing=slicing_frame,
                                output_name='GPI_FRAME')

        frame.data = np.asarray(frame.data, dtype='float64')
        frame.data = scipy.ndimage.median_filter(frame.data, 5)

        x_coord = frame.coordinate(x_coord_name)[0]
        y_coord = frame.coordinate(y_coord_name)[0]
        current_time = frame.coordinate('Time')[0][0,0]

        if levels is None:
            ax.contourf(x_coord, y_coord, frame.data, levels=51)
        else:
            ax.contourf(x_coord, y_coord, frame.data, levels=levels)

        if plot_separatrix:
            slicing = {'Time': current_time}
            
            d_sep_x_sliced = d_sep_x.slice_data(slicing=slicing)
            d_sep_y_sliced = d_sep_y.slice_data(slicing=slicing)
            
            separatrix_data = np.zeros([d_sep_x_sliced.shape[0], 2])
            separatrix_data[:,0] = d_sep_x_sliced.data
            separatrix_data[:,1] = d_sep_y_sliced.data
            
            if separatrix_data is not None:
                ax.plot(separatrix_data[:,0], separatrix_data[:,1], linewidth=1, color='red')
                
                if save_data_for_publication:
                    filename = f"{wd}/{exp_id}_{current_time}_separatrix.txt"
                    with open(filename, 'w+') as file1:
                        for i in range(len(separatrix_data[:,0])):
                            file1.write(f"{separatrix_data[i,0]}\t{separatrix_data[i,1]}\n")

        ax.set_aspect(1.0)
        
        # --- OOP STRUCTURE EXTRACTION ---
        target_idx = frame_sample_0 + i_frames + plot_example_structure_frames
        structures = dataset.frames[target_idx]

        if structures is not None and len(structures) > 0:
            R = np.arange(0, 2*np.pi, 0.01)
            
            for i_str, struct in enumerate(structures):
                
                # Check for a valid fit
                if not np.isnan(struct.fit_angle.value):
                    
                    # BUG FIX: Use the correct keys and extract the values!
                    phi = struct.fit_angle.value
                    a, b = struct.fit_axes_length[1].value, struct.fit_axes_length[0].value # major, minor
                    cx, cy = struct.fit_center[0].value, struct.fit_center[1].value

                    x_polygon, y_polygon = struct.x, struct.y

                    x_ellipse = cx + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
                    y_ellipse = cy + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
                    
                    if linewidth is None:
                        polygon_linewidth, ellipse_linewidth, semiaxis_linewidth = 1, 0.25, 0.5
                    else:
                        polygon_linewidth = linewidth['polygon']
                        ellipse_linewidth = linewidth['ellipse']
                        semiaxis_linewidth = linewidth['semiaxis']
                        
                    # Safely handle None labels
                    color_idx = int(np.mod(struct.label + 1, n_color)) if struct.label is not None else 0

                    _plot_ellipses_centers(ax,
                                           x_polygon, y_polygon,
                                           x_ellipse, y_ellipse,
                                           struct,
                                           polygon_color=colortable[color_idx],
                                           ellipse_color='black',
                                           polygon_linewidth=polygon_linewidth,
                                           ellipse_linewidth=ellipse_linewidth,
                                           semiaxis_linewidth=semiaxis_linewidth,
                                           plot_structure_mid=False)
                                           
                if save_data_for_publication:
                    # Save Polygon
                    filename_poly = f"{wd}/{exp_id}_{current_time}_half_path_no.{i_str}.txt"
                    with open(filename_poly, 'w+') as file1:
                        for i in range(len(x_polygon)):
                            file1.write(f"{x_polygon[i]}\t{y_polygon[i]}\n")

                    # Save Ellipse
                    filename_ellipse = f"{wd}/{exp_id}_{current_time}_fit_ellipse_no.{i_str}.txt"
                    with open(filename_ellipse, 'w+') as file1:
                        for i in range(len(x_ellipse)):
                            file1.write(f"{x_ellipse[i]}\t{y_ellipse[i]}\n")
                    
        if save_data_for_publication:
            filename_raw = f"{wd}/{exp_id}_{current_time}_raw_data.txt"
            with open(filename_raw, 'w+') as file1:
                for i in range(len(frame.data[0,:])):
                    string = '\t'.join([str(frame.data[j, i]) for j in range(len(frame.data[:,0]))]) + '\n'
                    file1.write(string)
                        
        ax.set_xlabel(f"{x_coord_name.replace('Device ', '')} {x_unit_name}")
        ax.set_ylabel(f"{y_coord_name.replace('Device ', '')} {y_unit_name}")
        
        if np.mod(i_frames, plot_ncol) != 0:
            ax.get_yaxis().set_visible(False)

        if i_frames < plot_nframe - plot_ncol:
            ax.get_xaxis().set_visible(False)

        ax.set_xlim([x_coord.min(), x_coord.max()])
        ax.set_ylim([y_coord.min(), y_coord.max()])
        ax.set_title(f"{current_time*1e3:.3f} ms", fontsize=8, y=0.95)

    plt.tight_layout(h_pad=0.3, w_pad=0.1)
    
    pdf_pages.savefig()
    pdf_pages.close()


def _plot_example_frames_results(exp_id=None, time_range=None, plot_time_range=None,
                                 dataset=None, wd=None, n_color=None,
                                 colortable=None, pdf=None, markersize=0.5,
                                 save_data_for_publication=False):
    
    """
    Plots the time evolution of specific structure properties in a multi-panel figure.

    

    This internal function generates a 6-row by 1-column figure displaying the time 
    traces of individual tracked structures. It plots the following properties 
    top-to-bottom: Radial position, Poloidal position, Area, Angle, Roundness, 
    and Total curvature. It handles both standard and differential properties 
    (which have a one-frame offset) and optionally exports the raw plot data to text files.

    Args:
        exp_id (int or str, optional): The experiment or shot ID.
        time_range (list or tuple, optional): The original [start, end] time range 
            of the calculated data. Used as the default x-axis limits.
        plot_time_range (list or tuple, optional): Specific [start, end] time range 
            to plot, overriding `time_range` for the x-axis limits if provided.
        frame_properties (dict, optional): Dictionary containing the analysis data. 
            Must include 'structures' (tracked data), 'data' (metadata/units for 
            standard properties), and 'derived' (metadata/units for differential properties).
        wd (str, optional): The working directory path where output files (PDFs and 
            data text files) will be saved.
        n_color (int, optional): The total number of distinct colors available in 
            the colortable.
        colortable (list or object, optional): A color sequence used to consistently 
            color distinct structures across all subplots.
        pdf (bool, optional): If True, saves the generated multi-panel figure to 
            a PDF file in the working directory.
        markersize (float, optional): The size of the markers on the line plots. 
            Defaults to 0.5.
        save_data_for_publication (bool, optional): If True, exports the time and 
            property value data for each tracked structure to individual `.txt` files 
            for external plotting. Defaults to False.

    Returns:
        None: The function generates plots, optionally saves files, and closes the 
        matplotlib figures.
    """

    set_matplotlib_for_publication(labelsize=8,
                                    linewidth=0.2,
                                    major_ticksize=2.,
                                    )
    
    #pdf_pages=PdfPages(wd+'/plots/plot_example_frame_results.pdf')
    if pdf:
        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                          time_range=time_range,
                                          working_directory=wd+'/plots',
                                          purpose='example_frame_results',
                                          extension='pdf')
        pdf_pages=PdfPages(filename)

    #return struct_by_struct
    #print('str_by_str_len: ',struct_by_struct)
    
    # ... (Keep your publication formatting and PDF setup here) ...

    labels=['a','b','c','d','e','f']
    fig, axes = plt.subplots(6,1,figsize=(17/2.54,12/2.54))
    
    for ind, key in enumerate(['Position radial fit', 'Position poloidal fit', 
                               'Area', 'Angle fit', 'Roundness', 'Total curvature']):
        ax = axes[ind]
        
        if save_data_for_publication:
            filename = f'{wd}/{labels[ind]}_{exp_id}_{time_range[0]}_{time_range[1]}_{key}_example.txt'
            file1 = open(filename, 'w+')
            
        for ind_str, struct in enumerate(dataset.tracked_structures):
            if len(struct.time) > 0:
                
                if key in struct.differential_parameters:
                    time_vec = np.asarray(struct.time[1:]) * 1e3
                    metric_array = struct.differential_parameters[key]
                elif key in struct.regular_parameters:
                    time_vec = np.asarray(struct.time) * 1e3
                    metric_array = struct.regular_parameters[key]
                else:
                    continue 

                # BUG FIX 1: Apply plot_time_range mask before plotting/saving!
                if plot_time_range is not None:
                    time_mask = (time_vec >= plot_time_range[0]*1e3) & (time_vec <= plot_time_range[1]*1e3)
                else:
                    time_mask = (time_vec >= time_range[0]*1e3) & (time_vec <= time_range[1]*1e3)
                    
                masked_time = time_vec[time_mask]
                masked_data = metric_array.value[time_mask]

                if len(masked_time) > 0:
                    try:
                        ax.plot(masked_time,
                                masked_data,
                                '-o',
                                markersize=markersize,
                                linewidth=0.2,
                                label=str(struct.label),
                                color=colortable[np.mod(int(struct.label)+1, n_color)])
                        
                        if save_data_for_publication:
                            file1.write(f'Structure #{struct.label}:\n')
                            for ind_save, time_point in enumerate(masked_time):
                                file1.write(f"{time_point}\t{masked_data[ind_save]}\n")
                            file1.write('\n')

                    except Exception as e:
                        print(f'Exception in plotting line 1748: {e}')
                    
        # Apply the units natively extracted from the Metric object!
        if metric_array.unit != '':
            ax.set_ylabel(f"{metric_array.plot_label} [{metric_array.unit}]", fontsize=8)
        else:
            ax.set_ylabel(metric_array.plot_label, fontsize=8)
                    
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        if ind < 5:
            ax.xaxis.label.set_visible(False)
            ax.set_xticklabels([])
            
        ax.text(-0.1, 0.9, f"({labels[ind]})", transform=ax.transAxes, size=8)
        ax.set_xlabel('Time [ms]', fontsize=8)
        
        if plot_time_range is not None:
            ax.set_xlim(np.asarray(plot_time_range) * 1e3) # Converted to ms
        else:
            ax.set_xlim(np.asarray(time_range) * 1e3)
            
        if ind == 0: ax.set_ylim([1.43, 1.55])
        if ind == 1: ax.set_ylim([0.07, 0.35])
        
        fig.tight_layout(pad=0.1)
        if save_data_for_publication: file1.close()
        
    if pdf:
       pdf_pages.savefig()
       pdf_pages.close()

def _structure_video_save(sample_0=None,
                         object_name=None,
                         exp_id=None,
                         n_frames=None,
                         d_sep_x=None,
                         d_sep_y=None,
                         d_flux=None,
                         video_resolution=None,
                         levels=None,
                         nocolorbar=None,
                         plot_flux_surfaces=None,
                         plot_separatrix=None,
                         str_finding_method=None,
                         video_framerate=None,
                         dataset=None, # Changed from frame_properties to your new StructureDataset!
                         colortable=None,
                         wd=None,
                         n_color=None,
                         video_start_frame=0):
    
    """
    Generates and saves a video animation of tracked GPI structures over time.

    

    This internal function renders a sequence of frames showing the raw Gas Puff 
    Imaging (GPI) data (median-filtered and contoured), overlaid with the magnetic 
    separatrix, flux surfaces, and identified structures (polygons and fitted ellipses). 
    It captures the matplotlib figure canvas for each frame and encodes them into an 
    MP4 video file using OpenCV.

    Args:
        --- General Inputs ---
        sample_0 (int, optional): The reference starting sample index for the data slice.
        object_name (str, optional): The name of the data object as registered in 
            the `flap` framework.
        exp_id (int or str, optional): The experiment or shot ID, used for the title 
            and the output filename.
        wd (str, optional): The working directory path where the video will be saved 
            (specifically in the `/plots` subdirectory).

        --- Data Inputs ---
        n_frames (int, optional): The total number of frames to iterate through.
        d_sep_x (object, optional): Data object containing separatrix R-coordinates.
        d_sep_y (object, optional): Data object containing separatrix Z-coordinates.
        d_flux (object, optional): Data object containing flux surface data.
        time (array-like, optional): Array of time values corresponding to each frame, 
            used for timestamping the video and the filename.
        frame_properties (dict, optional): Dictionary containing the tracked 
            'structures' data for each frame.

        --- Plot Formatting ---
        levels (int or array-like, optional): Contour levels for the background 
            data plot. If None, defaults to 51.
        nocolorbar (bool, optional): If True, suppresses the colorbar on the plot.
        plot_flux_surfaces (bool, optional): If True, overlays the flux surfaces 
            onto the frames.
        plot_separatrix (bool, optional): If True, overlays the magnetic separatrix.
        str_finding_method (str, optional): The name of the structure finding method 
            used (e.g., 'watershed'), appended to the output video filename.
        colortable (list or object, optional): A color sequence used to color distinct 
            structures based on their tracking label.
        n_color (int, optional): The total number of distinct colors in the colortable.

        --- Video Settings ---
        video_resolution (tuple or list, optional): The (width, height) resolution of 
            the output figure.
        video_framerate (int or float, optional): The frames per second (FPS) for the 
            encoded MP4 video.
        video_start_frame (int, optional): The frame offset to begin the video 
            rendering. Defaults to 0.

    Returns:
        None: The function saves an '.mp4' file to the working directory and 
        releases the video writer resources.
    """

    import scipy

    set_matplotlib_for_publication(labelsize=32,
                                   linewidth=2,
                                   major_ticksize=15.)

    slicing_frame={'Sample':sample_0}

    frame=flap.slice_data(object_name,
                          exp_id=exp_id,
                          slicing=slicing_frame,
                          output_name='GPI_FRAME')
                          
    x_coord_name, x_unit_name = 'Device R', '[m]'
    y_coord_name, y_unit_name = 'Device z', '[m]'

    x_coord=frame.coordinate(x_coord_name)[0]
    y_coord=frame.coordinate(y_coord_name)[0]
    
    # Initialize the video writer variable safely
    video = None

    for i_frames in range(video_start_frame, n_frames):
        slicing_frame={'Sample':sample_0+i_frames}

        frame=flap.slice_data(object_name,
                              exp_id=exp_id,
                              slicing=slicing_frame,
                              output_name='GPI_FRAME')

        frame.data = np.asarray(frame.data, dtype='float64')
        frame.data = scipy.ndimage.median_filter(frame.data, 5)

        slicing = {'Time':frame.coordinate('Time')[0][0,0]}
        
        if d_sep_x is not None and d_sep_y is not None:
            d_sep_x_sliced=d_sep_x.slice_data(slicing=slicing)
            d_sep_y_sliced=d_sep_y.slice_data(slicing=slicing)
    
            separatrix_data=np.zeros([d_sep_x_sliced.shape[0],2])
            separatrix_data[:,0]=d_sep_x_sliced.data
            separatrix_data[:,1]=d_sep_y_sliced.data
        else:
            separatrix_data = None
            
        surface_data_obj = d_flux.slice_data(slicing=slicing) if d_flux is not None else None

        my_dpi=80
        fig, ax = plt.subplots(figsize=(video_resolution[0]/my_dpi, video_resolution[1]/my_dpi), dpi=my_dpi)

        plt.contourf(x_coord, y_coord, frame.data, levels=51 if levels is None else levels)
        
        if not nocolorbar:
            plt.colorbar()
            
        if plot_flux_surfaces and surface_data_obj is not None:
            plt.contour(surface_data_obj.coordinate(x_coord_name)[0],
                        surface_data_obj.coordinate(y_coord_name)[0],
                        surface_data_obj.data.transpose(),
                        linewidth=0.5,
                        cmap='gist_ncar',
                        levels=51)
                        
        if plot_separatrix and separatrix_data is not None:
            plt.plot(separatrix_data[:,0], separatrix_data[:,1], linewidth=2, color='red')

        ax.set_aspect(1.0)

        # --- NATIVE OOP EXTRACTION ---
        structures = dataset.frames[i_frames]
        current_time = dataset.frame_times[i_frames]

        if structures is not None and len(structures) > 0:
            R = np.arange(0, 2*np.pi, 0.01)
            
            for struct in structures:
                # If it has a valid fit angle, we can plot the geometric shape!
                if not np.isnan(struct.fit_angle.value):

                    phi = struct.fit_angle.value
                    a, b = struct.fit_axes_length[1].value, struct.fit_axes_length[0].value # major, minor
                    cx, cy = struct.fit_center[0].value, struct.fit_center[1].value

                    x_polygon, y_polygon = struct.x, struct.y

                    x_ellipse = cx + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
                    y_ellipse = cy + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)

                    # Safely handle missing labels for coloring
                    color_idx = int(np.mod(struct.label, n_color)) if struct.label is not None else 0

                    _plot_ellipses_centers(ax,
                                           x_polygon, y_polygon,
                                           x_ellipse, y_ellipse,
                                           struct,
                                           polygon_color=colortable[color_idx],
                                           ellipse_color=colortable[color_idx],
                                           polygon_linewidth=3,
                                           ellipse_linewidth=1.5)

        ax.set_xlabel(x_coord_name.replace('Device ','') + ' '+ x_unit_name)
        ax.set_ylabel(y_coord_name.replace('Device ','') + ' '+ y_unit_name)
        ax.set_title(f"{exp_id} @ {current_time:.6f}")
        
        plt.xlim([x_coord.min(), x_coord.max()])
        plt.ylim([y_coord.min(), y_coord.max()])

        plt.title(f"{exp_id} @ {current_time * 1e3:.3f}ms")
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()

        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            if buf.shape[0] == h*2 * w*2 * 3:
                buf.shape = (h*2, w*2, 3)
            else:
                buf.shape = (h, w, 3)
                
            buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            
            # Create the video writer on the first successful pass
            if video is None:
                height, width = buf.shape[0], buf.shape[1]
                print(f'Canvas size is: {w} x {h}')
                print(f'Video resolution is: {width} x {height}')
                
                video_codec_code = 'mp4v'
                start_time, end_time = dataset.frame_times[0], dataset.frame_times[-1]
                filename = f"{wd}/plots/NSTX_GPI_{exp_id}_{start_time*1e3:.3f}_{end_time*1e3:.3f}_fit_structures_{str_finding_method}.mp4"
                
                video = cv2.VideoWriter(filename,
                                        cv2.VideoWriter_fourcc(*video_codec_code),
                                        float(video_framerate),
                                        (width, height),
                                        isColor=True)
                                        
            video.write(buf)
        except Exception as e:
            print(f'Video frame cannot be saved. Passing... ({e})')
        
        plt.close(fig)

    cv2.destroyAllWindows()
    if video is not None:
        video.release()

def _plot_str_by_str(dataset=None,
                     plot_scatter=True,
                     figsize=None,
                     colortable=None,
                     n_color=None,
                     time_range=None,
                     plot_for_publication=False,
                     pdf=False,
                     pdf_pages=None,
                     ):

    """
    Generates and saves individual time-series plots for each tracked property.

    

    This internal function creates a separate plot for every property defined in 
    the analyzed keys (e.g., Area, Angle, Position). For each property, it plots 
    the time evolution of every tracked structure on the same axes, differentiating 
    them by color. It handles both standard properties and differential properties 
    (which have one less time step) and can append the figures to an open PDF object.

    Args:
        frame_properties (dict, optional): The main data dictionary containing 
            tracked 'structures', 'data' (metadata for standard properties), and 
            'derived' (metadata for differential properties).
        plot_scatter (bool, optional): If True, plots lines with scatter markers 
            ('-o'). If False, plots solid lines ('-'). Defaults to True.
        figsize (tuple, optional): The (width, height) dimensions for the generated 
            matplotlib figures.
        colortable (list or object, optional): A sequence of colors used to assign 
            consistent colors to individual structures based on their index.
        n_color (int, optional): The total number of unique colors available in 
            the `colortable`.
        time_range (list or tuple, optional): The original [start, end] time range 
            used to set the x-axis limits (in seconds, converted to ms for plotting).
        plot_for_publication (bool, optional): If True, enforces a golden ratio 
            aspect ratio on the plot axes for publication-ready formatting. 
            Defaults to False.
        pdf (bool, optional): If True, saves each generated figure to the provided 
            `pdf_pages` object. Defaults to False.
        pdf_pages (matplotlib.backends.backend_pdf.PdfPages, optional): An open 
            PDF multi-page object where the plots will be saved.

    Returns:
        None: The function modifies the `pdf_pages` object in place or displays 
        figures interactively.
    """
    
    
    
# 1. We no longer need transform_frames_to_structures!
    # Just grab the list of tracked objects directly from the dataset.
    tracked_structs = dataset.tracked_structures
    
    if plot_scatter:
        linestyle = '-o'
    else:
        linestyle = '-'
        
    if not tracked_structs:
        print("No tracked structures found to plot.")
        return

    # Use the first structure to get the available keys
    first_struct = tracked_structs[0]
    param_groups = [
        ('Regular parameters', first_struct.regular_parameters.keys()),
        ('Differential parameters', first_struct.differential_parameters.keys())
    ]

    for param_type, keys in param_groups:
        for key in keys:
            
            fig, ax = plt.subplots(figsize=figsize)

            # BUG FIX 2: Safely extract the metric reference BEFORE the loop
            # so we are guaranteed to have the correct labels/units!
            if param_type == 'Regular parameters':
                metric_ref = first_struct.regular_parameters.get(key)
            else:
                metric_ref = first_struct.differential_parameters.get(key)
            
            for ind_str, struct in enumerate(tracked_structs):
                if len(struct.time) > 0:
                    
                    if param_type == 'Regular parameters':
                        if key not in struct.regular_parameters: continue
                        x_data = np.asarray(struct.time) * 1e3 
                        metric_array = struct.regular_parameters[key]
                    else:
                        if key not in struct.differential_parameters: continue
                        x_data = np.asarray(struct.time[1:]) * 1e3 
                        metric_array = struct.differential_parameters[key]
                    # Capture the reference metric ONCE, dynamically!
                    if metric_ref is None:
                        metric_ref = metric_array
                        
                    try:
                        ax.plot(x_data,
                                metric_array.value,
                                linestyle,
                                label=str(struct.label),
                                markersize=5,
                                color=colortable[np.mod(int(struct.label) + 1, n_color)])
                                
                    except Exception as e:
                        print(f'Exception in analyze_gpi_structures at line 2132: {e}')
            
            # --- Native OOP Labeling! ---
            if metric_ref is not None:
                if metric_ref.unit != '':
                    ax.set_ylabel(f"{metric_ref.plot_label} [{metric_ref.unit}]")
                else:
                    ax.set_ylabel(f"{metric_ref.plot_label}")
            else:
                ax.set_ylabel(key) # Safe fallback

            ax.set_xlabel('Time [ms]')
            ax.set_xlim(np.asarray(time_range) * 1e3)
            ax.set_title(f"{key} vs. time")
                
            fig.tight_layout(pad=0.1)

    
            if pdf:
                pdf_pages.savefig()
            plt.close(fig)