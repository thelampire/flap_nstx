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
                           identify_structures, track_structures, 
                           calculate_differential_structure_keys,
                           _plot_ellipses_centers)
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

#Importing and setting up the FLAP environment
import flap
import flap_nstx
flap_nstx.register()

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)


def analyze_gpi_structures(exp_id=None, time_range=None, data_object=None, 
                           x_range=None, y_range=None, normalize='simple',
                           normalize_f_kernel='Elliptic', normalize_f_high=1e3,
                           str_finding_method='watershed', ignore_side_structures=False,
                           ellipse_method='linalg', fit_shape='ellipse', subtraction_order=None, 
                           remove_interlaced_structures=True, nlevel=51, filter_level=5, 
                           global_levels=False, levels=None, threshold_method='variance', 
                           threshold_coeff=1.0, threshold_bg_range={'x':[54,65], 'y':[0,79]},
                           threshold_bg_multiplier=2., weighting='intensity', maxing='intensity', 
                           prev_str_weighting='intensity', str_size_lower_thres=0.00375*4, 
                           elongation_threshold=0.1, tracking='weighted', tracking_assignment='max_score', 
                           max_gap=1, smooth_contours=5, remove_orphans=True, min_structure_lifetime=10, 
                           calculate_rough_diff_velocities=False, structure_pixel_calc=False, 
                           score_threshold=0.7, matrix_weight={'iou':1,'cccf':0}, plot=True, 
                           pdf=False, plot_error=False, error_window=4., overplot_average=True, 
                           plot_tracking=True, plot_scatter=False, structure_video_save=False, 
                           video_start_frame=0, video_resolution=(1024,1024), video_framerate=24, 
                           nocolorbar=False, structure_pdf_save=False, plot_separatrix=True, 
                           plot_flux_surfaces=True, plot_time_range=None, plot_for_publication=False, 
                           plot_vertical_line_at=None, plot_str_by_str=False, plot_watershed_steps=False, 
                           plot_example_structure_frames=False, plot_example_frames_results=False, 
                           plot_nframe=None, plot_ncol=None, linewidth=None, filename=None, 
                           save_results=True, nocalc=True, recalc_tracking=False, return_results=False, 
                           return_pixel_displacement=False, cache_data=True, test=False, 
                           test_structures=False, test_histogram=False, save_data_for_publication=False, 
                           verbose=False, skip_mdsplus=False):
    
    """
    Analyzes Gas Puff Imaging (GPI) structures in plasma physics data.

    This function performs comprehensive structure identification, size processing,
    tracking, and velocity calculation on GPI data. It handles data normalization,
    contour/watershed segmentation, tracking structures over time, and generating 
    various diagnostic plots and video outputs.
    """

    # Input error handling
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
            filename_pdf=flap_nstx.tools.filename(exp_id=exp_id,
                                              working_directory=wd+'/plots',
                                              time_range=time_range,
                                              purpose='found structures',
                                              comment=comment,
                                              extension='pdf')
            pdf_structures=PdfPages(filename_pdf)
            
        # READ DATA
        if data_object is None:
            print("\n------- Reading NSTX GPI data --------")
            if cache_data:
                try:
                    data=flap.get_data_object('GPI',exp_id=exp_id)
                except:
                    print('Data is not cached, it needs to be read.')
                    data=flap.get_data('NSTX_GPI',exp_id=exp_id, name='', object_name='GPI')
            else:
                data=flap.get_data('NSTX_GPI',exp_id=exp_id, name='', object_name='GPI')
                
            if x_range is None or y_range is None:
                x_range=[0, data.data.shape[1]-1]
                y_range=[0, data.data.shape[2]-1]

            slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                     'Image x':flap.Intervals(x_range[0],x_range[1]),
                     'Image y':flap.Intervals(y_range[0],y_range[1])}

            data=flap.slice_data('GPI', exp_id=exp_id, slicing=slicing, output_name='GPI_SLICED_FULL')

        elif isinstance(data_object, str):
            if exp_id is None:
                exp_id='*'

            data=flap.get_data_object(data_object, exp_id=exp_id)
            time_range=[data.coordinate('Time')[0][0,0,0], data.coordinate('Time')[0][-1,0,0]]
            exp_id=data.exp_id
            object_name='GPI_SLICED_FULL'
            flap.add_data_object(data, object_name)

            if x_range is None: x_range=[0, data.data.shape[1]-1]
            if y_range is None: y_range=[0, data.data.shape[2]-1]

        elif isinstance(data_object, flap.DataObject):
            data=copy.deepcopy(data_object)
            object_name='GPI'
            flap.add_data_object(data, object_name)

            if x_range is None: x_range=[0, data.data.shape[1]-1]
            if y_range is None: y_range=[0, data.data.shape[2]-1]
            if time_range is None:
                time_range=[data.coordinate('Time')[0][:,0,0].min(),
                            data.coordinate('Time')[0][:,0,0].max()]
        else:
            raise TypeError(f"Invalid data_object type: {type(data_object)}")

        # NORMALIZATION PROCESS
        if normalize is not None and data_object is None:

            slicing_for_filtering=copy.deepcopy(slicing)
            slicing_for_filtering['Time']=flap.Intervals(time_range[0]-1/normalize_f_high*10,
                                                         time_range[1]+1/normalize_f_high*10)

            slicing_time_only={'Time':flap.Intervals(time_range[0], time_range[1])}

            flap.slice_data('GPI', exp_id=exp_id, slicing=slicing_for_filtering, output_name='GPI_SLICED_FOR_FILTERING')
            object_name='GPI_SLICED_FOR_FILTERING'
            coefficient=normalize_gpi(object_name,
                                      exp_id=exp_id,
                                      slicing_time=slicing_time_only,
                                      normalize=normalize,
                                      normalize_f_high=normalize_f_high,
                                      normalize_f_kernel=normalize_f_kernel,
                                      normalizer_object_name='GPI_LPF_INTERVAL',
                                      output_name='GPI_GAS_CLOUD')

            data_obj=flap.get_data_object('GPI_SLICED_FULL', exp_id=exp_id)
            data_obj.data = data_obj.data/coefficient
            flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_STR_SIZE')
            object_name='GPI_SLICED_DENORM_STR_SIZE'

        if subtraction_order is not None:
            if verbose: print("*** Subtracting the trend of the data ***")
            data=detrend_multidim(object_name, exp_id=exp_id, order=subtraction_order,
                                  coordinates=['Image x', 'Image y'], output_name='GPI_DETREND_STR_SIZE')
            object_name='GPI_DETREND_STR_SIZE'

        if global_levels:
            if levels is None:
                data=flap.get_data_object_ref(object_name)
                min_data=data.data.min()
                max_data=data.data.max()
                levels=np.arange(nlevel)/(nlevel-1)*(max_data-min_data)+min_data

        if threshold_method == 'variance':
            thres_obj_str_size=flap.slice_data(object_name, exp_id=exp_id,
                                               summing={'Image x':'Mean', 'Image y':'Mean'},
                                               output_name='GPI_SLICED_TIMETRACE')
            intensity_thres_level_str_size=np.sqrt(np.var(thres_obj_str_size.data))*threshold_coeff+np.mean(thres_obj_str_size.data)

        if threshold_method == 'background_average':
            intensity_thres_level_str_size=threshold_bg_multiplier*np.mean(flap.slice_data(object_name,
                                                                             slicing={'Image x':flap.Intervals(threshold_bg_range['x'][0], threshold_bg_range['x'][1]),
                                                                                      'Image y':flap.Intervals(threshold_bg_range['y'][0], threshold_bg_range['y'][1])}).data)
        
        # VARIABLE DEFINITION
        time_dim=data.get_coordinate_object('Time').dimension_list[0]
        n_frames=data.data.shape[time_dim]
        time=data.coordinate('Time')[0][:,0,0]
        sample_0=flap.get_data_object_ref('GPI_SLICED_FULL', exp_id=exp_id).coordinate('Sample')[0][0,0,0]
        
        if plot_flux_surfaces or plot_separatrix:
            try:
                if plot_separatrix:
                    d_sep_x=flap.get_data('NSTX_MDSPlus', name=r'\EFIT02::\RBDRY', exp_id=exp_id, object_name='SEP X OBJ')
                    d_sep_y=flap.get_data('NSTX_MDSPlus', name=r'\EFIT02::\ZBDRY', exp_id=exp_id, object_name='SEP Y OBJ')
                else:
                    d_sep_x, d_sep_y = None, None

                if plot_flux_surfaces:
                    d_flux=flap.get_data('NSTX_MDSPlus', name=r'\EFIT02::\PSIRZ', exp_id=exp_id, object_name='PSI RZ OBJ')
                else:
                    d_flux=None
            except Exception as e:
                print('Exception occurred in analyze_gpi_structures.py at EFIT loading.')
                print(e)
                d_sep_x, d_sep_y, d_flux = None, None, None
        
        if not ((structure_video_save or plot_example_structure_frames) and nocalc):
            
            raw_dataset = StructureDataset(mode='untracked', exp_id=exp_id)
            
            if test or test_structures or structure_pdf_save:
                fig_dpi=80
                plt.figure(figsize=(800/fig_dpi, 600/fig_dpi), dpi=fig_dpi)

            if not skip_mdsplus and data_object is None:
                elm_time=(time[-1]-time[0])/2
                try:
                    R_sep=flap.get_data('NSTX_MDSPlus', name='\EFIT02::\RBDRY', exp_id=exp_id, object_name='SEP R OBJ').slice_data(slicing={'Time':elm_time}).data
                    z_sep=flap.get_data('NSTX_MDSPlus', name='\EFIT02::\ZBDRY', exp_id=exp_id, object_name='SEP Z OBJ').slice_data(slicing={'Time':elm_time}).data
                    
                    coeff_r=np.asarray([3.75, 0,    1402.8097])/1000. 
                    coeff_z=np.asarray([0,    3.75, 70.544312])/1000. 
                    
                    z_bound_upper = coeff_z[2] + 79*coeff_z[0] + 64*coeff_z[1]
                    sep_GPI_ind = np.where((R_sep > coeff_r[2]) & (z_sep > coeff_z[2]) & (z_sep < z_bound_upper))
                    
                    sep_GPI_ind=np.asarray(sep_GPI_ind[0])
                    sep_GPI_ind=np.insert(sep_GPI_ind,0,sep_GPI_ind[0]-1)
                    sep_GPI_ind=np.insert(sep_GPI_ind,len(sep_GPI_ind),sep_GPI_ind[-1]+1)
                
                except Exception as e:
                    print(e)
                    print('\n Could not read EFIT data. Setting separatrix data to None')


            for i_frames in range(n_frames):
                print(f"\r{i_frames/(n_frames-1)*100.:.1f}% done from the calculation.", end="", flush=True)

                slicing_frame={'Sample':sample_0+i_frames}
                frame=flap.slice_data(object_name, exp_id=exp_id, slicing=slicing_frame, output_name='GPI_FRAME')
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

            with open(pickle_filename, 'wb') as f:
                pickle.dump(raw_dataset,f)
            if test:
                plt.close()
        else:
            print('\n\n--- Loading data from the pickle file ---')
            with open(pickle_filename, 'rb') as f:
                raw_dataset=pickle.load(f)
    else:
        print('\n\n--- Loading data from the pickle file ---')
        with open(pickle_filename, 'rb') as f:
            raw_dataset=pickle.load(f)
        
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
    
    tracked_dataset = calculate_differential_structure_keys(tracked_dataset)
    
    """
    PLOTTING THE RESULTS
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
                                       dataset=raw_dataset,
                                       plot_example_structure_frames=plot_example_structure_frames,
                                       plot_separatrix=plot_separatrix,
                                       separatrix_coordinates=(d_sep_x,d_sep_y),
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
    """
    
    if plot_results:
        import matplotlib
        matplotlib.use('QT5Agg')
    else:
        import matplotlib
        matplotlib.use('agg')

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
        figsize=(8.5/2.54, 8.5/2.54/1.618*1.1)
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
                                   dataset=None,
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
                if not np.isnan(struct.fit_angle):
                    
                    # Extracted the raw values smoothly!
                    phi = struct.fit_angle
                    a, b = struct.fit_axes_length[1], struct.fit_axes_length[0] # major, minor
                    cx, cy = struct.fit_center[0], struct.fit_center[1]

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
    """

    set_matplotlib_for_publication(labelsize=8,
                                    linewidth=0.2,
                                    major_ticksize=2.,
                                    )
    
    if pdf:
        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                          time_range=time_range,
                                          working_directory=wd+'/plots',
                                          purpose='example_frame_results',
                                          extension='pdf')
        pdf_pages=PdfPages(filename)

    labels=['a','b','c','d','e','f']
    fig, axes = plt.subplots(6,1,figsize=(17/2.54,12/2.54))
    
    if not dataset.tracked_structures:
        print("No tracked structures found for example frame plotting.")
        return

    for ind, key in enumerate(['Position radial fit', 'Position poloidal fit', 
                               'Area', 'Angle fit', 'Roundness', 'Total curvature']):
        ax = axes[ind]
        
        if save_data_for_publication:
            filename = f'{wd}/{labels[ind]}_{exp_id}_{time_range[0]}_{time_range[1]}_{key}_example.txt'
            file1 = open(filename, 'w+')

        # SAFELY EXTRACT THE METRIC REFERENCE FIRST
        first_struct = dataset.tracked_structures[0]
        metric_ref = first_struct.differential_parameters.get(key) or first_struct.regular_parameters.get(key)
        if metric_ref is None: continue
            
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

                # Apply plot_time_range mask
                if plot_time_range is not None:
                    time_mask = (time_vec >= plot_time_range[0]*1e3) & (time_vec <= plot_time_range[1]*1e3)
                else:
                    time_mask = (time_vec >= time_range[0]*1e3) & (time_vec <= time_range[1]*1e3)
                    
                masked_time = time_vec[time_mask]
                masked_data = metric_array.value[time_mask]

                if len(masked_time) > 0:
                    try:
                        ax.plot(masked_time, masked_data, '-o', markersize=markersize, linewidth=0.2,
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
        if metric_ref.unit != '':
            ax.set_ylabel(f"{metric_ref.plot_label} [{metric_ref.unit}]", fontsize=8)
        else:
            ax.set_ylabel(metric_ref.plot_label, fontsize=8)
                    
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        if ind < 5:
            ax.xaxis.label.set_visible(False)
            ax.set_xticklabels([])
            
        ax.text(-0.1, 0.9, f"({labels[ind]})", transform=ax.transAxes, size=8)
        ax.set_xlabel('Time [ms]', fontsize=8)
        
        if plot_time_range is not None:
            ax.set_xlim(np.asarray(plot_time_range) * 1e3) 
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
                         dataset=None,
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
                if not np.isnan(struct.fit_angle):

                    phi = struct.fit_angle
                    a, b = struct.fit_axes_length[1], struct.fit_axes_length[0] # major, minor
                    cx, cy = struct.fit_center[0], struct.fit_center[1]

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
                import cv2
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
    """
    
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

            # SAFELY EXTRACT THE METRIC REFERENCE BEFORE THE LOOP
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
                        
                    if metric_ref is None:
                        metric_ref = metric_array
                        
                    try:
                        # BUG FIX: Safely unpack the values from the MetricArray using .value
                        ax.plot(x_data,
                                metric_array.value,
                                linestyle,
                                label=str(struct.label),
                                markersize=5,
                                color=colortable[np.mod(int(struct.label) + 1, n_color)])
                                
                    except Exception as e:
                        print(f'Exception in analyze_gpi_structures at _plot_str_by_str: {e}')
            
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
        
        
def validate_structure(struct, x_coord, y_coord, elongation_threshold, str_size_lower_thres, str_size_upper_thres, ignore_side_structure):
    """Filters out invalid structures based on size, elongation, and boundary constraints."""
    
    # 1. NaN Check 
    if np.isnan(struct.fit_size[0]) or np.isnan(struct.fit_size[1]):
        struct.set_invalid()
        return False

    # 2. Elongation check (safely neutralize angle if it's too round)
    if struct.fit_elongation < elongation_threshold:
        struct._angle = np.nan 
        struct.update_regular_parameters()
        

    # 3. Size & Boundary Edge check 
    if str_size_lower_thres is not None and not np.isnan(struct.fit_size[1]):
        # Check lower/upper bounds
        if (struct.fit_size[0] < str_size_lower_thres or 
            struct.fit_size[1] < str_size_lower_thres or
            struct.fit_size[0] > str_size_upper_thres or 
            struct.fit_size[1] > str_size_upper_thres):
            struct.set_invalid()
            return False
            
        # Check edge touching
        if ignore_side_structure and (np.any(struct.x_data == x_coord.min()) or 
                                      np.any(struct.x_data == x_coord.max()) or
                                      np.any(struct.y_data == y_coord.min()) or 
                                      np.any(struct.y_data == y_coord.max()) or
                                      struct.fit_center[0] < x_coord.min() or 
                                      struct.fit_center[0] > x_coord.max() or
                                      struct.fit_center[1] < y_coord.min() or 
                                      struct.fit_center[1] > y_coord.max()):
            struct.set_invalid()
            return False
            
    return True # It survived all checks!