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

from flap_nstx.gpi import normalize_gpi, identify_structures, track_structures
from flap_nstx.tools import detrend_multidim, set_matplotlib_for_publication
from flap_nstx.tools import fringe_jump_correction

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

wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
#Constants for the calculation
#Using the spatial calibration to find the actual velocities.
coeff_r=np.asarray([3.75, 0,    1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
coeff_z=np.asarray([0,    3.75, 70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm

# Originally used coordinates for reference. (Vertical, radial geometrical coordinates)
# coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
# coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm


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
                           smooth_contours=5,                    #Smooths contours with the corner cutting technique this many times.
                           remove_orphans=True,                  #Structures which "live" shorter than min_structure_lifetime
                           min_structure_lifetime=10,            #are cut out from the calculation
                           calculate_rough_diff_velocities=False,#Calculate velocities from average or maximum structures (deprecated)
                           structure_pixel_calc=False,           #Calculate and plot the structure sizes in pixels

                           score_threshold=0.7,                  #Threshold for tracking of the structures based on the weighted tracking.
                           matrix_weight={'iou':1,'cccf':0},     #Weights for the tracking matrix. iou: intersection over union, cccf cross correlation coefficient funcion
                           #Fixing incorrect calculations
                           fix_structure_angles=False,
                           
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
    Calculate frame by frame average frame velocity of the NSTX GPI signal. The
    code takes subsequent frames, calculates the 2D correlation function between
    the two and finds the maximum. Based on the pixel shift and the sampling
    time of the signal, the radial and poloidal velocity is calculated.
    The code assumes that the structures present in the subsequent frames are
    propagating with the same velocity. If there are multiple structures
    propagating in e.g. different direction or with different velocities, their
    effects are averaged over.
    """

    #Input error handling
    if exp_id is None and data_object is None:
        raise ValueError('Either exp_id or data_object needs to be set for the calculation.')

    if data_object is None:
        if time_range is None and filename is None:
            raise ValueError('It takes too much time to calculate the entire shot, please set a time_range.')
        else:
            if type(time_range) is not list and filename is None:
                raise TypeError('time_range is not a list.')
            if filename is None and len(time_range) != 2:
                raise ValueError('time_range should be a list of two elements.')

    if weighting not in ['number', 'area', 'intensity']:
        raise ValueError("Weighting can only be by the 'number', 'area' or 'intensity' of the structures.")
    if maxing not in ['area', 'intensity']:
        raise ValueError("Maxing can only be by the 'area' or 'intensity' of the structures.")

    """
    SETTING UP THE FILENAME FOR DATA SAVING
    """

    if filename is None:
        comment=''
        if normalize is not None:
            comment+=normalize

        if remove_interlaced_structures:
            comment+='_nointer'
        comment+='_'+str_finding_method
        if data_object is not None:
            try:
                if type(data_object) == type(flap.DataObject):
                    exp_id=data_object.exp_id
                elif type(data_object) == str:
                    exp_id=flap.get_data_object_ref(data_object).exp_id
            except:
                exp_id=0

        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                          working_directory=wd+'/processed_data',
                                          time_range=time_range,
                                          purpose='structure char',
                                          comment=comment)

        filename_was_none=True
    else:
        filename_was_none=False

    plot_results=plot

    # if plot_for_publication:
    #     set_matplotlib_for_publication(labelsize=9,
    #                                    linewidth=0.5,
    #                                    major_ticksize=2.,
    #                                    )

    fit_shape=fit_shape.capitalize()

    pickle_filename=filename+'.pickle'
    
    if os.path.exists(pickle_filename) and nocalc:
        try:
            print('Loading '+pickle_filename)
            pickle.load(open(pickle_filename, 'rb'))
        except:
            print('The pickle file cannot be loaded. Recalculating the results.')
            nocalc=False
    elif nocalc:
        print(pickle_filename)
        print('The pickle file cannot be loaded. Recalculating the results.')
        nocalc=False

    if ((not test and not test_structures) or
        (not test and not plot_results and structure_pdf_save and test_structures)):
        import matplotlib
        matplotlib.use('agg')

    if not nocalc or structure_video_save or plot_example_structure_frames:
        if plot_flux_surfaces or plot_separatrix:
            try:
                if plot_separatrix:
                    d_sep_x=flap.get_data('NSTX_MDSPlus',
                                          name='\EFIT02::\RBDRY',
                                          exp_id=exp_id,
                                          object_name='SEP X OBJ'
                                          )

                    d_sep_y=flap.get_data('NSTX_MDSPlus',
                                          name='\EFIT02::\ZBDRY',
                                          exp_id=exp_id,
                                          object_name='SEP Y OBJ'
                                          )
                else:
                    d_sep_x=None
                    d_sep_y=None

                if plot_flux_surfaces:
                    d_flux=flap.get_data('NSTX_MDSPlus',
                                         name='\EFIT02::\PSIRZ',
                                         exp_id=exp_id,
                                         object_name='PSI RZ OBJ'
                                         )
                else:
                    d_flux=None
            except Exception as e:
                print('Exception occurred in analyze_gpi_structures.py at line 254.')
                print(e)

        if structure_pdf_save:
            filename=flap_nstx.tools.filename(exp_id=exp_id,
                                              working_directory=wd+'/plots',
                                              time_range=time_range,
                                              purpose='found structures',
                                              comment=comment,
                                              extension='pdf')
            pdf_structures=PdfPages(filename)
        """
        READING THE DATA
        """
        #Read data
        if data_object is None:
            print("\n------- Reading NSTX GPI data --------")
            if cache_data:
                try:
                    d=flap.get_data_object('GPI',exp_id=exp_id)
                except:
                    print('Data is not cached, it needs to be read.')
                    d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                    name='',
                                    object_name='GPI')
            else:
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                name='',
                                object_name='GPI')
            if x_range is None or y_range is None:
                x_range=[0, d.data.shape[1]-1]
                y_range=[0, d.data.shape[2]-1]

            slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                     'Image x':flap.Intervals(x_range[0],x_range[1]),
                     'Image y':flap.Intervals(y_range[0],y_range[1])}

            d=flap.slice_data('GPI',
                              exp_id=exp_id,
                              slicing=slicing,
                              output_name='GPI_SLICED_FULL')

        elif type(data_object) == str:
            if exp_id is None:
                exp_id='*'

            d=flap.get_data_object(data_object,
                                   exp_id=exp_id)
            time_range=[d.coordinate('Time')[0][0,0,0],
                        d.coordinate('Time')[0][-1,0,0]]
            exp_id=d.exp_id
            object_name='GPI_SLICED_FULL'
            flap.add_data_object(d, object_name)

            if x_range is None:
                x_range=[0, d.data.shape[1]-1]

            if y_range is None:
                y_range=[0, d.data.shape[2]-1]


        elif type(data_object) == type(flap.DataObject()):
            d=copy.deepcopy(data_object)
            object_name='GPI'
            flap.add_data_object(d, object_name)

            if x_range is None:
                x_range=[0, d.data.shape[1]-1]

            if y_range is None:
                y_range=[0, d.data.shape[2]-1]

            if time_range is None:
                time_range=[d.coordinate('Time')[0][:,0,0].min(),
                            d.coordinate('Time')[0][:,0,0].max()]
        else:
            raise TypeError('Data object should be of type str or flap.DataObject and not '+str(type(data_object)))

        """
        NORMALIZATION PROCESS
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
            d=detrend_multidim(object_name,
                               exp_id=exp_id,
                               order=subtraction_order,
                               coordinates=['Image x',
                                            'Image y'],
                               output_name='GPI_DETREND_STR_SIZE')

            object_name='GPI_DETREND_STR_SIZE'

        if global_levels:
            if levels is None:
                d=flap.get_data_object_ref(object_name)
                min_data=d.data.min()
                max_data=d.data.max()
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
            VARIABLE DEFINITION
        """
        #Calculate correlation between subsequent frames in the data
        #Setting the variables for the calculation
        time_dim=d.get_coordinate_object('Time').dimension_list[0]
        n_frames=d.data.shape[time_dim]
        time=d.coordinate('Time')[0][:,0,0]
        sample_time=time[1]-time[0]
        sample_0=flap.get_data_object_ref('GPI_SLICED_FULL',
                                          exp_id=exp_id).coordinate('Sample')[0][0,0,0]

        if not ((structure_video_save or plot_example_structure_frames) and nocalc):
            coordinate_names=[d.coordinates[i].unit.name for i in range(len(d.coordinates))]
            distance_unit='pix'
            time_unit='sample'
            for ind in range(len(coordinate_names)):
                if coordinate_names[ind] == 'Time':
                    time_unit=d.coordinates[ind].unit.unit
                if coordinate_names[ind] == 'Device R':
                    distance_unit=d.coordinates[ind].unit.unit

            frame_properties=frame_properties_dict(exp_id,time, time_unit,distance_unit)

            #Inicializing for frame handling
            frame=None
            structures_dict=None

            if test or test_structures or structure_pdf_save:
                fig_dpi=80
                plt.figure(figsize=(800/fig_dpi, 600/fig_dpi), dpi=fig_dpi)

            if not skip_mdsplus and data_object is None:
                elm_time=(frame_properties['Time'][-1]+frame_properties['Time'][0])/2

                R_sep=flap.get_data('NSTX_MDSPlus',
                                    name='\EFIT02::\RBDRY',
                                    exp_id=exp_id,
                                    object_name='SEP R OBJ').slice_data(slicing={'Time':elm_time}).data

                z_sep=flap.get_data('NSTX_MDSPlus',
                                    name='\EFIT02::\ZBDRY',
                                    exp_id=exp_id,
                                    object_name='SEP Z OBJ').slice_data(slicing={'Time':elm_time}).data

                sep_GPI_ind=np.where(np.logical_and(R_sep > coeff_r[2],
                                                    np.logical_and(z_sep > coeff_z[2],
                                                                   z_sep < coeff_z[2]+79*coeff_z[0]+64*coeff_z[1])))
                sep_GPI_ind=np.asarray(sep_GPI_ind[0])
                sep_GPI_ind=np.insert(sep_GPI_ind,0,sep_GPI_ind[0]-1)
                sep_GPI_ind=np.insert(sep_GPI_ind,len(sep_GPI_ind),sep_GPI_ind[-1]+1)

                z_sep_GPI=z_sep[(sep_GPI_ind)]
                R_sep_GPI=R_sep[sep_GPI_ind]
                GPI_z_vert=coeff_z[0]*np.arange(80)/80*64+coeff_z[1]*np.arange(80)+coeff_z[2]
                R_sep_GPI_interp=np.interp(GPI_z_vert,
                                           np.flip(z_sep_GPI),
                                           np.flip(R_sep_GPI))
                z_sep_GPI_interp=GPI_z_vert

            for i_frames in range(n_frames):

                print(str(int(i_frames/(n_frames-1)*100.))+"% done from the calculation.")

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
                                                      save_data_for_publication=save_data_for_publication)

                if plot_watershed_steps and i_frames == plot_watershed_steps:
                    plt.tight_layout(pad=0.1)
                    pdf_plot_watershed.savefig()
                    pdf_plot_watershed.close()

                frame_properties['structures'].append(structures_dict)

                if structure_pdf_save:
                    plt.title(str(exp_id)+' @ '+"{:.3f}".format(time[i_frames]*1e3)+'ms')
                    plt.show()
                    pdf_structures.savefig()

                """
                Structure size calculation based on the contours
                """

                #Crude average size calculation
                if structures_dict is not None and len(structures_dict) != 0: #Valid structure size
                    valid_structure_size=True
                else:
                    valid_structure_size=False
                    
                frame_properties=calculate_average_frame_properties(frame_properties,
                                                                    i_frames=i_frames,
                                                                    valid_structure_size=valid_structure_size,
                                                                    structures_dict=structures_dict,
                                                                    weighting=weighting,
                                                                    fit_shape=fit_shape,
                                                                    maxing=maxing,
                                                                    structure_pixel_calc=structure_pixel_calc,
                                                                    frame=frame,
                                                                    skip_mdsplus=skip_mdsplus,
                                                                    R_sep_GPI=R_sep_GPI,
                                                                    z_sep_GPI=z_sep_GPI,
                                                                    R_sep_GPI_interp=R_sep_GPI_interp,
                                                                    z_sep_GPI_interp=z_sep_GPI_interp,
                                                                    )
            if structure_pdf_save:
                pdf_structures.close()
            #Saving results into a pickle file
            if fix_structure_angles:
                frame_properties=_fix_structure_angles(frame_properties)
                
            pickle.dump(frame_properties,open(pickle_filename, 'wb'))
            if test:
                plt.close()
        else:
            print('--- Loading data from the pickle file ---')
            frame_properties=pickle.load(open(pickle_filename, 'rb'))
    else:
        print('--- Loading data from the pickle file ---')
        frame_properties=pickle.load(open(pickle_filename, 'rb'))
        #labels= 'label,born,died'
        
    """***************
    Structure tracking
    ***************"""
    
    frame_properties=track_structures(frame_properties=frame_properties,
                                      exp_id=exp_id,
                                      time_range=time_range,
                                      tracking=tracking,
                                      tracking_assignment=tracking_assignment,
                                      matrix_weight=matrix_weight,
                                      test=test,
                                      fit_shape=fit_shape,
                                      prev_str_weighting=prev_str_weighting,
                                      calculate_rough_diff_velocities=calculate_rough_diff_velocities,
                                      weighting=weighting,
                                      maxing=maxing,
                                      remove_orphans=remove_orphans,
                                      min_structure_lifetime=min_structure_lifetime,
                                      nocalc=nocalc,
                                      recalc_tracking=recalc_tracking,
                                      smooth_contours=smooth_contours,
                                      comment=comment,
                                      #fix_structure_angles=fix_structure_angles,
                                      )

    """*****************
    PLOTTING THE RESULTS
    *****************"""

    import matplotlib.colors as mcolors
    colortable=list(mcolors.TABLEAU_COLORS.keys())
    n_color=len(colortable)

    if time_range is None:
        time_range=[frame_properties['Time'][0],frame_properties['Time'][-1]]

    if not filename_was_none and not time_range is None:
        sample_time=frame_properties['Time'][1]-frame_properties['Time'][0]
        if time_range[0] < frame_properties['Time'][0]-sample_time or time_range[1] > frame_properties['Time'][-1]+sample_time:
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
                              time=time,
                              str_finding_method=str_finding_method,
                              video_framerate=video_framerate,
                              frame_properties=frame_properties,
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
                                       frame_properties=frame_properties,
                                       time=time,
                                       plot_example_structure_frames=plot_example_structure_frames,
                                       plot_separatrix=plot_separatrix,
                                       separatrix_coordinates=(d_sep_x,d_sep_y),
                                       # save_data_for_publication=save_data_for_publication,
                                       )

    #Plotting the results
    if plot_results or pdf:
        _plot_results(pdf=pdf,
                      plot_results=plot_results,
                      plot_time_range=plot_time_range,
                      time_range=time_range,
                      plot_str_by_str=plot_str_by_str,
                      comment=comment,
                      exp_id=exp_id,
                      frame_properties=frame_properties,
                      plot_for_publication=plot_for_publication,
                      plot_vertical_line_at=plot_vertical_line_at,
                      overplot_average=overplot_average,
                      plot_scatter=plot_scatter,
                      plot_tracking=plot_tracking,
                      n_color=n_color,
                      colortable=colortable,
                      )

    if plot_example_frames_results:
        _plot_example_frames_results(exp_id=exp_id,
                                     time_range=time_range,
                                     plot_time_range=plot_time_range,
                                     frame_properties=frame_properties,
                                     wd=wd,
                                     n_color=n_color,
                                     colortable=colortable,
                                     pdf=pdf,
                                     save_data_for_publication=save_data_for_publication,
                                     )

    if return_results:
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
    structure_2['Velocity radial centroid']=(structure_2['Polygon'].centroid[0]-
                                             structure_1['Polygon'].centroid[0])/sample_time
    structure_2['Velocity poloidal centroid']=(structure_2['Polygon'].centroid[1]-
                                               structure_1['Polygon'].centroid[1])/sample_time
    structure_2['Velocity radial position']=(structure_2[fit_shape].center[0]-
                                             structure_1[fit_shape].center[0])/sample_time
    structure_2['Velocity poloidal position']=(structure_2[fit_shape].center[1]-
                                               structure_1[fit_shape].center[1])/sample_time
    structure_2['Expansion fraction area']=np.sqrt(structure_2['Polygon'].area/
                                                   structure_1['Polygon'].area)
    structure_2['Expansion fraction axes']=np.sqrt(structure_2[fit_shape].axes_length[0]/
                                                   structure_1[fit_shape].axes_length[0]*
                                                   structure_2[fit_shape].axes_length[1]/
                                                   structure_1[fit_shape].axes_length[1])

    structure_2['Angular velocity angle']=(structure_2['Angle']-
                                           structure_1['Angle'])/sample_time
    try:
        structure_1['Angle of least inertia']
    except:
        #Not fringe jump corrected
        structure_1['Angle of least inertia']=structure_1['Polygon'].principal_axes_angle

    structure_2['Angular velocity ALI']=(structure_2['Angle of least inertia']-
                                         structure_1['Angle of least inertia'])/sample_time
    return structure_2



def calculate_average_frame_properties(frame_properties,
                                       i_frames=None,
                                       valid_structure_size=True,
                                       structures_dict=None,
                                       weighting='area',
                                       fit_shape='ellipse',
                                       maxing='intensity',
                                       structure_pixel_calc=False,
                                       frame=None,
                                       skip_mdsplus=False,
                                       R_sep_GPI=None,
                                       z_sep_GPI=None,
                                       R_sep_GPI_interp=None,
                                       z_sep_GPI_interp=None,
                                       ):
    
    if not valid_structure_size:
         #Setting np.nan if no structure is available
         for key in frame_properties['data'].keys():
             frame_properties['data'][key]['avg'][i_frames]=np.nan
             frame_properties['data'][key]['max'][i_frames]=np.nan
             frame_properties['data'][key]['stddev'][i_frames]=np.nan

         frame_properties['data']['Str number']['max'][i_frames]=0.
         frame_properties['data']['Frame COG radial']['max'][i_frames]=np.nan
         frame_properties['data']['Frame COG poloidal']['max'][i_frames]=np.nan
         # frame_properties['Structures'][i_frames]=None
         return frame_properties
    
    #Calculating the average properties of the structures present in one frame
    n_str=len(structures_dict)
    areas=np.zeros(len(structures_dict))
    intensities=np.zeros(len(structures_dict))

    for i_str in range(n_str):
        #Average size calculation based on the number of structures
        areas[i_str]=structures_dict[i_str]['Area']
        intensities[i_str]=structures_dict[i_str]['Intensity']

    #Calculating the averages based on the input setting
    if weighting == 'number':
        weight=np.zeros(n_str)
        weight[:]=1./n_str
    elif weighting == 'intensity':
        weight=intensities/np.sum(intensities)
    elif weighting == 'area':
        weight=areas/np.sum(areas)

    for i_str in range(n_str):
        #Quantities from Ellipse fitting
        fit_obj_cur=structures_dict[i_str][fit_shape]
        frame_properties['data']['Size radial']['avg'][i_frames]+=fit_obj_cur.size[0]*weight[i_str]
        frame_properties['data']['Size poloidal']['avg'][i_frames]+=fit_obj_cur.size[1]*weight[i_str]
        frame_properties['data']['Position radial']['avg'][i_frames]+=fit_obj_cur.center[0]*weight[i_str]
        frame_properties['data']['Position poloidal']['avg'][i_frames]+=fit_obj_cur.center[1]*weight[i_str]
        frame_properties['data']['Angle']['avg'][i_frames]+=fit_obj_cur.angle*weight[i_str]
        frame_properties['data']['Elongation']['avg'][i_frames]+=fit_obj_cur.elongation*weight[i_str]
        frame_properties['data']['Axes length minor']['avg'][i_frames]+=np.min(fit_obj_cur.axes_length)*weight[i_str]
        frame_properties['data']['Axes length major']['avg'][i_frames]+=np.max(fit_obj_cur.axes_length)*weight[i_str]

        #Quantities from polygons
        polygon_cur=structures_dict[i_str]['Polygon']
        frame_properties['data']['Centroid radial']['avg'][i_frames]+=polygon_cur.centroid[0]*weight[i_str]
        frame_properties['data']['Centroid poloidal']['avg'][i_frames]+=polygon_cur.centroid[1]*weight[i_str]
        frame_properties['data']['Center of gravity radial']['avg'][i_frames]+=polygon_cur.center_of_gravity[0]*weight[i_str]
        frame_properties['data']['Center of gravity poloidal']['avg'][i_frames]+=polygon_cur.center_of_gravity[1]*weight[i_str]
        frame_properties['data']['Area']['avg'][i_frames]+=polygon_cur.area*weight[i_str]
        frame_properties['data']['Angle of least inertia']['avg'][i_frames]+=polygon_cur.principal_axes_angle*weight[i_str]
        frame_properties['data']['Roundness']['avg'][i_frames]+=polygon_cur.roundness*weight[i_str]
        frame_properties['data']['Solidity']['avg'][i_frames]+=polygon_cur.solidity*weight[i_str]
        frame_properties['data']['Convexity']['avg'][i_frames]+=polygon_cur.convexity*weight[i_str]
        frame_properties['data']['Total curvature']['avg'][i_frames]+=polygon_cur.total_curvature*weight[i_str]
        frame_properties['data']['Total bending energy']['avg'][i_frames]+=polygon_cur.total_bending_energy*weight[i_str]

    #Calculating the properties of the structure having the maximum area or intensity
    if maxing == 'area':
        ind_max=np.argmax(areas)
    elif maxing == 'intensity':
        ind_max=np.argmax(intensities)

    #Properties of the max structure:
    fit_obj_cur=structures_dict[ind_max][fit_shape]
    frame_properties['data']['Size radial']['max'][i_frames]=fit_obj_cur.size[0]
    frame_properties['data']['Size poloidal']['max'][i_frames]=fit_obj_cur.size[1]
    frame_properties['data']['Position radial']['max'][i_frames]=fit_obj_cur.center[0]
    frame_properties['data']['Position poloidal']['max'][i_frames]=fit_obj_cur.center[1]
    frame_properties['data']['Angle']['max'][i_frames]=fit_obj_cur.angle
    frame_properties['data']['Elongation']['max'][i_frames]=fit_obj_cur.elongation
    frame_properties['data']['Axes length minor']['max'][i_frames]=np.min(fit_obj_cur.axes_length)
    frame_properties['data']['Axes length major']['max'][i_frames]=np.max(fit_obj_cur.axes_length)

    polygon_cur=structures_dict[ind_max]['Polygon']
    frame_properties['data']['Centroid radial']['max'][i_frames]=polygon_cur.centroid[0]
    frame_properties['data']['Centroid poloidal']['max'][i_frames]=polygon_cur.centroid[1]
    frame_properties['data']['Center of gravity radial']['max'][i_frames]=polygon_cur.center_of_gravity[0]
    frame_properties['data']['Center of gravity poloidal']['max'][i_frames]=polygon_cur.center_of_gravity[1]
    frame_properties['data']['Area']['max'][i_frames]=polygon_cur.area
    frame_properties['data']['Angle of least inertia']['max'][i_frames]=polygon_cur.principal_axes_angle
    frame_properties['data']['Roundness']['max'][i_frames]=polygon_cur.roundness
    frame_properties['data']['Solidity']['max'][i_frames]=polygon_cur.solidity
    frame_properties['data']['Convexity']['max'][i_frames]=polygon_cur.convexity
    frame_properties['data']['Total curvature']['max'][i_frames]=polygon_cur.total_curvature
    frame_properties['data']['Total bending energy']['max'][i_frames]=polygon_cur.total_bending_energy

    #The center of gravity for the entire frame
    if structure_pixel_calc:
        frame_properties['data']['Frame COG radial']['max'][i_frames]=np.sum(frame.coordinate('Image x')[0]*frame.data)/np.sum(frame.data)
    else:
        frame_properties['data']['Frame COG radial']['max'][i_frames]=np.sum(frame.coordinate('Device R')[0]*frame.data)/np.sum(frame.data)
    frame_properties['data']['Frame COG radial']['avg'][i_frames]=frame_properties['data']['Frame COG radial']['max'][i_frames]

    if structure_pixel_calc:
        frame_properties['data']['Frame COG radial']['max'][i_frames]=np.sum(frame.coordinate('Image y')[0]*frame.data)/np.sum(frame.data)
    else:
        frame_properties['data']['Frame COG poloidal']['max'][i_frames]=np.sum(frame.coordinate('Device z')[0]*frame.data)/np.sum(frame.data)
    frame_properties['data']['Frame COG poloidal']['avg'][i_frames]=frame_properties['data']['Frame COG poloidal']['max'][i_frames]

    #The number of structures in a frame
    frame_properties['data']['Str number']['max'][i_frames]=n_str
    frame_properties['data']['Str number']['avg'][i_frames]=n_str

    #Calculate the distance from the separatrix
    try:
    # if True:
        for key in ['max','avg']:
            if not skip_mdsplus:
                frame_properties['data']['Separatrix dist'][key][i_frames]=np.min(np.sqrt((frame_properties['data']['Position radial'][key][i_frames]-R_sep_GPI_interp)**2 +
                                                                                          (frame_properties['data']['Position poloidal'][key][i_frames]-z_sep_GPI_interp)**2))
            else:
                frame_properties['data']['Separatrix dist'][key][i_frames]=np.nan
            ind_z_min=np.argmin(np.abs(z_sep_GPI-frame_properties['data']['Position poloidal'][key][i_frames]))
            if z_sep_GPI[ind_z_min] >= frame_properties['data']['Position poloidal'][key][i_frames]:
                ind1=ind_z_min
                ind2=ind_z_min+1
            else:
                ind1=ind_z_min-1
                ind2=ind_z_min

            radial_distance=frame_properties['data']['Position radial'][key][i_frames]- \
                ((frame_properties['data']['Position poloidal'][key][i_frames]-z_sep_GPI[ind2])/ \
                 (z_sep_GPI[ind1]-z_sep_GPI[ind2])*(R_sep_GPI[ind1]-R_sep_GPI[ind2])+R_sep_GPI[ind2])
            if radial_distance < 0:
                frame_properties['data']['Separatrix dist'][key][i_frames]*=-1
    except:
        frame_properties['data']['Separatrix dist'][key][i_frames]=np.nan
    
    return frame_properties



def transform_frames_to_structures(frame_properties):
    max_str_label=0
    for i_frames in range(len(frame_properties['structures'])):
        if frame_properties['structures'][i_frames] is not None:
            for j_str in range(len(frame_properties['structures'][i_frames])):
                current_label=frame_properties['structures'][i_frames][j_str]['Label']
                if current_label is not None and current_label > max_str_label:
                    max_str_label=current_label
                    
    struct_by_struct=[]
    for ind in range(max_str_label+1):
        struct_by_struct.append({'Time':[],}.copy())

    analyzed_keys=read_analyzed_keys()

    for i_frames in range(len(frame_properties['structures'])):
        if frame_properties['structures'][i_frames] is not None:
            for j_str in range(len(frame_properties['structures'][i_frames])):
                if (frame_properties['structures'][i_frames][j_str]['Label'] is not None and
                    frame_properties['structures'][i_frames][j_str]['Label'] != []):

                    ind_structure=int(frame_properties['structures'][i_frames][j_str]['Label'])
                    for key_str in frame_properties['structures'][i_frames][j_str].keys():
                        if key_str in analyzed_keys:
                            if key_str not in struct_by_struct[ind_structure].keys():
                                struct_by_struct[ind_structure][key_str]=[]
                            try:
                                if frame_properties['structures'][i_frames][j_str][key_str] != 0:
                                    struct_by_struct[ind_structure][key_str].append(frame_properties['structures'][i_frames][j_str][key_str])
                                else:
                                    struct_by_struct[ind_structure][key_str].append(np.nan)
                            except Exception as e:
                                print('Exception occurred in analyze_gpi_structures at line 2056: ', e)
                                print(key_str, frame_properties['structures'][i_frames][j_str][key_str])
                                print(frame_properties['structures'][i_frames][j_str].keys())
                                struct_by_struct[ind_structure][key_str].append(np.nan)
                    struct_by_struct[ind_structure]['Time'].append(frame_properties['Time'][i_frames])
    return struct_by_struct




def _fix_structure_angles(frame_properties):
    for i_frames in range(len(frame_properties['structures'])):
        if frame_properties['structures'][i_frames] is not None:
            for j_str in range(len(frame_properties['structures'][i_frames])):
                angle=frame_properties['structures'][i_frames][j_str]['Angle']
                #frame_properties['structures'][i_frames][j_str]['Angle of least inertia']
                
                #THis fixes the angles and puts them between -np.pi and np.pi
                while (angle > np.pi or angle < -np.pi):
                    if angle > np.pi:
                        angle -= 2*np.pi
                    if angle < -np.pi:
                        angle += 2*np.pi
                #ALI needs to be recalculated due to the bug with arcsin(sin)

                mu=np.zeros([2,2])
                cog=frame_properties['structures'][i_frames][j_str]['Center of gravity']
                data_inside_structure=frame_properties['structures'][i_frames][j_str]['Data']
                x_data=frame_properties['structures'][i_frames][j_str]['X coord']
                y_data=frame_properties['structures'][i_frames][j_str]['Y coord']
                
                if not np.isnan(cog[0]):
                    mu[0,0]=np.sum(data_inside_structure*(y_data-cog[1])**2)
                    mu[0,1]=-np.sum(data_inside_structure*(x_data-cog[0])*(y_data-cog[1]))
                    mu[1,0]=mu[0,1]
                    mu[1,1]=np.sum(data_inside_structure*(x_data-cog[0])**2)
                else:
                    mu[:,:]=np.nan

                try:
                    eigvalues,eigvectors=np.linalg.eig(mu)
                    eig_ind=np.argmax(eigvalues)
                    angle_ali=np.arctan(eigvectors[1,eig_ind]/
                                        eigvectors[0,eig_ind])
                    angle_ali=np.mod(angle_ali, np.pi)
                    
                except:
                    angle_ali=np.nan
                frame_properties['structures'][i_frames][j_str]['Angle of least inertia']=angle_ali
                frame_properties['structures'][i_frames][j_str]['Angle']=angle
    return frame_properties

def _plot_results(pdf=False,
                  plot_results=False,
                  plot_time_range=None,
                  time_range=None,
                  plot_str_by_str=False,
                  comment=None,
                  exp_id=None,
                  frame_properties=None,
                  plot_for_publication=False,
                  plot_vertical_line_at=None,
                  overplot_average=False,
                  plot_scatter=False,
                  plot_tracking=False,
                  n_color=None,
                  colortable=None,
                  ):

    differential_keys=list(frame_properties['derived'].keys())
    
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
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
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

    if not plot_str_by_str:
        _plot_avg_results(frame_properties=frame_properties,
                          time_range=time_range,
                          figsize=figsize,
                          plot_vertical_line_at=plot_vertical_line_at,
                          overplot_average=overplot_average,
                          plot_for_publication=plot_for_publication,
                          pdf=pdf,
                          pdf_pages=pdf_pages,
                          plot_scatter=plot_scatter,
                          )
    else:
        _plot_str_by_str(frame_properties=frame_properties,
                         plot_scatter=plot_scatter,
                         figsize=figsize,
                         differential_keys=differential_keys,
                         plot_tracking=plot_tracking,
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



def _plot_example_structure_frames(exp_id=None,
                                   time_range=None,
                                   plot_time_range=None,
                                   
                                   sample_0=None,
                                   wd=None,
                                   object_name=None,
                                   plot_nframe=None,
                                   plot_ncol=None,
                                   levels=None,
                                   frame_properties=None,
                                   time=None,
                                   plot_example_structure_frames=None,
                                   save_data_for_publication=False,
                                   plot_separatrix=False,
                                   separatrix_coordinates=None,
                                   linewidth=None,
                                   ):
    
    (d_sep_x,d_sep_y)=separatrix_coordinates
    
    if plot_time_range is not None:
        if plot_time_range[0] < time_range[0] or plot_time_range[1] > time_range[1]:
            raise ValueError('The plot time range is not in the interval of the original time range.')
        time_range=plot_time_range
    
    import matplotlib.colors as mcolors
    colortable=list(mcolors.TABLEAU_COLORS.keys())
    n_color=len(colortable)


    pdf_filenames_frames=flap_nstx.tools.filename(exp_id=exp_id,
                                                  time_range=time_range,
                                                  purpose="example_frames",
                                                  extension='.pdf',
                                                  comment='s0'+str(sample_0))

    pdf_pages=PdfPages(wd+'/plots/'+pdf_filenames_frames)
    from flap_nstx.gpi import _plot_ellipses_centers
    import scipy
    
    if plot_time_range:
        data_object=flap.get_data_object(object_name)
        sliced_data=data_object.slice_data(slicing={'Time':flap.Intervals(plot_time_range[0],
                                                                          plot_time_range[1])})
        frame_sample_0=sliced_data.coordinate('Sample')[0][0,0,0]-sample_0
        sample_0=sliced_data.coordinate('Sample')[0][0,0,0]
        
    slicing_frame={'Sample':sample_0}

    frame=flap.slice_data(object_name,
                          exp_id=exp_id,
                          slicing=slicing_frame,
                          output_name='GPI_FRAME')
    x_coord_name='Device R'
    x_unit_name='[m]'
    y_coord_name='Device z'
    y_unit_name='[m]'

    x_coord=frame.coordinate(x_coord_name)[0]
    y_coord=frame.coordinate(y_coord_name)[0]

    if plot_nframe is None and plot_ncol is None:

        plot_nframe=15
        plot_ncol=5


    fig,axes=plt.subplots(int(plot_nframe/plot_ncol),plot_ncol,
                          figsize=(8.5/2.54,3.5*plot_nframe/plot_ncol/2.54),
                          )
    for i_frames in range(0,plot_nframe):
        ax=axes[i_frames//plot_ncol,
                np.mod(i_frames,plot_ncol)]

        slicing_frame={'Sample':sample_0+i_frames+plot_example_structure_frames}

        frame=flap.slice_data(object_name,
                              exp_id=exp_id,
                              slicing=slicing_frame,
                              output_name='GPI_FRAME')

        frame.data=np.asarray(frame.data, dtype='float64')
        frame.data=scipy.ndimage.median_filter(frame.data, 5)


        if levels is None:
            ax.contourf(x_coord, y_coord, frame.data, levels=51)
        else:
            ax.contourf(x_coord, y_coord, frame.data, levels=levels)

        if plot_separatrix:
            slicing={'Time':frame.coordinate('Time')[0][0,0]}
            
            d_sep_x_sliced=d_sep_x.slice_data(slicing=slicing)
            d_sep_y_sliced=d_sep_y.slice_data(slicing=slicing)
            
            print(slicing)
            separatrix_data=np.zeros([d_sep_x_sliced.shape[0],2])
            separatrix_data[:,0]=d_sep_x_sliced.data
            separatrix_data[:,1]=d_sep_y_sliced.data
            
            print(d_sep_x_sliced.coordinate('Time'))
            
            if plot_separatrix and separatrix_data is not None:
                ax.plot(separatrix_data[:,0],
                        separatrix_data[:,1],
                        linewidth=1,
                        color='red')
                # if save_data_for_publication:
                #     exp_id=data_object.exp_id
                #     time=frame.coordinate('Time')[0][0,0]
                #     wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
                #     filename=wd+'/'+str(exp_id)+'_'+str(time)+'_separatrix.txt'
                #     file1=open(filename, 'w+')
                #     for i in range(len(separatrix_data[:,0])):
                #         file1.write(str(separatrix_data[i,0])+'\t'+str(separatrix_data[i,1])+'\n')
                #     file1.close()

        ax.set_aspect(1.0)
        structures=frame_properties['structures'][frame_sample_0+i_frames+plot_example_structure_frames]

        if structures is not None and len(structures) > 0:
            #Parametric reproduction of the Ellipse
            R=np.arange(0,2*np.pi,0.01)
            for i_str in range(len(structures)):
                if (structures[i_str]['Half path'] is not None and
                    structures[i_str]['Ellipse'] is not None):

                    phi=structures[i_str]['Angle']
                    a,b=structures[i_str]['Axes length']

                    x_polygon=structures[i_str]['Polygon'].x
                    y_polygon=structures[i_str]['Polygon'].y

                    x_ellipse = (structures[i_str]['Center'][0] +
                                 a*np.cos(R)*np.cos(phi) -
                                 b*np.sin(R)*np.sin(phi))
                    y_ellipse = (structures[i_str]['Center'][1] +
                                 a*np.cos(R)*np.sin(phi) +
                                 b*np.sin(R)*np.cos(phi))
                    if linewidth is None:
                        polygon_linewidth=1
                        ellipse_linewidth=0.25
                        semiaxis_linewidth=0.5
                    else:
                        polygon_linewidth=linewidth['polygon']
                        ellipse_linewidth=linewidth['ellipse']
                        semiaxis_linewidth=linewidth['semiaxis']
                    _plot_ellipses_centers(ax,
                                           x_polygon, y_polygon,
                                           x_ellipse, y_ellipse,
                                           structures[i_str],
                                           polygon_color=colortable[int(np.mod(structures[i_str]['Label']+1,n_color))],
                                           ellipse_color='black',
                                           polygon_linewidth=polygon_linewidth,
                                           ellipse_linewidth=ellipse_linewidth,
                                           semiaxis_linewidth=semiaxis_linewidth,
                                           plot_structure_mid=False,
                                           )
                elif structures[i_str]['Half path'] is not None:
                        ax.plot(x_polygon,
                                y_polygon,
                                color=colortable[int(np.mod(structures[i_str]['Label']+1,n_color))],
                                linewidth=1
                                )
        #         if save_data_for_publication:
        #             exp_id=data_object.exp_id
        #             time=frame.coordinate('Time')[0][0,0]
        #             wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        #             filename=wd+'/'+str(exp_id)+'_'+str(time)+'_half_path_no.'+str(i_str)+'.txt'
        #             file1=open(filename, 'w+')
        #             for i in range(len(x_polygon)):
        #                 file1.write(str(x_polygon[i])+'\t'+str(y_polygon[i])+'\n')
        #             file1.close()

        #             filename=wd+'/'+str(exp_id)+'_'+str(time)+'_fit_ellipse_no.'+str(i_str)+'.txt'
        #             file1=open(filename, 'w+')
        #             for i in range(len(x_ellipse)):
        #                 file1.write(str(x_ellipse[i])+'\t'+str(y_ellipse[i])+'\n')
        #             file1.close()
                    
        # if save_data_for_publication:
        #     time=frame.coordinate('Time')[0][0,0]
        #     wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        #     filename=wd+'/'+str(exp_id)+'_'+str(time)+'_raw_data.txt'
        #     file1=open(filename, 'w+')
        #     data=frame.data
        #     for i in range(len(data[0,:])):
        #         string=''
        #         for j in range(len(data[:,0])):
        #             string+=str(data[j,i])+'\t'
        #         string+='\n'
        #         file1.write(string)
        #     file1.close()
                        
                        
        ax.set_xlabel('R' + ' '+ x_unit_name)
        ax.set_ylabel('z' + ' '+ y_unit_name)
        
        if np.mod(i_frames,plot_ncol) != 0:
            ax.get_yaxis().set_visible(False)

        if i_frames < plot_nframe-plot_ncol:
            ax.get_xaxis().set_visible(False)

        #ax.set_title(str(exp_id)+' @ '+str(frame.coordinate('Time')[0][0,0]))

        plt.show()

        ax.set_xlim([x_coord.min(),x_coord.max()])
        ax.set_ylim([y_coord.min(),y_coord.max()])
        ax.set_title("{:.3f}".format(frame.coordinate('Time')[0][0,0]*1e3)+' ms',
                     fontsize=8, y=0.95)

    plt.tight_layout(h_pad=0.3,
                     w_pad=0.1)
    
    pdf_pages.savefig()
    pdf_pages.close()





def _plot_example_frames_results(exp_id=None,
                                 time_range=None,
                                 plot_time_range=None,
                                 frame_properties=None,
                                 wd=None,
                                 n_color=None,
                                 colortable=None,
                                 pdf=None,
                                 markersize=0.5,
                                 save_data_for_publication=False,
                                 ):

    set_matplotlib_for_publication(labelsize=8,
                                    linewidth=0.2,
                                    major_ticksize=2.,
                                    )
    
    differential_keys=list(frame_properties['derived'].keys())
    
    #pdf_pages=PdfPages(wd+'/plots/plot_example_frame_results.pdf')
    if pdf:
        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                          time_range=time_range,
                                          working_directory=wd+'/plots',
                                          purpose='example_frame_results',
                                          extension='pdf')
        pdf_pages=PdfPages(filename)

    struct_by_struct=transform_frames_to_structures(frame_properties)
    #return struct_by_struct
    #print('str_by_str_len: ',struct_by_struct)
    labels=['a','b','c','d','e','f']
    #Area, Angle, Elongation, Roundness
    fig, axes = plt.subplots(6,1,figsize=(17/2.54,12/2.54))
    for ind,key in enumerate(['Position radial', 'Position poloidal', 'Area','Angle','Roundness','Total curvature']):
        ax=axes[ind]
        
        if save_data_for_publication:
            labels=['a','b','c','d','e','f']
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename=f'{wd}/{labels[ind]}_{exp_id}_{time_range[0]}_{time_range[1]}_{key}_example.txt'
            file1=open(filename, 'w+')
            
        if key == 'Area':
            frame_properties['data'][key]['label']='A'
            frame_properties['data'][key]['unit']='$mm^2$'
        for ind_str in range(len(struct_by_struct)):
            if struct_by_struct[ind_str]['Time'] !=[]:
                if key not in differential_keys:
                    if key == 'Area':
                        struct_by_struct[ind_str][key] = np.asarray(struct_by_struct[ind_str][key]) * 1e5
                        
                    try:
                        ax.plot(np.asarray(struct_by_struct[ind_str]['Time']),
                                struct_by_struct[ind_str][key],
                                '-o',
                                markersize=markersize,
                                linewidth=0.2,
                                label=str(ind_str),
                                color=colortable[np.mod(int(ind_str)+1,n_color)], 
                                )
                        
                        if save_data_for_publication:
                            file1.write(f'Structure #{ind_str}:\n')
                            for ind_save in range(len(np.asarray(struct_by_struct[ind_str]['Time']))):
                                file1.write(str(np.asarray(struct_by_struct[ind_str]['Time'][ind_save]))+'\t'+str(struct_by_struct[ind_str][key][ind_save])+'\n')
                            file1.write('\n')


                    except Exception as e:
                        print(str(e))
                        
                    if frame_properties['data'][key]['unit'] != '':
                        ax.set_ylabel(frame_properties['data'][key]['label']+' '+
                                      '['+frame_properties['data'][key]['unit']+']', fontsize=8)
                    else:
                        if frame_properties['data'][key]['label'] == 'Round.':
                            frame_properties['data'][key]['label']='Roundness'
                        ax.set_ylabel(frame_properties['data'][key]['label'], fontsize=8)
                else:
                    try:
                        ax.plot(np.asarray(struct_by_struct[ind_str]['Time'][1:]),
                                struct_by_struct[ind_str][key],
                                '-o',
                                markersize=markersize,
                                linewidth=0.2,
                                label=str(ind_str),
                                color=colortable[int(np.mod(ind_str+1,n_color))],
                                )
                        
                        if save_data_for_publication:
                            file1.write(f'Structure #{ind_str}:\n')
                            for ind_save in range(len(np.asarray(struct_by_struct[ind_str]['Time'][1:]))):
                                file1.write(str(np.asarray(struct_by_struct[ind_str]['Time'][ind_save]))+'\t'+str(struct_by_struct[ind_str][key][ind_save])+'\n')
                            file1.write('\n')
                            
                    except Exception as e:
                        print(str(e))
                        
                    if frame_properties['data'][key]['unit'] != '':
                        ax.set_ylabel(frame_properties['derived'][key]['label']+' '+
                                      '['+frame_properties['derived'][key]['unit']+']', fontsize=8)
                    else:
                        if frame_properties['derived'][key]['label'] == 'Round.':
                            frame_properties['derived'][key]['label']='Roundness'
                        ax.set_ylabel(frame_properties['derived'][key]['label'], fontsize=8)
                        
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        if ind < 5:
            ax.xaxis.label.set_visible(False)
            ax.set_xticklabels([])
        ax.text(-0.1, 0.9, '('+labels[ind]+')', transform=ax.transAxes, size=8)
        ax.set_xlabel('Time [s]', fontsize=8)
        if plot_time_range is not None:
            ax.set_xlim(np.asarray(plot_time_range))
        else:
            ax.set_xlim(np.asarray(time_range))
        if ind == 0:
            ax.set_ylim([1.43,1.55])
        if ind == 1:
            ax.set_ylim([0.07,0.35])
        # ax.set_title(str(key)+ ' vs. time', fontsize=8)
        fig.tight_layout(pad=0.1)
    if save_data_for_publication:
        file1.close()
        
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
                         time=None,
                         str_finding_method=None,
                         video_framerate=None,
                         frame_properties=None,
                         colortable=None,
                         wd=None,
                         n_color=None,
                         video_start_frame=0,
                         ):

    from flap_nstx.gpi import _plot_ellipses_centers
    import scipy

    set_matplotlib_for_publication(labelsize=32,
                                   linewidth=2,
                                   major_ticksize=15.,
                                   )

    slicing_frame={'Sample':sample_0}

    frame=flap.slice_data(object_name,
                          exp_id=exp_id,
                          slicing=slicing_frame,
                          output_name='GPI_FRAME')
    x_coord_name='Device R'
    x_unit_name='[m]'
    y_coord_name='Device z'
    y_unit_name='[m]'

    x_coord=frame.coordinate(x_coord_name)[0]
    y_coord=frame.coordinate(y_coord_name)[0]


    for i_frames in range(video_start_frame,n_frames):
        slicing_frame={'Sample':sample_0+i_frames}

        frame=flap.slice_data(object_name,
                              exp_id=exp_id,
                              slicing=slicing_frame,
                              output_name='GPI_FRAME')

        frame.data=np.asarray(frame.data, dtype='float64')
        frame.data=scipy.ndimage.median_filter(frame.data, 5)

        slicing={'Time':frame.coordinate('Time')[0][0,0]}
        d_sep_x_sliced=d_sep_x.slice_data(slicing=slicing)
        d_sep_y_sliced=d_sep_y.slice_data(slicing=slicing)

        separatrix_data=np.zeros([d_sep_x_sliced.shape[0],2])
        separatrix_data[:,0]=d_sep_x_sliced.data
        separatrix_data[:,1]=d_sep_y_sliced.data
        if d_flux is not None:
            surface_data_obj=d_flux.slice_data(slicing=slicing)

        my_dpi=80
        fig,ax=plt.subplots(figsize=(video_resolution[0]/my_dpi,
                                     video_resolution[1]/my_dpi),
                            dpi=my_dpi)

        if levels is None:
            plt.contourf(x_coord,
                         y_coord,
                         frame.data,
                         levels=51)
        else:
            plt.contourf(x_coord,
                         y_coord,
                         frame.data,
                         levels=levels)
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
            plt.plot(separatrix_data[:,0],
                     separatrix_data[:,1],
                     linewidth=2,
                     color='red')

        ax.set_aspect(1.0)


        structures=frame_properties['structures'][i_frames]

        if structures is not None and len(structures) > 0:
            #Parametric reproduction of the Ellipse
            R=np.arange(0,2*np.pi,0.01)
            for i_str in range(len(structures)):
                if (structures[i_str]['Half path'] is not None and
                    structures[i_str]['Ellipse'] is not None):

                    phi=structures[i_str]['Angle']
                    a,b=structures[i_str]['Axes length']

                    x_polygon=structures[i_str]['Polygon'].x
                    y_polygon=structures[i_str]['Polygon'].y

                    x_ellipse = (structures[i_str]['Center'][0] +
                                 a*np.cos(R)*np.cos(phi) -
                                 b*np.sin(R)*np.sin(phi))
                    y_ellipse = (structures[i_str]['Center'][1] +
                                 a*np.cos(R)*np.sin(phi) +
                                 b*np.sin(R)*np.cos(phi))

                    _plot_ellipses_centers(ax,
                                           x_polygon, y_polygon,
                                           x_ellipse, y_ellipse,
                                           structures[i_str],
                                           polygon_color=colortable[int(np.mod(structures[i_str]['Label'],n_color))],
                                           ellipse_color=colortable[int(np.mod(structures[i_str]['Label'],n_color))],
                                           polygon_linewidth=3,
                                           ellipse_linewidth=1.5)

        ax.set_xlabel(x_coord_name.replace('Device ','') + ' '+ x_unit_name)
        ax.set_ylabel(y_coord_name.replace('Device ','') + ' '+ y_unit_name)
        ax.set_title(str(exp_id)+' @ '+str(frame.coordinate('Time')[0][0,0]))
        plt.show()
        plt.pause(0.001)

        plt.xlim([x_coord.min(),x_coord.max()])
        plt.ylim([y_coord.min(),y_coord.max()])


        fig = plt.gcf()
        plt.title(str(exp_id)+' @ '+"{:.3f}".format(time[i_frames]*1e3)+'ms')
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()

        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            if buf.shape[0] == h*2 * w*2 * 3:
                buf.shape = ( h*2, w*2, 3 )
            else:
                buf.shape = ( h, w, 3 )
            buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            try:
                video
            except NameError:
                print('Canvas size is: '+str(w)+' x '+str(h))
                height = buf.shape[0]
                width = buf.shape[1]
                video_codec_code='mp4v'
                filename=wd+'/plots/NSTX_GPI_'+str(exp_id)+'_'+\
                    "{:.3f}".format(time[0]*1e3)+'_'+\
                    "{:.3f}".format(time[-1]*1e3)+\
                    '_fit_structures_'+str_finding_method+'.mp4'
                video = cv2.VideoWriter(filename,
                                        cv2.VideoWriter_fourcc(*video_codec_code),
                                        float(video_framerate),
                                        (width,height),
                                        isColor=True)
                print('Video resolution is: '+str(width)+' x '+str(height))
            video.write(buf)
        except:
            print('Video frame cannot be saved. Passing...')

    cv2.destroyAllWindows()
    video.release()
    del video

def _plot_str_by_str(frame_properties=None,
                     plot_scatter=True,
                     figsize=None,
                     differential_keys=None,
                     plot_tracking=None,
                     colortable=None,
                     n_color=None,
                     time_range=None,
                     plot_for_publication=False,
                     pdf=False,
                     pdf_pages=None,
                     ):
#            frame_properties['structures'][i_frames]
    struct_by_struct=transform_frames_to_structures(frame_properties)

    analyzed_keys=read_analyzed_keys()
    #return struct_by_struct
    #print('str_by_str_len: ',struct_by_struct)
    if plot_scatter:
        linestyle='-o'
    else:
        linestyle='-'

    for key in analyzed_keys:
        fig, ax = plt.subplots(figsize=figsize)
        for ind_str in range(len(struct_by_struct)):
            if struct_by_struct[ind_str]['Time'] !=[]:
                if key not in differential_keys:
                    try:
                        if plot_tracking:

                            ax.plot(np.asarray(struct_by_struct[ind_str]['Time'])*1e3,
                                    struct_by_struct[ind_str][key],
                                    linestyle,
                                    label=str(ind_str),
                                    markersize=5,
                                    color=colortable[np.mod(int(ind_str)+1,n_color)]
                                    )
                    except Exception as e:
                        print(str(e))
                    ax.set_ylabel(frame_properties['data'][key]['label']+' '+'['+frame_properties['data'][key]['unit']+']')
                else:
                    try:
                        if plot_tracking:
                            if key in ['Angular velocity angle', 'Angular velocity ALI']:
                                pass
                            ax.plot(np.asarray(struct_by_struct[ind_str]['Time'][1:])*1e3,
                                    struct_by_struct[ind_str][key],
                                    linestyle,
                                    label=str(ind_str),
                                    markersize=5,
                                    color=colortable[int(np.mod(ind_str+1,n_color))],
                                    )

                    except Exception as e:
                        print(str(e))
                    ax.set_ylabel(frame_properties['derived'][key]['label']+' '+'['+frame_properties['derived'][key]['unit']+']')

        ax.set_xlabel('Time [ms]')
        ax.set_xlim(np.asarray(time_range)*1e3)
        ax.set_title(str(key)+ ' vs. time')

        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
        fig.tight_layout(pad=0.1)

        if pdf:
            pdf_pages.savefig()
            plt.cla()


def _plot_avg_results(frame_properties=None,
                      time_range=None,
                      figsize=None,
                      plot_vertical_line_at=None,
                      overplot_average=False,
                      plot_for_publication=False,
                      pdf=False,
                      pdf_pages=None,
                      plot_scatter=True,
                      ):

    keys=list(frame_properties['data'].keys())
    plot_index_structure=np.logical_and(np.logical_not(np.isnan(frame_properties['data']['Elongation']['avg'])),
                                        np.logical_and(frame_properties['Time'] >= time_range[0],
                                                       frame_properties['Time'] <= time_range[1]))
    for i in range(len(keys)):
        fig, ax = plt.subplots(figsize=figsize)
        if plot_vertical_line_at is not None:
            ax.axvline(x=plot_vertical_line_at,
                     color='red')
        ax.plot(frame_properties['Time'][plot_index_structure],
                frame_properties['data'][keys[i]]['max'][plot_index_structure],
                markersize=5,
                marker='o',
                color='tab:blue')

        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure],
                    frame_properties['data'][keys[i]]['avg'][plot_index_structure],
                    linewidth=0.5,
                    color='red',)

        ax.set_xlabel('Time $[\mu s]$')
        ax.set_ylabel(frame_properties['data'][keys[i]]['label']+' '+'['+frame_properties['data'][keys[i]]['unit']+']')
        ax.set_xlim(time_range)
        ax.set_title(str(keys[i])+ ' vs. time')
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
        fig.tight_layout(pad=0.1)

        if pdf:
            pdf_pages.savefig()

    keys=list(frame_properties['derived'].keys())

    for i in range(len(keys)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index_structure],
                frame_properties['derived'][keys[i]]['max'][plot_index_structure],
                color='tab:blue')

        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure],
                       frame_properties['derived'][keys[i]]['max'][plot_index_structure],
                        s=5,
                        marker='o',
                        color='tab:blue')

        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure],
                    frame_properties['derived'][keys[i]]['avg'][plot_index_structure],
                    linewidth=0.5,
                    color='red',)

        ax.set_xlabel('Time $[\mu s]$')
        ax.set_ylabel(frame_properties['derived'][keys[i]]['label']+' '+'['+frame_properties['derived'][keys[i]]['unit']+']')
        ax.set_xlim(time_range)
        ax.set_title(str(keys[i])+ ' vs. time')

        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
        fig.tight_layout(pad=0.1)

        if pdf:
            pdf_pages.savefig()



def frame_properties_dict(exp_id, time, time_unit, distance_unit):

    data_dict={'max':np.zeros([len(time)]),
               'avg':np.zeros([len(time)]),
               'stddev':np.zeros([len(time)]),
               'raw':np.zeros([len(time)]),

               'unit':None,
               'label':None,
               }

    frame_properties={'shot':exp_id,
                      'Time':time,
                      'data':{},
                      'derived':{},
                      'structures':[],
                      }

    """
    Frame characterizing parameters
    """

    key='Angle'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$\phi$'
    frame_properties['data'][key]['unit']='rad'

    key='Angle of least inertia'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$\phi_{ALI}$'
    frame_properties['data'][key]['unit']='rad'

    key='Area'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='Area'
    frame_properties['data'][key]['unit']='$'+distance_unit+'^2$'

    key='Axes length minor'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='a'
    frame_properties['data'][key]['unit']=distance_unit

    key='Axes length major'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='a'
    frame_properties['data'][key]['unit']=distance_unit

    key='Center of gravity radial'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$COG_{rad}$'
    frame_properties['data'][key]['unit']=distance_unit

    key='Center of gravity poloidal'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$COG_{pol}$'
    frame_properties['data'][key]['unit']=distance_unit

    key='Centroid radial'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='Centr. rad.'
    frame_properties['data'][key]['unit']=distance_unit

    key='Centroid poloidal'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='Centr. pol.'
    frame_properties['data'][key]['unit']=distance_unit

    key='Convexity'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='Convexity'
    frame_properties['data'][key]['unit']=''

    key='Elongation'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='Elong.'
    frame_properties['data'][key]['unit']=''

    key='Frame COG radial'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$COG_{frame,rad}$'
    frame_properties['data'][key]['unit']=distance_unit

    key='Frame COG poloidal'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$COG_{frame,rad}$'
    frame_properties['data'][key]['unit']=distance_unit

    key='Position radial'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='R'
    frame_properties['data'][key]['unit']=distance_unit

    key='Position poloidal'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='z'
    frame_properties['data'][key]['unit']=distance_unit

    key='Roundness'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='Roundness'
    frame_properties['data'][key]['unit']=''

    key='Separatrix dist'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$r-r_{sep}$'
    frame_properties['data'][key]['unit']=distance_unit

    key='Size radial'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$d_{rad}$'
    frame_properties['data'][key]['unit']=distance_unit

    key='Size poloidal'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$d_{pol}$'
    frame_properties['data'][key]['unit']=distance_unit

    key='Solidity'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='Solidity'
    frame_properties['data'][key]['unit']=''

    key='Str number'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='N'
    frame_properties['data'][key]['unit']=''

    key='Total bending energy'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$E_{bend}$'
    frame_properties['data'][key]['unit']=''

    key='Total curvature'
    frame_properties['data'][key]=copy.deepcopy(data_dict)
    frame_properties['data'][key]['label']='$\kappa_{tot}$'
    frame_properties['data'][key]['unit']=''

    """
    Differential parameters
    """

    key='Angular velocity angle'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']=''
    frame_properties['derived'][key]['unit']='rad/'+time_unit

    key='Angular velocity ALI'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']=''
    frame_properties['derived'][key]['unit']='rad/'+time_unit

    key='Expansion fraction area'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']='$f_E$'
    frame_properties['derived'][key]['unit']='1/'+time_unit

    key='Expansion fraction axes'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']='$f_{E,area}$'
    frame_properties['derived'][key]['unit']='1/'+time_unit

    key='Velocity radial position'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']='$v_{rad,pos}$'
    frame_properties['derived'][key]['unit']=distance_unit+'/'+time_unit

    key='Velocity poloidal position'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']='$v_{pol,pos}$'
    frame_properties['derived'][key]['unit']=distance_unit+'/'+time_unit

    key='Velocity radial COG'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']='$v_{rad,COG}$'
    frame_properties['derived'][key]['unit']=distance_unit+'/'+time_unit

    key='Velocity poloidal COG'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']='$v_{pol,COG}$'
    frame_properties['derived'][key]['unit']=distance_unit+'/'+time_unit

    key='Velocity radial centroid'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']='$v_{rad,centroid}$'
    frame_properties['derived'][key]['unit']=distance_unit+'/'+time_unit

    key='Velocity poloidal centroid'
    frame_properties['derived'][key]=copy.deepcopy(data_dict)
    frame_properties['derived'][key]['label']='$v_{pol,centroid}$'
    frame_properties['derived'][key]['unit']=distance_unit+'/'+time_unit

    return frame_properties



def read_analyzed_keys():

    analyzed_keys=[#'Centroid radial', 'Centroid poloidal',
                   'Position radial', 'Position poloidal',
                   'Center of gravity radial', 'Center of gravity poloidal',
                   #'Center radial', 'Center poloidal', #added later, might cause trouble
                   
                   'Axes length minor','Axes length major',
                   'Size radial', 'Size poloidal',
                   'Area',
                   'Elongation',

                   'Angle', 'Angle of least inertia',
                   'Velocity radial COG', 'Velocity poloidal COG',
                   'Velocity radial centroid', 'Velocity poloidal centroid',
                   'Velocity radial position', 'Velocity poloidal position',
                   'Expansion fraction area', 'Expansion fraction axes',
                   'Angular velocity angle', 'Angular velocity ALI',

                   'Convexity', 'Solidity', 'Roundness', 'Total curvature',
                   'Total bending energy',
                   
                   #'Born', 'Died', 'Splits', 'Merges', #added later, might cause trouble
                   ]

    return analyzed_keys
