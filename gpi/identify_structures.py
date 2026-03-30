#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 02:12:04 2022

@author: mlampert
"""

#Core imports
import os
import copy
#Importing and setting up the FLAP environment
import flap

import flap_nstx
flap_nstx.register()

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)
#Scientific library imports
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
from flap_nstx.tools import Polygon, FitEllipse, FitGaussian

import cv2

import imutils

import matplotlib.pyplot as plt
#from matplotlib.patches import Ellipse

import numpy as np
import scipy

from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed


def identify_structures(#General inputs
                        data_object=None,                       #Name of the FLAP.data_object
                        exp_id='*',                             #Shot number (if data_object is not used)
                        time=None,                              #Time when the structures need to be evaluated (when exp_id is used)
                        sample=None,                            #Sample number where the structures need to be evaluated (when exp_id is used)
                        spatial=False,                          #Calculate the results in real spatial coordinates
                        pixel=False,                            #Calculate the results in pixel coordinates
                        mfilter_range=5,                        #Range of the median filter

                        ignore_side_structure=False,            #Ignore the structures on the side of the frame
                        ignore_large_structure=False,           #Ignore structures larger than the frame size.
                        smooth_contours=0,                      #smooth the perimeter with corner cutting this many times
                        ellipse_method='linalg',                #linalg or skimage
                        fit_shape='ellipse',                    #ellipse or gaussian
                        str_size_lower_thres=4*0.00375,         #lower threshold for structures
                        str_size_upper_thres=64*0.00375,        #Upper theshold for structures

                        elongation_threshold=0.1,               #Threshold for angle definition, the angle of circular structures cannot be determined.

                        str_finding_method='contour',           #Contour or watershed for now

                        #Contour segmentation inputs
                        nlevel=51,                              #The number of contours to be used for the calculation (default:ysize/mfilter_range=80//5)
                        levels=None,                            #Contour levels from an input and not from automatic calculation
                        threshold_level=None,                   #Threshold level over which it is considered to be a structure
                                                                #if set, the value is subtracted from the data and contours are found after that.
                                                                #Negative values are substituted with 0.
                        filter_struct=True,                     #Filter out the structures with less than filter_level number of contours
                        filter_level=None,                      #The number of contours threshold for structures filtering (default:nlevel//4)
                        remove_interlaced_structures=False,     #Filter out the structures which are interlaced. Only the largest structures is preserved, others are removed.
                        video_resolution=None,
                        structure_video_save=False,
                        #Watershed specific inputs
                        threshold_method='otsu',

                        #Plotting and testing
                        plot_full=False,                        #Plot the steps of the watershed segmentation along with the fitted structures
                        plot_full_for_publication=False,        #Plot it for a publication with correct sizes and all.
                        plot_result=False,                      #Test the result only (plot the contour and the found structures)

                        plot_flux_surfaces=True,                #Plot the magnetic surfaces on the video/frames.
                        surface_data_obj=None,                  #FLAP data object for magnetic surfaces in the same coordinates as the plotting is
                        plot_separatrix=True,                   #Plot the separatrix onto the video/frames.
                        separatrix_data=None,                   #Data from the separatrix in the form of [N,2] [:.0] is horizontal, [:,1] is vertical.

                        verbose=False,
                        test=False,                             #Test the contours and the structures before any kind of processing
                        save_data_for_publication=False,        #Save the data for publication
                        ):

    """
    Identifies, segments, and parametrizes plasma structures within a single 2D GPI frame.

    This core engine isolates turbulent structures (blobs) using either intensity 
    contour mapping or watershed segmentation. It extracts the topology of the 
    structures at their Full Width at Half Maximum (FWHM), filters out artifacts, 
    and fits geometric models (ellipses or 2D Gaussians) to quantify their kinematics.

    Args:
        data_object (flap.DataObject, str, np.ndarray, optional): The 2D frame data.
        exp_id (int or str, optional): Shot number if loading data directly. Defaults to '*'.
        time (float, optional): Exact time step to analyze.
        sample (int, optional): Exact sample index to analyze.
        spatial (bool, optional): If True, coordinates are returned in real space [m].
        pixel (bool, optional): If True, coordinates are returned in pixels.
        mfilter_range (int, optional): Kernel size for the initial median filter.
        ignore_side_structure (bool, optional): Drops structures touching the frame boundaries.
        ignore_large_structure (bool, optional): Drops structures exceeding frame dimensions.
        smooth_contours (int, optional): Number of refinement iterations for polygon smoothing.
        ellipse_method (str, optional): Math method for ellipse fitting ('linalg' or 'skimage').
        fit_shape (str, optional): Geometric model to fit ('ellipse' or 'gaussian').
        str_size_lower_thres (float, optional): Minimum absolute size for a valid structure.
        str_size_upper_thres (float, optional): Maximum absolute size for a valid structure.
        elongation_threshold (float, optional): Axis ratio threshold below which angle is undefined.
        str_finding_method (str, optional): Segmentation algorithm ('contour' or 'watershed').
        nlevel (int, optional): Number of contours used in 'contour' segmentation.
        levels (list, optional): Explicit contour intensity levels.
        threshold_level (float, optional): Absolute intensity cutoff for background subtraction.
        filter_struct (bool, optional): If True, filters structures with too few contours.
        filter_level (int, optional): The minimum contour count threshold.
        remove_interlaced_structures (bool, optional): Removes internal sub-peaks if True.
        threshold_method (str, optional): Method for watershed binary cutoff ('otsu').
        plot_full (bool, optional): Plots the full segmentation pipeline.
        plot_result (bool, optional): Plots the final fitted structures.
        save_data_for_publication (bool, optional): Exports step-by-step arrays to .txt files.
        (Other standard plotting args passed automatically...)

    Returns:
        list of dict: A list representing identified structures, where each dict contains:
            - 'Polygon', 'Half path', 'Vertices', 'X coord', 'Y coord', 'Data'
            - 'Area', 'Intensity', 'Convexity', 'Solidity', 'Roundness'
            - 'Center', 'Centroid', 'Center of gravity' (both radial and poloidal)
            - 'Size', 'Axes length', 'Angle', 'Elongation'
            - 'Ellipse' or 'Gaussian' (The raw fit object)
    """

    if type(data_object) is str:
        data_object=flap.get_data_object_ref(data_object, exp_id=exp_id)
        if len(data_object.data.shape) != 2:
            raise IOError('The inpud data_object is not 2D. The method only processes 2D data.')

    elif data_object is None:
        if (exp_id is None) or ((time is None) and (sample is None)):
            raise IOError('exp_id and time needs to be set if data_object is not set.')
        try:
            data_object=flap.get_data_object_ref('GPI', exp_id=exp_id)
        except:
            print('---- Reading GPI data ----')
            data_object=flap.get_data('NSTX_GPI', exp_id=exp_id, name='', object_name='GPI')

        if (time is not None) and (sample is not None):
            raise IOError('Either time or sample can be set, not both.')

        if time is not None:
            data_object=data_object.slice_data(slicing={'Time':time})
        if sample is not None:
            data_object=data_object.slice_data(slicing={'Sample':sample})

    if type(data_object) is type(flap.DataObject()):
        try:
            data_object.data
        except:
            raise IOError('The input data object should be a flap.DataObject')

        if len(data_object.data.shape) != 2:
            raise TypeError('The frame dataobject needs to be a 2D object without a time coordinate.')

        if pixel:
            x_coord_name='Image x'
            x_unit_name='[pix]'

            y_coord_name='Image y'
            y_unit_name='[pix]'

        elif spatial:
            x_coord_name='Device R'
            x_unit_name='[m]'

            y_coord_name='Device z'
            y_unit_name='[m]'
        else:
            raise TypeError('Cannot do pixel and spatial calculation at the same time.')

        x_coord=data_object.coordinate(x_coord_name)[0]
        y_coord=data_object.coordinate(y_coord_name)[0]

        x_coord_pix=data_object.coordinate('Image x')[0]
        y_coord_pix=data_object.coordinate('Image y')[0]

        if test:
            print(x_coord.shape,y_coord.shape)
        data = scipy.ndimage.median_filter(data_object.data, mfilter_range)

    elif type(data_object) is np.ndarray:
        data=data_object
        x_coord=np.arange(data_object.shape[0])
        y_coord=np.arange(data_object.shape[1])
        x_coord_pix=x_coord
        y_coord_pix=y_coord
    else:
        raise ValueError('Input is not str, None, flap.DataObject or np.ndarray')
    structures=[]

    one_structure={'Polygon':None,  #Calculated during segmentation
                   'Half path':None,

                   'Vertices':None, #Calculated after segmentation
                   'X coord':None,
                   'Y coord':None,
                   'Data':None,

                   'Born':False,    #Calculated during tracking in track_structures
                   'Died':False,
                   
                   'Missing frames':None,
                   'Active':False,
                   
                   'Splits':False,
                   'Merges':False,
                   'Label':None,
                   'Parent':[],
                   'Child':[],
                   }

    """
    ----------------
    READING THE DATA
    ----------------
    """


    if test:
        plt.cla()

    if threshold_level is not None:
        if data.max() < threshold_level:
            if verbose: print('The maximum of the signal doesn\'t reach the threshold level.')
            return
        data_thresholded = data - threshold_level
        data_thresholded[np.where(data_thresholded < 0)] = 0.
    else:
        data_thresholded = data

    if str_finding_method == 'contour':

        if levels is None:
            levels=np.arange(nlevel)/(nlevel-1)*(data_thresholded.max()-data_thresholded.min())+data_thresholded.min()
        else:
            nlevel=len(levels)

        try:
            structure_contours=plt.contourf(x_coord,
                                            y_coord,
                                            data_thresholded, levels=levels)
        except:
            plt.cla()
            plt.close()
            print('Failed to create the contours for the structures.')
            return None

        prelim_structures=[]
        pre_one_struct={'Paths':[None],
                        'Levels':[None]}

        if test:
            print('Plotting levels')
        else:
            plt.close()
        #The following lines are the core of the code. It separates the structures
        #from each other and stores the in the structure list.

        """
        Steps of the algorithm:

            1st step: Take the paths at the highest level and store them. These
                      create the initial structures
            2nd step: Take the paths at the second highest level
                2.1 step: if either of the previous paths contain either of
                          the paths at this level, the corresponding
                          path is appended to the contained structure from the
                          previous step.
                2.2 step: if none of the previous structures contain the contour
                          at this level, a new structure is created.
            3rd step: Repeat the second step until it runs out of levels.
            4th step: Delete those structures from the list which doesn't have
                      enough paths to be called a structure.

        (Note: a path is a matplotlib path, a structure is a processed path)
        """
        for i_lev in range(len(structure_contours.collections)-1,-1,-1):
            cur_lev_paths=structure_contours.collections[i_lev].get_paths()
            n_paths_cur_lev=len(cur_lev_paths)

            if len(cur_lev_paths) > 0:
                if len(prelim_structures) == 0:
                    for i_str in range(n_paths_cur_lev):
                        prelim_structures.append(copy.deepcopy(pre_one_struct))
                        prelim_structures[i_str]['Paths'][0]=cur_lev_paths[i_str]
                        prelim_structures[i_str]['Levels'][0]=levels[i_lev]
                else:
                    for i_cur in range(n_paths_cur_lev):
                        new_path=True
                        cur_path=cur_lev_paths[i_cur]
                        for j_prev in range(len(prelim_structures)):
                            if cur_path.contains_path(prelim_structures[j_prev]['Paths'][-1]):
                                prelim_structures[j_prev]['Paths'].append(cur_path)
                                prelim_structures[j_prev]['Levels'].append(levels[i_lev])
                                new_path=False
                        if new_path:
                            prelim_structures.append(copy.deepcopy(pre_one_struct))
                            prelim_structures[-1]['Paths'][0]=cur_path
                            prelim_structures[-1]['Levels'][0]=levels[i_lev]
                        if test:
                            x=cur_lev_paths[i_cur].to_polygons()[0][:,0]
                            y=cur_lev_paths[i_cur].to_polygons()[0][:,1]
                            plt.plot(x,y)
                            plt.axis('equal')
                            plt.pause(0.001)

        #Cut the structures based on the filter level
        if filter_level is None:
            filter_level=nlevel//5

        if filter_struct:
            cut_structures=[]
            for i_str in range(len(prelim_structures)):
                if len(prelim_structures[i_str]['Levels']) > filter_level:
                    cut_structures.append(prelim_structures[i_str])
        prelim_structures=cut_structures

        if test:
            print('Plotting structures')
            plt.cla()
            for struct in prelim_structures:
                plt.contourf(x_coord, y_coord, data, levels=levels)
                for path in struct['Paths']:
                    x=path.to_polygons()[0][:,0]
                    y=path.to_polygons()[0][:,1]
                    plt.plot(x,y)
                plt.pause(0.001)
                plt.cla()

            plt.axis('equal')
            plt.contourf(x_coord, y_coord, data, levels=levels)
            plt.colorbar()

        #Finding the contour at the half level for each structure and
        #calculating its properties
        if len(prelim_structures) > 1:
            #Finding the paths at FWHM
            paths_at_half=[]
            for i_str in range(len(prelim_structures)):
                half_level=(prelim_structures[i_str]['Levels'][-1]+prelim_structures[i_str]['Levels'][0])/2.
                ind_at_half=np.argmin(np.abs(prelim_structures[i_str]['Levels']-half_level))
                paths_at_half.append(prelim_structures[i_str]['Paths'][ind_at_half])

            #Process the structures which are embedded (cut the inner one)
            if remove_interlaced_structures:
                structures_to_be_removed=[]
                for ind_path1 in range(len(paths_at_half)):
                    for ind_path2 in range(ind_path1,len(paths_at_half),1):
                        if ind_path1 != ind_path2:
                            if paths_at_half[ind_path2].contains_path(paths_at_half[ind_path1]):
                                structures_to_be_removed.append(ind_path1)
                            if paths_at_half[ind_path2] == paths_at_half[ind_path1]:
                                structures_to_be_removed.append(ind_path2)
                structures_to_be_removed=np.unique(structures_to_be_removed)
                cut_structures=[]
                for i_str in range(len(prelim_structures)):
                    if i_str not in structures_to_be_removed:
                        cut_structures.append(prelim_structures[i_str])
                prelim_structures=cut_structures
        if test: print('N after removing interlaced:',len(prelim_structures))
        
        for i_str in range(len(prelim_structures)):

            str_levels=prelim_structures[i_str]['Levels']
            half_level=(str_levels[-1]+str_levels[0])/2.
            ind_at_half=np.argmin(np.abs(str_levels-half_level))
            n_path=len(prelim_structures[i_str]['Levels'])

            polygon_areas=np.zeros(n_path)
            polygon_centroids=np.zeros([n_path,2])
            polygon_intensities=np.zeros(n_path)

            for i_path in range(n_path):
                polygon=prelim_structures[i_str]['Paths'][i_path].to_polygons()
                if polygon != []:
                    polygon=polygon[0]
                    polygon_areas[i_path]=flap_nstx.tools.Polygon(polygon[:,0],polygon[:,1]).area
                    polygon_centroids[i_path,:]=flap_nstx.tools.Polygon(polygon[:,0],polygon[:,1]).centroid
                if i_path == 0:
                    polygon_intensities[i_path]=polygon_areas[i_path]*str_levels[i_path]
                else:
                    polygon_intensities[i_path]=(polygon_areas[i_path]-polygon_areas[i_path-1])*str_levels[i_path]

            half_coords=prelim_structures[i_str]['Paths'][ind_at_half].to_polygons()[0]

            """
            These lines cut down the computation cost of .contains_points() by
            having only a rectangular frame around the polygon to be checked not
            the entire frame of measurement.
            """

            coords_2d=[]
            data_inside_half=[]

            x_inds_of_half=list(np.where(np.logical_and(x_coord[:,0] >= np.min(half_coords[:,0]),
                                                        x_coord[:,0] <= np.max(half_coords[:,0])))[0])
            y_inds_of_half=list(np.where(np.logical_and(y_coord[0,:] >= np.min(half_coords[:,1]),
                                                        y_coord[0,:] <= np.max(half_coords[:,1])))[0])
            for i_coord_x in x_inds_of_half:
                for j_coord_y in y_inds_of_half:
                    coords_2d.append([x_coord[:,0][i_coord_x],
                                      y_coord[0,:][j_coord_y]])
                    data_inside_half.append(data_thresholded[i_coord_x,j_coord_y])

            coords_2d=np.asarray(coords_2d)
            data_inside_half=np.asarray(data_inside_half)

            try:
                ind_inside_half_path=prelim_structures[i_str]['Paths'][ind_at_half].contains_points(coords_2d)
            except Exception as e:
                print('Exception in identify structures line 392: ',e)

            x_data=coords_2d[ind_inside_half_path,0]
            y_data=coords_2d[ind_inside_half_path,1]
            enclosed_data=data_inside_half[ind_inside_half_path]

            half_polygon=Polygon(x=half_coords[:,0],
                                 y=half_coords[:,1],
                                 x_data=np.asarray(x_data),
                                 y_data=np.asarray(y_data),
                                 x_data_pix=x_coord_pix[ind_inside_half_path],
                                 y_data_pix=y_coord_pix[ind_inside_half_path],
                                 data=np.asarray(enclosed_data),
                                 test=test)

            if smooth_contours > 0:
                half_polygon.smooth(refinements=smooth_contours)

            structures.append(copy.deepcopy(one_structure))
            structures[-1]['Half path']=prelim_structures[i_str]['Paths'][ind_at_half]
            structures[-1]['Polygon']=half_polygon

    elif str_finding_method == 'watershed':

        thresh = threshold_otsu(data_thresholded)
        binary = np.asarray(data_thresholded > thresh, dtype='uint8')

        #distance_transformed = ndimage.distance_transform_edt(data_thresholded) #THIS IS UNNECESSARY AS THE STRUCTURES DO NOT HAVE DISTINCT BORDERS
        localMax = peak_local_max(copy.deepcopy(data_thresholded),
                                  min_distance=5,
                                  #indices=False,
                                  labels=binary)

        peaks_mask = np.zeros_like(data_thresholded,
                                    dtype=bool)

        peaks_mask[tuple(localMax.T)] = True

        markers = scipy.ndimage.label(#localMax,
                                      peaks_mask,
                                      structure=np.ones((3, 3)))[0]

        labels = watershed(-data_thresholded, markers, mask=binary)

        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue
            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(data.shape, dtype="uint8")
            mask[labels == label] = 255
            	# detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

            cnts = imutils.grab_contours(cnts)
            max_contour_prelim = max(cnts, key=cv2.contourArea)
            max_contour = np.squeeze(max_contour_prelim)

            try:

                if len(max_contour.shape) == 2:
                    if spatial:
                        max_contour=np.asarray([x_coord[max_contour[:,1],
                                                        max_contour[:,0]],
                                                y_coord[max_contour[:,1],
                                                        max_contour[:,0]]])

                    else:
                        max_contour=max_contour.T
                else:
                    continue
            except Exception as e:
                if verbose:
                    print('Exception at flap_nstx.gpi.identify_structures line 438:')
                    print(e)

            from matplotlib.path import Path

            if max_contour.shape[0] != 1:
                indices=np.where(labels == label)
                codes=[Path.MOVETO]
                for i_code in range(1,len(max_contour.transpose()[:,0])):
                    codes.append(Path.CURVE4)
                codes.append(Path.CLOSEPOLY)

                max_contour_looped=np.zeros([max_contour.shape[1]+1,
                                             max_contour.shape[0]])
                max_contour_looped[0:-1,:]=max_contour.transpose()
                max_contour_looped[-1,:]=max_contour[:,0]
                vertices=copy.deepcopy(max_contour_looped)

                try:
                    full_polygon=Polygon(x=vertices[:,0],
                                         y=vertices[:,1],
                                         x_data=x_coord[indices],
                                         y_data=y_coord[indices],
                                         x_data_pix=x_coord_pix[indices],
                                         y_data_pix=y_coord_pix[indices],
                                         data=data[indices],
                                         test=test)
                    if smooth_contours > 0:
                        full_polygon.smooth(refinements=smooth_contours)

                    structures.append(copy.deepcopy(one_structure))
                    structures[-1]['Half path']=Path(max_contour_looped,codes)
                    structures[-1]['Polygon']=full_polygon

                except Exception as e:
                    if verbose: 
                        print('Exception in flap_nstx.gpi.identify_structures at line 506:')
                        print(e)
                    #continue


    #Calculate the ellipse and its properties for the half level contours
# --- 5. Geometric Extraction & Fitting ---
    fitted_structures = []
    for struct in structures:
        poly = struct['Polygon']
        
        struct.update({
            'Vertices': poly.vertices, 
            'X coord': poly.x_data, 
            'Y coord': poly.y_data, 
            'Data': poly.data,
            
            'Centroid': poly.centroid, 
            'Centroid radial': poly.centroid[0], 
            'Centroid poloidal': poly.centroid[1],
            
            'Area': poly.area, 
            'Intensity': poly.intensity, 
            
            'Center of gravity': poly.center_of_gravity,
            'Center of gravity radial': poly.center_of_gravity[0], 
            'Center of gravity poloidal': poly.center_of_gravity[1],
            
            'Convexity': poly.convexity, 
            'Solidity': poly.solidity, 
            'Roundness': poly.roundness,
            'Total bending energy': poly.total_bending_energy, 
            'Total curvature': poly.convexity,
            'Angle of least inertia': poly.principal_axes_angle
        })

        if fit_shape == 'ellipse':
            fit_struct = FitEllipse(x=poly.x, y=poly.y, method=ellipse_method, verbose=verbose)
            struct['Ellipse'] = fit_struct
            
        elif fit_shape == 'gaussian':
            fit_struct = FitGaussian(x=poly.x_data, y=poly.y_data, data=poly.data, verbose=verbose)
            struct['Gaussian'] = fit_struct

        struct.update({
            'Axes length': fit_struct.axes_length,
            'Axes length minor': fit_struct.axes_length[0], 
            'Axes length major': fit_struct.axes_length[1],
            
            'Center': fit_struct.center, 
            'Center radial': fit_struct.center[0], 
            'Center poloidal': fit_struct.center[1],
            
            'Position': fit_struct.center, 
            'Position radial': fit_struct.center[0], 
            'Position poloidal': fit_struct.center[1],
            
            'Size': fit_struct.size, 
            'Size radial': fit_struct.size[0], 
            'Size poloidal': fit_struct.size[1],
            
            'Angle': fit_struct.angle, 
            'Elongation': fit_struct.elongation
        })

        if struct['Axes length'][1] / struct['Axes length'][0] < elongation_threshold:
            struct['Angle'] = np.nan

        if np.iscomplex(fit_struct.size[0]) or np.iscomplex(fit_struct.size[1]):
            fit_struct.set_invalid()

        if ignore_large_structure and (fit_struct.size[0] > (x_coord.max() - x_coord.min()) or 
                                       fit_struct.size[1] > (y_coord.max() - y_coord.min())):
            fit_struct.set_invalid()

        # --- Validation & Filtering ---
        if str_size_lower_thres is not None and struct['Size'] is not None:
            if (struct['Size'][0] < str_size_lower_thres or struct['Size'][1] < str_size_lower_thres or
                struct['Size'][0] > str_size_upper_thres or struct['Size'][1] > str_size_upper_thres):
                continue # Skip this structure
                
            if ignore_side_structure and (np.any(struct['X coord'] == x_coord.min()) or np.any(struct['X coord'] == x_coord.max()) or
                                          np.any(struct['Y coord'] == y_coord.min()) or np.any(struct['Y coord'] == y_coord.max()) or
                                          fit_struct.center[0] < x_coord.min() or fit_struct.center[0] > x_coord.max() or
                                          fit_struct.center[1] < y_coord.min() or fit_struct.center[1] > y_coord.max()):
                continue # Skip this structure

        fitted_structures.append(struct)

    structures = fitted_structures
    
    if test: print('Number of structures after fitting:',len(structures))

    if plot_result:
        if structure_video_save:
            my_dpi=80
            fig,ax=plt.subplots(figsize=(video_resolution[0]/my_dpi,
                                         video_resolution[1]/my_dpi),
                                dpi=my_dpi)
        else:
            fig,ax=plt.subplots(figsize=(8.5/2.54, 8.5/2.54))

        if levels is not None:
            plt.contourf(x_coord,
                         y_coord,
                         data,
                         levels=nlevel)

            if plot_flux_surfaces and surface_data_obj is not None:
                plt.contour(surface_data_obj.coordinate(x_coord_name)[0],
                            surface_data_obj.coordinate(y_coord_name)[0],
                            surface_data_obj.data.transpose(),
                            levels=51)

            if plot_separatrix and separatrix_data is not None:
                plt.plot(separatrix_data[:,0],
                         separatrix_data[:,1],
                         color='red')

        else:
            plt.contourf(x_coord, y_coord, data, levels=levels)

        ax.set_aspect(1.0)
        plt.colorbar()

    elif plot_full and str_finding_method == 'watershed':
        if not plot_full_for_publication:
            plt.cla()
            fig,axes=plt.subplots(2,2,
                                  figsize=(10,10))

            ax=axes[0,0]
            ax.contourf(x_coord,
                        y_coord,
                        data,
                        levels=nlevel)
            ax.set_title('data')

            ax=axes[0,1]
            ax.contourf(x_coord,
                        y_coord,
                        data_thresholded)
            ax.set_title('thresholded')

            ax=axes[1,0]
            ax.contourf(x_coord,
                        y_coord,
                        binary)
            ax.set_title('binary')

            ax=axes[1,1]
            ax.contourf(x_coord,
                        y_coord,
                        labels)
            ax.set_title('labels')

            for ax in np.ravel(axes):
                ax.set_aspect(1.0)
                ax.set_xlabel('Image x')
                ax.set_ylabel('Image y')
                ax.set_xlim([x_coord.min(),x_coord.max()])
                ax.set_ylim([y_coord.min(),y_coord.max()])
        else:
            plt.cla()
            np.set_printoptions(threshold=np.inf)
            fig,axes=plt.subplots(2,2,
                                  figsize=(8.5/2.54,10/2.54))
            xpos, ypos= (-0.4,1.1)
            ax=axes[0,0]
            ax.contourf(x_coord,
                        y_coord,
                        data,
                        levels=nlevel)
            ax.set_title('Pre-processed frame')
            ax.set_aspect(1.0)
            ax.set_xlabel('x [pix]')
            ax.set_ylabel('y [pix]')
            ax.set_xlim([x_coord.min(),x_coord.max()])
            ax.set_ylim([y_coord.min(),y_coord.max()])
            ax.text(xpos, ypos, '(a)', transform=ax.transAxes, size=9)
         
            if save_data_for_publication:
                with open(f"{wd}+'/a_preproc_frame.txt", 'w+') as f:
                    f.write(str(data))
                
            ax=axes[0,1]
            ax.contourf(x_coord,
                        y_coord,
                        data_thresholded)
            ax.set_title('Thresholded frame')
            ax.set_aspect(1.0)
            ax.set_xlabel('x [pix]')
            #ax.set_ylabel('y [pix]')
            # ax.get_yaxis().set_visible(False)
            ax.set_xlim([x_coord.min(),x_coord.max()])
            ax.set_ylim([y_coord.min(),y_coord.max()])
            ax.text(xpos, ypos, '(b)', transform=ax.transAxes, size=9)

            if save_data_for_publication:
                with open(wd+'/b_thresholded.txt','w+') as file1:
                    file1.write(str(data_thresholded))

            ax=axes[1,0]
            ax.contourf(x_coord,
                        y_coord,
                        binary)
            ax.set_title('Binary frame')
            ax.set_aspect(1.0)
            ax.set_xlabel('x [pix]')
            #ax.set_ylabel('y [pix]')
            # ax.get_yaxis().set_visible(False)
            ax.set_xlim([x_coord.min(),x_coord.max()])
            ax.set_ylim([y_coord.min(),y_coord.max()])
            ax.text(xpos, ypos, '(c)', transform=ax.transAxes, size=9)

            if save_data_for_publication:
                with open(wd+'/c_binary.txt','w+') as file1:
                    file1.write(str(binary))

            ax=axes[1,1]
            ax.contourf(x_coord,
                        y_coord,
                        labels)
            ax.set_title('Segmented frame')
            ax.set_aspect(1.0)
            ax.set_xlabel('x [pix]')
            #ax.set_ylabel('y [pix]')
            # ax.get_yaxis().set_visible(False)
            ax.set_xlim([x_coord.min(),x_coord.max()])
            ax.set_ylim([y_coord.min(),y_coord.max()])
            ax.text(xpos, ypos, '(d)', transform=ax.transAxes, size=9)

            if save_data_for_publication:
                with open(wd+'/d_segmented.txt','w+') as file1:
                    file1.write(str(labels))

            # plt.tight_layout(pad=0.1)

    elif plot_full and str_finding_method == 'contour':
        raise ValueError('plot_full cannot be set when contour segmentation is performed.')

    else:
        pass

    if len(structures) > 0:
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

                if plot_result or plot_full: #This plots the structures and the fit ellipses one by one
                    if not plot_full_for_publication:
                        fig,axes=plt.subplots(1,4,
                                              figsize=(17/2.54,8.5/2.54))
                    if plot_result:
                        _plot_ellipses_centers(ax,
                                               x_polygon,
                                               y_polygon,
                                               x_ellipse,
                                               y_ellipse,
                                               structures[i_str],
                                               ellipse_linewidth=0.5)
                    if plot_full:
                        for ax_cur in np.ravel(axes):
                            _plot_ellipses_centers(ax_cur, x_polygon, y_polygon, x_ellipse, y_ellipse, structures[i_str])


                if save_data_for_publication:
                    exp_id=data_object.exp_id
                    time=data_object.coordinate('Time')[0][0,0]
                    filename=wd+'/'+str(exp_id)+'_'+str(time)+'_half_path_no.'+str(i_str)+'.txt'
                    file1=open(filename, 'w+')
                    for i in range(len(x_polygon)):
                        file1.write(str(x_polygon[i])+'\t'+str(y_polygon[i])+'\n')
                    file1.close()

                    filename=wd+'/'+str(exp_id)+'_'+str(time)+'_fit_ellipse_no.'+str(i_str)+'.txt'
                    file1=open(filename, 'w+')
                    for i in range(len(x_ellipse)):
                        file1.write(str(x_ellipse[i])+'\t'+str(y_ellipse[i])+'\n')
                    file1.close()

            if plot_result:
                ax.set_xlabel(x_coord_name.replace('Device ','') + ' '+ x_unit_name)
                ax.set_ylabel(y_coord_name.replace('Device ','') + ' '+ y_unit_name)
                ax.set_title(str(exp_id)+' @ '+str(data_object.coordinate('Time')[0][0,0]))
                plt.show()
                plt.pause(0.001)

        plt.xlim([x_coord.min(),x_coord.max()])
        plt.ylim([y_coord.min(),y_coord.max()])

        if save_data_for_publication:
            exp_id=data_object.exp_id
            time=data_object.coordinate('Time')[0][0,0]
            filename=wd+'/'+str(exp_id)+'_'+str(time)+'_raw_data.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()

    return structures

def _plot_ellipses_centers(ax_cur,
                           x_polygon, y_polygon,
                           x_ellipse, y_ellipse,
                           structure,
                           polygon_color=None,
                           ellipse_color=None,
                           polygon_linewidth=1,
                           ellipse_linewidth=1,
                           semiaxis_linewidth=1,
                           plot_structure_mid=False):
    """Internal helper to overlay geometric fits on frame axes."""
    
    # Plot bounds and fit
    poly_args = {'color': polygon_color} if polygon_color else {}
    el_args = {'color': ellipse_color} if ellipse_color else {}
    
    ax_cur.plot(x_polygon, y_polygon, linewidth=polygon_linewidth, **poly_args)
    ax_cur.plot(x_ellipse, y_ellipse, linewidth=ellipse_linewidth, **el_args)

    # Plot Semi-axis
    cx, cy = structure['Center']
    a, angle = structure['Axes length'][0], structure['Angle']
    ax_cur.plot([cx - a*np.cos(angle), cx + a*np.cos(angle)],
                [cy - a*np.sin(angle), cy + a*np.sin(angle)],
                color='magenta', linewidth=semiaxis_linewidth)
    
    # Plot Centers
    if plot_structure_mid:
        ax_cur.scatter(*structure['Centroid'], color='yellow')
        ax_cur.scatter(*structure['Center of gravity'], color='red')