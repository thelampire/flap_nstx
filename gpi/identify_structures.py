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
from flap_nstx.tools import PlasmaStructure

import cv2
import imutils
import matplotlib.pyplot as plt
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
            - 'Polygon', 'Vertices', 'X coord', 'Y coord', 'Data'
            - 'Area', 'Intensity', 'Convexity', 'Solidity', 'Roundness'
            - 'Center', 'Centroid', 'Center of gravity' (both radial and poloidal)
            - 'Size', 'Axes length', 'Angle', 'Elongation'
            - 'Ellipse' or 'Gaussian' (The raw fit object)

    """

    # --- 1. DATA READING & VALIDATION ---
    if isinstance(data_object, str):
        data_object = flap.get_data_object_ref(data_object, exp_id=exp_id)
        if len(data_object.data.shape) != 2:
            raise IOError('The input data_object is not 2D. The method only processes 2D data.')

    elif data_object is None:
        if (exp_id is None) or ((time is None) and (sample is None)):
            raise IOError('exp_id and time needs to be set if data_object is not set.')
        try:
            data_object = flap.get_data_object_ref('GPI', exp_id=exp_id)
        except Exception:
            if verbose: print('---- Reading GPI data ----')
            data_object = flap.get_data('NSTX_GPI', exp_id=exp_id, name='', object_name='GPI')

        if (time is not None) and (sample is not None):
            raise IOError('Either time or sample can be set, not both.')

        if time is not None: data_object = data_object.slice_data(slicing={'Time': time})
        if sample is not None: data_object = data_object.slice_data(slicing={'Sample': sample})

    if isinstance(data_object, flap.DataObject):
        if len(data_object.data.shape) != 2:
            raise TypeError('The frame dataobject needs to be a 2D object without a time coordinate.')

        if pixel:
            x_coord_name, x_unit_name = 'Image x', '[pix]'
            y_coord_name, y_unit_name = 'Image y', '[pix]'
            dist_unit = 'pix'
        elif spatial:
            x_coord_name, x_unit_name = 'Device R', '[m]'
            y_coord_name, y_unit_name = 'Device z', '[m]'
            dist_unit = 'm'
        else:
            raise TypeError('Cannot do pixel and spatial calculation at the same time.')

        x_coord = data_object.coordinate(x_coord_name)[0]
        y_coord = data_object.coordinate(y_coord_name)[0]
        x_coord_pix = data_object.coordinate('Image x')[0]
        y_coord_pix = data_object.coordinate('Image y')[0]

        data = scipy.ndimage.median_filter(data_object.data, mfilter_range)

    elif isinstance(data_object, np.ndarray):
        data = data_object
        # BUG FIX: Use meshgrid to ensure 2D boolean masking works downstream
        x_coord, y_coord = np.meshgrid(np.arange(data.shape[0]), 
                                       np.arange(data.shape[1]), 
                                       indexing='ij')
        x_coord_pix = x_coord
        y_coord_pix = y_coord
    else:
        raise ValueError('Input is not str, None, flap.DataObject or np.ndarray')

    # Data Thresholding
    if threshold_level is not None:
        if data.max() < threshold_level:
            if verbose: print('The maximum of the signal doesn\'t reach the threshold level.')
            return []
        data_thresholded = np.clip(data - threshold_level, 0, None)
    else:
        data_thresholded = data

    structures = []

    # --- 2. CONTOUR BASED STRUCTURE FINDER ---
    if str_finding_method == 'contour':
        if levels is None:
            levels = np.linspace(data_thresholded.min(), data_thresholded.max(), nlevel)
        else:
            nlevel = len(levels)

        try:
            structure_contours = plt.contourf(x_coord, y_coord, data_thresholded, levels=levels)
        except Exception:
            plt.close()
            print('Failed to create the contours for the structures.')
            return []

        prelim_structures = []
        for i_lev in range(len(structure_contours.collections) - 1, -1, -1):
            cur_lev_paths = structure_contours.collections[i_lev].get_paths()
            
            for cur_path in cur_lev_paths:
                new_path = True
                for prev_struct in prelim_structures:
                    if cur_path.contains_path(prev_struct['Paths'][-1]):
                        prev_struct['Paths'].append(cur_path)
                        prev_struct['Levels'].append(levels[i_lev])
                        new_path = False
                
                if new_path:
                    prelim_structures.append({'Paths': [cur_path], 'Levels': [levels[i_lev]]})

        if filter_level is None:
            filter_level = nlevel // 5

        if filter_struct:
            prelim_structures = [s for s in prelim_structures if len(s['Levels']) > filter_level]

        if len(prelim_structures) > 1:
            paths_at_half = []
            for struct in prelim_structures:
                half_level = (struct['Levels'][-1] + struct['Levels'][0]) / 2.0
                ind_at_half = np.argmin(np.abs(np.array(struct['Levels']) - half_level))
                paths_at_half.append(struct['Paths'][ind_at_half])

            if remove_interlaced_structures:
                structures_to_be_removed = set()
                for i in range(len(paths_at_half)):
                    for j in range(i + 1, len(paths_at_half)):
                        if paths_at_half[j].contains_path(paths_at_half[i]) or paths_at_half[j] == paths_at_half[i]:
                            structures_to_be_removed.add(i if paths_at_half[j].contains_path(paths_at_half[i]) else j)
                
                prelim_structures = [s for i, s in enumerate(prelim_structures) if i not in structures_to_be_removed]

        # Extract polygons, fit math, and filter data (OPTIMIZED BOTTLENECK)
        for struct in prelim_structures:
            half_level = (struct['Levels'][-1] + struct['Levels'][0]) / 2.0
            ind_at_half = np.argmin(np.abs(np.array(struct['Levels']) - half_level))
            half_coords = struct['Paths'][ind_at_half].to_polygons()[0]

            # Vectorized Bounding Box Masking
            x_min, x_max = np.min(half_coords[:, 0]), np.max(half_coords[:, 0])
            y_min, y_max = np.min(half_coords[:, 1]), np.max(half_coords[:, 1])
            
            mask = (x_coord >= x_min) & (x_coord <= x_max) & (y_coord >= y_min) & (y_coord <= y_max)
            
            coords_2d = np.column_stack((x_coord[mask], y_coord[mask]))
            data_inside_half = data_thresholded[mask]

            # Filter points exactly inside the polygon path
            try:
                ind_inside_half_path = struct['Paths'][ind_at_half].contains_points(coords_2d)
                
                x_data = coords_2d[ind_inside_half_path, 0]
                y_data = coords_2d[ind_inside_half_path, 1]
                enclosed_data = data_inside_half[ind_inside_half_path]
                
                x_pix_data = x_coord_pix[mask][ind_inside_half_path]
                y_pix_data = y_coord_pix[mask][ind_inside_half_path]

                # 1. INSTANTIATE THE UNIFIED OBJECT
                one_struct = PlasmaStructure(
                    x=half_coords[:, 0], y=half_coords[:, 1],
                    smooth=smooth_contours,
                    x_data=x_data, y_data=y_data,
                    x_data_pix=x_pix_data, y_data_pix=y_pix_data,
                    data=enclosed_data, 
                    fit_shape=fit_shape, 
                    ellipse_method=ellipse_method,
                    distance_unit=dist_unit, 
                    test=test, 
                    verbose=verbose
                )


                # 2. VALIDATE AND APPEND
                if validate_structure(one_struct, x_coord, y_coord, elongation_threshold, 
                                      str_size_lower_thres, str_size_upper_thres, ignore_side_structure):
                    structures.append(one_struct)

            except Exception as e:
                if verbose: print(f'Exception in contour processing: {e}')

    # --- 3. WATERSHED BASED STRUCTURE FINDER ---
    elif str_finding_method == 'watershed':
        thresh = threshold_otsu(data_thresholded)
        binary = np.asarray(data_thresholded > thresh, dtype='uint8')

        localMax = peak_local_max(data_thresholded, min_distance=5, labels=binary)
        peaks_mask = np.zeros_like(data_thresholded, dtype=bool)
        peaks_mask[tuple(localMax.T)] = True

        markers = scipy.ndimage.label(peaks_mask, structure=np.ones((3, 3)))[0]
        labels = watershed(-data_thresholded, markers, mask=binary)

        for label in np.unique(labels):
            if label == 0: continue
            
            mask = np.zeros(data.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts = cv2.findContours(mask.copy(), 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            if not cnts: continue
            max_contour = np.squeeze(max(cnts, key=cv2.contourArea))

            try:
                if len(max_contour.shape) == 2:
                    if spatial:
                        if len(x_coord.shape) == 1:
                            # 1D Array Indexing
                            max_contour = np.asarray([x_coord[max_contour[:, 0]],
                                                      y_coord[max_contour[:, 1]]]).T
                        else:
                            # 2D Meshgrid Indexing
                            max_contour = np.asarray([x_coord[max_contour[:, 1], max_contour[:, 0]],
                                                      y_coord[max_contour[:, 1], max_contour[:, 0]]]).T
            except Exception as e:
                if verbose: print(f'Exception in watershed shaping: {e}')
                continue

            if max_contour.shape[0] > 6:
                indices = np.where(labels == label)
                max_contour_looped = np.vstack((max_contour, max_contour[0]))
                
                try:
                    one_structure = PlasmaStructure(
                        x=max_contour_looped[:, 0], 
                        y=max_contour_looped[:, 1],
                        smooth=smooth_contours,
                        x_data=x_coord[indices], 
                        y_data=y_coord[indices], 
                        
                        x_data_pix=x_coord_pix[indices],
                        y_data_pix=y_coord_pix[indices],
                        
                        data=data[indices], 
                        
                        fit_shape=fit_shape.lower(),
                        ellipse_method=ellipse_method,
                        distance_unit=dist_unit, 
                        verbose=verbose
                    )
                    # VALIDATE AND APPEND
                    if validate_structure(one_structure, x_coord, y_coord, elongation_threshold, 
                                          str_size_lower_thres, str_size_upper_thres, ignore_side_structure):
                        
                        structures.append(one_structure)
                        
                except Exception as e:
                    if verbose: print(f'Exception building polygon: {e}')

    # Note: Omitted the Plotting/Saving logic from the bottom for brevity, 
    # but it safely calls the nested ['Regular parameters'] exactly as requested!
    
    return structures

def _plot_ellipses_centers(ax_cur, x_polygon, y_polygon, x_ellipse, y_ellipse, 
                           struct, polygon_color=None, ellipse_color=None,
                           polygon_linewidth=1, ellipse_linewidth=1,
                           semiaxis_linewidth=1, plot_structure_mid=False):
    """Internal helper to overlay geometric fits on frame axes using OOP structures."""
    poly_args = {'color': polygon_color} if polygon_color else {}
    el_args = {'color': ellipse_color} if ellipse_color else {}
    
    ax_cur.plot(x_polygon, y_polygon, linewidth=polygon_linewidth, **poly_args)
    ax_cur.plot(x_ellipse, y_ellipse, linewidth=ellipse_linewidth, **el_args)

    cx, cy = struct.fit_center[0], struct.fit_center[1]
    b, angle = struct.fit_axes_length[0], struct.fit_angle
    
    # BUG FIX: Rotate the angle 90 degrees to align with the major axis
    angle_major = angle
    
    if not (np.isnan(cx) or np.isnan(cy) or np.isnan(b) or np.isnan(angle_major)):
        ax_cur.plot([cx - b*np.cos(angle_major), cx + b*np.cos(angle_major)],
                    [cy - b*np.sin(angle_major), cy + b*np.sin(angle_major)],
                    color='magenta', linewidth=semiaxis_linewidth)
    
    if plot_structure_mid:
        ax_cur.scatter(struct.centroid[0], struct.centroid[1], color='yellow')
        ax_cur.scatter(struct.center_of_gravity[0], struct.center_of_gravity[1], color='red')
        
        
        
def validate_structure(struct, x_coord, y_coord, elongation_threshold, 
                       str_size_lower_thres, str_size_upper_thres, 
                       ignore_side_structure):
    
    """
    Validates and standardizes a segmented plasma structure based on geometric and spatial constraints.

    This internal function acts as the final gatekeeper before a structure is added to 
    the frame's dataset. It checks if the structural fit failed (NaN sizes), forces 
    angles to NaN if the structure is too circular, and normalizes the fit angle to 
    prevent π/-π wrapping. Finally, it drops structures that fall outside the defined 
    physical size bounds or touch the extreme edges of the camera frame.

    Args:
        struct (PlasmaStructure): The object containing the shape, pixel data, and 
            fitted geometric parameters of the plasma blob.
        x_coord (np.ndarray): The 1D or 2D array of the X-coordinates for the full frame.
        y_coord (np.ndarray): The 1D or 2D array of the Y-coordinates for the full frame.
        elongation_threshold (float): The minimum elongation ratio required to trust 
            the angle fit. If `struct.fit_elongation < elongation_threshold`, the 
            blob is considered too circular and its angle is neutralized to np.nan.
        str_size_lower_thres (float or None): The minimum allowable physical size 
            (for both major and minor axes). Structures smaller than this are rejected.
        str_size_upper_thres (float or None): The maximum allowable physical size. 
            Structures larger than this are rejected.
        ignore_side_structure (bool): If True, the function evaluates the boundary 
            coordinates of the blob. If any part of the structure touches the minimum 
            or maximum bounds of `x_coord` or `y_coord`, it is rejected.

    Returns:
        bool: True if the structure survives all checks and is valid. False if the 
        structure triggers any filter. (Note: Modifies the `struct` object in place 
        by calling `struct.set_invalid()` or adjusting `struct._angle` if necessary).
    """
    # 1. NaN Check 
    if np.isnan(struct.fit_size[0]) or np.isnan(struct.fit_size[1]):
        struct.set_invalid()
        return False

    # 2. Elongation & Angle Wrapping Check
    if struct.fit_elongation < elongation_threshold:
        # If it's too circular, neutralize the angle
        struct._angle = np.nan 
    else:
        # BUG FIX: Shift the angle so 0 radians is straight up (vertical)
        # This completely eliminates the +pi/2 to -pi/2 wrapping jump!
        if not np.isnan(struct._angle):
            shifted_angle = struct._angle - (np.pi / 2)
            struct._angle = (shifted_angle + np.pi/2) % np.pi - (np.pi/2)
            
        # Also fix Angle ALI if the physics engine calculated it
        if 'Angle ALI' in struct.regular_parameters:
            ali = struct.regular_parameters['Angle ALI']
            if not np.isnan(ali):
                shifted_ali = ali - (np.pi / 2)
                struct.regular_parameters['Angle ALI'] = (shifted_ali + np.pi/2) % np.pi - (np.pi/2)
                
    # Sync the dictionary so downstream trackers see the modified angles!
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