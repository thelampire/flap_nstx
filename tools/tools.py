#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:37:37 2019

@author: mlampert
"""
#Core imports
import os
import copy
import time
#Importing and setting up the FLAP environment
import flap
import flap_nstx
flap_nstx.register()

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

import matplotlib.pyplot as plt

import numpy as np

from scipy.signal import find_peaks_cwt
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
from scipy.sparse import issparse  # $scipy/sparse/csr.py
import random
import pickle

def calculate_nstx_gpi_norm_coeff(exp_id=None,              # Experiment ID
                                  f_high=1e2,               # Low pass filter frequency in Hz
                                  design='Chebyshev II',    # IIR filter design (from scipy)
                                  test=False,               # Testing input
                                  filter_data=True,         # IIR LPF the data
                                  time_range=None,          # Timer range for the averaging in ms [t1,t2]
                                  calc_around_max=False,    # Calculate the average around the maximum of the GPI signal
                                  time_window=50.,          # The time window for the calc_around_max calculation
                                  cache_data=True,          #
                                  verbose=False,
                                  output_name='GPI_NORMALIZER',
                                  #add_flux_r=False,
                                  ):

    #This function calculates the GPI normalizer image with which all the GPI
    #images should be divided. Returns a flap data object. The inputs are
    #expained next to the inputs.

    normalizer_options={'LP freq':f_high,
                        'Filter design': design,
                        'Filter data':filter_data,
                        'Time range':time_range,
                        'Calc around max': calc_around_max,
                        'Time window': time_window,
                        #'Flux R': add_flux_r,
                        }

    if not cache_data:
        flap.delete_data_object(output_name,
                                'GPI_*_FILTERED_*',
                                'GPI_MEAN')

    if calc_around_max and time_range is not None:
        print('Both calc_around_max and time_range cannot be set.')
        print('Setting calc_around_max to False')
        calc_around_max=False

    #Get the data from the cine file
    if exp_id is not None:
        try:
            flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
        except:
            if verbose or test:
                print('Data is not cached, it needs to be read.')
                print("\n------- Reading NSTX GPI data --------")
            flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        object_name='GPI'
        #if add_flux_r:
        #    flap.add_coordinate(object_name, exp_id=exp_id, coordinates='Flux r')
    else:
        raise ValueError('The experiment ID needs to be set.')

    if time_range is not None:
        sliced_object_name=object_name+'_'+str(time_range[0])+'_'+str(time_range[1])
        try:
            flap.get_data_object_ref(exp_id=exp_id,object_name=sliced_object_name)
        except:
            flap.slice_data(object_name,exp_id=exp_id,
                            slicing={'Time':flap.Intervals(time_range[0],time_range[1])},
                            output_name=object_name+'_'+str(time_range[0])+'_'+str(time_range[1])
                            )
        object_name=sliced_object_name


    #Highpass filter the data to get rid of the spikes
    if filter_data:
        filtered_data_object_name=object_name+'_FILTERED_LP_'+str(f_high)+'_'+design.replace(' ','')
        try:
            flap.get_data_object_ref(exp_id=exp_id,object_name=filtered_data_object_name)
        except:
            if verbose or test:
                print('Filtered data is not cached, it needs to be filtered.')
                print("\n------- Filtering NSTX GPI data --------")
            flap.filter_data(object_name,exp_id=exp_id,
                             coordinate='Time',
                             options={'Type':'Lowpass',
                                      'f_high':f_high,
                                      'Design':design},
                                      output_name=filtered_data_object_name)
        object_name=filtered_data_object_name

    if calc_around_max and time_range is None:
        #Calculate the average image for a time window around the maximum signal
        d=flap.slice_data(object_name,
                          summing={'Image x':'Mean','Image y':'Mean'},
                          output_name='GPI_MEAN')

        max_time_index=np.argmax(d.data)
        max_time=d.coordinate('Time')[0][max_time_index]
        flap.slice_data(object_name,exp_id=exp_id,
                        slicing={'Time':flap.Intervals(max_time-time_window,max_time+time_window)},
                        summing={'Time':'Mean'},
                        output_name=output_name)
    else:
        #Calculate the average image for the entire shot
        d=flap.slice_data(object_name,exp_id=exp_id,
                          summing={'Time':'Mean'},
                          output_name=output_name)
    object_name=output_name

    d.info['Normalizer options']=normalizer_options

    if test:
        plt.figure()
        flap.plot(object_name,
                  axes=['Device R', 'Device z'],
                  exp_id=exp_id,
                  plot_type='contour',
                  plot_options={'levels':21}
                  )
    return d

def calculate_nstx_gpi_reference(object_name=None,
                                 exp_id=None,
                                 time_range=None,
                                 reference_pixel=None,
                                 reference_area=None,
                                 reference_position=None,
                                 reference_flux=None,
                                 filter_low=None,
                                 filter_high=None,
                                 filter_design='Chebyshev II',
                                 output_name=None
                                 ):
    try:
        input_object=flap.get_data_object_ref(object_name, exp_id=exp_id)
    except:
        raise IOError('The given object_name doesn\'t exist in the FLAP storage.')
    if output_name is None:
        output_name=object_name+'_REF'

    if reference_pixel is None and reference_position is None and reference_flux is None:
        raise ValueError('There is no reference given. Please set reference_pixel or reference_position or reference_flux.')

    if filter_low is not None or filter_high is not None:
        if filter_low is not None and filter_high is None:
            filter_type='Highpass'
        if filter_low is None and filter_high is not None:
            filter_type='Lowpass'
        if filter_low is not None and filter_high is None:
            filter_type='Bandpass'

        flap.filter_data(object_name,exp_id=exp_id,
                         coordinate='Time',
                         options={'Type':filter_type,
                                  'f_low':filter_low,
                                  'f_high':filter_high,
                                  'Design':filter_design},
                         output_name=object_name+'_FILTERED')
        object_name=object_name+'_FILTERED'
    slicing_dict={}

    if time_range is not None:
        slicing_dict['Time']=flap.Intervals(time_range[0],time_range[1])

    if reference_pixel is not None:
        #Single pixel correlation
        if reference_area is None:
            slicing_dict['Image x']=reference_pixel[0]
            slicing_dict['Image y']=reference_pixel[1]
            summing_dict=None
        else:
            if type(reference_area) is not list:
                reference_area=[reference_area,reference_area]
            #Handling the edges:
            if reference_pixel[0]-reference_area[0] < 0:
                reference_pixel[0]=reference_area[0]
            if reference_pixel[1]-reference_area[0] < 0:
                reference_pixel[1]=reference_area[0]

            if reference_pixel[0]+reference_area[1] > input_object.data.shape[1]:
                reference_pixel[0]=input_object.data.shape[1]-reference_area[1]
            if reference_pixel[1]+reference_area[1] > input_object.data.shape[2]:
                reference_pixel[1]=input_object.data.shape[2]-reference_area[1]

            slicing_dict['Image x']=flap.Intervals(reference_pixel[0]-reference_area[0],
                                                   reference_pixel[0]+reference_area[0])
            slicing_dict['Image y']=flap.Intervals(reference_pixel[1]-reference_area[1],
                                                   reference_pixel[1]+reference_area[1])
            summing_dict={'Image x':'Mean', 'Image y':'Mean'}

    if reference_position is not None:
        if reference_area is None:
            try:
                slicing_dict['Device R']=reference_position[0]
                slicing_dict['Device z']=reference_position[1]
                summing_dict=None
            except:
                raise ValueError('Reference position is outside the measurement range.')
        else:
            if type(reference_area) is not list:
                reference_area=[reference_area,reference_area]
            try:
            #Multiple pixel correlation (averaged)
                slicing_dict['Device R']=flap.Intervals(reference_position[0]-reference_area[0],
                                                        reference_position[0]+reference_area[0])
                slicing_dict['Device z']=flap.Intervals(reference_position[1]-reference_area[1],
                                                        reference_position[1]+reference_area[1])
                summing_dict={'Device R':'Mean', 'Device z':'Mean'}
            except:
                raise ValueError('Reference position is outside the measurement range.')

    if reference_flux is not None:
        if len(reference_flux) != 2:
            raise ValueError('The reference position needs to be a 2 element list (Psi,z).')
        if reference_area is None:
            try:
                slicing_dict['Flux r']=flap.Intervals(reference_flux[0])
                slicing_dict['Device z']=flap.Intervals(reference_flux[1])
                summing_dict=None
            except:
                raise ValueError('Reference position is outside the measurement range.')
        else:
            if len(reference_area) !=2:
                 raise ValueError('The reference area needs to be a 2 element list (Psi,z).')
            try:
            #Multiple pixel correlation (averaged)
                slicing_dict['Flux r']=flap.Intervals(reference_flux[0]-reference_area[0],
                                                      reference_flux[0]+reference_area[0])
                slicing_dict['Device z']=flap.Intervals(reference_flux[1]-reference_area[1],
                                                        reference_flux[1]+reference_area[1])
                summing_dict={'Flux r':'Mean', 'Device z':'Mean'}
            except:
                raise ValueError('Reference position is outside the measurement range.')

    reference_signal=flap.slice_data(object_name, exp_id=exp_id,
                                     slicing=slicing_dict,
                                     summing=summing_dict,
                                     output_name=output_name)
    return reference_signal

def find_filaments(data_object=None,      #FLAP data objectCould be set instead of exp_id and time_range
                   exp_id=None,           #Shot number
                   time_range=None,       #Time range for the filament finding
                   frange=[0.1e3,100e3],  #Frequency range to pre-condition the data
                   normalize=False,       #Normalize (divide) the data with the time average in time_range
                   ref_pixel=[10,40],     #The pixel to find the peak in.
                   horizontal_sum=False,  #Sum up the pixels vertically in xrange
                   xrange=[0,32],         #Range for summing up the pixels
                   vertical_sum=False,    #Sum up all the pixels vertically in yrange
                   yrange=[10,70],        #Range for summing up the pixels
                   width_range=[1,30],    #The width range for the CWT peak finding algorithm
                   cache_data=False,      #Try to gather the cached data (exp_id, timerange input)
                   return_index=False,     #Return the peak times instaed of the peak indices
                   test=False):           #Plot the resulting data along with the peaks

    #Read signal
    if data_object is None:
        data_object='GPI_FILAMENTS'
        if time_range is None:
            print('The time range needs to set for the calculation.')
            return
        else:
            if (type(time_range) is not list and len(time_range) != 2):
                raise TypeError('time_range needs to be a list with two elements.')
        if exp_id is not None:
            print("\n------- Reading NSTX GPI data --------")
            if cache_data:
                try:
                    d=flap.get_data_object_ref(exp_id=exp_id,object_name=data_object)
                except:
                    print('Data is not cached, it needs to be read.')
                    d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name=data_object)
            else:
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name=data_object)
        else:
            raise ValueError('The experiment ID needs to be set.')
        slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                 'Image x':ref_pixel[0],
                 'Image y':ref_pixel[1]}
        summing=None
        if normalize:
            normalizer=flap.slice_data(data_object,
                                       slicing={'Time':flap.Intervals(time_range[0],
                                                                      time_range[1])},
                                       summing={'Time':'Mean'}).data

            for i_x in range(d.data.shape[1]):
                for i_y in range(d.data.shape[2]):
                    d.data[:,i_x,i_y]=d.data[:,i_x,i_y]/normalizer[i_x,i_y]
    else:
        if normalize:
            normalizer=flap.slice_data(data_object,summing={'Time':'Mean'}).data
            d=flap.get_data_object_ref(data_object).data
            for i_x in range(d.data.shape[1]):
                for i_y in range(d.data.shape[2]):
                    d.data[:,i_x,i_y]=d.data[:,i_x,i_y]/normalizer[i_x,i_y]
        slicing={'Image x':ref_pixel[0],
                 'Image y':ref_pixel[1]}
        summing=None
    if vertical_sum or horizontal_sum:
        summing={}
        if horizontal_sum:
            slicing['Image x']=flap.Intervals(xrange[0],xrange[1])
            summing['Image x']='Mean'
        if vertical_sum:
            slicing['Image y']=flap.Intervals(yrange[0],yrange[1])
            summing['Image y']='Mean'
    flap.slice_data(data_object,
                    slicing=slicing,
                    summing=summing,
                    output_name='GPI_SLICED_FILAMENTS')

    #Filter signal to HPF 100Hz
    d=flap.filter_data('GPI_SLICED_FILAMENTS',
                       coordinate='Time',
                       options={'Type':'Bandpass',
                                'f_low':frange[0],
                                'f_high':frange[1],
                                'Design':'Chebyshev II'},
                       output_name='GPI_FILTERED_FILAMENTS')

    #ind=find_peaks(d.data, distance=25, threshold=threshold)[0]         #This method needs quite a lot of tinkering, it is deprecated
    ind=find_peaks_cwt(d.data, np.arange(width_range[0],width_range[1])) #This method is working quite well without any data preconditioning except the filtering
    try:
        flap.delete_data_object('GPI_FILAMENTS')
    except:
        pass
    flap.delete_data_object('GPI_SLICED_FILAMENTS')
    flap.delete_data_object('GPI_FILTERED_FILAMENTS')
    if test:
        plt.figure()
        plt.plot(d.coordinate('Time')[0],d.data)
        plt.scatter(d.coordinate('Time')[0][ind],d.data[ind], color='red')
        plt.xlabel('Time [s]')
        plt.ylabel('GPI signal [a.u.]')
        plt.show()
    if return_index:
        return ind
    else:
        return d.coordinate('Time')[0][ind]

def detrend_multidim(data_object=None,
                     exp_id=None,
                     coordinates=None,
                     order=None,
                     test=False,
                     return_trend=False,
                     output_name=None,
                     ):
    """
    Performs 2D or 3D polynomial detrending (background subtraction) on a multidimensional FLAP data object.

    This function generates a polynomial basis matrix of the specified order, fits it 
    to the data using a least-squares pseudo-inverse, and subtracts the resulting 
    trend. It supports processing high-dimensional data (e.g., detrending 2D spatial 
    frames across a 3D time-series) via vectorized matrix multiplication.

    Args:
        data_object (flap.DataObject or str): The FLAP data object or its registry name.
        exp_id (int, optional): The experiment ID, if loading by name. Defaults to None.
        coordinates (list of str): A list of 2 or 3 coordinate names to detrend along 
            (e.g., ['Image x', 'Image y']).
        order (int): The maximum polynomial order for the detrending surface.
        test (bool, optional): If True and detrending a pure 2D dataset, plots the 
            original, trend, and detrended data using matplotlib. Defaults to False.
        return_trend (bool, optional): If True, returns the raw numpy array of the 
            calculated trend instead of the detrended FLAP object. Defaults to False.
        output_name (str, optional): If provided, registers the detrended FLAP object 
            under this name. Defaults to None.

    Returns:
        flap.DataObject or numpy.ndarray: The detrended FLAP data object, or the 
        calculated trend array if `return_trend` is True.
    """

    # --- 1. Object Loading & Dimensionality Checks ---
    if exp_id is not None:
        d = copy.deepcopy(flap.get_data_object(data_object, exp_id=exp_id))
    else:
        d = copy.deepcopy(flap.get_data_object(data_object))

    total_dim = len(d.data.shape)
    if total_dim > 4:
        raise TypeError('Datasets over 4 dimensions are not supported.')

    ndim = len(coordinates)
    if ndim not in [2, 3]:
        raise ValueError('Detrend is only supported for 2D and 3D coordinates.')

    # Aggregate the target dimensions to detrend along
    detrend_dims = np.unique(np.concatenate([d.get_coordinate_object(c).dimension_list for c in coordinates]))
    shape = d.data.shape
    d_shape = [shape[i] for i in detrend_dims]

    # --- 2. Build Polynomial Basis Matrix ---
    if ndim == 2:
        nx, ny = d_shape
        points = np.array([
            [i**k * j**l for k in range(order + 1) for l in range(order - k + 1)]
            for i in range(nx) for j in range(ny)
        ])
    elif ndim == 3:
        nx, ny, nz = d_shape
        points = np.array([
            [i**l * j**m * k**n for l in range(order + 1) 
                                for m in range(order - l + 1) 
                                for n in range(order - l - m + 1)]
            for i in range(nx) for j in range(ny) for k in range(nz)
        ])

    # --- 3. Vectorized Least Squares Fit ---
    # pseudo_inv maps the data to the polynomial coefficients safely: (X^T * X)^-1 * X^T
    pseudo_inv = np.linalg.pinv(points)

    # To vectorize over non-detrended dimensions (like Time), we permute the array 
    # to push the detrended dimensions to the front, then flatten.
    other_dims = [i for i in range(total_dim) if i not in detrend_dims]
    perm = list(detrend_dims) + other_dims
    
    data_permuted = np.transpose(d.data, perm)
    
    n_pixels = np.prod(d_shape)
    n_other = np.prod([shape[i] for i in other_dims]) if other_dims else 1
    
    # Flatten into 2D: (pixels, slices)
    data_flat = data_permuted.reshape((n_pixels, n_other))

    # Bulk matrix multiplication fits all slices simultaneously!
    coeffs = pseudo_inv @ data_flat      # Shape: (n_polynomial_terms, n_slices)
    trend_flat = points @ coeffs         # Shape: (n_pixels, n_slices)

    # Reshape the trend back to the permuted state, then reverse the permutation
    trend_permuted = trend_flat.reshape([shape[i] for i in perm])
    inv_perm = np.argsort(perm)
    trend = np.transpose(trend_permuted, inv_perm)

    # --- 4. Subtraction & Output ---
    d.data = d.data - trend

    if test and total_dim == 2 and ndim == 2:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].contourf(trend.T, levels=51)
        axes[0].set_title('Trend')
        axes[1].contourf((d.data + trend).T, levels=51)
        axes[1].set_title('Original Data')
        axes[2].contourf(d.data.T, levels=51)
        axes[2].set_title('Detrended Data')
        plt.show()

    if output_name is not None:
        flap.add_data_object(d, output_name)
        
    if return_trend:
        return trend
        
    return d

def filename(exp_id=None,
             time_range=None,
             working_directory=None,
             purpose=None,
             frange=None,
             comment=None,
             extension=None):
    """
    Generates a standardized, formatted filename for NSTX data analysis outputs.

    Constructs a filename by concatenating the provided experiment parameters 
    separated by underscores. It ensures consistent naming conventions across 
    the module and safely handles directory path joining and extension appending.

    Args:
        exp_id (int or str): The experiment shot number. Required.
        time_range (list or tuple of float, optional): A 2-element list specifying 
            the start and end times. If None, appends 'whole'. Defaults to None.
        working_directory (str, optional): The base directory path. If provided, 
            the returned string will be a full absolute/relative path. Defaults to None.
        purpose (str, optional): A short description of the file's purpose. Spaces 
            will be automatically replaced with underscores. Defaults to None.
        frange (list or tuple of float, optional): A 2-element list specifying 
            the frequency range [f_min, f_max]. Defaults to None.
        comment (str, optional): Additional text to append to the filename. Spaces 
            will be replaced with underscores. Defaults to None.
        extension (str, optional): File extension (e.g., 'pdf', 'pickle'). 
            Can be provided with or without a leading dot. Defaults to None.

    Raises:
        ValueError: If `exp_id` is missing, or if `time_range`/`frange` are invalid lengths.
        TypeError: If `purpose`, `comment`, or `extension` are not strings.

    Returns:
        str: The fully constructed filename (or absolute path if a working directory was given).
    """

    if exp_id is None:
        raise ValueError('The exp_id needs to be set for the filename.')

    # Collect filename components in a list
    parts = [f"NSTX_{exp_id}"]

    # --- Time Range ---
    if time_range is None:
        parts.append('whole')
    elif isinstance(time_range, (list, tuple, np.ndarray)) and len(time_range) == 2:
        parts.append(f"{time_range[0]:.6f}_{time_range[1]:.6f}")
    else:
        raise ValueError('Time range should be a two-element list or tuple.')

    # --- Purpose ---
    if purpose is not None:
        if isinstance(purpose, str):
            parts.append(purpose.replace(' ', '_'))
        else:
            raise TypeError('Purpose should be a string.')

    # --- Frequency Range ---
    if frange is not None:
        if isinstance(frange, (list, tuple)) and len(frange) == 2:
            parts.append(f"freq_{frange[0]}_{frange[1]}")
        else:
            raise ValueError('Frequency range should be a two-element list or tuple if not None.')

    # --- Comment ---
    if comment is not None:
        if isinstance(comment, str):
            parts.append(comment.replace(' ', '_'))
        else:
            raise TypeError('Comment should be a string.')

    # --- Join Base Filename ---
    base_filename = "_".join(parts)

    # --- Extension ---
    if extension is not None:
        if isinstance(extension, str):
            # .lstrip('.') ensures it works whether the user passes "pdf" or ".pdf"
            base_filename += f".{extension.lstrip('.')}"
        else:
            raise TypeError('Extension should be a string.')

    # --- Path Construction ---
    if working_directory is not None:
        return os.path.join(working_directory, base_filename)
        
    return base_filename



def polyfit_2D(x=None,
               y=None,
               values=None,
               sigma=None,
               order=None,
               irregular=False,
               return_covariance=False,
               return_fit=False):
    """
    Fits a 2D polynomial surface to regular grid or irregular scatter data.

    Uses Weighted Least Squares (WLS) to fit a 2D polynomial of a specified order 
    to the given data. Capable of handling both 2D image matrices and 1D arrays 
    of scattered (irregular) coordinate points.

    Args:
        x (np.ndarray, optional): X-coordinates. Defaults to array indices if None.
        y (np.ndarray, optional): Y-coordinates. Defaults to array indices if None.
        values (np.ndarray): The Z-values to fit the surface to.
        sigma (np.ndarray, optional): 1-sigma uncertainties for WLS weighting. 
            Defaults to uniform weighting of 1.0.
        order (int): The maximum order of the 2D polynomial.
        irregular (bool, optional): If True, treats x, y, and values as 1D arrays 
            of scattered points. Defaults to False.
        return_covariance (bool, optional): If True, returns a tuple of 
            (coefficients, covariance_matrix). Defaults to False.
        return_fit (bool, optional): If True, evaluates the polynomial on the 
            input coordinates and returns the fitted surface array. Defaults to False.

    Returns:
        np.ndarray or tuple: The calculated polynomial coefficients. Can optionally 
        return the covariance matrix and/or the fitted data array.
    """
    
    if order is None:
        raise ValueError('The polynomial order must be set.')
    if values is None:
        raise ValueError('Values must be provided.')

    # --- 1. Validate, Unify, and Flatten Inputs ---
    original_shape = values.shape

    if not irregular:
        if len(original_shape) != 2:
            raise ValueError('Values must be 2D when irregular=False.')

        if x is None and y is None:
            # Equivalent to the original `for i... for j...` indexing
            x, y = np.indices(original_shape)
        elif x is None or y is None:
            raise ValueError('Either both or neither x and y must be set.')
        elif x.shape != original_shape or y.shape != original_shape:
            raise ValueError('x and y shapes must match values shape.')

        # Flatten arrays for generalized linear algebra
        x_flat, y_flat, v_flat = x.flatten(), y.flatten(), values.flatten()
    else:
        if len(original_shape) != 1 or x.shape != original_shape or y.shape != original_shape:
            raise ValueError('x, y, and values must be 1D arrays of the same length when irregular=True.')
        
        x_flat, y_flat, v_flat = x, y, values

    # Handle uncertainties (weights)
    if sigma is None:
        s_flat = np.ones_like(v_flat)
    else:
        if sigma.shape != original_shape:
            raise ValueError('The shape of sigma must match the shape of values.')
        s_flat = sigma.flatten()

    # --- 2. Build Polynomial Basis Matrices ---
    # Generate exponent pairs (k, l) such that k + l <= order
    powers = [(k, l) for k in range(order + 1) for l in range(order - k + 1)]

    # Phi is the unweighted basis matrix (Vandermonde matrix)
    Phi = np.column_stack([(x_flat**k) * (y_flat**l) for k, l in powers])

    # V is the weighted basis matrix for solving
    V = Phi / s_flat[:, np.newaxis]
    v_weighted = v_flat / s_flat

    # --- 3. Solve Weighted Least Squares ---
    # Pseudo-inverse is numerically safer than standard inverse for polynomials
    covariance_matrix = np.linalg.pinv(V.T @ V)
    coefficients = covariance_matrix @ V.T @ v_weighted

    # --- 4. Handle Returns ---
    if return_fit:
        # Evaluate the fit on the unweighted coordinates
        fit_flat = Phi @ coefficients
        return fit_flat.reshape(original_shape)

    if return_covariance:
        return coefficients, covariance_matrix

    return coefficients



def subtract_photon_peak_2D(autocorr=None,     #INPUT autocorrelation metrix
                            order=2,           #Order of the fitting
                            neglect_range=1,   #Range to be substituted by the fit 1=middle value, 2=+-1 area around middle etc.
                            fitting_range=2    #Range to be fit with the polynom
                            ):
    if autocorr is None:
        raise ValueError('No input is given.')
    index=[0] * 2
    middle_index=[0] * 2
    for i in range(2):
        index[i]=slice(autocorr.shape[i]//2-fitting_range,autocorr.shape[i]//2+fitting_range+1)
        if neglect_range == 1:
            middle_index[i]=autocorr.shape[i]//2
        else:
            middle_index[i]=slice(autocorr.shape[i]//2-(neglect_range-1),autocorr.shape[i]//2+(neglect_range-1)+1)
    x=np.zeros(autocorr.shape)
    y=np.zeros(autocorr.shape)
    for j in range(x.shape[1]):
        x[:,j]=np.arange(x.shape[0])
    for i in range(y.shape[0]):
        y[i,:]=np.arange(y.shape[1])
    _autocorr=copy.deepcopy(autocorr)
    _autocorr[tuple(middle_index)]=np.nan
    to_be_fit_index=np.logical_not(np.isnan(_autocorr[tuple(index)]))

    x_to_be_fit=(x[tuple(index)])[to_be_fit_index]
    y_to_be_fit=(y[tuple(index)])[to_be_fit_index]
    autocorr_to_be_fit=(_autocorr[tuple(index)])[to_be_fit_index]

    coeff=flap_nstx.analysis.polyfit_2D(x=x_to_be_fit,
                                        y=y_to_be_fit,
                                        values=autocorr_to_be_fit,
                                        order=order,
                                        irregular=True)
    points=np.asarray([[x[k,l]**i * y[k,l]**j for k in range(x.shape[0]) for l in range(x.shape[1]) ] for i in range(order+1) for j in range(order-i+1)], dtype='float64')
    fit = np.dot(coeff,points)
    fit=np.reshape(fit, autocorr.shape)
    _autocorr[tuple(middle_index)]=fit[tuple(middle_index)]
    return _autocorr


def make_plot_cursor_format(current, other):
    """
    The method is for displaying double cursors for the overplotted correlations
    in the velocity calculation.
    """
    # current and other are axes
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x,y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Left: {:<40}    Right: {:<}'
                .format(*['({:.6f}, {:.6f})'.format(x, y) for x,y in coords]))
    return format_coord

def signal_windowed_avg_err(x,windowsize):
    """
    Returns the average and the square root of the variance of signal x in a
    defined window size.
    """
    if len(x) < windowsize:
        raise ValueError('The window size is larger than the data\'s length')
    data_len=len(x)
    return_data=np.zeros(data_len)
    return_error=np.zeros(data_len)

    return_data[0:windowsize]=np.mean(x[0:windowsize])
    return_error[0:windowsize]=np.sqrt(np.var(x[0:windowsize]))
    for i_data in range(windowsize,data_len):
        return_data[i_data]=np.mean(x[i_data-windowsize:i_data])
        return_error[i_data]=np.sqrt(np.var(x[i_data-windowsize:i_data]))
    return return_data,return_error


def kmeans( X, centres, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1 ):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    if verbose:
        print("kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s" %(X.shape, centres.shape, delta, maxiter, metric))
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = cdist_sparse( X, centres, metric=metric, p=p )  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print("kmeans: av |X - nearest centre| = %.4g" % avdist)
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print("kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc))
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print("kmeans: cluster 50 % radius", r50.astype(int))
        print("kmeans: cluster 90 % radius", r90.astype(int))
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances

#...............................................................................
def kmeanssample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centres
    """
        # merge w kmeans ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centres = randomsample( X, int(k) )
    samplecentres = kmeans( Xsample, pass1centres, **kwargs )[0]
    return kmeans( X, samplecentres, **kwargs )

def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs)
    d = np.empty((X.shape[0], Y.shape[0]), np.float64)
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist(x.todense(), Y, **kwargs)[0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist(X, y.todense(), **kwargs)[0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist(x.todense(), y.todense(), **kwargs)[0]
    return d

def randomsample(X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample(range(X.shape[0]), int(n))
    return X[sampleix]

def nearestcentres(X, centres, metric="euclidean", p=2):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist(X, centres, metric=metric, p=p)  # |X| x |centres|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()

#...............................................................................
class Kmeans:
    """ km = Kmeans( X, k= or centres=, ... )
        in: either initial centres= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centres, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centres[jcentre]
                J indexes e.g. X[J], classes[J]
    """
    def __init__( self, X, k=0, centres=None, nsample=0, **kwargs ):
        self.X = X
        if centres is None:
            self.centres, self.Xtocentre, self.distances = kmeanssample(X, k=k, nsample=nsample, **kwargs )
        else:
            self.centres, self.Xtocentre, self.distances = kmeans(X, centres, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)


def calculate_corr_acceptance_levels(n_data=200,
                                     n_rand=10000,
                                     recalc=False,
                                     verbose=False):

    corr_accept_filename=wd+'/processed_data/correlation_coefficient_significance_threshold_'+str(n_data)+'_'+str(n_rand)+'.pickle'
    if not os.path.exists(corr_accept_filename) or recalc:

        result=np.zeros([n_data,n_rand])

        start_time=time.time()
        for i_rand in range(n_rand):
            for i_data in range(n_data):
                a=np.random.rand(i_data)
                b=np.random.rand(i_data)
                a=a-np.mean(a)
                b=b-np.mean(b)
                result[i_data,i_rand]=np.abs(np.sum((a)*(b))/np.sqrt((np.sum((a)**2)*(np.sum((b)**2)))))
            one_time=time.time()-start_time
            rem_time=one_time*(n_rand-i_rand)
            #print(rem_time)
            print('Remaining time from the calculation:'+str(int(rem_time/3600.))+'h '+str(int(np.mod(rem_time,3600.)/60.))+'min.')
        corr_accept={'avg':np.mean(result, axis=1),
                     'stddev':np.sqrt(np.var(result, axis=1)),
                     'result':result}
        pickle.dump(corr_accept,open(corr_accept_filename,'wb'))
    else:
        corr_accept=pickle.load(open(corr_accept_filename,'rb'))

    return corr_accept



def plot_pearson_matrix(matrix,
                        xlabels=None,
                        ylabels=None,
                        title='',
                        colormap='seismic',
                        figsize=(8.5/2.54,8.5/2.54*1.2),
                        
                        charsize=9,
                        charsize_score=9/1.5,
                        charcolor_score='white',
                        
                        zrange=[-1,1],
                        plot_large=True,
                        plot_values=True,
                        plot_colorbar=True,
                        linewidth=1,
                        ticksize=6,
                        
                        minor_ticksize=False,
                        major_ticksize=False,
                        
                        colorbar_ticks=None,
                        fig_ax=None
                        ):
    if plot_large:
        plt.rcParams['lines.linewidth'] = linewidth
        plt.rcParams['axes.linewidth'] = linewidth
        plt.rcParams['axes.labelsize'] = charsize
        plt.rcParams['axes.titlesize'] = charsize

        plt.rcParams['xtick.labelsize'] = charsize
        
        

        plt.rcParams['ytick.labelsize'] = charsize
        
        
        if major_ticksize:
            plt.rcParams['xtick.major.size'] = major_ticksize
            plt.rcParams['ytick.major.size'] = major_ticksize
        else:
            plt.rcParams['xtick.major.size'] = ticksize
            plt.rcParams['ytick.major.size'] = ticksize
        
        plt.rcParams['xtick.major.width'] = linewidth
        plt.rcParams['xtick.minor.width'] = linewidth
        
        plt.rcParams['ytick.major.width'] = linewidth
        plt.rcParams['ytick.minor.width'] = linewidth
        
        if minor_ticksize:
            plt.rcParams['xtick.minor.size'] = minor_ticksize
            plt.rcParams['ytick.minor.size'] = minor_ticksize
        else:
            plt.rcParams['xtick.minor.size'] = ticksize/2
            plt.rcParams['ytick.minor.size'] = ticksize/2

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if fig_ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    else:
        fig,ax=fig_ax
        
    im=ax.matshow(matrix,
                  cmap=colormap,
                  vmin=zrange[0],
                  vmax=zrange[1],
                  )

    ax.set_xticks(ticks=np.arange(matrix.shape[1]),
                  labels=xlabels,
                  rotation=45,
                  ha='left',
                  rotation_mode='anchor'
                  )
    ax.set_yticks(ticks=np.arange(matrix.shape[0]),
                  labels=ylabels)
    
    if plot_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', ticks=colorbar_ticks)
        if colorbar_ticks is not None:
            print(colorbar_ticks)
            # cax.set_yticks(colorbar_ticks, labels=colorbar_ticks)
        
    ax.set_title(title)

    # ax.set_xticks(np.arange(0, len(xlabels), 1))
    # ax.set_yticks(np.arange(0, len(ylabels), 1))
    ax.set_xticks(np.arange(-.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=linewidth)
    if plot_values:
        for (i, j), z in np.ndenumerate(matrix):
            ax.text(j, i, '{:0.1f}'.format(z),
                    ha='center',
                    va='center',
                    color=charcolor_score,
                    size=charsize_score)

    
    

def set_matplotlib_for_publication(labelsize=8.,
                                   linewidth=0.5,
                                   major_ticksize=2.,
                                   minor_ticksize=1.,
                                   ):

    plt.rc('font', family='serif', serif='Helvetica')
    plt.rc('text', usetex=False)                                            #usetex doesnt work with the current installation but works with $$ somehow.
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['axes.linewidth'] = linewidth
    plt.rcParams['axes.labelsize'] = labelsize
    plt.rcParams['axes.titlesize'] = labelsize

    plt.rcParams['xtick.labelsize'] = labelsize
    plt.rcParams['xtick.major.size'] = major_ticksize
    plt.rcParams['xtick.major.width'] = linewidth
    plt.rcParams['xtick.minor.width'] = linewidth/2
    plt.rcParams['xtick.minor.size'] = minor_ticksize

    plt.rcParams['ytick.labelsize'] = labelsize
    plt.rcParams['ytick.major.width'] = linewidth
    plt.rcParams['ytick.major.size'] = major_ticksize
    plt.rcParams['ytick.minor.width'] = linewidth/2
    plt.rcParams['ytick.minor.size'] = minor_ticksize
    plt.rcParams['legend.fontsize'] = labelsize

def fringe_jump_correction(data,                                                #Data input
                           fringe_size=np.pi,                                 #Size of the jumps need to be corrected
                           tolerance=0.5,                                       #Tolerance for fringes, e.g., 0.2 means 2*np.pi * (1 - 0.2) still needs to be corrected.
                           ):

    if data is None:
        raise ValueError('data input must be provided')

    while True:
        fringe_exists=0
        for ind in range(len(data)-1):
            if data[ind+1]-data[ind] > fringe_size*(1-tolerance):
                data[ind+1] -= fringe_size
                fringe_exists+=1
            elif data[ind+1]-data[ind] < -(fringe_size*(1-tolerance)):
                data[ind+1] += fringe_size
                fringe_exists+=1
            else:
                pass
        if fringe_exists == 0:
            break
    return data

def mutual_information(x, 
                       y, 
                       bins=None, 
                       normalized=True):
    
    if bins is None:
        bins=9

    def shan_entropy(c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized* np.log2(c_normalized))  
        return H

    c_XY = np.histogram2d(x,y,bins)[0]
    c_X = np.histogram(x,bins)[0]
    c_Y = np.histogram(y,bins)[0]
 
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    
    if normalized:
        MI = (H_X + H_Y - H_XY)/np.sqrt(H_X * H_Y)
    else:
        MI = (H_X + H_Y - H_XY)
        
    return MI

def correlation(data1,data2,
                threshold_correlation=False,
                correlation_accept=None,
                confidence_sigma=None):
    
    data1 = data1 - np.mean(data1)
    data2 = data2 - np.mean(data2)
    
    correlation = np.sum(data1 * data2) / (np.sqrt(np.sum(data1**2) * np.sum(data2**2)))
    
    if threshold_correlation:
        try:
            if not (np.abs(correlation) > (correlation_accept['avg'][len(data1)] +
                                       confidence_sigma*correlation_accept['stddev'][len(data1)])):
                correlation=np.nan
        except:
            pass
    return correlation

def calculate_plasma_squareness(R, z, 
                                upper=False, 
                                lower=False,
                                test=False):
    ind_R_mid=np.argmax(R)
    R_mid=R[ind_R_mid]
    z_mid=z[ind_R_mid]
    
    ind_z_top=np.argmax(z)
    ind_z_bot=np.argmin(z)
    
    R_top=R[ind_z_top]
    R_bot=R[ind_z_bot]
    z_top=z[ind_z_top]
    z_bot=z[ind_z_bot]
    
    R_ellipse_intersection_top, z_ellipse_intersection_top = ellipse_line_intersection(R_top,z_mid, R_mid-R_top, z_top-z_mid, R_top,z_mid,R_mid,z_top)[0]
    R_separatrix_intersection_top, z_separatrix_intersection_top = path_line_intersections(R, z, R_top,z_mid,R_mid,z_top)[0]
    
    R_ellipse_intersection_bot, z_ellipse_intersection_bot = ellipse_line_intersection(R_bot,z_mid, (R_mid-R_bot), (z_bot-z_mid), R_bot,z_mid,R_mid,z_bot)[0]
    R_separatrix_intersection_bot, z_separatrix_intersection_bot = path_line_intersections(R, z, R_bot,z_mid,R_mid,z_bot)[0]
    
    def _distance(x1,y1,x2,y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    AB_top=_distance(R_top,z_mid, R_separatrix_intersection_top,z_separatrix_intersection_top)
    AC_top=_distance(R_top,z_mid, R_ellipse_intersection_top,z_ellipse_intersection_top)
    CD_top=_distance(R_ellipse_intersection_top, z_ellipse_intersection_top, R_mid,z_top)
    
    AB_bot=_distance(R_bot,z_mid, R_separatrix_intersection_bot,z_separatrix_intersection_bot)
    AC_bot=_distance(R_bot,z_mid, R_ellipse_intersection_bot,z_ellipse_intersection_bot)
    CD_bot=_distance(R_ellipse_intersection_bot,z_ellipse_intersection_bot, R_mid,z_bot)
    
    upper_squareness=(AB_top-AC_top)/CD_top
    lower_squareness=(AB_bot-AC_bot)/CD_bot
    
    if test:
        plt.figure()
        plt.plot(R,z)
        plt.scatter([R_top,R_bot,R_mid,R_bot,R_top,R_mid,R_mid],
                    [z_mid,z_mid,z_mid,z_bot,z_top,z_top,z_bot])
        
        plt.scatter([R_separatrix_intersection_bot,R_separatrix_intersection_top],
                    [z_separatrix_intersection_bot,z_separatrix_intersection_top])
        
        plt.scatter([R_ellipse_intersection_bot,R_ellipse_intersection_top],
                    [z_ellipse_intersection_bot,z_ellipse_intersection_top])
        
        print([R_ellipse_intersection_bot,R_ellipse_intersection_top],
              [z_ellipse_intersection_bot,z_ellipse_intersection_top])
        
        print([R_separatrix_intersection_bot,R_separatrix_intersection_top],
              [z_separatrix_intersection_bot,z_separatrix_intersection_top])
        
    if upper: return upper_squareness 
    elif lower: return lower_squareness
    else: return {'lower': lower_squareness, 'upper':upper_squareness}

def line_segment_intersection(x1,y1,x2,y2, x3,y3,x4,y4, tol=1e-12):
    """
    Intersection of two finite line segments P1P2 and P3P4.
    Returns (x,y) if they intersect, else None.
    """
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < tol:
        return None  # segments are parallel or coincident
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / denom

    if 0-tol <= t <= 1+tol and 0-tol <= u <= 1+tol:
        xi = x1 + t*(x2-x1)
        yi = y1 + t*(y2-y1)
        return (xi, yi)
    return None

def path_line_intersections(path_x, path_y, x1,y1,x2,y2):
    """
    Find intersections between a polyline path (path_x, path_y)
    and a finite line segment (x1,y1)-(x2,y2).

    Returns a list of (x,y) intersection points.
    """
    pts = []
    for i in range(len(path_x)-1):
        px1, py1 = path_x[i],   path_y[i]
        px2, py2 = path_x[i+1], path_y[i+1]
        p = line_segment_intersection(x1,y1,x2,y2, px1,py1,px2,py2)
        if p is not None:
            pts.append(p)
    return pts
    
    
def ellipse_line_intersection(xc, yc, a, b, x1, y1, x2, y2, segment_only=True, tol=1e-12):
    """
    Find intersection points between ellipse and line (or segment).

    Ellipse: ( (x - xc)^2 / a^2 ) + ( (y - yc)^2 / b^2 ) = 1
    Line: through (x1,y1) and (x2,y2)

    Parameters
    ----------
    xc, yc : float
        Ellipse center coordinates.
    a, b : float
        Semi-major and semi-minor axes (along x and y).
    x1, y1, x2, y2 : float
        Coordinates of line endpoints.
    segment_only : bool, default=True
        If True, only return intersections lying within the segment [P1,P2].
    tol : float
        Tolerance for numerical comparisons.

    Returns
    -------
    intersections : list of (x,y)
        List of intersection points (0, 1, or 2).
    """

    dx = x2 - x1
    dy = y2 - y1

    # Parametric line: (x,y) = (x1 + t*dx, y1 + t*dy)
    # Substitute into ellipse equation
    A = (dx**2)/(a**2) + (dy**2)/(b**2)
    B = 2*((x1-xc)*dx/(a**2) + (y1-yc)*dy/(b**2))
    C = ((x1-xc)**2)/(a**2) + ((y1-yc)**2)/(b**2) - 1

    # Quadratic At^2 + Bt + C = 0
    disc = B**2 - 4*A*C
    pts = []
    if disc < -tol:
        return pts  # no intersection
    elif abs(disc) <= tol:
        t = -B/(2*A)
        if (not segment_only) or (0-tol <= t <= 1+tol):
            pts.append((x1 + t*dx, y1 + t*dy))
    else:
        sqrt_disc = np.sqrt(disc)
        t1 = (-B + sqrt_disc)/(2*A)
        t2 = (-B - sqrt_disc)/(2*A)
        for t in (t1, t2):
            if (not segment_only) or (0-tol <= t <= 1+tol):
                pts.append((x1 + t*dx, y1 + t*dy))
    return pts

from scipy.interpolate import RegularGridInterpolator
import skimage.measure

import numpy as np
import flap
import flap_nstx

def get_flux_coord(shot=None,
                   time=None,
                   R_target=None, 
                   z_target=None):
    """
    Calculates the normalized poloidal flux and arc-length poloidal angle 
    for a specific list of absolute (R, Z) points.
    
    Args:
        shot (int): Experimental shot number.
        time (float): Time slice to extract equilibrium data.
        R_target (np.ndarray): 1D array of Target Major Radius coordinates (from center of torus).
        z_target (np.ndarray): 1D array of Target Vertical coordinates.
        
    Returns:
        dict: 'psi_norm' and 'theta_arc' arrays corresponding to the target points.
    """
    R_target = np.atleast_1d(R_target)
    Z_target = np.atleast_1d(z_target)
    
    # =========================================================================
    # 1. Read data for flux coordinate calculation
    # =========================================================================
    psi_rz_obj = flap.get_data('NSTX_MDSPlus',
                               name=r'\EFIT02::\PSIRZ',
                               exp_id=shot,
                               object_name='PSIRZ_FOR_COORD'
                               ).slice_data(slicing={'Time': time})
    
    R_grid_1d = psi_rz_obj.coordinate('Device R')[0][:, 0]
    z_grid_1d = psi_rz_obj.coordinate('Device z')[0][0, :]
    psi_grid_2d = psi_rz_obj.data
    
    # Extract structural scalars (wrapped in float() to unpack single-element arrays)
    R_axis = flap.get_data('NSTX_MDSPlus',
                                 name=r'\EFIT02::\RMAXIS',
                                 exp_id=shot,
                                 object_name='RMAXIS_FOR_COORD'
                                 ).slice_data(slicing={'Time': time}).data
                                 
    Z_axis = float(flap.get_data('NSTX_MDSPlus',
                                 name=r'\EFIT02::\ZMAXIS',
                                 exp_id=shot,
                                 object_name='ZMAXIS_FOR_COORD'
                                 ).slice_data(slicing={'Time': time}).data)
                                 
    psi_axis = float(flap.get_data('NSTX_MDSPlus',
                                   name=r'\EFIT02::\SSIMAG',
                                   exp_id=shot,
                                   object_name='PSI0_FOR_COORD'
                                   ).slice_data(slicing={'Time': time}).data)
                                   
    psi_bdry = float(flap.get_data('NSTX_MDSPlus',
                                   name=r'\EFIT02::\SSIBRY',
                                   exp_id=shot,
                                   object_name='PSIBDY_FOR_COORD'
                                   ).slice_data(slicing={'Time': time}).data)
    # =========================================================================
    # 2. Interpolate psi at the target locations
    # =========================================================================
    # RegularGridInterpolator strictly maps absolute R and absolute Z to psi.
    interp_func = RegularGridInterpolator((R_grid_1d, z_grid_1d), psi_grid_2d, 
                                          bounds_error=False, fill_value=np.nan)
    
    target_coords = np.column_stack((R_target, Z_target))
    psi_target = interp_func(target_coords)
    # Calculate Normalized Flux
    psi_norm_target = (psi_target - psi_axis) / (psi_bdry - psi_axis)
    # =========================================================================
    # 3. Trace the contour for each point to calculate physical arc length
    # =========================================================================
    theta_arc_target = np.zeros_like(psi_target)
    
    for i, (r_pt, z_pt, psi_val) in enumerate(zip(R_target, Z_target, psi_target)):
        if np.isnan(psi_val):
            theta_arc_target[i] = np.nan
            continue
            
        # If the point is exactly on the axis, theta is degenerate
        if np.isclose(psi_val, psi_axis, atol=1e-5):
            theta_arc_target[i] = 0.0
            continue
            
        # Find the contour for this specific psi value
        contours = skimage.measure.find_contours(psi_grid_2d, psi_val)
        if not contours:
            theta_arc_target[i] = np.nan
            continue
            
        # Assume the longest contour is our closed flux surface
        contour = max(contours, key=len)
        
        # Map skimage pixel indices back to physical absolute R, Z coordinates
        # (Assuming psi_grid_2d is shape [len(R), len(Z)])
        R_idx, Z_idx = contour[:, 0], contour[:, 1]
        R_c = np.interp(R_idx, np.arange(len(R_grid_1d)), R_grid_1d)
        Z_c = np.interp(Z_idx, np.arange(len(z_grid_1d)), z_grid_1d)
        
        # Find Outboard Midplane (OMP) to use as theta = 0
        # Since R is measured from the center, the outboard side is strictly R > R_axis
        omp_mask = R_c > R_axis
        if np.any(omp_mask):
            omp_idx = np.where(omp_mask)[0][np.argmin(np.abs(Z_c[omp_mask] - Z_axis))]
        else:
            # Fallback for highly distorted topologies
            omp_idx = np.argmax(R_c)
            
        # Roll arrays to start at the OMP
        R_c = np.roll(R_c, -omp_idx)
        Z_c = np.roll(Z_c, -omp_idx)
        
        # Ensure standard counter-clockwise orientation (Z should increase initially)
        if len(Z_c) > 1 and Z_c[1] < Z_c[0]:
            R_c = np.insert(R_c[1:][::-1], 0, R_c[0])
            Z_c = np.insert(Z_c[1:][::-1], 0, Z_c[0])
            
        # Calculate cumulative arc lengths along this specific surface
        dR = np.diff(R_c)
        dZ = np.diff(Z_c)
        ds = np.sqrt(dR**2 + dZ**2)
        s_cumulative = np.insert(np.cumsum(ds), 0, 0.0)
        s_total = s_cumulative[-1]
        
        # Find exactly where our target blob sits on this contour
        distances = (R_c - r_pt)**2 + (Z_c - z_pt)**2
        closest_idx = np.argmin(distances)
        
        # Normalize the distance to [0, 2pi]
        theta_arc_target[i] = 2 * np.pi * (s_cumulative[closest_idx] / s_total)
        
    return (psi_norm_target, theta_arc_target)