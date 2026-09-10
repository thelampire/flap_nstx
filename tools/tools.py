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
import string

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

def place_subplot_labels(axes,
                         labels=None,
                         label_format='({})',
                         fontsize=8.,
                         pad_points=3.,
                         y=1.0,
                         va='top',
                         make_room=True,
                         n_iteration=5,
                         **text_kwargs):
    """Place the (a), (b), ... subplot labels next to the axes dynamically.

    The horizontal position of the labels is calculated from the actual extent
    of the axes decorations (tick labels, axis labels), therefore the labels
    never overlap with the content of the plots regardless of how wide the
    tick labels are. All the labels of a call are put into the same column
    (the one belonging to the widest axes decoration) and each of them sits at
    the same relative height of its own axes, hence the labels are equidistant
    for equidistant subplots.

    The placement is repeated before every rendering of the figure, so a
    tight_layout() call or a resize done after this function cannot break it.

    Parameters:
        axes (list): the axes the labels belong to, in reading order.
        labels (list): the label texts, e.g. ['a','b']. Defaults to a,b,c,...
        label_format (str): format string applied to each label text.
        fontsize (float): font size of the labels.
        pad_points (float): gap between the axes decorations and the labels.
        y (float): vertical position of the labels in axes coordinates.
        va (str): vertical alignment of the labels.
        make_room (bool): if True, the subplot parameters are adjusted so the
            labels fit next to the axes. Call the function after the final
            tight_layout() call, that would overwrite the adjustment.
        n_iteration (int): maximum number of iterations for making room.

    Return:
        The list of the created matplotlib Text objects.
    """
    axes = [ax for ax in np.asarray(axes).flatten() if ax is not None]
    if len(axes) == 0:
        return []

    if labels is None:
        labels = [string.ascii_lowercase[ind % 26] for ind in range(len(axes))]
    if len(labels) < len(axes):
        raise ValueError('Not enough subplot labels for the number of axes.')

    # Axes sitting in different columns of the grid need their own label
    # column, otherwise every label would be placed next to the leftmost
    # axes and would overlap the panels of the columns on the right.
    columns = {}
    for ind, ax in enumerate(axes):
        columns.setdefault(round(ax.get_position().x0, 6), []).append(ind)

    if len(columns) > 1:
        texts = [None]*len(axes)
        for x_column in sorted(columns):
            indices = columns[x_column]
            column_texts = place_subplot_labels([axes[ind] for ind in indices],
                                                labels=[labels[ind] for ind in indices],
                                                label_format=label_format,
                                                fontsize=fontsize,
                                                pad_points=pad_points,
                                                y=y,
                                                va=va,
                                                make_room=make_room,
                                                n_iteration=n_iteration,
                                                **text_kwargs)
            for ind, text in zip(indices, column_texts):
                texts[ind] = text
        return texts

    fig = axes[0].figure

    texts = []
    for label in labels[0:len(axes)]:
        text = fig.text(0., 0., label_format.format(label),
                        size=fontsize,
                        ha='right',
                        va=va,
                        **text_kwargs)
        # The labels are excluded from the layout calculations, otherwise they
        # would push themselves further and further away from the axes.
        text.set_in_layout(False)
        texts.append(text)

    def _label_column(renderer):
        """Left edge (figure coords) of the label column and its width."""
        inverse = fig.transFigure.inverted()
        x_axes = min(ax.get_tightbbox(renderer).transformed(inverse).x0
                     for ax in axes)
        pad = pad_points/72./fig.get_figwidth()
        width = max(text.get_window_extent(renderer).transformed(inverse).width
                    for text in texts)
        return x_axes - pad, width

    def _free_space(renderer, x_label):
        """Free space on the left of the label column in figure coordinates.

        The space is either limited by the edge of the figure or by the
        decorations of the axes sitting in a column left of the labelled ones.
        """
        inverse = fig.transFigure.inverted()
        x_group = min(ax.get_position().x0 for ax in axes)
        x_left = 0.
        for other in fig.axes:
            if other in axes or other.get_position().x1 > x_group:
                continue
            x_left = max(x_left,
                         other.get_tightbbox(renderer).transformed(inverse).x1)
        return x_label - x_left

    def _make_room(renderer):
        """Rearrange the subplots so the label column fits next to the axes.

        The figure margin (leftmost group of axes) or the gap between the
        columns (any other group) is widened by the part of the labels which
        doesn't fit into the free space. Both keep the axes equally sized.
        """
        for _ in range(n_iteration):
            x_label, width = _label_column(renderer)
            missing = width - _free_space(renderer, x_label)
            if missing < 1e-3:
                break
            subplotpars = fig.subplotpars
            if (min(ax.get_position().x0 for ax in axes) ==
                    min(other.get_position().x0 for other in fig.axes)):
                fig.subplots_adjust(left=subplotpars.left + missing)
            else:
                # wspace is measured in units of the average axes width.
                average_width = np.mean([other.get_position().width
                                         for other in fig.axes])
                fig.subplots_adjust(wspace=subplotpars.wspace +
                                    missing/average_width)
            renderer = _get_figure_renderer(fig)
        return renderer

    def _reposition(renderer):
        """Put every label into the same column, next to its own axes."""
        x_label, _ = _label_column(renderer)
        for ax, text in zip(axes, texts):
            position = ax.get_position()
            text.set_position((x_label, position.y0 + y*position.height))

    def _place(renderer):
        """Make room for the labels and put them next to their axes."""
        if make_room:
            renderer = _make_room(renderer)
        _reposition(renderer)

    _place(_get_figure_renderer(fig))

    # The placement is repeated right before every rendering, this way a
    # tight_layout() call, a resize of the figure or a moved axes done after
    # this function cannot break the alignment. The placement is based on the
    # rendered extents, so repeating it doesn't move anything once it's good.
    placers = getattr(fig, '_subplot_label_placers', None)
    if placers is None:
        placers = []
        fig._subplot_label_placers = placers
        original_draw = fig.draw

        def draw(draw_renderer, *args, **kwargs):
            for place in placers:
                place(draw_renderer)
            return original_draw(draw_renderer, *args, **kwargs)

        fig.draw = draw
    placers.append(_place)

    return texts


def _get_figure_renderer(fig):
    """Return a renderer for the figure, working for every backend."""
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        from matplotlib.backend_bases import _get_renderer
        return _get_renderer(fig)


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

def read_equilibrium_data(shot=None, verbose=True):
    """
    Reads the EFIT02 equilibrium data objects needed for the flux coordinate
    calculation of a shot.

    This is the only function performing MDSplus reading, and it is meant to be
    called once per shot by the parent routine. The resulting object is then
    passed down to `get_equilibrium_slice` and `get_flux_coord`, so the slow
    reading is never repeated for the individual structures.

    Args:
        shot (int): Experimental shot number.
        verbose (bool, optional): Print a message when the reading fails.
            Defaults to True.

    Returns:
        dict or None: Dictionary with the 'psi_rz', 'r_axis', 'z_axis',
        'psi_axis' and 'psi_bdry' flap data objects, or None if the
        equilibrium data is not available for the shot.
    """
    signals = {'psi_rz': r'\EFIT02::\PSIRZ',
               'r_axis': r'\EFIT02::\RMAXIS',
               'z_axis': r'\EFIT02::\ZMAXIS',
               'psi_axis': r'\EFIT02::\SSIMAG',
               'psi_bdry': r'\EFIT02::\SSIBRY'}

    equilibrium = {}
    try:
        for key, signal_name in signals.items():
            equilibrium[key] = flap.get_data('NSTX_MDSPlus',
                                             name=signal_name,
                                             exp_id=shot,
                                             object_name=f'{key.upper()}_FOR_COORD_{shot}')
    except Exception as e:
        if verbose:
            print(f'The EFIT02 equilibrium data is unavailable for shot #{shot}: {e}')
            print('The flux coordinates are going to be filled with NaNs.')
        return None

    return equilibrium


def snap_to_equilibrium_time(equilibrium=None, time=None, shot=None, verbose=True):
    """
    Snaps a requested time onto the nearest EFIT reconstruction time.

    EFIT is reconstructed on a coarse time base (typically every 1ms), while
    every tracked structure asks for the equilibrium at its own mean time.
    Snapping onto the reconstruction times keeps the results of the structures
    belonging to the same EFIT sample identical.

    Args:
        equilibrium (dict): Output of `read_equilibrium_data`.
        time (float): Requested time.
        shot (int, optional): Shot number, only used in the message.
        verbose (bool, optional): Print a message when the time base cannot be
            determined. Defaults to True.

    Returns:
        float or None: The nearest EFIT time, or the unchanged input if the
        time base cannot be determined.
    """
    if time is None or equilibrium is None:
        return time

    try:
        efit_time = equilibrium['psi_rz'].coordinate('Time')[0]
        efit_time = np.unique(np.asarray(efit_time, dtype=float).ravel())
        return float(efit_time[np.argmin(np.abs(efit_time - float(time)))])
    except Exception as e:
        if verbose:
            print(f'The EFIT time base of shot #{shot} cannot be determined: {e}')
        return time


def get_equilibrium_slice(equilibrium=None, time=None, shot=None, verbose=True):
    """
    Returns the time sliced equilibrium quantities of a shot.

    Args:
        equilibrium (dict): Output of `read_equilibrium_data`.
        time (float): Time of the equilibrium slice.
        shot (int, optional): Shot number, only used in the messages.
        verbose (bool, optional): Print a message when the slicing fails.
            Defaults to True.

    Returns:
        dict or None: Dictionary with 'R_grid', 'z_grid', 'psi_grid', 'R_axis',
        'Z_axis', 'psi_axis' and 'psi_bdry', or None if the equilibrium is not
        available.
    """
    if equilibrium is None:
        return None

    #Snap onto the EFIT time base so that all the structures falling between
    #two reconstructions get consistent results.
    time = snap_to_equilibrium_time(equilibrium=equilibrium, time=time,
                                    shot=shot, verbose=verbose)

    try:
        psi_rz_obj = copy.deepcopy(equilibrium['psi_rz']).slice_data(slicing={'Time': time})

        sliced = {'R_grid': psi_rz_obj.coordinate('Device R')[0][:, 0],
                  'z_grid': psi_rz_obj.coordinate('Device z')[0][0, :],
                  'psi_grid': psi_rz_obj.data,
                  'R_axis': copy.deepcopy(equilibrium['r_axis']).slice_data(slicing={'Time': time}).data,
                  'Z_axis': float(copy.deepcopy(equilibrium['z_axis']).slice_data(slicing={'Time': time}).data),
                  'psi_axis': float(copy.deepcopy(equilibrium['psi_axis']).slice_data(slicing={'Time': time}).data),
                  'psi_bdry': float(copy.deepcopy(equilibrium['psi_bdry']).slice_data(slicing={'Time': time}).data),
                  }
    except Exception as e:
        if verbose:
            print(f'The EFIT02 equilibrium of shot #{shot} cannot be sliced at {time}s: {e}')
        return None

    return sliced


from matplotlib.path import Path as _matplotlib_path
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay as _Delaunay


def _select_flux_contour(contours, R_grid_1d, z_grid_1d, R_axis, Z_axis):
    """
    Picks the physically meaningful branch out of the contours belonging to one
    psi level and converts it to physical (R, z) coordinates.

    A single psi level set is generally not a single curve: close to the
    separatrix it splits into the core flux surface and the divertor legs
    around the X-point. Selecting simply the longest branch (as the previous
    implementation did) makes the choice flip between neighbouring frames,
    which quantizes the resulting poloidal angle. Here the branch which
    actually encircles the magnetic axis is preferred, and only if no such
    branch exists is the one closest to the outboard midplane used.

    Args:
        contours (list): Output of `skimage.measure.find_contours` (index space).
        R_grid_1d (np.ndarray): Physical R coordinates of the psi grid.
        z_grid_1d (np.ndarray): Physical z coordinates of the psi grid.
        R_axis (float): Major radius of the magnetic axis.
        Z_axis (float): Vertical position of the magnetic axis.

    Returns:
        tuple or None: (R_contour, z_contour, is_closed) of the selected branch,
        or None if no usable branch was found.
    """
    best_enclosing = None
    best_enclosing_len = -1.0
    best_fallback = None
    best_fallback_dist = np.inf

    for contour in contours:
        if len(contour) < 4:
            continue

        R_c = np.interp(contour[:, 0], np.arange(len(R_grid_1d)), R_grid_1d)
        z_c = np.interp(contour[:, 1], np.arange(len(z_grid_1d)), z_grid_1d)

        #A contour is closed if skimage returned identical end points.
        is_closed = (np.isclose(contour[0, 0], contour[-1, 0]) and
                     np.isclose(contour[0, 1], contour[-1, 1]))

        if is_closed:
            polygon = _matplotlib_path(np.column_stack((R_c, z_c)))
            if polygon.contains_point((R_axis, Z_axis)):
                #Among the surfaces enclosing the axis the outermost one is the
                #relevant flux surface for the requested psi value.
                circumference = np.sum(np.hypot(np.diff(R_c), np.diff(z_c)))
                if circumference > best_enclosing_len:
                    best_enclosing_len = circumference
                    best_enclosing = (R_c, z_c, True)
                continue

        #Open (SOL / divertor leg) branches: keep the one passing closest to the
        #outboard midplane, which is where the GPI structures live.
        omp_distance = np.min(np.hypot(R_c - R_axis, z_c - Z_axis) *
                              np.where(R_c > R_axis, 1.0, 1e3))
        if omp_distance < best_fallback_dist:
            best_fallback_dist = omp_distance
            best_fallback = (R_c, z_c, is_closed)

    if best_enclosing is not None:
        return best_enclosing
    return best_fallback


def _orient_and_reference_contour(R_c, z_c, is_closed, R_axis, Z_axis):
    """
    Applies a globally consistent orientation and theta=0 reference to a contour.

    Both the direction of travel and the origin of the arc length must be
    derived from a robust global property of the curve, otherwise near-ties
    make the choice flip between frames and the resulting angle becomes
    discretized. The direction is fixed from the total signed winding around
    the magnetic axis (counter-clockwise) instead of comparing two neighbouring
    samples, and the origin is the point where the curve crosses the outboard
    midplane.

    Args:
        R_c (np.ndarray): R coordinates of the contour.
        z_c (np.ndarray): z coordinates of the contour.
        is_closed (bool): Whether the contour is a closed curve.
        R_axis (float): Major radius of the magnetic axis.
        Z_axis (float): Vertical position of the magnetic axis.

    Returns:
        tuple: (R_c, z_c, s_cumulative, s_total, s_reference) with the contour
        oriented counter-clockwise and the arc length origin at the OMP.
    """
    geometric_angle = np.arctan2(z_c - Z_axis, R_c - R_axis)

    #Global orientation: the net winding decides, not a single sample pair.
    net_winding = np.sum(np.diff(np.unwrap(geometric_angle)))
    if net_winding < 0:
        R_c = R_c[::-1]
        z_c = z_c[::-1]
        geometric_angle = geometric_angle[::-1]

    if is_closed:
        #Roll the closed curve so that it starts at the outboard midplane.
        outboard = R_c > R_axis
        if np.any(outboard):
            candidate_indices = np.where(outboard)[0]
            start_index = candidate_indices[np.argmin(np.abs(geometric_angle[candidate_indices]))]
        else:
            start_index = int(np.argmax(R_c))

        R_c = np.roll(R_c, -start_index)
        z_c = np.roll(z_c, -start_index)
        #Close the curve explicitly so the arc length spans the full loop.
        R_c = np.append(R_c, R_c[0])
        z_c = np.append(z_c, z_c[0])
        geometric_angle = np.arctan2(z_c - Z_axis, R_c - R_axis)

    segment_lengths = np.hypot(np.diff(R_c), np.diff(z_c))
    s_cumulative = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    s_total = s_cumulative[-1]

    if s_total <= 0:
        return None

    if is_closed:
        s_reference = 0.0
    else:
        #Open branch: reference the arc length to its OMP crossing so that the
        #angle stays comparable with the closed surfaces.
        outboard = R_c > R_axis
        if np.any(outboard):
            candidate_indices = np.where(outboard)[0]
            reference_index = candidate_indices[np.argmin(np.abs(geometric_angle[candidate_indices]))]
        else:
            reference_index = int(np.argmax(R_c))
        s_reference = s_cumulative[reference_index]

    return (R_c, z_c, s_cumulative, s_total, s_reference)


def _build_theta_map(equilibrium,
                     n_levels=200,
                     psi_norm_range=(0.02, 1.6),
                     ):
    """
    Builds a continuous poloidal angle map theta(R, z) for one equilibrium slice.

    The map is constructed once for a whole time slice by tracing a family of
    flux surfaces with a consistent branch selection, orientation and arc length
    origin, and then interpolating the resulting scattered angle samples. This
    replaces the previous per-point contour tracing, where every structure
    position triggered an independent `find_contours` call whose branch choice,
    direction and origin could differ from point to point and therefore
    quantized the angle onto a few discrete values.

    The angle is interpolated through its cosine and sine components so the
    0 <-> 2pi seam does not leak into the interpolation.

    Args:
        equilibrium (dict): Output of `get_equilibrium_slice`.
        n_levels (int, optional): Number of traced flux surfaces. Defaults to 200.
        psi_norm_range (tuple, optional): Normalized flux range covered by the
            traced surfaces. Defaults to (0.02, 1.6). The upper limit has to
            reach well into the SOL, otherwise the structures sitting outside
            the traced family fall back to nearest neighbour interpolation,
            which reintroduces exactly the staircase-like quantization this
            map is meant to remove.

    Returns:
        dict or None: Interpolators for the cosine and sine of the angle plus a
        nearest neighbour fallback, or None if no surface could be traced.
    """
    R_grid_1d = equilibrium['R_grid']
    z_grid_1d = equilibrium['z_grid']
    psi_grid_2d = equilibrium['psi_grid']
    R_axis = float(np.atleast_1d(equilibrium['R_axis'])[0])
    Z_axis = equilibrium['Z_axis']
    psi_axis = equilibrium['psi_axis']
    psi_bdry = equilibrium['psi_bdry']

    sample_points = []
    sample_cos = []
    sample_sin = []

    for psi_norm_level in np.linspace(psi_norm_range[0], psi_norm_range[1], n_levels):
        psi_level = psi_axis + psi_norm_level * (psi_bdry - psi_axis)

        contours = skimage.measure.find_contours(psi_grid_2d, psi_level)
        if not contours:
            continue

        selected = _select_flux_contour(contours, R_grid_1d, z_grid_1d, R_axis, Z_axis)
        if selected is None:
            continue

        oriented = _orient_and_reference_contour(*selected, R_axis=R_axis, Z_axis=Z_axis)
        if oriented is None:
            continue

        R_c, z_c, s_cumulative, s_total, s_reference = oriented

        theta_c = 2 * np.pi * np.mod((s_cumulative - s_reference) / s_total, 1.0)

        sample_points.append(np.column_stack((R_c, z_c)))
        sample_cos.append(np.cos(theta_c))
        sample_sin.append(np.sin(theta_c))

    if not sample_points:
        return None

    points = np.vstack(sample_points)
    cos_values = np.concatenate(sample_cos)
    sin_values = np.concatenate(sample_sin)

    # The Delaunay triangulation dominates the cost of building the map, and it
    # only depends on the sample positions. Building it once and reusing it for
    # both the cosine and the sine interpolator halves the build time.
    triangulation = _Delaunay(points)

    return {'cos_linear': LinearNDInterpolator(triangulation, cos_values),
            'sin_linear': LinearNDInterpolator(triangulation, sin_values),
            'cos_nearest': NearestNDInterpolator(points, cos_values),
            'sin_nearest': NearestNDInterpolator(points, sin_values),
            }


def get_theta_map(equilibrium_slice=None, shot=None, time=None, verbose=True):
    """
    Builds the poloidal angle map of an already sliced equilibrium.

    Args:
        equilibrium_slice (dict): Output of `get_equilibrium_slice`.
        shot (int, optional): Shot number, only used in the message.
        time (float, optional): Time of the slice, only used in the message.
        verbose (bool, optional): Print a message when the map cannot be built.
            Defaults to True.

    Returns:
        dict or None: The angle map built by `_build_theta_map`, or None if the
        equilibrium is unavailable or the map cannot be built.
    """
    if equilibrium_slice is None:
        return None

    try:
        return _build_theta_map(equilibrium_slice)
    except Exception as e:
        if verbose:
            print(f'The poloidal angle map cannot be built for shot #{shot} at {time}s: {e}')
        return None


def fold_poloidal_angle(theta):
    """
    Folds a poloidal angle from [0, 2pi) into [0, pi/2].

    This is the legacy behaviour of `get_flux_coord`. It is lossy: the two
    reflections map four physically different poloidal positions onto the same
    value and clamp the traces at the fold axes, so it should only be used for
    reproducing older results.

    Args:
        theta (np.ndarray): Poloidal angle in radians.

    Returns:
        np.ndarray: The folded angle.
    """
    theta = np.asarray(theta, dtype=float)
    folded_theta = np.copy(theta)

    finite_mask = np.isfinite(theta)
    if np.any(finite_mask):
        folded = np.mod(theta[finite_mask], 2 * np.pi)
        #Fold [pi, 2pi) down onto [0, pi) (reflection about the midplane)
        folded = np.where(folded > np.pi, 2 * np.pi - folded, folded)
        #Fold (pi/2, pi] onto [0, pi/2] (reflection about the vertical axis)
        folded = np.where(folded > np.pi / 2, np.pi - folded, folded)
        folded_theta[finite_mask] = folded

    return folded_theta


#Approximate GPI field of view, only used for the sanity check of the angles.
#Derived from `spatial_calibration_coeffs`: R = 1.34..1.64 m, z = 0.07..0.32 m.
GPI_FOV_THETA_LIMITS = (0.0, 0.7)


def get_geometric_poloidal_angle(R_target, z_target, R_axis, Z_axis):
    """
    Returns the geometric poloidal angle measured from the magnetic axis.

    theta = atan2(z - z_axis, R - R_axis), i.e. zero at the outboard midplane
    and increasing upwards.

    Unlike the arc length based angle this is defined everywhere, both inside
    and outside the separatrix, it needs no contour tracing, no branch
    selection and no arc length normalization. That makes it the appropriate
    choice for an outboard midplane diagnostic such as GPI, whose field of view
    lies largely in the SOL where closed flux surfaces simply do not exist.

    Args:
        R_target (np.ndarray): Major radius coordinates of the targets.
        z_target (np.ndarray): Vertical coordinates of the targets.
        R_axis (float): Major radius of the magnetic axis.
        Z_axis (float): Vertical position of the magnetic axis.

    Returns:
        np.ndarray: The poloidal angle in radians, wrapped into [0, 2pi).
    """
    return np.mod(np.arctan2(np.asarray(z_target, dtype=float) - Z_axis,
                             np.asarray(R_target, dtype=float) - R_axis),
                  2 * np.pi)


def check_gpi_theta_range(theta, shot=None, limits=GPI_FOV_THETA_LIMITS, verbose=True):
    """
    Warns if the poloidal angles fall outside the range the GPI field of view
    can physically produce.

    The GPI field of view covers roughly R = 1.34..1.64 m and z = 0.07..0.32 m
    on the low field side, which corresponds to a geometric poloidal angle of
    about 0.12..0.67 rad around a typical NSTX magnetic axis. Angles far
    outside this band mean that the angle definition or the equilibrium is
    wrong, so this check is meant to surface such errors immediately instead of
    letting them show up in a plot.

    Args:
        theta (np.ndarray): Poloidal angles in radians.
        shot (int, optional): Shot number, only used in the message.
        limits (tuple, optional): Accepted (min, max) angle in radians.
            Defaults to `GPI_FOV_THETA_LIMITS`.
        verbose (bool, optional): Print the warning. Defaults to True.

    Returns:
        bool: True if all the finite angles are inside the limits.
    """
    theta = np.asarray(theta, dtype=float)
    finite_theta = theta[np.isfinite(theta)]

    if len(finite_theta) == 0:
        return True

    #The angle is wrapped into [0, 2pi), so a point slightly below the midplane
    #shows up just under 2pi. Map it back onto a small negative angle before
    #comparing against the limits.
    signed_theta = np.where(finite_theta > np.pi, finite_theta - 2 * np.pi, finite_theta)

    outside = (signed_theta < limits[0] - 0.2) | (signed_theta > limits[1] + 0.2)
    if np.any(outside):
        if verbose:
            print(f'Warning: {np.count_nonzero(outside)}/{len(finite_theta)} poloidal '
                  f'angles of shot #{shot} lie outside the range the GPI field of view '
                  f'can produce ({limits[0]:.2f}..{limits[1]:.2f} rad). '
                  f'Observed {signed_theta.min():.2f}..{signed_theta.max():.2f} rad.')
        return False

    return True


def get_flux_coord(shot=None,
                   time=None,
                   R_target=None, 
                   z_target=None,
                   equilibrium=None,
                   equilibrium_slice=None,
                   theta_map=None,
                   theta_method='geometric',
                   fold_angle=False,
                   check_theta_range=True,
                   verbose=True):
    """
    Calculates the normalized poloidal flux and the poloidal angle for a
    specific list of absolute (R, Z) points.
    
    The EFIT equilibrium is expected to be read once by the parent routine and
    passed in through `equilibrium` (or, already sliced, through
    `equilibrium_slice`), so that the slow MDSplus reading is not repeated for
    every structure. If nothing is passed, the equilibrium is read here as a
    convenience. If the equilibrium data is not available on the server, NaN
    arrays are returned instead of raising an exception.
    
    Two angle definitions are available:
    
    - 'geometric' (default): theta = atan2(z - z_axis, R - R_axis). Defined
      everywhere, including the SOL, and directly comparable to the geometric
      position of the structures. This is the appropriate choice for GPI, whose
      field of view is on the low field side and lies mostly outside the
      separatrix.
    - 'arclength': the normalized arc length along the flux surface, evaluated
      from a theta(R, z) map (see `get_theta_map`). Only meaningful on
      closed flux surfaces; outside the separatrix the arc length is normalized
      by a grid clipped curve length, which makes the value arbitrary. It is
      also strongly compressed with respect to the geometric angle on shaped
      surfaces (a geometric 0.35 rad maps to ~0.21 rad at kappa = 2.2).
    
    Args:
        shot (int): Experimental shot number.
        time (float): Time slice to extract equilibrium data.
        R_target (np.ndarray): 1D array of Target Major Radius coordinates (from center of torus).
        z_target (np.ndarray): 1D array of Target Vertical coordinates.
        equilibrium (dict, optional): Output of `read_equilibrium_data`. Read
            here if neither this nor `equilibrium_slice` is given.
        equilibrium_slice (dict, optional): Output of `get_equilibrium_slice`.
            Takes precedence over `equilibrium`, avoiding the re-slicing.
        theta_map (dict, optional): Output of `get_theta_map`, only used when
            `theta_method` is 'arclength'. Built here if not given.
        theta_method (str, optional): Either 'geometric' or 'arclength'.
            Defaults to 'geometric'.
        fold_angle (bool, optional): Fold the angle into [0, pi/2] the way the
            original implementation did. Lossy, only kept for reproducing older
            results. Defaults to False.
        check_theta_range (bool, optional): Warn if the resulting angles cannot
            be produced by the GPI field of view. Defaults to True.
        verbose (bool, optional): Print a message when the equilibrium reading
            fails. Defaults to True.
        
    Returns:
        tuple: ('psi_norm', 'theta') arrays corresponding to the target points.
    """
    if theta_method not in ['geometric', 'arclength']:
        raise ValueError("theta_method can only be 'geometric' or 'arclength'.")
    
    R_target = np.atleast_1d(R_target)
    Z_target = np.atleast_1d(z_target)
    
    # =========================================================================
    # 1. Obtain the equilibrium slice (passed in by the parent routine)
    # =========================================================================
    if equilibrium_slice is None:
        if equilibrium is None:
            equilibrium = read_equilibrium_data(shot=shot, verbose=verbose)
        equilibrium_slice = get_equilibrium_slice(equilibrium=equilibrium,
                                                  time=time,
                                                  shot=shot,
                                                  verbose=verbose)
    
    if equilibrium_slice is None:
        nan_array = np.full(len(R_target), np.nan)
        return (nan_array, np.full(len(R_target), np.nan))
    
    R_grid_1d = equilibrium_slice['R_grid']
    z_grid_1d = equilibrium_slice['z_grid']
    psi_grid_2d = equilibrium_slice['psi_grid']
    
    psi_axis = equilibrium_slice['psi_axis']
    psi_bdry = equilibrium_slice['psi_bdry']
    
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
    # 3. Poloidal angle
    # =========================================================================
    if theta_method == 'geometric':
        theta_target = get_geometric_poloidal_angle(R_target,
                                                    Z_target,
                                                    float(np.atleast_1d(equilibrium_slice['R_axis'])[0]),
                                                    equilibrium_slice['Z_axis'])
    else:
        if theta_map is None:
            theta_map = get_theta_map(equilibrium_slice=equilibrium_slice,
                                      shot=shot, time=time, verbose=verbose)
        
        if theta_map is None:
            theta_target = np.full(len(R_target), np.nan)
        else:
            cos_target = theta_map['cos_linear'](target_coords)
            sin_target = theta_map['sin_linear'](target_coords)
            
            # Points falling just outside the traced surface family are filled
            # from the nearest neighbour interpolator instead of being dropped
            # to NaN. This fallback is piecewise constant, so it can
            # reintroduce a staircase-like quantization.
            outside = ~np.isfinite(cos_target) | ~np.isfinite(sin_target)
            if np.any(outside):
                cos_target[outside] = theta_map['cos_nearest'](target_coords[outside])
                sin_target[outside] = theta_map['sin_nearest'](target_coords[outside])
                if verbose and np.count_nonzero(outside) > 0.1 * len(R_target):
                    print(f'Warning: {np.count_nonzero(outside)}/{len(R_target)} points of '
                          f'shot #{shot} lie outside the traced flux surfaces, their '
                          f'poloidal angle is nearest neighbour interpolated.')
            
            theta_target = np.mod(np.arctan2(sin_target, cos_target), 2 * np.pi)
        
        # The arc length angle is meaningless outside the separatrix, where the
        # curve is open and its length depends on the EFIT grid boundary.
        open_surface = np.isfinite(psi_norm_target) & (psi_norm_target > 1.0)
        if np.any(open_surface):
            theta_target[open_surface] = np.nan
            if verbose:
                print(f'Warning: {np.count_nonzero(open_surface)}/{len(R_target)} points of '
                      f'shot #{shot} lie outside the separatrix, where the arc length '
                      f"angle is undefined. Use theta_method='geometric' instead.")
    
    # The angle is meaningless where psi itself could not be interpolated.
    theta_target[~np.isfinite(psi_target)] = np.nan
    
    # =========================================================================
    # 4. Optional legacy folding into [0, pi/2]
    # =========================================================================
    if fold_angle:
        theta_target = fold_poloidal_angle(theta_target)
    elif check_theta_range:
        check_gpi_theta_range(theta_target, shot=shot, verbose=verbose)

    return (psi_norm_target, theta_target)
