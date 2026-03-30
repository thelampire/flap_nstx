#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:48:59 2021

@author: mlampert
"""

import copy
import numpy as np
import flap

def normalize_gpi(input_object,
                  exp_id=None,
                  normalize='roundtrip',
                  normalize_f_high=1e3,
                  normalize_f_kernel='Elliptic',
                  normalizer_object_name='GPI_NORMALIZER',
                  slicing_time=None,
                  output_name=None,
                  return_object='coefficient'
                  ):
    """
    Normalizes Gas Puff Imaging (GPI) data using temporal low-pass filtering.

    This function extracts the low-frequency background of a GPI signal (the "gas cloud" 
    contribution) and uses it to normalize the raw data. It supports standard low-pass 
    filtering, zero-phase filtering (roundtrip), and an ELM-specific split filtering 
    (halved) to handle sharp signal transients.

    Args:
        input_object (str): The name of the input FLAP data object to be normalized.
        exp_id (int, optional): The experiment shot number. Defaults to None.
        normalize (str, optional): The normalization strategy. 
            Options:
                - 'simple': Standard forward low-pass filter.
                - 'roundtrip': Forward and backward low-pass filter (zero-phase).
                - 'halved': Splits filtering before and after the ELM peak.
                - None: Skips calculation and returns None.
            Defaults to 'roundtrip'.
        normalize_f_high (float, optional): The cutoff frequency for the low-pass 
            filter in Hz. Defaults to 1e3.
        normalize_f_kernel (str, optional): The filter design type (e.g., 'Elliptic', 
            'Butterworth'). Defaults to 'Elliptic'.
        normalizer_object_name (str, optional): Name to assign the intermediate 
            filtered background data object in the FLAP registry. 
            Defaults to 'GPI_NORMALIZER'.
        slicing_time (dict, optional): A dictionary containing FLAP slicing intervals 
            to restrict the output time range. Defaults to None.
        output_name (str, optional): Name to assign the final output data object in 
            the FLAP registry. Defaults to None.
        return_object (str, optional): Dictates the returned value. 
            Options:
                - 'coefficient': Returns the raw NumPy array of the background.
                - 'data subtract': Returns a FLAP object of (raw - background).
                - 'data divide': Returns a FLAP object of (raw / background).
            Defaults to 'coefficient'.

    Returns:
        np.ndarray or flap.DataObject: The normalized data or background coefficient, 
        depending on `return_object`. Returns None if `normalize` is None.
    """

    if normalize is None:
        return None

    # Base filter options used across multiple branches
    filter_options = {
        'Type': 'Lowpass',
        'f_high': normalize_f_high,
        'Design': normalize_f_kernel
    }

    # --- 1. Simple Filtering ---
    if normalize == 'simple':
        flap.filter_data(input_object, exp_id=exp_id, coordinate='Time',
                         options=filter_options, output_name=normalizer_object_name)
                         
        coefficient = flap.slice_data(normalizer_object_name, exp_id=exp_id,
                                      slicing=slicing_time, output_name=output_name).data

    # --- 2. Roundtrip (Zero-Phase) Filtering ---
    elif normalize == 'roundtrip':
        # Forward pass
        flap.filter_data(input_object, exp_id=exp_id, coordinate='Time',
                         options=filter_options, output_name=normalizer_object_name)

        # Reverse data array safely
        norm_obj = flap.get_data_object(normalizer_object_name)
        norm_obj.data = np.flip(norm_obj.data, axis=0)
        flap.add_data_object(norm_obj, normalizer_object_name)

        # Backward pass
        flap.filter_data(normalizer_object_name, exp_id=exp_id, coordinate='Time',
                         options=filter_options, output_name=normalizer_object_name)

        # Reverse data array back to original time orientation
        norm_obj = flap.get_data_object(normalizer_object_name)
        norm_obj.data = np.flip(norm_obj.data, axis=0)
        flap.add_data_object(norm_obj, normalizer_object_name)

        coefficient = flap.slice_data(normalizer_object_name, exp_id=exp_id,
                                      slicing=slicing_time, output_name=output_name).data

    # --- 3. Halved Filtering (ELM Specific) ---
    elif normalize == 'halved':
        # Find the peak of the signal (proxy for ELM time)
        data_obj_trace = flap.get_data_object_ref(input_object).slice_data(summing={'Image x': 'Mean', 'Image y': 'Mean'})
        ind_peak = np.argmax(data_obj_trace.data)

        # Create reversed input object
        data_obj_reverse = copy.deepcopy(flap.get_data_object(input_object))
        data_obj_reverse.data = np.flip(data_obj_reverse.data, axis=0)
        input_object_rev = input_object + '_REV'
        flap.add_data_object(data_obj_reverse, input_object_rev)

        # Forward pass
        flap.filter_data(input_object, exp_id=exp_id, coordinate='Time',
                         options=filter_options, output_name=normalizer_object_name)
        coefficient1_sliced = flap.slice_data(normalizer_object_name, exp_id=exp_id, slicing=slicing_time)

        # Backward pass
        normalizer_object_name_reverse = normalizer_object_name + '_REV'
        flap.filter_data(input_object_rev, exp_id=exp_id, coordinate='Time',
                         options=filter_options, output_name=normalizer_object_name_reverse)

        # Reverse the backward-filtered result
        coeff2_obj = flap.get_data_object(normalizer_object_name_reverse)
        coeff2_obj.data = np.flip(coeff2_obj.data, axis=0)
        flap.add_data_object(coeff2_obj, normalizer_object_name_reverse)
        
        coefficient2_sliced = flap.slice_data(normalizer_object_name_reverse, exp_id=exp_id, slicing=slicing_time)

        # Stitch them together, giving a 4-frame buffer before the peak
        split_idx = max(0, ind_peak - 4)
        coeff1_first_half = coefficient1_sliced.data[:split_idx, :, :]
        coeff2_second_half = coefficient2_sliced.data[split_idx:, :, :]
        
        coefficient = np.concatenate((coeff1_first_half, coeff2_second_half), axis=0)
        
        # Save combined coefficient as FLAP object
        coefficient_dataobject = copy.deepcopy(coefficient1_sliced)
        coefficient_dataobject.data = coefficient
        
        if output_name is not None:
            flap.add_data_object(coefficient_dataobject, output_name)

    # --- 4. Return Handling ---
    if return_object == 'coefficient':
        return coefficient
    
    # Calculate derived normalized object
    data_obj_orig = flap.get_data_object_ref(input_object)
    data_obj_normalized = data_obj_orig.slice_data(slicing=slicing_time)

    if return_object == 'data subtract':
        data_obj_normalized.data = data_obj_normalized.data - coefficient
    elif return_object == 'data divide':
        data_obj_normalized.data = data_obj_normalized.data / coefficient

    if output_name is not None:
        flap.add_data_object(data_obj_normalized, output_name)

    return data_obj_normalized