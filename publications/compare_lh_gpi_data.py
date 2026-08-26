#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:38:15 2026

@author: mlampert
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add flap_repos to Python path if not already there
sys.path.append('/Users/mlampert/work/repos/flap_repos')

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

from flap_nstx.gpi import normalize_gpi

# Make sure flap_nstx is configured
thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir, "flap_nstx.cfg")
flap.config.read(file_name=fn)

def get_normalized_data(exp_id, time_range):
    print(f"Loading data for shot {exp_id}...")
    data_obj_name = 'GPI'
    
    # Try to load cached data, if not read it
    try:
        data = flap.get_data_object(data_obj_name, exp_id=exp_id)
    except:
        data = flap.get_data('NSTX_GPI', exp_id=exp_id, name='', object_name=data_obj_name)
    
    slicing_time = {'Time': flap.Intervals(time_range[0], time_range[1])}
    data_sliced = data.slice_data(slicing=slicing_time)
    
    data_sliced_name = f"{data_obj_name}_{exp_id}_sliced"
    flap.add_data_object(data_sliced, data_sliced_name)
    
    normalized_name = f"GPI_NORMALIZED_{exp_id}"
    print(f"Normalizing data for shot {exp_id}...")
    
    # Core normalizing function from analyze_gpi_structures
    normalize_gpi(data_sliced_name,
                  exp_id=exp_id,
                  slicing_time=slicing_time,
                  normalize='simple',
                  normalize_f_high=1e3,
                  normalize_f_kernel='Elliptic',
                  output_name=normalized_name)
    
    # Get the normalized data object reference
    norm_obj = flap.get_data_object_ref(normalized_name)
    data_arr = norm_obj.data
    time_coords = norm_obj.coordinate('Time')[0]
    
    return data_arr, time_coords

# Parameters
hmode_shot = 141319
hmode_time = [0.552, 0.5525]

lmode_shot = 141998
lmode_time = [0.22012, 0.221]

# Get normalized data arrays
hmode_data, hmode_t = get_normalized_data(hmode_shot, hmode_time)
lmode_data, lmode_t = get_normalized_data(lmode_shot, lmode_time)

# Determine the time axis based on the data shape
if hmode_data.shape[0] == len(hmode_t) or (hmode_data.ndim == 3 and hmode_data.shape[0] > 100):
    time_axis = 0
else:
    time_axis = -1

num_frames_to_plot = 5

# Select 5 evenly spaced indices across the time window
n_frames_h = hmode_data.shape[time_axis]
idx_h = np.linspace(0, n_frames_h-1, num_frames_to_plot, dtype=int)

n_frames_l = lmode_data.shape[time_axis]
idx_l = np.linspace(0, n_frames_l-1, num_frames_to_plot, dtype=int)

# Set up the plot
fig, axes = plt.subplots(2, num_frames_to_plot, figsize=(15, 6))

for i in range(num_frames_to_plot):
    # H-mode
    if time_axis == 0:
        frame_h = hmode_data[idx_h[i], :, :]
        frame_l = lmode_data[idx_l[i], :, :]
    else:
        frame_h = hmode_data[:, :, idx_h[i]]
        frame_l = lmode_data[:, :, idx_l[i]]
        
    ax_h = axes[0, i]
    ax_h.imshow(frame_h, cmap='inferno', origin='lower')
    ax_h.set_title(f"H-mode t={hmode_t[idx_h[i]]*1000}ms")
    ax_h.axis('off')
    
    # L-mode
    ax_l = axes[1, i]
    ax_l.imshow(frame_l, cmap='inferno', origin='lower')
    ax_l.set_title(f"L-mode t={lmode_t[idx_l[i]]*1000}ms")
    ax_l.axis('off')
    
plt.tight_layout()
plt.suptitle('NSTX GPI L-mode vs H-mode Comparison', fontsize=16, y=1.05)

plt.show()