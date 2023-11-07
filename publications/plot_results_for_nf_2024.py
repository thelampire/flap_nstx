#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:15:30 2023

@author: mlampert
"""
class Hell(Exception):pass

import os
import copy


import flap
import flap_nstx

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

from flap_nstx.gpi import calculate_nstx_gpi_angular_velocity, show_nstx_gpi_video_frames
from flap_nstx.gpi import analyze_gpi_structures
from flap_nstx.test import test_angular_displacement_estimation
from flap_nstx.analysis import calculate_blob_parameter_histograms,plot_blob_blob_parameter_trends

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

import numpy as np
from skimage.filters import window, difference_of_gaussians

wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/publication_figures/rsi_2022'


plt.rc('font', family='serif', serif='Helvetica')
labelsize=9.
linewidth=0.5
major_ticksize=2.
plt.rc('text', usetex=False)
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
plt.rcParams['xtick.minor.size'] = major_ticksize/2

plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['ytick.major.width'] = linewidth
plt.rcParams['ytick.major.size'] = major_ticksize
plt.rcParams['ytick.minor.width'] = linewidth/2
plt.rcParams['ytick.minor.size'] = major_ticksize/2
plt.rcParams['legend.fontsize'] = labelsize


def plot_results_for_iaea_2023(plot_figure=2,
                             save_data_into_txt=False,
                             plot_all=False,
                             nocalc=False):

    if plot_all:
        plot_figure=-1
        for i in range(15):
            plot_results_for_iaea_2023(plot_figure=i,
                                     save_data_into_txt=save_data_into_txt)

    """
    GPI plot
    """
    if plot_figure == 1:
        raise ValueError('GPI plot, no need to create figure.')

    """
    Watershed segmentation plot horizontal
    """
    if plot_figure == 2:
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.552,0.5522],
                               ignore_side_structures=True,
                               pdf=True,
                               plot_watershed_steps=4, #4th frame
                               plot_for_publication=True,
                               nocalc=False,
                               plot=False,
                               )

    """
    Flowchart plot
    """
    if plot_figure == 3:
        raise ValueError('Flowchart plot, no need to create figure.')

    """Blob evolution frames"""
    if plot_figure == 4:
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.552,0.5522],
                               ignore_side_structures=True,
                               pdf=True,
                               plot_example_structure_frames=1,
                               nocalc=False,
                               plot=False,
                               plot_for_publication=True,
                               min_structure_lifetime=1,
                               )

    """Blob evolution results"""
    if plot_figure == 5:
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.551,0.5522],
                               ignore_side_structures=True,
                               pdf=True, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=False,
                               min_structure_lifetime=4,
                               nocalc=True,
                               plot_scatter=True,
                               plot_tracking=True,
                               plot_example_frames_results=True,
                               )

    """Blob histrogram distribution results"""
    if plot_figure == 6:
        calculate_blob_parameter_histograms(nocalc=True,
                                            pdf=True,
                                            plot_for_publication=True,
                                            calc_mean_distribution=False)

    """Blob blob trends"""
    if plot_figure == 7:
        plot_blob_blob_parameter_trends(plot_for_publication=True,
                                        calc_mean_distribution=False)