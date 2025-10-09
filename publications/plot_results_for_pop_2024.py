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

from flap_nstx.analysis import calculate_blob_parameter_histograms
from flap_nstx.analysis import plot_blob_blob_parameter_trends
from flap_nstx.analysis import calculate_blob_blob_parameter_correlation_matrix
from flap_nstx.analysis import plot_blob_plasma_parameter_trends
from flap_nstx.analysis import calculate_blob_plasma_parameter_correlation_matrix
from flap_nstx.analysis import plot_blob_regime_graph,plot_well_known_parameter_dependences

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

import numpy as np
from skimage.filters import window, difference_of_gaussians

wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/publication_figures/rsi_2022'

flap_nstx.tools.set_matplotlib_for_publication(labelsize=8.,
                                               linewidth=0.5,
                                               major_ticksize=2.,
                                               minor_ticksize=1.)

"""
DOES NOT YET HAVE THE DATA AVAILABILITY OUTPUT
"""


def plot_results_for_pop_2024(plot_figure=2,
                               save_data_into_txt=False,
                               plot_all=False,
                               nocalc=False):

    if plot_all:
        plot_figure=-1
        for i in range(15):
            plot_results_for_pop_2024(plot_figure=i,
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
        # analyze_gpi_structures(exp_id=141319,
        #                        time_range=[0.552+2.5e-6,0.5522],
        #                        ignore_side_structures=True,
        #                        pdf=True,
        #                        plot_example_structure_frames=1,
        #                        nocalc=False,
        #                        plot=False,
        #                        plot_for_publication=True,
        #                        min_structure_lifetime=8,
        #                        plot_ncol=3, plot_nframe=9
        #                        )
        
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.540, 0.5522],
                               plot_time_range=[0.5465,0.5466],
                               ignore_side_structures=True,
                               pdf=True,
                               plot_example_structure_frames=1,
                               nocalc=True,
                               plot=False,
                               plot_for_publication=True,
                               min_structure_lifetime=0,
                               plot_ncol=3, plot_nframe=9,
                               plot_separatrix=True
                               )

    """Blob evolution results"""
    if plot_figure == 5:
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.540, 0.5522],
                               plot_time_range=[0.544, 0.548],
                               ignore_side_structures=True,
                               pdf=True, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=True,
                               min_structure_lifetime=10,
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
                                            calc_mean_distribution=False,
                                            min_structure_lifetime=10)


    if plot_figure == 7:
        calculate_blob_blob_parameter_correlation_matrix(threshold_corr=True,
                                                         calc_mean_distribution=False,
                                                         plot_interesting_only=True,
                                                         min_structure_lifetime=10
                                                         )
    """Blob blob trends"""
    if plot_figure == 8:
        plot_blob_blob_parameter_trends(plot_for_publication=True,
                                        calc_mean_distribution=False,
                                        min_structure_lifetime=10)

    if plot_figure == 9:
        plot_blob_plasma_parameter_trends(plot_for_publication=True)
        
    
    if plot_figure== 10:
        plot_blob_regime_graph(nocalc=True)
        
    if plot_figure==11:
        plot_well_known_parameter_dependences(pdf_filename=None,
                                              nocalc=True,
                                              )
        
    if plot_figure == 12:
        calculate_blob_plasma_parameter_correlation_matrix(nocalc=False, 
                                                           averaging='shot', 
                                                           average='avg', 
                                                           str_finding_method='watershed', 
                                                           quantity='correlation', 
                                                           plot_for_publication=True, 
                                                           colormap='seismic', 
                                                           plot_full=True, 
                                                           threshold_corr=False, 
                                                           linewidth=1, 
                                                           ticksize=3, 
                                                           charsize=9)
        

    if plot_figure == 13:
        calculate_blob_plasma_parameter_correlation_matrix(nocalc=False, 
                                                           averaging='shot', 
                                                           average='avg', 
                                                           str_finding_method='watershed', 
                                                           quantity='mutual_information', 
                                                           plot_for_publication=True, 
                                                           colormap='seismic', 
                                                           plot_full=True, 
                                                           threshold_corr=False, 
                                                           linewidth=1, 
                                                           ticksize=3, 
                                                           charsize=9,
                                                           plot_colorbar=False)
    if plot_figure == 14:
        calculate_blob_plasma_parameter_correlation_matrix(nocalc=False, 
                                                           averaging='shot', 
                                                           average='avg', 
                                                           str_finding_method='watershed', 
                                                           quantity='predictive_power', 
                                                           plot_for_publication=True, 
                                                           colormap='seismic', 
                                                           plot_full=True, 
                                                           threshold_corr=False, 
                                                           linewidth=1, 
                                                           ticksize=3, 
                                                           charsize=9,
                                                           plot_colorbar=False)
        
