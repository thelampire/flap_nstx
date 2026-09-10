#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:50:26 2026

@author: mlampert
"""
class Hell(Exception):pass

import os
import copy


import flap
import flap_nstx

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
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

import string
abc=string.ascii_lowercase

wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/publication_figures/rsi_2022'

flap_nstx.tools.set_matplotlib_for_publication(labelsize=8.,
                                               linewidth=0.5,
                                               major_ticksize=2.,
                                               minor_ticksize=1.)

def plot_results_for_pop_2026(plot_figure=2,
                              save_data_into_txt=False,
                              plot_all=False,
                              nocalc=False):

    if plot_all:
        plot_figure=-1
        for i in range(15):
            if i not in [0,1,3]:
                plot_results_for_pop_2026(plot_figure=i,
                                          save_data_into_txt=save_data_into_txt)

    """
    GPI plot
    """
    if plot_figure == 1:
        raise ValueError('GPI plot, no need to create figure.')
    
    if plot_figure == 2:    #these are put next to each other manually
        analyze_gpi_structures(exp_id=141998,
                               time_range=[0.215, 0.235],
                               plot_time_range=[0.220895,0.221],
                               ignore_side_structures=True,
                               pdf=True,
                               plot_example_structure_frames=1,
                               nocalc=True,
                               plot=False,
                               plot_for_publication=True,
                               min_structure_lifetime=0,
                               plot_ncol=1, plot_nframe=5,
                               plot_separatrix=True,
                               
                               )
        
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.527, 0.547],
                               plot_time_range=[0.532430,0.547],
                               ignore_side_structures=True,
                               pdf=True,
                               plot_example_structure_frames=1,
                               nocalc=True,
                               plot=False,
                               plot_for_publication=True,
                               min_structure_lifetime=0,
                               plot_ncol=1, plot_nframe=5,
                               plot_separatrix=True,
                               
                               )
    
    if plot_figure == 3:
        
        analyze_gpi_structures(exp_id=141998,
                               time_range=[0.215,0.235],
                               plot_time_range=[0.22012,0.221],
                               ignore_side_structures=True,
                               pdf=True,
                               plot_example_structure_frames=1,
                               nocalc=True,
                               plot=False,
                               plot_for_publication=True,
                               min_structure_lifetime=2,
                               plot_ncol=3, plot_nframe=9,
                               plot_separatrix=True
                               )
        
    if plot_figure == 4:
        options={}
        options['keys_to_plot']={}
        options['keys_to_plot']['Centroid radial']={'label':'R',
                                                    'unit':'m',
                                                    'range':[1.37,1.6],
                                                    'multiplier':1}
        
        options['keys_to_plot']['Normalized flux coordinate']={'label':r'$\psi_{norm}$',
                                                    'unit':None,
                                                    'range':[0.6,2.0],#[1.37,1.6],
                                                    'multiplier':1,
                                                    'hline_at':1.0,}
                         
        options['keys_to_plot']['Centroid poloidal']={'label':'z',
                                                      'unit':'m',
                                                      'range':[0.05,0.35],
                                                      'multiplier':1}
    
                                 
        options['keys_to_plot']['Poloidal angle']={'label':r'$\theta$',
                                                   'unit':'rad',
                                                   'range':[0.,1.],#[0.05,0.35],
                                                   'multiplier':1}        
        
        # options['keys_to_plot']['Velocity radial centroid']={'label':'$v_{rad}$',
        #                                                      'unit':'km/s',
        #                                                      'range':[-5,5],
        #                                                      'multiplier':1e-3}
        
        # options['keys_to_plot']['Velocity poloidal centroid']={'label':'$v_{pol}$',
        #                                                        'unit':'km/s',
        #                                                        'range':[-10,10],
        #                                                        'multiplier':1e-3}
        
        # options['keys_to_plot']['Angle fit']={'label':r'$\phi$',
        #                                       'unit':'rad',
        #                                       'range':[-2,2],
        #                                       'multiplier':1}
        
        # options['keys_to_plot']['Angular velocity angle fit']={'label':r'$\omega$',
        #                                                        'unit':'krad/s',
        #                                                        'range':[-100,100],
        #                                                        'multiplier':1e-3}
        nplot=len(list(options['keys_to_plot'].keys()))
        fig, axes=plt.subplots(nplot,2, figsize=[8.5/2.54,2*nplot/2.54])
        
        options['fig_axes']=(fig,axes[:,0])
        options['hide_y_labels']=False
        options['subplot_labels']=abc[0:nplot]
        options['title']='L-mode #141998'
        pdf_pages=PdfPages(wd+'/plots/fig4_lh_single_shot.pdf')
        
        window_length=1e-3
        
        analyze_gpi_structures(exp_id=141998,
                               time_range=[0.215,0.235],
                               plot_time_range=[0.222,0.222+window_length],
                               ignore_side_structures=True,
                               pdf=False, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=True,
                               min_structure_lifetime=10,
                               nocalc=True,
                               plot_scatter=True,
                               plot_tracking=True,
                               plot_example_results=True,
                               plot_example_results_options=options
                               )
        
        options['hide_y_labels']=True
        options['fig_axes']=(fig,axes[:,1])
        options['subplot_labels']=abc[nplot:2*nplot]
        options['title']='H-mode #141319'
        
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.527, 0.547],
                               plot_time_range=[0.530, 0.530+window_length],
                               ignore_side_structures=True,
                               pdf=False, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=True,
                               min_structure_lifetime=10,
                               nocalc=True,
                               plot_scatter=True,
                               plot_tracking=True,
                               plot_example_results=True,
                               plot_example_results_options=options
                               )
        
        fig.tight_layout(pad=0.1)
        pdf_pages.savefig()
        pdf_pages.close()
        
    if plot_figure == 5:
        
        pdf_pages=PdfPages(wd+'/plots/fig5_lh_single_shot_vel.pdf')
        
        options={}
        options['keys_to_plot']={} 
        
        options['keys_to_plot']['Velocity radial centroid']={'label':'$v_{rad}$',
                                                             'unit':'km/s',
                                                             'range':[-5,5],
                                                             'multiplier':1e-3,
                                                             'hline_at':0.,}
        
        options['keys_to_plot']['Normalized flux coordinate velocity']={'label':'$\\partial_{t} \\psi_{N}$',
                                                             'unit':'1/ms',
                                                             'range':[-15,15],
                                                             'multiplier':1e-3,
                                                             'hline_at':0.}
        
       
        options['keys_to_plot']['Velocity poloidal centroid']={'label':'$v_{pol}$',
                                                               'unit':'km/s',
                                                               'range':[-10,10],
                                                               'multiplier':1e-3,
                                                               'hline_at':0.}
        
        options['keys_to_plot']['Poloidal angular velocity']={'label':'$\\partial_{t} \\theta$',
                                                               'unit':'krad/s',
                                                               'range':[-20,20],
                                                               'multiplier':1e-3,
                                                               'hline_at':0.}
        
        nplot=len(list(options['keys_to_plot'].keys()))
        fig, axes=plt.subplots(nplot,2, figsize=[8.5/2.54,2*nplot/2.54])
        options['markersize']=0.1
        options['fig_axes']=(fig,axes[:,0])
        options['hide_y_labels']=False
        options['subplot_labels']=abc[0:nplot]
        options['title']='L-mode #141998'
        
        window_length=1e-3
        
        analyze_gpi_structures(exp_id=141998,
                               time_range=[0.215,0.235],
                               plot_time_range=[0.222,0.222+window_length],
                               ignore_side_structures=True,
                               pdf=False, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=True,
                               min_structure_lifetime=10,
                               nocalc=True,
                               plot_scatter=True,
                               plot_tracking=True,
                               plot_example_results=True,
                               plot_example_results_options=options
                               )
        
        options['hide_y_labels']=True
        options['fig_axes']=(fig,axes[:,1])
        options['subplot_labels']=abc[nplot:2*nplot]
        options['title']='H-mode #141319'
        
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.527, 0.547],
                               plot_time_range=[0.530, 0.530+window_length],
                               ignore_side_structures=True,
                               pdf=False, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=True,
                               min_structure_lifetime=10,
                               nocalc=True,
                               plot_scatter=True,
                               plot_tracking=True,
                               plot_example_results=True,
                               plot_example_results_options=options
                               )
        
        fig.tight_layout(pad=0.1)
        pdf_pages.savefig()
        pdf_pages.close()
        
    if plot_figure == 6:
        
        pdf_pages=PdfPages(wd+'/plots/fig6_lh_single_shot_rotation.pdf')
        
        options={}
        options['keys_to_plot']={}
        
        options['keys_to_plot']['Angle fit']={'label':r'$\phi$',
                                              'unit':'rad',
                                              'range':[-np.pi/2,np.pi/2],
                                              'multiplier':1,
                                              'hline_at':0.}

        options['keys_to_plot']['Angle ALI']={'label':r'$\phi_{ALI}$',
                                              'unit':'rad',
                                              'range':[-np.pi/2,np.pi/2],
                                              'multiplier':1,
                                              'hline_at':0.}
        
        options['keys_to_plot']['Angular velocity angle fit']={'label':r'$\omega$',
                                                               'unit':'krad/s',
                                                               'range':[-100,100],
                                                               'multiplier':1e-3,
                                                               'hline_at':0.}

        
        options['keys_to_plot']['Angular velocity ALI']={'label':r'$\omega$',
                                                               'unit':'krad/s',
                                                               'range':[-100,100],
                                                               'multiplier':1e-3,
                                                               'hline_at':0.}        
        
        
        nplot=len(list(options['keys_to_plot'].keys()))
        fig, axes=plt.subplots(nplot,2, figsize=[8.5/2.54,2*nplot/2.54])
        
        options['fig_axes']=(fig,axes[:,0])
        options['hide_y_labels']=False
        options['subplot_labels']=abc[0:nplot]
        options['title']='L-mode #141998'
        
        window_length=1e-3
        
        analyze_gpi_structures(exp_id=141998,
                               time_range=[0.215,0.235],
                               plot_time_range=[0.222,0.222+window_length],
                               ignore_side_structures=True,
                               pdf=False, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=True,
                               min_structure_lifetime=10,
                               nocalc=True,
                               plot_scatter=True,
                               plot_tracking=True,
                               plot_example_results=True,
                               plot_example_results_options=options
                               )
        
        options['hide_y_labels']=True
        options['fig_axes']=(fig,axes[:,1])
        options['subplot_labels']=abc[nplot:2*nplot]
        options['title']='H-mode #141319'
        
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.527, 0.547],
                               plot_time_range=[0.530, 0.530+window_length],
                               ignore_side_structures=True,
                               pdf=False, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=True,
                               min_structure_lifetime=10,
                               nocalc=True,
                               plot_scatter=True,
                               plot_tracking=True,
                               plot_example_results=True,
                               plot_example_results_options=options
                               )
        
        fig.tight_layout(pad=0.1)
        pdf_pages.savefig()
        pdf_pages.close()

    if plot_figure == 7:
        
        pdf_pages=PdfPages(wd+'/plots/fig7_lh_single_shot_sizes.pdf')
        
        options={}
        options['keys_to_plot']={}
        
        options['keys_to_plot']['Size radial fit']={'label':r'$d_{rad}$',
                                                    'unit':'mm',
                                                    'range':None,
                                                    'multiplier':1e3,
                                                    }

        options['keys_to_plot']['Size poloidal fit']={'label':r'$d_{pol}$',
                                                      'unit':'mm',
                                                      'range':None,
                                                      'multiplier':1e3,
                                                      }
        
        options['keys_to_plot']['Elongation fit']={'label':'Elongation',
                                                   'unit':'',
                                                   'range':None,
                                                   'multiplier':1,}
   
        
        
        nplot=len(list(options['keys_to_plot'].keys()))
        fig, axes=plt.subplots(nplot,2, figsize=[8.5/2.54,2*nplot/2.54])
        
        options['fig_axes']=(fig,axes[:,0])
        options['hide_y_labels']=False
        options['subplot_labels']=abc[0:nplot]
        options['title']='L-mode #141998'
        
        window_length=1e-3
        
        analyze_gpi_structures(exp_id=141998,
                               time_range=[0.215,0.235],
                               plot_time_range=[0.222,0.222+window_length],
                               ignore_side_structures=True,
                               pdf=False, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=True,
                               min_structure_lifetime=10,
                               nocalc=True,
                               plot_scatter=True,
                               plot_tracking=True,
                               plot_example_results=True,
                               plot_example_results_options=options
                               )
        
        options['hide_y_labels']=True
        options['fig_axes']=(fig,axes[:,1])
        options['subplot_labels']=abc[nplot:2*nplot]
        options['title']='H-mode #141319'
        
        analyze_gpi_structures(exp_id=141319,
                               time_range=[0.527, 0.547],
                               plot_time_range=[0.530, 0.530+window_length],
                               ignore_side_structures=True,
                               pdf=False, plot=True,
                               plot_str_by_str=True,
                               plot_for_publication=True,
                               min_structure_lifetime=10,
                               nocalc=True,
                               plot_scatter=True,
                               plot_tracking=True,
                               plot_example_results=True,
                               plot_example_results_options=options
                               )
        
        fig.tight_layout(pad=0.1)
        pdf_pages.savefig()
        pdf_pages.close()

    if plot_figure == 8:

        options={}
        options['keys_to_plot']={}

        options['keys_to_plot']['Centroid radial']={'label':'R',
                                                    'unit':'m',
                                                    'range':[1.37,1.60],
                                                    'multiplier':1}
        
        options['keys_to_plot']['Normalized flux coordinate']={'label':r'$\psi_{norm}$',
                                                               'unit':None,
                                                               'range':[0.5,1.5],
                                                               'multiplier':1,
                                                               'vline_at':1.0}
        
        options['keys_to_plot']['Centroid poloidal']={'label':'z',
                                                      'unit':'m',
                                                      'range':[0.05,0.35],
                                                      'multiplier':1}


        options['keys_to_plot']['Poloidal angle']={'label':r'$\theta$',
                                                   'unit':'rad',
                                                   'range':[-0.4,0.9],
                                                   'multiplier':1}

        nplot=len(list(options['keys_to_plot'].keys()))
        fig, axes=plt.subplots(nplot,1, figsize=[8.5/2.54,3.5*nplot/2.54])

        options['fig_axes']=(fig,axes)
        options['nrow']=nplot
        options['ncol']=1
        options['subplot_labels']=abc[0:nplot]
        options['pdf_filename']=wd+'/plots/fig8_lh_position_histograms.pdf'

        calculate_blob_parameter_histograms(analyze_lh_diff=True,
                                            nocalc=True,
                                            pdf=True,
                                            plot=True,
                                            plot_for_publication=True,
                                            min_structure_lifetime=10,
                                            str_finding_method='watershed',
                                            save_data_for_publication=save_data_into_txt,
                                            options=options,
                                            )

    if plot_figure == 9:

        options={}
        options['keys_to_plot']={}

        options['keys_to_plot']['Velocity radial centroid']={'label':'$v_{rad}$',
                                                             'unit':'km/s',
                                                             'range':[-5,7],
                                                             'multiplier':1e-3,
                                                             'vline_at':0.}

        options['keys_to_plot']['Normalized flux coordinate velocity']={'label':'$\\partial_{t} \\psi_{N}$',
                                                                        'unit':'1/ms',
                                                                        'range':[-20,20],
                                                                        'multiplier':1e-3,
                                                                        'vline_at':0.}

        options['keys_to_plot']['Velocity poloidal centroid']={'label':'$v_{pol}$',
                                                               'unit':'km/s',
                                                               'range':[-10,10],
                                                               'multiplier':1e-3,
                                                               'vline_at':0.}

        options['keys_to_plot']['Poloidal angular velocity']={'label':'$\\partial_{t} \\theta$',
                                                              'unit':'krad/s',
                                                              'range':[-20,20],
                                                              'multiplier':1e-3,
                                                              'vline_at':0.}

        nplot=len(list(options['keys_to_plot'].keys()))
        fig, axes=plt.subplots(nplot,1, figsize=[8.5/2.54,3.5*nplot/2.54])

        options['fig_axes']=(fig,axes)
        options['nrow']=nplot
        options['ncol']=1
        options['subplot_labels']=abc[0:nplot]
        options['pdf_filename']=wd+'/plots/fig9_lh_velocity_histograms.pdf'

        calculate_blob_parameter_histograms(analyze_lh_diff=True,
                                            nocalc=True,
                                            pdf=True,
                                            plot=True,
                                            plot_for_publication=True,
                                            min_structure_lifetime=10,
                                            str_finding_method='watershed',
                                            save_data_for_publication=save_data_into_txt,
                                            options=options,
                                            )

    if plot_figure == 10:

        options={}
        options['keys_to_plot']={}

        options['keys_to_plot']['Angle fit']={'label':r'$\phi$',
                                              'unit':'rad',
                                              'range':[-np.pi/2,np.pi/2],
                                              'multiplier':1,
                                              'vline_at':np.pi/2}

        options['keys_to_plot']['Angle ALI']={'label':r'$\phi_{ALI}$',
                                              'unit':'rad',
                                              'range':[-np.pi/2,np.pi/2],
                                              'multiplier':1,
                                              'vline_at':np.pi/2}

        options['keys_to_plot']['Angular velocity angle fit']={'label':r'$\omega$',
                                                               'unit':'krad/s',
                                                               'range':[-250,250],
                                                               'multiplier':1e-3,
                                                               'vline_at':0.}

        options['keys_to_plot']['Angular velocity ALI']={'label':r'$\omega_{ALI}$',
                                                               'unit':'krad/s',
                                                               'range':[-250,250],
                                                               'multiplier':1e-3,
                                                               'vline_at':0.}

        nplot=len(list(options['keys_to_plot'].keys()))
        fig, axes=plt.subplots(nplot,1, figsize=[8.5/2.54,3.5*nplot/2.54])

        options['fig_axes']=(fig,axes)
        options['nrow']=nplot
        options['ncol']=1
        options['subplot_labels']=abc[0:nplot]
        options['pdf_filename']=wd+'/plots/fig10_lh_rotation_histograms.pdf'

        calculate_blob_parameter_histograms(analyze_lh_diff=True,
                                            nocalc=True,
                                            pdf=True,
                                            plot=True,
                                            plot_for_publication=True,
                                            min_structure_lifetime=10,
                                            str_finding_method='watershed',
                                            save_data_for_publication=save_data_into_txt,
                                            options=options,
                                            )
        
    if plot_figure == 11: #Not yet finished
        
        options={}
        options['keys_to_plot']={}

        options['keys_to_plot']['Size radial fit']={'label':r'$d_{rad}$',
                                                    'unit':'mm',
                                                    'range':[0,100],
                                                    'multiplier':1e3,
                                                    }
        
        options['keys_to_plot']['Size radial fit diff']={'label':r'$\partial_t d_{rad}$',
                                                    'unit':'km/s',
                                                    'range':[-10,10],
                                                    'multiplier':1e-3,
                                                    }

        options['keys_to_plot']['Size poloidal fit']={'label':r'$d_{pol}$',
                                                      'unit':'mm',
                                                      'range':[0,150],
                                                      'multiplier':1e3,
                                                      }
        
        options['keys_to_plot']['Size poloidal fit diff']={'label':r'$\partial_t d_{pol}$',
                                                      'unit':'km/s',
                                                      'range':[-15,15],
                                                      'multiplier':1e-3,
                                                      }
        
        options['keys_to_plot']['Elongation fit']={'label':'$\kappa$',
                                                   'unit':'',
                                                   'range':None,
                                                   'multiplier':1,}
        
        options['keys_to_plot']['Elongation fit diff']={'label':'$\partial_t \kappa$',
                                                   'unit':'$1/ms$',
                                                   'range':None,
                                                   'multiplier':1e-3,}
        
        



        


        nplot=len(list(options['keys_to_plot'].keys()))
        ncol=2
        fig, axes = plt.subplots(int(nplot/2),ncol, figsize=[8.5/2.54,3.5*nplot/ncol/2.54])

        options['fig_axes']=(fig,axes)
        options['nrow']=nplot
        options['ncol']=ncol
        options['subplot_labels']=abc[0:nplot]
        options['pdf_filename']=wd+'/plots/fig11_lh_filament_size_histograms.pdf'

        calculate_blob_parameter_histograms(analyze_lh_diff=True,
                                            nocalc=True,
                                            pdf=True,
                                            plot=True,
                                            plot_for_publication=True,
                                            min_structure_lifetime=10,
                                            str_finding_method='watershed',
                                            save_data_for_publication=save_data_into_txt,
                                            options=options,
                                            )
    if plot_figure == 12: #Pearson matrix
        calculate_blob_blob_parameter_correlation_matrix(threshold_corr=True,
                                                         nocalc=True,
                                                         plot_all_matrices_for_lh=True,
                                                         calc_mean_distribution=False,
                                                         plot_interesting_only=True,
                                                         analyze_lh_difference=True,
                                                         min_structure_lifetime=10
                                                         )
    
    if plot_figure == 13:   #2D histrograms, maybe a few figures with them
        pass
    
    if plot_figure == 14:   #Conditional average plots, the interesting ones.
        pass
    
    if plot_figure == 15:   #Correlation matrix between filament parameter and blob parameters
        pass
    
    if plot_figure == 16:   #Trends between plasma parameter and filament parameter along with correlations and R2 values
        pass