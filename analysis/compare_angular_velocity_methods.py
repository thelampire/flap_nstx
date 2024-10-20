#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 23:52:22 2022

@author: mlampert
"""
from flap_nstx.gpi import calculate_nstx_gpi_angular_velocity
from flap_nstx.gpi import analyze_gpi_structures

import matplotlib.pyplot as plt
import numpy as np

def compare_angular_velocity_methods(exp_id=139901,
                                     time_range=[0.3245,0.3255],
                                     nocalc=[True,False,False],
                                     avg_or_max='max',
                                     ):
    result_ccf=calculate_nstx_gpi_angular_velocity(exp_id=exp_id,
                                                   time_range=time_range,
                                                   return_results=True,
                                                   new_scheme_return=True,
                                                   nocalc=nocalc[0],
                                                   plot=False,
                                                   pdf=False)

    result_contour=analyze_gpi_structures(exp_id=exp_id,
                                          time_range=time_range,
                                          nocalc=nocalc[1],
                                          str_finding_method='contour',
                                          test_structures=False,
                                          return_results=True,
                                          plot=False,
                                          pdf=False,
                                          ellipse_method='linalg',
                                          fit_shape='ellipse',
                                          prev_str_weighting='max_intensity')

    result_watershed=analyze_gpi_structures(exp_id=exp_id,
                                            time_range=time_range,
                                            plot=False,
                                            pdf=False,
                                            nocalc=nocalc[2],
                                            str_finding_method='watershed',
                                            test_structures=False,
                                            return_results=True,
                                            ellipse_method='linalg',
                                            fit_shape='ellipse',
                                            prev_str_weighting='max_intensity')
    import matplotlib
    matplotlib.use('qt5agg')
    fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))

    ax.plot(result_ccf['time'],
            result_ccf['data']['Angular velocity ccf log']['raw'],
            label='ccf')
    ax.plot(result_contour['Time'],
            result_contour['derived']['Angular velocity angle'][avg_or_max],
            label='con,ang')
    ax.plot(result_contour['Time'],
            result_contour['derived']['Angular velocity ALI'][avg_or_max],
            label='con,ALI')
    ax.plot(result_watershed['Time'],
            result_watershed['derived']['Angular velocity angle'][avg_or_max],
            label='wsh,ang')
    ax.plot(result_watershed['Time'],
            result_watershed['derived']['Angular velocity ALI'][avg_or_max],
            label='wsh,ALI')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('$\omega$ [rad/s]')
    plt.legend()
    plt.tight_layout(pad=0.1)

    fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))

    ax.plot(result_ccf['time'],
            result_ccf['data']['Expansion velocity ccf FLAP']['raw'],
            label='ccf')
    ax.plot(result_contour['Time'],
            result_contour['derived']['Expansion fraction area'][avg_or_max]-1,
            label='con,ang')
    ax.plot(result_contour['Time'],
            result_contour['derived']['Expansion fraction axes'][avg_or_max]-1,
            label='con,ALI')
    ax.plot(result_watershed['Time'],
            result_watershed['derived']['Expansion fraction area'][avg_or_max]-1,
            label='wsh,ang')
    ax.plot(result_watershed['Time'],
            result_watershed['derived']['Expansion fraction axes'][avg_or_max]-1,
            label='wsh,ALI')
    ax.set_ylim([-0.5,0.5])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('$f_{E}$ [rad/s]')
    plt.legend()
    plt.tight_layout(pad=0.1)

    fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))

    ax.plot(result_ccf['time'],
            result_ccf['data']['Velocity ccf FLAP radial']['raw'],
            label='ccf')

    ax.plot(result_contour['Time'],
            result_contour['derived']['Velocity radial position'][avg_or_max],
            label='con,pos')

    ax.plot(result_contour['Time'],
            result_contour['derived']['Velocity radial position'][avg_or_max],
            label='wsh,pos')

    ax.plot(result_contour['Time'],
            result_contour['derived']['Velocity radial COG'][avg_or_max],
            label='con,COG')

    ax.plot(result_contour['Time'],
            result_contour['derived']['Velocity radial COG'][avg_or_max],
            label='wsh,COG')

    ax.plot(result_contour['Time'],
            result_contour['derived']['Velocity radial centroid'][avg_or_max],
            label='con,cen')

    ax.plot(result_contour['Time'],
            result_contour['derived']['Velocity radial centroid'][avg_or_max],
            label='wsh,cen')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('$v_{rad}$ [m/s]')
    plt.legend()
    plt.tight_layout(pad=0.1)

    fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))


    ax.plot(result_contour['Time'],
            result_contour['data']['Angle'][avg_or_max],
            label='con,ang')
    ax.plot(result_contour['Time'],
            result_contour['data']['Angle of least inertia'][avg_or_max],
            label='con,ALI')
    ax.plot(result_watershed['Time'],
            result_watershed['data']['Angle'][avg_or_max],
            label='wsh,ang')
    ax.plot(result_watershed['Time'],
            result_watershed['data']['Angle of least inertia'][avg_or_max],
            label='wsh,ALI')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('angle [rad]')
    plt.legend()
    plt.tight_layout(pad=0.1)

    plt.show()