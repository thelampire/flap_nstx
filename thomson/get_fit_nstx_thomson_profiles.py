#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:45:24 2021

@author: mlampert
"""

import os
import copy
import time as time_mod
import pickle

#FLAP imports and settings
import flap
import flap_nstx
import flap_mdsplus

from flap_nstx.tools import tanh_function, mtanh_function
flap_nstx.register()
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

#Scientific imports
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

import matplotlib.pyplot as plt
#Other necessary imports

def get_fit_nstx_thomson_profiles(exp_id=None,                                      #Shot number
                                  pressure=False,                                   #Return the pressure profile paramenters
                                  temperature=False,                                #Return the temperature profile parameters
                                  density=False,                                    #Return the density profile parameters

                                  spline_data=False,                                #Calculate the results from the spline data (no error is going to be taken into account)

                                  modified_tanh=False,
                                  average_profiles=None,

                                  device_coordinates=False,                          #Calculate the results as a function of device coordinates
                                  radial_range=None,                                #Radial range of the pedestal (only works when the device coorinates is set)

                                  flux_coordinates=False,                           #Calculate the results in flux coordinates
                                  flux_range=None,                                  #The normalaized flux coordinates range for returning the results

                                  outboard_only=True,                           #Use only the outboard profile
                                  force_overlap=False,                          #Shifts the inboard and outboard profiles of the TS to match

                                  max_iter=1200,                                #Maximum iteration for the shifting
                                  max_err=1e-5,                                 #difference between iteration steps to be reached

                                  output_name=None,
                                  plot_time_vec=None,
                                  pdf_object=None,

                                  test=False,
                                  test_time_vec=False,
                                  nocalc=False,
                                  ):
    """

    Returns a dataobject which has the largest corresponding gradient based on the tanh fit.

    Fitting is based on publication https://aip.scitation.org/doi/pdf/10.1063/1.4961554
    The linear background is not usitlized, instead of the mtanh, only tanh is used.
    """
    if force_overlap:
        outboard_only=False

    if force_overlap and outboard_only:
        raise ValueError('force_overlap and outboard_only cannot be set at the same time_vec.')

    if ((device_coordinates and flux_range is not None) or
        (flux_coordinates and radial_range is not None)):
        raise ValueError('When flux or device coordinates are set, only flux or radial range can be set! Returning...')
    if test_time_vec:
        start_time_vec=time_mod.time_vec()

    comment=''
    if density:         comment+='_ne'
    if temperature:     comment+='_te'
    if pressure:        comment+='_pe'
    if spline_data:     comment+="_spl"
    if modified_tanh:   comment+='_mtanh'
    if device_coordinates:  comment+='_dev'
    if flux_coordinates:    comment+='_flux'

    pickle_filename=wd+'/processed_data/TS_'+str(exp_id)+comment+'.pickle'

    if nocalc and os.path.exists(pickle_filename):
        thomson_profiles=pickle.load(open(pickle_filename,'rb'))
    else:
        flap_options={'pressure':pressure,
                      'temperature':temperature,
                      'density':density,
                      'spline_data':False,
                      'add_flux_coordinates':True,
                      'force_mdsplus':False}

        d=flap.get_data('NSTX_THOMSON',
                        exp_id=exp_id,
                        name='',
                        object_name='THOMSON_DATA',
                        options=flap_options)

        time_vec=d.coordinate('Time')[0][0,:]
        if test_time_vec:
            print('Fit 1st in ',time_mod.time_vec()-start_time_vec)
            start_time_vec=time_mod.time_vec()

        if flux_coordinates:
            r_coord_name='Flux r'
            if flux_range is None:
                flux_range=[0.,1.1]

        if device_coordinates or not flux_coordinates:
            r_coord_name='Device R'
            if radial_range is None:
                radial_range=[np.min(d.coordinate(r_coord_name)[0]),
                              np.max(d.coordinate(r_coord_name)[0])]

        # print(time_vec.shape,d.data.shape, d.coordinate('Flux r')[0].shape)
        thomson_profiles={'time_vec':time_vec,
                          'Data':d.data,
                          'Device R':d.coordinate('Device R')[0],
                          'Flux r':d.coordinate('Flux r')[0],
                          
                          'Fit parameters':np.zeros([time_vec.shape[0],5]),
                          'Fit parameter errors':np.zeros([time_vec.shape[0],5]),
                          'Fit parameter covariance':np.zeros([time_vec.shape[0],5,5]),
                          
                          'a':np.zeros(time_vec.shape),
                          'Height':np.zeros(time_vec.shape),
                          'Width':np.zeros(time_vec.shape),
                          'Global gradient':np.zeros(time_vec.shape),
                          'Position':np.zeros(time_vec.shape),
                          'Position r':np.zeros(time_vec.shape),
                          'SOL offset':np.zeros(time_vec.shape),
                          'Max gradient':np.zeros(time_vec.shape),
                          'Value at max':np.zeros(time_vec.shape),
                          'SOL avg':np.zeros(time_vec.shape),
                          
                          'Error':{'Data':d.error,
                                   'Height':np.zeros(time_vec.shape),
                                   'SOL offset':np.zeros(time_vec.shape),
                                   'Position':np.zeros(time_vec.shape),
                                   'Position r':np.zeros(time_vec.shape),
                                   'Width':np.zeros(time_vec.shape),
                                   'Global gradient':np.zeros(time_vec.shape),
                                   'Max gradient':np.zeros(time_vec.shape),
                                   'Value at max':np.zeros(time_vec.shape),
                                   'SOL avg':np.zeros(time_vec.shape),
                                  },
                         }
        if modified_tanh:
            thomson_profiles['Slope']=np.zeros(time_vec.shape)
            thomson_profiles['Error']['Slope']=np.zeros(time_vec.shape)

        if test:
            plt.figure()
        if flux_range is not None:
            x_range=flux_range
        if radial_range is not None:
            x_range=radial_range

        if not modified_tanh:
            tanh_fit_function=tanh_function
        else:
            tanh_fit_function=mtanh_function

        if test_time_vec:
            print('Fit 2nd in ',time_mod.time_vec()-start_time_vec)
            start_time_vec=time_mod.time_vec()
            
        rmaxis=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\RMAXIS',
                             exp_id=exp_id,
                             object_name='RMAXIS')
        
        d2=flap.get_data('NSTX_THOMSON',
                        exp_id=exp_id,
                        name='',
                        object_name='THOMSON_DATA',
                        options={'pressure':False,
                                 'temperature':True,
                                 'density':False,
                                 'spline_data':False,
                                 'add_flux_coordinates':True,
                                 'force_mdsplus':False})
        if test_time_vec:
            print('Fit 3rd in ',time_mod.time_vec()-start_time_vec)
            start_time_vec=time_mod.time_vec()

        for i_time_vec, cur_time in enumerate(time_vec):
            if r_coord_name =='Flux r':
                x_data=d.coordinate('Flux r')[0][:,i_time_vec]
                y_data=d.data[:,i_time_vec]
                y_data_error=d.error[:,i_time_vec]

                ind_time_vec_efit=np.argmin(np.abs(rmaxis.coordinate('Time')[0]-time_vec[i_time_vec]))
                r_maxis_cur=rmaxis.data[ind_time_vec_efit]
                ind_maxis=np.argmin(np.abs(d.coordinate('Device R')[0][:,i_time_vec]-r_maxis_cur))

                if outboard_only or device_coordinates:
                    if not outboard_only and device_coordinates and i_time_vec == 0:
                        print('outboard_only=False and device_coordinates=True are not compatible')
                        print('Using only outboard TS points.')
                    x_data=x_data[ind_maxis:]
                    y_data=y_data[ind_maxis:]
                    y_data_error=d.error[ind_maxis:,i_time_vec]
                else:
                    x_data_in=x_data[:ind_maxis]
                    x_data_out=x_data[ind_maxis:]
                    y_data_in=y_data[:ind_maxis]
                    y_data_out=y_data[ind_maxis:]

                    if not temperature:                                                 #Only temperature is a flux function on NSTX
                        temp_data_in=d2.data[:,i_time_vec][:ind_maxis]
                        temp_data_out=d2.data[:,i_time_vec][ind_maxis:]
                    else:
                        temp_data_in=y_data[:ind_maxis]
                        temp_data_out=y_data[ind_maxis:]

                    psi_shift_0=max(x_data_out[2:]-x_data_out[:-2])
                    psi_shift_1=-psi_shift_0
                    for ind_iter in range(max_iter):

                        s_in_neg=UnivariateSpline(x_data_in-psi_shift_0, temp_data_in)
                        s_out_neg=UnivariateSpline(x_data_out+psi_shift_0, temp_data_out)

                        integral_difference_neg = s_out_neg.integral(0, np.inf) - s_in_neg.integral(0, np.inf)

                        s_in_pos=UnivariateSpline(x_data_in-psi_shift_1, temp_data_in)
                        s_out_pos=UnivariateSpline(x_data_out+psi_shift_1, temp_data_out)

                        integral_difference_pos = s_out_pos.integral(0, np.inf) - s_in_pos.integral(0, np.inf)

                        if integral_difference_pos > integral_difference_neg:
                            psi_shift=None

                    x_data=np.concatenate([x_data_in,x_data_out])
                    y_data=np.concatenate([y_data_in,y_data_out])
                    sort_ind=np.argsort(x_data)
                    x_data=x_data[sort_ind]
                    y_data=y_data[sort_ind]
            else:
                #By default it's outboard only, the full profile is not tanh
                if average_profiles is not None and i_time_vec+1 > average_profiles:
                    ind_max=np.argmax(d.data[:,i_time_vec])
                    x_data=d.coordinate('Device R')[0][ind_max:,i_time_vec]
                    y_data=(np.sum(d.data[ind_max:,i_time_vec-average_profiles+1:i_time_vec+1] *
                                   d.error[ind_max:,i_time_vec-average_profiles+1:i_time_vec+1]) /
                            average_profiles/np.sum(d.error[ind_max:,i_time_vec-average_profiles+1:i_time_vec+1],axis=1)
                            )
                    y_data_error=np.sqrt(np.sum(d.error[ind_max:,i_time_vec-average_profiles+1:i_time_vec+1]**2,axis=1))/np.sqrt(average_profiles)
                else:
                    ind_max=np.argmax(d.data[:,i_time_vec])
                    x_data=d.coordinate('Device R')[0][ind_max:,i_time_vec]
                    y_data=d.data[ind_max:,i_time_vec]
                    y_data_error=d.error[ind_max:,i_time_vec]

            if np.sum(np.isinf(x_data)) != 0:
                continue

            #Further adjustment based on the set x_range
            ind_coord=np.where(np.logical_and(x_data > x_range[0],
                                              x_data <= x_range[1]))
            x_data=x_data[ind_coord]
            y_data=y_data[ind_coord]
            y_data_error=y_data_error[ind_coord]

            try:
                if not modified_tanh:
                    p0=[y_data[0],                                      #b_height
                        y_data[-1],                                     #b_sol
                        (x_data[0]+x_data[-1])/2,                       #b_pos
                        np.abs((x_data[-1]-x_data[0])/2),                      #b_width
                        #(y_data[0]-y_data[-1])/(x_data[0]-x_data[-1]), #b_slope this is supposed to be some kind of linear modification to the
                                                                        #tanh function called mtanh. It messes up the fitting quite a bit and it's not useful at all.
                        ]

                else:
                    p0=[y_data[0],                                      #b_height
                        y_data[-1],                                     #b_sol
                        (x_data[0]+x_data[-1])/2,                       #b_pos
                        np.abs((x_data[-1]-x_data[0])/2),                  #b_width
                        (y_data[0]-y_data[-1])/(x_data[0]-x_data[-1]),  #b_slope this is supposed to be some kind of linear modification to the
                                                                        #tanh function called mtanh. It messes up the fitting quite a bit and it's not useful at all.

                        ]
            except:
                print('Missing TS data for shot '+str(exp_id)+', time_vec: '+ str(time_vec[i_time_vec]))

            try:
                popt, pcov = curve_fit(tanh_fit_function,
                                       x_data,
                                       y_data,
                                       sigma=y_data_error,
                                       p0=p0)
                perr = np.sqrt(np.diag(pcov))
                successful_fitting=True
            except:
                if modified_tanh:
                    popt=[np.nan,np.nan,np.nan,np.nan,np.nan]
                    perr=[np.nan,np.nan,np.nan,np.nan,np.nan]
                    pcov=np.zeros([5,5])
                    pcov[:,:]=np.nan
                else:
                    popt=[np.nan,np.nan,np.nan,np.nan]
                    perr=[np.nan,np.nan,np.nan,np.nan]
                    pcov=np.zeros([4,4])
                    pcov[:,:]=np.nan
                    
                successful_fitting=False
            if test or (plot_time_vec is not None and i_time_vec==np.argmin(np.abs(plot_time_vec-time_vec))):
                plt.cla()
                if successful_fitting:
                    color='tab:blue'
                else:
                    color='red'
                plt.scatter(x_data,
                            y_data,
                            color=color)
                plt.errorbar(x_data,
                             y_data,
                             yerr=y_data_error,
                             marker='o',
                             color=color,
                             ls='')
                plt.plot(x_data,tanh_fit_function(x_data,*popt), color=color)

                if flux_coordinates:
                    xlabel='PSI_norm'
                else:
                    xlabel='Device R [m]'

                if temperature:
                    profile_string='temperature'
                    ylabel='Temperature [keV]'

                elif density:
                    profile_string='density'
                    ylabel='Density [1/m3]'
                elif pressure:
                    profile_string='pressure'
                    ylabel='Pressure [kPa]'

                time_vec_string=' @ '+str(time_vec[i_time_vec])
                plt.title('Fit '+profile_string+' profile of '+str(exp_id)+time_vec_string)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)


                if pdf_object is not None:
                    pdf_object.savefig()
            else:
                pass

            if modified_tanh:
                thomson_profiles['Fit parameters'][i_time_vec,:]=popt
                thomson_profiles['Fit parameter errors'][i_time_vec,:]=perr
                thomson_profiles['Fit parameter covariance'][i_time_vec,:,:]=pcov
            else:
                thomson_profiles['Fit parameters'][i_time_vec,0:4]=popt
                thomson_profiles['Fit parameter errors'][i_time_vec,0:4]=perr
                thomson_profiles['Fit parameter covariance'][i_time_vec,0:4,0:4]=pcov

            thomson_profiles['Height'][i_time_vec]=popt[0]
            thomson_profiles['SOL offset'][i_time_vec]=popt[1]
            thomson_profiles['Position'][i_time_vec]=popt[2]

            try:
            #if True:
                thomson_profiles['Position r'][i_time_vec]=np.interp(popt[2],
                                                                     d.coordinate('Flux r')[0][np.argmin(d.coordinate('Flux r')[0][:,i_time_vec]):,i_time_vec],
                                                                     d.coordinate('Device R')[0][np.argmin(d.coordinate('Flux r')[0][:,i_time_vec]):,i_time_vec])
            except:
                print('Interpolation failed.')
                thomson_profiles['Position r'][i_time_vec]=np.nan

            thomson_profiles['Width'][i_time_vec]=popt[3]
            if r_coord_name == 'Flux r':
                ind_max=np.argmin(d.coordinate('Flux r')[0][:,i_time_vec])
                ind_sol=np.where(d.coordinate('Flux r')[0][ind_max:,i_time_vec] > 1.0)
                
                
                
            else:
                
                R_separatrix=flap.get_data('NSTX_MDSPlus',
                                           name='\EFIT02::\RMIDOUT',
                                           exp_id=exp_id,
                                           ).slice_data(slicing={'Time':cur_time}).data-0.02
                
                
                ind_max=np.argmax(d.data[:,i_time_vec])
                
                if temperature:
                    sol_limit=0.05
                if density:
                    sol_limit=5e18
                if pressure:
                    sol_limit=5e18*0.05*11606*1e3
                    
                ind_sol=np.where(np.logical_and(d.coordinate('Device R')[0][ind_max:,i_time_vec] > R_separatrix, 
                                                d.data[ind_max:,i_time_vec] < sol_limit))
                
            thomson_profiles['SOL avg'][i_time_vec]=np.mean((d.data[ind_max:,i_time_vec])[ind_sol])
            # thomson_profiles['Error']['SOL avg'][i_time_vec]=np.sqrt(np.var((d.data[ind_max:,i_time_vec])[ind_sol]))
            thomson_profiles['Error']['SOL avg'][i_time_vec]=np.sqrt(np.sum(d.error[ind_max:,i_time_vec][ind_sol]**2))/len(ind_sol)
            
            if modified_tanh:
                thomson_profiles['Slope'][i_time_vec]=popt[4]

            thomson_profiles['Error']['Height'][i_time_vec]=perr[0]
            thomson_profiles['Error']['SOL offset'][i_time_vec]=perr[1]
            thomson_profiles['Error']['Position'][i_time_vec]=perr[2]
            thomson_profiles['Error']['Width'][i_time_vec]=perr[3]
            thomson_profiles['Error']['Value at max'][i_time_vec]=(perr[0]+perr[1])/2
            thomson_profiles['Error']['Global gradient'][i_time_vec]=(perr[0]/popt[3]+
                                                                      perr[1]/popt[3]+
                                                                      np.abs((-popt[1]+popt[0])/popt[3]**2)*perr[3])
            thomson_profiles['Error']['Max gradient'][i_time_vec]=(np.abs(1/(4*popt[3])*perr[1])+
                                                                   np.abs(1/(4*popt[3])*perr[0]))

            if modified_tanh:
                thomson_profiles['Error']['Slope'][i_time_vec]=perr[4]

        thomson_profiles['Max gradient']=(thomson_profiles['SOL offset']-thomson_profiles['Height'])/(4*thomson_profiles['Width'])
        thomson_profiles['Value at max']=(thomson_profiles['SOL offset']+thomson_profiles['Height'])/2.
        thomson_profiles['Global gradient']=(thomson_profiles['SOL offset']-thomson_profiles['Height'])/(4*thomson_profiles['Width'])
        if test_time_vec:
            print('Fit 4th in ',time_mod.time_vec()-start_time_vec)

        pickle.dump(thomson_profiles,open(pickle_filename,'wb'))

    return thomson_profiles