#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 11:22:29 2025

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

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

#Scientific imports
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

import matplotlib.pyplot as plt
#Other necessary imports

import MDSplus as mds

def fit_nstx_profiles(exp_id=None,                                              #Shot number
                      electron=False,                                           #Thomson data
                      ion=False,                                                #CHERS data
                      
                      pressure=False,                                           #Return the pressure profile paramenters
                      temperature=False,                                        #Return the temperature profile parameters
                      density=False,                                            #Return the density profile parameters
                      
                      spline_data=False,                                        #Calculate the results from the spline data (no error is going to be taken into account)
                        
                      modified_tanh=False,
                      average_profiles=None,

                      device_coordinates=False,                                 #Calculate the results as a function of device coordinates
                      radial_range=None,                                        #Radial range of the pedestal (only works when the device coorinates is set)

                      flux_coordinates=False,                                   #Calculate the results in flux coordinates
                      flux_range=None,                                          #The normalaized flux coordinates range for returning the results

                      outboard_only=True,                                       #Use only the outboard profile
                      force_overlap=False,                                      #Shifts the inboard and outboard profiles of the TS to match

                      max_iter=1200,                                            #Maximum iteration for the shifting
                      max_err=1e-5,                                             #difference between iteration steps to be reached

                      output_name=None,
                      plot_time_diag=None,
                      pdf_object=None,

                      test=False,
                      test_time_diag=False,
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
            raise ValueError('force_overlap and outboard_only cannot be set at the same time_diag.')

        if ((device_coordinates and flux_range is not None) or
            (flux_coordinates and radial_range is not None)):
            raise ValueError('When flux or device coordinates are set, only flux or radial range can be set! Returning...')
        if test_time_diag:
            start_time_diag=time_mod.time_diag()

        comment=''
        if density:         comment+='_ne'
        if temperature:     comment+='_te'
        if pressure:        comment+='_pe'
        if spline_data:     comment+="_spl"
        if modified_tanh:   comment+='_mtanh'
        if device_coordinates:  comment+='_dev'
        if flux_coordinates:    comment+='_flux'
        if electron:
            pickle_filename=wd+'/processed_data/TS_'+str(exp_id)+comment+'.pickle'
        elif ion:
            pickle_filename=wd+'/processed_data/CHERS_'+str(exp_id)+comment+'.pickle'

        if nocalc and os.path.exists(pickle_filename):
            profile_data=pickle.load(open(pickle_filename,'rb'))
        else:
            
            conn=mds.Connection('skylark.pppl.gov')

            if test_time_diag:
                print('Fit 1st in ',time_mod.time_diag()-start_time_diag)
                start_time_diag=time_mod.time_diag()
    
            if flux_coordinates:
                r_coord_name='Flux r'
                if flux_range is None:
                    flux_range=[0.,1.1]
                    
        
            try:
                conn.openTree('EFIT02',exp_id)    
                
                data_psirz=conn.get('\\PSIRZ').data()
                time_psirz=conn.get('dim_of(\\PSIRZ,0)').data()
                rad_coord_psirz=conn.get('dim_of(\\PSIRZ,1)').data()

                data_ssimag=conn.get('\\SSIMAG').data()

                data_ssibry=conn.get('\\SSIBRY').data()

                
                # R_data=flap.get_data('NSTX_MDSPlus',
                #                      name='\EFIT02::\R',
                #                      exp_id=exp_id,
                #                      object_name='R_FOR_COORD')
                # PSI_norm_data=flap.get_data('NSTX_MDSPlus',
                #                          name='\EFIT02::\PSIN',
                #                          exp_id=exp_id,
                #                          object_name='PSIN')
                
            except:
                raise ValueError("The PSIRZ MDSPlus node cannot be reached.")
            
            psi_n=(data_psirz-data_ssimag[:,None,None])/(data_ssibry-data_ssimag)[:,None,None]
            psi_n[np.isnan(psi_n)]=0.
            psi_n_efit=psi_n[:,32,:] #psi_n is transposed originally
            time_efit=time_psirz
            R_efit=rad_coord_psirz
            
            conn.openTree('ACTIVESPEC', exp_id)

            if electron:
                node_validity='\\TS_BEST:VALID'
                if not spline_data:
                    if pressure:
                        node='\\TS_BEST:FIT_PE'
                        node_error='\\TS_BEST:FIT_PE_ERR'
                    elif temperature:
                        node='\\TS_BEST:FIT_TE'
                        node_error='\\TS_BEST:FIT_TE_ERR'
                    elif density:
                        node='\\TS_BEST:FIT_NE'
                        node_error='\\TS_BEST:FIT_NE_ERR'
                    else:
                        raise ValueError('pressure, temperature or density needs to be set.')
                else:
                    raise ValueError('Spline data is not support for Thomson scattering due to the lack of uncertainty data.')
                # else:
                #     if pressure:
                #         node='\\TS_BEST:SPLINE_PE'
                #         node_error='\\TS_BEST:FIT_PE_ERR'
                #     elif temperature:
                #         node='\\TS_BEST:SPLINE_TE'
                #         node_error='\\TS_BEST:FIT_TE_ERR'
                #     elif density:
                #         node='\\TS_BEST:SPLINE_NE'
                #         node_error='\\TS_BEST:FIT_NE_ERR'
                #     else:
                #         raise ValueError('pressure, temperature or density needs to be set.')
                
            elif ion:
                if temperature:
                    node='\\TOP.CHERS.ANALYSIS.CT1:TI'
                    node_error='\\TOP.CHERS.ANALYSIS.CT1:DTI'
                elif density:
                    node='\\TOP.CHERS.ANALYSIS.CT1:ND'
                    node_error='\\TOP.CHERS.ANALYSIS.CT1:DND'
                elif pressure:
                    node='\\TOP.CHERS.ANALYSIS.CT1:PI'
                    node_error='\\TOP.CHERS.ANALYSIS.CT1:DPI'
                
                node_validity='\\TOP.CHERS.ANALYSIS.CT1:VALID'
            #END OF THOMSON DATA ANALYSIS
                if spline_data and not pressure: node += 'S'
                
            try:
                data=conn.get(node).data()
                error=conn.get(node_error).data()
                data_validity=conn.get(node_validity).data()
                
                if ion: 
                    t_dim=1
                    R_dim=0
                elif electron:
                    t_dim=0
                    R_dim=1
                    
                time_diag=conn.get(f'dim_of({node},{t_dim})').data()              #For some reason the second coordinate is time, flap_mdsplus cannot read that.
                R_diag=conn.get(f'dim_of({node},{R_dim})').data()/100.      #Originally in cm
                
            except Exception as e:
                print(e)
                raise ValueError('The data could not be read')

            #CHERS measures both LFS and HFS sometimes, only LFS is considered
            
            try:
                conn.openTree('EFIT02', exp_id)
                R_mag_axis_data=conn.get('\\RMAXIS').data()
                R_mag_axis_efit_time=conn.get('dim_of(\\RMAXIS,0)')

                R_mag_axis_diag_time=np.interp(time_diag,
                                               R_mag_axis_efit_time,
                                               R_mag_axis_data)
                        
            except Exception as e:
                print(e)
                print('Failed to read RMAXIS for shot ',exp_id)
                R_mag_axis_data=np.nan
                
            try:
                for ind_time, _ in enumerate(time_diag):
                    data[ind_time, R_diag < R_mag_axis_diag_time[ind_time]] = np.nan
            except:
                pass
        
            # if device_coordinates or not flux_coordinates:
            #     r_coord_name='Device R'
            #     if radial_range is None:
            #         radial_range=[np.min(rad_coord),
            #                       np.max(rad_coord)]
            #     flux_coord=None    
    
            #Piecewise 2D interpolation from t_EFIT,R_EFIT to t_chers,R_chers for psi_N
            
            n_time_diag=len(time_diag)
            n_radius_efit=len(R_efit)
            # print(psi_n_efit[ind_efit_time,:])
            # print(psi_n_chers_R_efit_t[ind_efit_time,:])
            
            psi_n_R_efit_t_diag=np.zeros([n_radius_efit,n_time_diag])
            # print(radial_coordinate, R_on_grid, psi_N_on_grid)
            print(psi_n_efit.shape, R_efit.shape, time_efit.shape)
            
            for ind_r_efit in range(n_radius_efit):
                psi_n_R_efit_t_diag[:,ind_r_efit]=np.interp(time_diag, 
                                                            time_efit, 
                                                            psi_n_efit[:,ind_r_efit])
        
            

            
            #Do the interpolation
            #psi_values_spat_interpol=np.zeros([thomson_r_coord.shape[0],
            #                                   psi_t_coord.shape[0]])
            psi_n_diag=np.zeros([data.shape[0],
                                 time_diag.shape[0]])
            
            for index_t in range(len(time_diag)):
                ind_t_efit=np.argmin(np.abs(time_efit-time_diag[index_t]))
                psi_n_diag[:,index_t]=np.interp(R_diag,
                                                R_efit[ind_t_efit,:],
                                                psi_n_R_efit_t_diag[ind_t_efit,:])
                
            if test:
                for index_t in range(len(time_diag)):
                    plt.cla()
                    plt.plot(R_diag,psi_n_diag[:,index_t])
                    # plt.plot(thomson_r_coord,psi_values_at_ts[:,index_t])
                    plt.pause(0.5)
                
            psi_n_diag[np.isnan(psi_n_diag)]=0.

            if test_time_diag:
                print('Fit 1st in ',time_mod.time_diag()-start_time_diag)
                start_time_diag=time_mod.time_diag()


            if device_coordinates or not flux_coordinates:
                r_coord_name='Device R'
                if radial_range is None:
                    radial_range=[np.min(R_diag),
                                  np.max(R_diag)]
            if flux_coordinates:
                fitting_string='psi_n'
            elif device_coordinates:
                fitting_string='R'

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

            if test_time_diag:
                print('Fit 2nd in ',time_mod.time_diag()-start_time_diag)
                start_time_diag=time_mod.time_diag()

                
            # print(time_diag.shape,d.data.shape, d.coordinate('Flux r')[0].shape)
            profile_data={'time_diag':time_diag,
                          'Data':data,
                          
                          'Device R':R_diag,
                          'Flux r':psi_n_diag,
                          
                          'R magnetic axis':R_mag_axis_diag_time,
                                  
                          'Fitting':fitting_string,
                          'Fit parameters':np.zeros([time_diag.shape[0],5]),
                          'Fit parameter errors':np.zeros([time_diag.shape[0],5]),
                          'Fit parameter covariance':np.zeros([time_diag.shape[0],5,5]),
                         
                          'a':np.zeros(time_diag.shape),
                          'Height':np.zeros(time_diag.shape),
                          'Width':np.zeros(time_diag.shape),
                          'Global gradient':np.zeros(time_diag.shape),
                          'Position':np.zeros(time_diag.shape),
                          'Position r':np.zeros(time_diag.shape),
                          'SOL offset':np.zeros(time_diag.shape),
                          'Max gradient':np.zeros(time_diag.shape),
                          'Value at max':np.zeros(time_diag.shape),
                          'SOL avg':np.zeros(time_diag.shape),
                          }
            
            if modified_tanh:
                profile_data['Slope']=np.zeros(time_diag.shape)
                                     
            profile_data['Error']={'Data':error,
                                   'Height':np.zeros(time_diag.shape),
                                   'SOL offset':np.zeros(time_diag.shape),
                                   'Position':np.zeros(time_diag.shape),
                                   'Position r':np.zeros(time_diag.shape),
                                   'Width':np.zeros(time_diag.shape),
                                   'Global gradient':np.zeros(time_diag.shape),
                                   'Max gradient':np.zeros(time_diag.shape),
                                   'Value at max':np.zeros(time_diag.shape),
                                   'SOL avg':np.zeros(time_diag.shape),
                                   }
            
            if modified_tanh:
                profile_data['Error']['Slope']=np.zeros(time_diag.shape)

            if test_time_diag:
                print('Fit 3rd in ',time_mod.time_diag()-start_time_diag)
                start_time_diag=time_mod.time_diag()

            for i_time_diag, cur_time in enumerate(time_diag):
                if r_coord_name =='Flux r':
                    x_data=psi_n_diag
                    y_data=data[:,i_time_diag]
                    y_data_error=error[:,i_time_diag]

                    r_maxis_cur=R_mag_axis_diag_time[i_time_diag]
                    ind_maxis=np.argmin(np.abs(R_diag-r_maxis_cur))

                    if electron and (outboard_only or device_coordinates and not temperature):
                        if not outboard_only and device_coordinates and i_time_diag == 0:
                            print('outboard_only=False and device_coordinates=True are not compatible')
                            print('Using only outboard TS points.')
                        x_data=x_data[ind_maxis:]
                        y_data=y_data[ind_maxis:]
                        y_data_error=error[ind_maxis:,i_time_diag]
                    else:
                        x_data_in=x_data[:ind_maxis]
                        x_data_out=x_data[ind_maxis:]
                        y_data_in=y_data[:ind_maxis]
                        y_data_out=y_data[ind_maxis:]

                        if not temperature:                                                 #Only temperature is a flux function on NSTX
                            temp_data_in=data[:,i_time_diag][:ind_maxis]
                            temp_data_out=data[:,i_time_diag][ind_maxis:]
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
                    if average_profiles is not None and i_time_diag+1 > average_profiles:
                        ind_max=np.argmax(data[:,i_time_diag])
                        x_data=R_diag[ind_max:]
                        y_data=(np.sum(data[ind_max:,i_time_diag-average_profiles+1:i_time_diag+1] *
                                       error[ind_max:,i_time_diag-average_profiles+1:i_time_diag+1]) /
                                average_profiles/np.sum(error[ind_max:,i_time_diag-average_profiles+1:i_time_diag+1],axis=1)
                                )
                        y_data_error=np.sqrt(np.sum(error[ind_max:,i_time_diag-average_profiles+1:i_time_diag+1]**2,axis=1))/np.sqrt(average_profiles)
                    else:
                        ind_max=np.argmax(data[:,i_time_diag])
                        x_data=R_diag[ind_max:]
                        y_data=data[ind_max:,i_time_diag]
                        y_data_error=error[ind_max:,i_time_diag]

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
                    print('Missing TS data for shot '+str(exp_id)+', time_diag: '+ str(time_diag[i_time_diag]))

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
                
                if test or (plot_time_diag is not None and i_time_diag==np.argmin(np.abs(plot_time_diag-time_diag))):
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

                    time_diag_string=' @ '+str(time_diag[i_time_diag])
                    plt.title('Fit '+profile_string+' profile of '+str(exp_id)+time_diag_string)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)


                    if pdf_object is not None:
                        pdf_object.savefig()
                else:
                    pass

                if modified_tanh:
                    profile_data['Fit parameters'][i_time_diag,:]=popt
                    profile_data['Fit parameter errors'][i_time_diag,:]=perr
                    profile_data['Fit parameter covariance'][i_time_diag,:,:]=pcov
                else:
                    profile_data['Fit parameters'][i_time_diag,0:4]=popt
                    profile_data['Fit parameter errors'][i_time_diag,0:4]=perr
                    profile_data['Fit parameter covariance'][i_time_diag,0:4,0:4]=pcov

                profile_data['Height'][i_time_diag]=popt[0]
                profile_data['SOL offset'][i_time_diag]=popt[1]
                profile_data['Position'][i_time_diag]=popt[2]

                try:
                #if True:
                    profile_data['Position r'][i_time_diag]=np.interp(popt[2],
                                                                         psi_n_diag[np.argmin(psi_n_diag[:,i_time_diag]):,i_time_diag],
                                                                         R_diag[np.argmin(psi_n_diag[:,i_time_diag]):])
                except:
                    print('Interpolation failed.')
                    profile_data['Position r'][i_time_diag]=np.nan

                profile_data['Width'][i_time_diag]=popt[3]
                if r_coord_name == 'Flux r':
                    ind_max=np.argmin(psi_n_diag[:,i_time_diag])
                    ind_sol=np.where(psi_n_diag[ind_max:,i_time_diag] > 1.0)
                    
                    
                    
                else:
                    
                    R_separatrix=flap.get_data('NSTX_MDSPlus',
                                               name='\EFIT02::\RMIDOUT',
                                               exp_id=exp_id,
                                               ).slice_data(slicing={'Time':cur_time}).data-0.02
                    
                    
                    ind_max=np.argmax(data[:,i_time_diag])
                    
                    if temperature:
                        sol_limit=0.05
                    if density:
                        sol_limit=5e18
                    if pressure:
                        sol_limit=5e18*0.05*11606*1e3
                        
                    ind_sol=np.where(np.logical_and(R_diag[ind_max:] > R_separatrix, 
                                                    data[ind_max:,i_time_diag] < sol_limit))
                    
                profile_data['SOL avg'][i_time_diag]=np.mean((data[ind_max:,i_time_diag])[ind_sol])
                # profile_data['Error']['SOL avg'][i_time_diag]=np.sqrt(np.var((d.data[ind_max:,i_time_diag])[ind_sol]))
                profile_data['Error']['SOL avg'][i_time_diag]=np.sqrt(np.sum(error[ind_max:,i_time_diag][ind_sol]**2))/len(ind_sol)
                
                if modified_tanh:
                    profile_data['Slope'][i_time_diag]=popt[4]

                profile_data['Error']['Height'][i_time_diag]=perr[0]
                profile_data['Error']['SOL offset'][i_time_diag]=perr[1]
                profile_data['Error']['Position'][i_time_diag]=perr[2]
                profile_data['Error']['Width'][i_time_diag]=perr[3]
                profile_data['Error']['Value at max'][i_time_diag]=(perr[0]+perr[1])/2
                profile_data['Error']['Global gradient'][i_time_diag]=(perr[0]/popt[3]+
                                                                          perr[1]/popt[3]+
                                                                          np.abs((-popt[1]+popt[0])/popt[3]**2)*perr[3])
                profile_data['Error']['Max gradient'][i_time_diag]=(np.abs(1/(4*popt[3])*perr[1])+
                                                                       np.abs(1/(4*popt[3])*perr[0]))

                if modified_tanh:
                    profile_data['Error']['Slope'][i_time_diag]=perr[4]

            profile_data['Max gradient']=(profile_data['SOL offset']-profile_data['Height'])/(4*profile_data['Width'])
            profile_data['Value at max']=(profile_data['SOL offset']+profile_data['Height'])/2.
            profile_data['Global gradient']=(profile_data['SOL offset']-profile_data['Height'])/(4*profile_data['Width'])
            if test_time_diag:
                print('Fit 4th in ',time_mod.time_diag()-start_time_diag)

            pickle.dump(profile_data,open(pickle_filename,'wb'))

        return profile_data