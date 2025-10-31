#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:10:17 2025

@author: mlampert
"""

import os
os.environ["MKL_DISABLE_WARNINGS"] = "1"

import time as time_mod
import pickle


#FLAP imports and settings
import flap
import flap_nstx
import flap_mdsplus

import MDSplus as mds

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

import matplotlib.pyplot as plt
#Other necessary imports


def get_fit_nstx_chers_profiles(exp_id=None,                                      #Shot number
                                ion_temperature=False,                                   #Return the pressure profile paramenters
                                ion_density=False,
                                ion_pressure=False,
                                
                                toroidal_velocity=False,
                                effective_charge_state=False,
                                carbon_density=False,
                                
                                spline_data=False,
                                
                                modified_tanh=False,
                                average_profiles=None,

                                device_coordinates=False,                          #Calculate the results as a function of device coordinates
                                radial_range=None,                                #Radial range of the pedestal (only works when the device coorinates is set)

                                flux_coordinates=False,                           #Calculate the results in flux coordinates
                                flux_range=None,                                  #The normalaized flux coordinates range for returning the results

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

    if ((device_coordinates and flux_range is not None) or
        (flux_coordinates and radial_range is not None)):
        raise ValueError('When flux or device coordinates are set, only flux or radial range can be set! Returning...')
    
    if (not device_coordinates and not flux_coordinates) or (device_coordinates and flux_coordinates):
        raise ValueError('Either device_coordinates or flux_coordinates need to be set, not neither or both.')
    
    if test_time_vec:
        start_time_vec=time_mod.time_vec()

    comment=''
    if ion_density:         comment+='_ni'
    elif ion_temperature:     comment+='_ti'
    elif toroidal_velocity:        comment+='_vtor'
    elif effective_charge_state:        comment+='_zeff'
    elif carbon_density: comment+='_nc'
    elif ion_pressure: comment+='_pi'
    
    if spline_data: comment+='_spline'
    
    if modified_tanh:   comment+='_mtanh'
    if device_coordinates:  comment+='_R'
    if flux_coordinates:    comment+='_psi'

    pickle_filename=wd+'/processed_data/CHERS_'+str(exp_id)+comment+'.pickle'

    if nocalc and os.path.exists(pickle_filename):
        chers_profiles=pickle.load(open(pickle_filename,'rb'))
    else:

        conn=mds.Connection('skylark.pppl.gov:8505')
        conn.openTree('ACTIVESPEC', exp_id)
        
        if ion_temperature:
            node='\\TOP.CHERS.ANALYSIS.CT1:TI'
            node_error='\\TOP.CHERS.ANALYSIS.CT1:DTI'
        elif ion_density:
            node='\\TOP.CHERS.ANALYSIS.CT1:ND'
            node_error='\\TOP.CHERS.ANALYSIS.CT1:DND'
        elif carbon_density:
            node='\\TOP.CHERS.ANALYSIS.CT1:NC'
            node_error='\\TOP.CHERS.ANALYSIS.CT1:DNC'
        elif ion_pressure:
            node='\\TOP.CHERS.ANALYSIS.CT1:PI'
            node_error='\\TOP.CHERS.ANALYSIS.CT1:DPI'
        elif toroidal_velocity:
            node='\\TOP.CHERS.ANALYSIS.CT1:VT'
            node_error='\\TOP.CHERS.ANALYSIS.CT1:DVT'
        elif effective_charge_state:
            node='\\TOP.CHERS.ANALYSIS.CT1:ZEFF'
            node_error='\\TOP.CHERS.ANALYSIS.CT1:DZEFF'
            
        if spline_data and not ion_pressure: node += 'S'
        
        try:
            data_chers=conn.get(node).data()
            data_chers_error=conn.get(node_error).data()
            
            time_chers=conn.get(f'dim_of({node},1)').data()              #For some reason the second coordinate is time, flap_mdsplus cannot read that.
            
            R_chers=conn.get(f'dim_of({node},0)').data()/100.      #Originally in cm
            data_validity=conn.get('\\TOP.CHERS.ANALYSIS.CT1:VALID').data()
            
        except Exception as e:
            print(e)
            raise ValueError('The data could not be read')

        #CHERS measures both LFS and HFS sometimes, only LFS is considered
        
        try:
            R_mag_axis=flap.get_data('NSTX_MDSPlus',
                                 name='\EFIT02::\RMAXIS',
                                 exp_id=exp_id,
                                 )
            R_mag_axis_data=R_mag_axis.data
            R_mag_axis_efit_time=R_mag_axis.coordinate('Time')[0]
            R_mag_axis_chers_time=np.interp(time_chers,R_mag_axis_efit_time,R_mag_axis_data)
                    
        except Exception as e:
            print(e)
            print('Failed to read RMAXIS for shot ',exp_id)
            R_mag_axis=np.nan
            
        try:
            for ind_time, _ in enumerate(time_chers):
                data_chers[ind_time, R_chers < R_mag_axis_chers_time[ind_time]] = np.nan
        except:
            pass




        if test_time_vec:
            print('Fit 1st in ',time_mod.time_vec()-start_time_vec)
            start_time_vec=time_mod.time_vec()

        if flux_range is None:
            flux_range=[0.,1.1]
            
        if radial_range is None:
            radial_range=[1.0,1.55]
        
        try:
            conn.openTree('EFIT02',exp_id)
            psirz=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\PSIRZ',
                                exp_id=exp_id,
                                object_name='PSIRZ_FOR_COORD')
            
            ssimag=flap.get_data('NSTX_MDSPlus',
                                  name='\EFIT02::\SSIMAG',
                                  exp_id=exp_id,
                                  object_name='SSIMAG_FOR_COORD')
            
            ssibry=flap.get_data('NSTX_MDSPlus',
                                    name='\EFIT02::\SSIBRY',
                                    exp_id=exp_id,
                                    object_name='SSIBRY_FOR_COORD')
            
        except Exception as e:
            print(e)
            raise ValueError("The PSIRZ MDSPlus node cannot be reached.")
        
        psi_n=(psirz.data-ssimag.data[:,None,None])/(ssibry.data-ssimag.data)[:,None,None]
        psi_n[np.isnan(psi_n)]=0.
        
        # return psi_n
        psi_n_efit=psi_n[:,32,:] #GTFO, psi_n is transposed originally
        time_efit=psirz.coordinate('Time')[0][:,0,0]
        R_efit=psirz.coordinate('Device R')[0][0,:,32]            #midplane is the middle coordinate in the array
        
        
        #Piecewise 2D interpolation from t_EFIT,R_EFIT to t_chers,R_chers for psi_N
        
        n_time_efit=len(psi_n_efit[:,0])
        n_radius_chers=len(R_chers)
        
        psi_n_chers_R_efit_t=np.zeros([n_time_efit,n_radius_chers])
        # print(radial_coordinate, R_on_grid, psi_N_on_grid)
        for ind_efit_time in range(n_time_efit):
            psi_n_chers_R_efit_t[ind_efit_time,:]=np.interp(R_chers, 
                                                            R_efit, 
                                                            psi_n_efit[ind_efit_time,:])
            # print(psi_n_efit[ind_efit_time,:])
            # print(psi_n_chers_R_efit_t[ind_efit_time,:])
        
        psi_n_chers_tR=np.zeros([len(time_chers),len(R_chers)])
        
        for ind_chers_R, _ in enumerate(R_chers):
            psi_n_chers_tR[:,ind_chers_R]=np.interp(time_chers,
                                                    time_efit,
                                                    psi_n_chers_R_efit_t[:,ind_chers_R])

        if flux_coordinates:
            fitting_string='psi_n'
        elif device_coordinates:
            fitting_string='R'
        
        # print(time_chers.shape,data_chers.shape, psi_n_chers_tR.shape)
        chers_profiles={'time_vec':time_chers,
                        'Data':data_chers.T,
                        
                        'Device R':R_chers,
                        'Flux r':psi_n_chers_tR.T,
                        'R magnetic axis':R_mag_axis_chers_time,
                        
                        'Fitting': fitting_string,
                        'Fit parameters':np.zeros([time_chers.shape[0],5]),
                        'Fit parameter errors':np.zeros([time_chers.shape[0],5]),
                        'Fit parameter covariance':np.zeros([time_chers.shape[0],5,5]),
                         
                        'a':np.zeros(time_chers.shape),
                        'Height':np.zeros(time_chers.shape),
                        'Width':np.zeros(time_chers.shape),
                        'Global gradient':np.zeros(time_chers.shape),
                        'Position':np.zeros(time_chers.shape),
                        'SOL offset':np.zeros(time_chers.shape),
                        
                        'Max gradient':np.zeros(time_chers.shape),
                        'Value at max':np.zeros(time_chers.shape),
                        'SOL avg':np.zeros(time_chers.shape),
                        }
        
        if modified_tanh:
            chers_profiles['Slope']=np.zeros(time_chers.shape)
            
        chers_profiles['Error']={'Data':data_chers_error.T,
                                 'Height':np.zeros(time_chers.shape),
                                 'SOL offset':np.zeros(time_chers.shape),
                                 'Position':np.zeros(time_chers.shape),
                                 'Width':np.zeros(time_chers.shape),
                                 'Global gradient':np.zeros(time_chers.shape),
                                 'Max gradient':np.zeros(time_chers.shape),
                                 'Value at max':np.zeros(time_chers.shape),
                                 'SOL avg':np.zeros(time_chers.shape),
                                 }
        
        if modified_tanh:
            chers_profiles['Error']['Slope']=np.zeros(time_chers.shape)
        
        for ind_time, curr_time in enumerate(time_chers):
            #data_validity_curr=data_validity[ind_time,:]

            data_validity_curr=data_validity[ind_time,:]
            
            ind_valid_data=np.where(np.logical_and(data_validity_curr != 0.,
                                                   ~np.isnan(data_chers[ind_time,:])))
            
            data_curr=(data_chers[ind_time,:])[ind_valid_data]
            data_error_curr=data_chers_error[ind_time,:][ind_valid_data]
            psi_n_curr=psi_n_chers_tR[ind_time,:][ind_valid_data]
            R_chers_curr=R_chers[ind_valid_data]

            if test:
                plt.figure()
                
            if flux_coordinates:
                x_range=flux_range
                
            elif device_coordinates:
                x_range=radial_range
    
            if not modified_tanh:
                tanh_fit_function=tanh_function
            else:
                tanh_fit_function=mtanh_function
                
            if flux_coordinates:
                x_data=psi_n_curr
                
            elif device_coordinates:
                x_data=R_chers_curr
                
            y_data=data_curr
            y_data_error=data_error_curr
            
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
                        np.abs((x_data[-1]-x_data[0])/2),               #b_width
                        #(y_data[0]-y_data[-1])/(x_data[0]-x_data[-1]), #b_slope
                        ]

                else:
                    p0=[y_data[0],                                      #b_height
                        y_data[-1],                                     #b_sol
                        (x_data[0]+x_data[-1])/2,                       #b_pos
                        np.abs((x_data[-1]-x_data[0])/2),               #b_width
                        (y_data[0]-y_data[-1])/(x_data[0]-x_data[-1]),  #b_slope 
                        ]
            except:
                print('Missing CHERS data for shot '+str(exp_id)+', time_vec: '+ str(curr_time))

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

            if test or (plot_time_vec is not None and ind_time==np.argmin(np.abs(plot_time_vec-time_chers))):
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
                    
                elif device_coordinates:
                    xlabel='Device R [m]'

                if ion_temperature:
                    profile_string='ion_temperature'
                    ylabel='Ion temperature [keV]'

                elif ion_density:
                    profile_string='ion_density'
                    ylabel='Ion density [1/m3]'
                    
                elif ion_pressure:
                    profile_string='ion_pressure'
                    ylabel='Ion pressure [kPa]'
                    
                elif toroidal_velocity:
                    profile_string='v_tor'
                    ylabel='v_tor [m/s]'
                    
                elif effective_charge_state:
                    profile_string='Z_eff'
                    ylabel='Z_eff'
                    
                elif carbon_density:
                    profile_string='n_C'
                    ylabel='n_c [1/m3]'
                    
                else:
                    profile_string=''
                    ylabel=''

                time_vec_string=' @ '+str(curr_time)
                plt.title('Fit '+profile_string+' profile of '+str(exp_id)+time_vec_string)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)


                if pdf_object is not None:
                    pdf_object.savefig()


            if successful_fitting:
                
                if modified_tanh:
                    chers_profiles['Fit parameters'][ind_time,:]=popt
                    chers_profiles['Fit parameter errors'][ind_time,:]=perr
                    chers_profiles['Fit parameter covariance'][ind_time,:,:]=pcov
                else:
                    chers_profiles['Fit parameters'][ind_time,0:4]=popt
                    chers_profiles['Fit parameter errors'][ind_time,0:4]=perr
                    chers_profiles['Fit parameter covariance'][ind_time,0:4,0:4]=pcov
    
                chers_profiles['Height'][ind_time]=popt[0]
                chers_profiles['SOL offset'][ind_time]=popt[1]
                chers_profiles['Position'][ind_time]=popt[2]
                chers_profiles['Width'][ind_time]=popt[3]    
                
                if modified_tanh: chers_profiles['Slope'][ind_time]=popt[4]
    
                if flux_coordinates:
                    ind_sol=np.where(psi_n_curr > 1.0)
                    
                elif device_coordinates:
                    R_separatrix=flap.get_data('NSTX_MDSPlus',
                                               name='\EFIT02::\RMIDOUT',
                                               exp_id=exp_id,
                                               ).slice_data(slicing={'Time':curr_time}).data-0.02
    
                    if ion_temperature:
                        sol_limit=0.05
                    if ion_density:
                        sol_limit=5e18
                    if ion_pressure:
                        sol_limit=5e18*0.05*11606*1e3
                    else:
                        sol_limit=0.
                        
                    ind_sol=np.where(np.logical_and(R_chers_curr > R_separatrix, 
                                                    data_curr < sol_limit))
                    
                chers_profiles['SOL avg'][ind_time]=np.mean(data_curr[ind_sol])
                
    
                chers_profiles['Error']['Height'][ind_time]=perr[0]
                chers_profiles['Error']['SOL offset'][ind_time]=perr[1]
                chers_profiles['Error']['Position'][ind_time]=perr[2]
                chers_profiles['Error']['Width'][ind_time]=perr[3]
                
                if modified_tanh: chers_profiles['Error']['Slope'][ind_time]=perr[4]
                    
                chers_profiles['Error']['Value at max'][ind_time]=(perr[0]+perr[1])/2
                chers_profiles['Error']['Global gradient'][ind_time]=(perr[0]/popt[3]+
                                                                      perr[1]/popt[3]+
                                                                      np.abs((-popt[1]+popt[0])/popt[3]**2)*perr[3])
                chers_profiles['Error']['Max gradient'][ind_time]=(np.abs(1/(4*popt[3])*perr[1])+
                                                                   np.abs(1/(4*popt[3])*perr[0]))
    
    
                chers_profiles['Error']['SOL avg'][ind_time]=np.sqrt(np.sum(data_error_curr[ind_sol]**2))/len(ind_sol)
                #np.sqrt(np.var(data_curr[ind_sol]))
                
            else: #successful_fit=False
                calculate_keys=['Height',
                                'SOL offset',
                                'Position',
                                'Width',
                                'Global gradient',
                                'Max gradient',
                                'Value at max',
                                'SOL avg',
                                ]
                for key in calculate_keys:
                    if key !='Error':
                        chers_profiles[key][ind_time]=np.nan
                        chers_profiles['Error'][key][ind_time]=np.nan
                        
        chers_profiles['Max gradient']=(chers_profiles['SOL offset']-chers_profiles['Height'])/(4*chers_profiles['Width'])
        chers_profiles['Value at max']=(chers_profiles['SOL offset']+chers_profiles['Height'])/2.
        chers_profiles['Global gradient']=(chers_profiles['SOL offset']-chers_profiles['Height'])/(4*chers_profiles['Width'])
        
        if test_time_vec:
            print('Fit 4th in ',time_mod.time_vec()-start_time_vec)

        pickle.dump(chers_profiles,open(pickle_filename,'wb'))
        
    conn.disconnect()
    return chers_profiles