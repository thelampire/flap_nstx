#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 15:53:35 2025

@author: mlampert
"""

import os
os.environ["MKL_DISABLE_WARNINGS"] = "1"
import copy
import time as time_mod
import pickle
import warnings
warnings.filterwarnings("ignore")

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')

from flap_nstx.analysis import read_blob_database
from flap_nstx.thomson import get_fit_nstx_thomson_profiles

from flap_nstx.pedestal_db import nstx_pedestal_database_header,nstx_pedestal_database_dictionary

from flap_nstx.tools import mtanh_function, calculate_plasma_squareness

import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas

#Plot settings for publications
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/plots'


def build_nstx_pedestal_database(nocalc=False,
                                 time_range_around_peak=50e-3,
                                 calculate_oak_database=False,
                                 plot_profile_fits=False,
                                 return_database=False,
                                 ):
    time_start=time_mod.time()
    if plot_profile_fits:
        import matplotlib
        matplotlib.use('agg')
        
    if calculate_oak_database:
        shot_db=read_oak_jrt_db()
        db_file_name=wd+'/db/pedestal_db_oak.pickle'
    else:
        shot_db=read_blob_database() #Same as H-mode db but different times
        db_file_name=wd+'/db/pedestal_db_mlampert.pickle'
    
    full_database=[]
    
    
    if not os.path.exists(db_file_name) and nocalc:
        print('The db file does not exist. Recalculating...')    
    
    if not os.path.exists(db_file_name) or not nocalc:
        for ind,(shot,time) in enumerate(zip((shot_db['shot']),(shot_db['time']))):
            # try:
            if True:
                print(shot, time)
                pedestal_db=read_nstx_pedestal_data(shot=int(shot),
                                                    time_range=[time-time_range_around_peak,
                                                                time+time_range_around_peak],
                                                    plot_profile_fits=plot_profile_fits)
                
                full_database.append(pedestal_db)
                pickle.dump(full_database,open(db_file_name+'_tmp','wb'))
            # except Exception as e:
            #     print(e)
                
            time_cur=time_mod.time()
            time_rem=(time_cur-time_start)/(ind+1)*(len(shot_db['shot'])-(ind+1))
            
            # Convert seconds -> h:m:s
            hours, rem = divmod(int(time_rem), 3600)
            minutes, seconds = divmod(rem, 60)

            print(
                f"\n**************************************\n\n"
                f"{hours}h {minutes}m {seconds}s remain from the calculation\n\n"
                f"**************************************\n"
                )
            
            # print(f"\n**************************************\n\n{time_rem/3600} hours remain from the calculation\n\n**************************************\\n")
            
        pickle.dump(full_database,open(db_file_name,'wb'))
    else:
        full_database=pickle.load(open(db_file_name,'rb'))
    if plot_profile_fits:
        matplotlib.use('qt5agg')
        
    if return_database:
        return full_database

def read_nstx_pedestal_data(shot=None,
                            time_range=None,
                            plot_profile_fits=False,
                            ):
    
    if plot_profile_fits:
        filename=flap_nstx.tools.filename(exp_id=shot,
                                          time_range=time_range,
                                          working_directory=wd+'/plots',
                                          purpose='profile_fits',
                                          extension='pdf')
        
        pdf_pages=PdfPages(filename)
    else:
        pdf_pages=None
    
    pedestal_db=nstx_pedestal_database_dictionary()
    
    pedestal_db['Shot data']['shot']=shot
    pedestal_db['Shot data']['time_range']=time_range
    

    
    for key_param in list(pedestal_db.keys())[1:]:
        # print(pedestal_db[key_param])
        if pedestal_db[key_param]['source'] == 'mdsplus':
            pedestal_db=read_pedestal_mdsplus_data(pedestal_db,
                                                   key_param=key_param)
            # print(pedestal_db)            
        elif pedestal_db[key_param]['source'] == 'mdsplus time derivative':
            pedestal_db=read_pedestal_mdsplus_time_derivative(pedestal_db,
                                                              key_param=key_param)                      
            # print(pedestal_db)            
        elif pedestal_db[key_param]['source'] == 'mdsplus threshold time':
            pedestal_db=read_pedestal_mdsplus_threshold_time(pedestal_db,
                                                             key_param=key_param)
            # print(pedestal_db)    
        elif (pedestal_db[key_param]['source'] == 'TS profile fitting' or 
              pedestal_db[key_param]['source'] == 'CHERS profile fitting'):
            pedestal_db=calculate_pedestal_profile_fitting(pedestal_db,
                                                           key_param=key_param,
                                                           plot_profile_fits=plot_profile_fits,
                                                           pdf_pages=pdf_pages)
            # print(pedestal_db)
        elif pedestal_db[key_param]['source'] == 'Squareness calculation':
            pedestal_db=calculate_squareness(pedestal_db)
        
        elif pedestal_db[key_param]['source'] == 'H98 calculation':
            pass
            #pedestal_db=calculate_H98_for_db(pedestal_db)
        
        elif pedestal_db[key_param]['source'] == 'H89 calculation':
            pass
        
        elif pedestal_db[key_param]['source'] == 'Aspect ratio calculation':
            pedestal_db=calculate_pedestal_aspect_ratio(pedestal_db)
            # print(pedestal_db)
        elif pedestal_db[key_param]['source'] == 'Greenwald density calculation':
            pedestal_db=calculate_greenwald_density(pedestal_db)
            # print(pedestal_db)
        elif pedestal_db[key_param]['source'] is None:
            print(f'Source for "{key_param}" is missing.')
            
        else:
            source=pedestal_db[key_param]['source']
            print(f'Unknown source for {key_param}: "{source}"')
    pdf_pages.close()
    return pedestal_db


def save_db_in_csv_format(db_file_name=wd+'/db/pedestal_db.pickle_tmp',
                          csv_db_file_name=wd+'/db/pedestal_db.csv'):
    
    
    full_database=pickle.load(open(db_file_name,'rb'))            

    db_for_csv=np.full([len(full_database),106], '', dtype=np.dtype('U100'))
    
    for ind_shot,curr_dict in enumerate(full_database):
        
        db_for_csv[ind_shot,0]='NSTX'
        db_for_csv[ind_shot,1]=curr_dict['Shot data']['shot']
        db_for_csv[ind_shot,2]=curr_dict['Shot data']['regime']
        db_for_csv[ind_shot,3]=curr_dict['Shot data']['wall conditioning']
        
        db_for_csv[ind_shot,4]=curr_dict['Shot data']['time_range'][0]
        db_for_csv[ind_shot,5]=curr_dict['Shot data']['time_range'][1]
        
        db_for_csv[ind_shot,6]=curr_dict['Shot data']['gas_a']
        db_for_csv[ind_shot,7]=curr_dict['Shot data']['gas_z']
        
        try:
            db_for_csv[ind_shot,8]=curr_dict['Shot data']['gas_minority_a']
        except:
            db_for_csv[ind_shot,8]=curr_dict['Shot data']['gas_minority']
            
        try:
            db_for_csv[ind_shot,9]=curr_dict['Shot data']['gas_minority_b']
        except Exception as e:
            db_for_csv[ind_shot,9]=6
            
        db_for_csv[ind_shot,10]=curr_dict['Shot data']['author']
        db_for_csv[ind_shot,11]=curr_dict['Shot data']['equilibrium']
        
        for key in list(curr_dict.keys())[1:]:
            if curr_dict[key]['db column #'] is not None:
                col_num=int(curr_dict[key]['db column #'])
                try:
                    data=curr_dict[key]['data']
                    error=curr_dict[key]['data error']
                    data_string=f'({data},{error})'
                    db_for_csv[ind_shot,col_num+11]=data_string
                except Exception as e:
                    print(e, key)
                    db_for_csv[ind_shot,col_num+11]='(0,0)'
    df=pandas.DataFrame(db_for_csv.T, 
                        index=nstx_pedestal_database_header())
    
    df.transpose().to_csv(csv_db_file_name)
    return df


def read_csv_into_db_format(original_db_file_name=wd+'/db/pedestal_db.pickle',
                            new_db_filename=wd+'/db/pedestal_db_new.pickle',
                            csv_db_file_name=wd+'/db/pedestal_db.csv',
                            ind_mod=0): #set this -1 for Oak's db
    
    full_database_original=pickle.load(open(original_db_file_name,'rb'))    
    
    new_db=[full_database_original[0]]
    for key in list(new_db[0]['Shot data'].keys()):
        new_db[0]['Shot data'][key]=None
    
    for key in list(new_db[0].keys()):
        if key != 'Shot data':
            for key2 in list(new_db[0][key].keys()):
                new_db[0][key][key2]=None
            
    db_for_csv=np.asarray(pandas.read_csv(csv_db_file_name))
    
    for ind_shot,shot in enumerate(db_for_csv[:,2]):
        new_db.append(copy.deepcopy(new_db[0]))
        new_db[-1]['Shot data']['shot']=db_for_csv[ind_shot,2+ind_mod]
        new_db[-1]['Shot data']['regime']=db_for_csv[ind_shot,3+ind_mod]
        new_db[-1]['Shot data']['wall conditioning']=db_for_csv[ind_shot,4+ind_mod]
        new_db[-1]['Shot data']['time_range']=[db_for_csv[ind_shot,5+ind_mod],db_for_csv[ind_shot,6+ind_mod]]
        new_db[-1]['Shot data']['gas_a']=db_for_csv[ind_shot,7+ind_mod]
        new_db[-1]['Shot data']['gas_z']=db_for_csv[ind_shot,8+ind_mod]
        new_db[-1]['Shot data']['gas_minority_a']=db_for_csv[ind_shot,9+ind_mod]
        new_db[-1]['Shot data']['gas_minority_n']=db_for_csv[ind_shot,10+ind_mod]
        new_db[-1]['Shot data']['author']=db_for_csv[ind_shot,11+ind_mod]
        new_db[-1]['Shot data']['equilibrium']=db_for_csv[ind_shot,12+ind_mod]
        
        for key in list(new_db[-1].keys())[1:]:
            if full_database_original[1][key]['db column #'] is not None:
                col_num=int(full_database_original[1][key]['db column #'])
                # try:
                if True:
                    full_data=db_for_csv[ind_shot,col_num+12+ind_mod]
                    try:
                    # if True:
                        data=float(full_data.split(',')[0].replace('(',''))
                        error=float(full_data.split(',')[1].replace(')',''))
                        new_db[-1][key]['data']=data
                        new_db[-1][key]['error']=error
                    except:
                        pass
                # except Exception as e:
                #     print(e, key)
                #     new_db[-1][key]['data']=np.nan
                #     new_db[-1][key]['error']=np.nan

    
    new_db=new_db[1:]
    pickle.dump(new_db,open(new_db_filename,'wb'))
    return new_db


def calculate_H98_for_db(pedestal_db):
    I_p=None
    B_T=None
    n_e_avg=None
    P_SOL=None
    R_geo=None
    kappa_a=None
    epsilon=None
    capital_m=None
    
    tau_e_98y2=(0.0562 * I_p**0.93 * B_T**0.15 * n_e_avg**0.41 * P_SOL**-0.69 
                * R_geo**1.97 * kappa_a**0.78 * epsilon**0.58 * capital_m**0.19)
    
    return pedestal_db
                
def calculate_greenwald_density(pedestal_db):
    
    shot=pedestal_db['Shot data']['shot']
    time=pedestal_db['Shot data']['time_range']
    
    try:
    # if True:
        current=flap.get_data('NSTX_MDSPlus',
                              name='\EFIT02::\IPMEAS',
                              exp_id=shot,
                              ).slice_data(slicing={'Time':time}).data
        
        minor_radius=flap.get_data('NSTX_MDSPlus',
                                   name='\EFIT02::\AMINOR',
                                   exp_id=shot,
                                   ).slice_data(slicing={'Time':time}).data
        pedestal_db['Density Greenwald']['data']=np.mean(current/(np.pi*minor_radius**2))
        pedestal_db['Density Greenwald']['data error']=np.sqrt(np.var((current/(np.pi*minor_radius**2))))
        
    except Exception as e:
        print('Cannot read and calculate Greenwald density')
        print(e)
        
    return pedestal_db

def read_pedestal_mdsplus_data(pedestal_db, 
                               key_param=None):
    
    shot=pedestal_db['Shot data']['shot']
    time=pedestal_db['Shot data']['time_range']
    time_slicing={'Time':flap.Intervals(time[0], time[1])}
    
    node=pedestal_db[key_param]['mds_node']
    tree=pedestal_db[key_param]['mds_tree']
    mds_name='\\'+tree+'::\\'+node
    
    if tree == '' or node == '' or tree is None or node is None:
        pedestal_db[key_param]['data']=0
        pedestal_db[key_param]['data error']=0
    else:
        try:
        # if True:
            data=flap.get_data('NSTX_MDSPlus',
                               name=mds_name,
                               exp_id=shot,
                               ).slice_data(slicing=time_slicing).data
            pedestal_db[key_param]['data']=np.mean(data)
            pedestal_db[key_param]['data error']=np.sqrt(np.var(data))
        except Exception as e:
            print(e)
            pedestal_db[key_param]['data']=0
            pedestal_db[key_param]['data error']=0
            
    return pedestal_db
            
def read_pedestal_mdsplus_time_derivative(pedestal_db,
                                          key_param=None):
        
    shot=pedestal_db['Shot data']['shot']
    time=pedestal_db['Shot data']['time_range']
    time_slicing={'Time':flap.Intervals(time[0], time[1])}
    
    node=pedestal_db[key_param]['mds_node']
    tree=pedestal_db[key_param]['mds_tree']
    mds_name='\\'+tree+'::\\'+node
    
    if tree == '' or node == '' or tree is None or node is None:
        pedestal_db[key_param]['data']=0
        pedestal_db[key_param]['data error']=0
    else:
        try:
            data=flap.get_data('NSTX_MDSPlus',
                               name=mds_name,
                               exp_id=shot,
                               ).slice_data(slicing={'Time':time_slicing}).data
            pedestal_db[key_param]['data']=np.mean(np.gradient(data))
            pedestal_db[key_param]['data error']=np.sqrt(np.var(np.gradient(data)))
        except:
            pedestal_db[key_param]['data']=0
            pedestal_db[key_param]['data error']=0
    return pedestal_db

def read_pedestal_mdsplus_threshold_time(pedestal_db,
                                         key_param=None):
    
    shot=pedestal_db['Shot data']['shot']
    time=pedestal_db['Shot data']['time_range']
    time_slicing={'Time':flap.Intervals(time[0], time[1])}
    
    node=pedestal_db[key_param]['mds_node']
    tree=pedestal_db[key_param]['mds_tree']
    mds_name='\\'+tree+'::\\'+node
    
    if tree == '' or node == '' or tree is None or node is None:
        pedestal_db[key_param]['data']=0
        pedestal_db[key_param]['data error']=0
    else:
        if node == 'PNB':
            try:
                data=flap.get_data('NSTX_MDSPlus',
                                   name=mds_name,
                                   exp_id=shot,
                                   ).slice_data(slicing={'Time':time_slicing})
                                                        
                nbi_switch_on_time=data.coordinate('Time')[0][np.where(data.data > (np.max(data.data)-np.min(data.data))/4)[0][0]]
                pedestal_db[key_param]['data']=nbi_switch_on_time
                pedestal_db[key_param]['data error']=(data.coordinate('Time')[0][1]-data.coordinate('Time')[0][0])/2
            except:
                pedestal_db[key_param]['data']=0
                pedestal_db[key_param]['data error']=0
    return pedestal_db

def calculate_pedestal_aspect_ratio(pedestal_db):
    
    try:
        pedestal_db['Aspect ratio']['data']=pedestal_db['Magnetic axis R']['data']/pedestal_db['Minor radius']['data']
    except:
        pedestal_db=read_pedestal_mdsplus_data(pedestal_db,key_param='Magnetic axis R')
        pedestal_db=read_pedestal_mdsplus_data(pedestal_db,key_param='Minor radius')
        
    try:        
        pedestal_db['Aspect ratio']['data']=pedestal_db['Magnetic axis R']['data']/pedestal_db['Minor radius']['data']
    except:
        pedestal_db['Aspect ratio']['data']=np.nan
    
    return pedestal_db

def calculate_squareness(pedestal_db):
    
    shot=pedestal_db['Shot data']['shot']
    time_range=pedestal_db['Shot data']['time_range']
    
    radial_coordinates=flap.get_data('NSTX_MDSPlus',
                                     name='\\EFIT02::\\RBDRY',
                                     exp_id=shot,
                                     object_name='RMAXIS').slice_data(slicing={'Time':flap.Intervals(time_range[0],time_range[1])}).data
    
    vertical_coordinates=flap.get_data('NSTX_MDSPlus',
                                       name='\\EFIT02::\\ZBDRY',
                                       exp_id=shot,
                                       object_name='RMAXIS').slice_data(slicing={'Time':flap.Intervals(time_range[0],time_range[1])}).data
    lower_squareness=0
    upper_squareness=0
    
    for ind_time, _ in enumerate(vertical_coordinates[:,0]):
        R=radial_coordinates[ind_time,:]
        z=vertical_coordinates[ind_time,:]
        
        squareness = calculate_plasma_squareness(R,z)
        
        lower_squareness += squareness['lower']
        upper_squareness += squareness['upper']
        
    lower_squareness /= len(vertical_coordinates[:,0])
    upper_squareness /= len(vertical_coordinates[:,0])
    
    pedestal_db['Lower squareness']['data']=lower_squareness
    pedestal_db['Upper squareness']['data']=upper_squareness
    
    return pedestal_db
    
def calculate_pedestal_profile_fitting(pedestal_db,
                                       key_param=None,
                                       plot_profile_fits=False,
                                       pdf_pages=None):
    
    """For now this calculates the average 
    fitted profile and not the fitted average profile"""
    
    shot=pedestal_db['Shot data']['shot']
    time_range=pedestal_db['Shot data']['time_range']
    fit_ion=False
    if 'Temperature' in key_param: 
        param_boolean=[True,False,False,False,False,False]                      #[Temperature, Density, Pressure, Toroidal velocity, Effective charge state, Carbon density]
        param_string='Temperature'
        
    if 'Density' in key_param: 
        param_boolean=[False,True,False,False,False,False]
        param_string='Density'
        
    if 'Pressure' in key_param: 
        param_boolean=[False,False,True,False,False,False]
        param_string='Pressure'
        
    if ' ion ' in key_param:
        param_string+=' ion'
        fit_ion=True
        
    if 'Density carbon' in key_param:
        param_boolean=[False,False,False,False,False,True]
        param_string='Density C6'
        fit_ion=True
    
    if 'Velocity toroidal' in key_param:
        param_boolean=[False,False,False,True,False,False]
        param_string='Velocity toroidal'
        fit_ion=True
        
    if 'Effective charge state' in key_param:
        param_boolean=[False,False,False,False,True,False]
        param_string='Effective charge state'
        fit_ion=True

    try:
        pedestal_db[key_param]['data']
        #If the data is available in the key_param parameter then it does not have to be read again
    except:
        #THESE READ THE ENTIRE SHOT"S PROFILES AND FIT THEM
        #for now I calculate average of fitted profiles and not fitted average profiles
        
        params=get_fit_nstx_thomson_profiles(exp_id=shot,
                                             
                                             electron=not fit_ion,
                                             ion=fit_ion,
                                             
                                             temperature=param_boolean[0],
                                             density=param_boolean[1],
                                             pressure=param_boolean[2],
                                             
                                             spline_data=True,
                                             modified_tanh=True,
                                             
                                             flux_coordinates=True,
                                             flux_range=[0.,1.1],
                                             )        

        ind_time=np.where(np.logical_and(params['time_vec'] > time_range[0],
                                         params['time_vec'] < time_range[1]))[0]
        
        if 'Temperature' in key_param or 'Density' in key_param or 'Pressure' in key_param:
    
            try:
                pedestal_db[param_string+' pedestal location']['data']=np.mean(params['Position'][ind_time])
                pedestal_db[param_string+' pedestal location']['data error']=np.sqrt(np.sum(params['Error']['Position'][ind_time]**2))/len(ind_time)
            except Exception as e:
                print(f'Error in writing {param_string} pedestal location')
                print(e)
                pedestal_db[param_string+' pedestal location']['data error']=np.nan
                pedestal_db[param_string+' pedestal location']['data']=np.nan
                
            try:
                pedestal_db[param_string+' pedestal height']['data']=np.mean(params['Height'][ind_time])
                pedestal_db[param_string+' pedestal height']['data error']=np.sqrt(np.sum(params['Error']['Height'][ind_time]**2))/len(ind_time)
            except Exception as e:
                print(f'Error in writing {param_string} pedestal height')
                print(e)
                pedestal_db[param_string+' pedestal height']['data']=np.nan
                pedestal_db[param_string+' pedestal height']['data error']=np.nan                
                
            try:
                pedestal_db[param_string+' pedestal width']['data']=np.mean(params['Width'][ind_time])
                pedestal_db[param_string+' pedestal width']['data error']=np.sqrt(np.sum(params['Error']['Width'][ind_time]**2))/len(ind_time)
            except Exception as e:
                print(f'Error in writing {param_string} pedestal width')
                print(e)
                pedestal_db[param_string+' pedestal width']['data']=np.nan
                pedestal_db[param_string+' pedestal width']['data error']=np.nan
                
            try:
                pedestal_db[param_string+' SOL']['data']=np.mean(params['SOL avg'][ind_time])
                # pedestal_db[param_string+' SOL']['data error']=np.sqrt(np.var(params['SOL avg'][ind_time]))
                pedestal_db[param_string+' SOL']['data error']=np.sqrt(np.sum(params['Error']['SOL avg'][ind_time]**2))/len(ind_time)
            except Exception as e:
                print(f'Error in writing {param_string} SOL')
                print(e)
                pedestal_db[param_string+' SOL']['data']=np.nan
                pedestal_db[param_string+' SOL']['data error']=np.nan
            
            popt=params['Fit parameters'][ind_time,:]
            pcov=params['Fit parameter covariance'][ind_time,:,:]
            
            def _calculate_mtanh_error(x, popt, pcov, eps=1e-8, yerr=None):
                
                grad = np.zeros(len(popt))
                
                for ind_popt, curr_popt in enumerate(popt):
                    dp = np.zeros(len(popt))
                    dp[ind_popt] = curr_popt*eps
                    gradient=(mtanh_function(x, *(popt+dp)) - 
                              mtanh_function(x, *(popt-dp))) / (2*dp[ind_popt])
                    grad[ind_popt] = gradient
                    
                # propagate parameter uncertainty
                var_params = grad @ pcov @ grad
                # include measurement error if given
                if yerr is not None:
                    var_total = var_params + yerr**2
                else:
                    var_total = var_params
                return mtanh_function(x, *popt), np.sqrt(var_total)
            
            full_data=np.zeros([len(ind_time),4])
            full_data_error=np.zeros([len(ind_time),4])
            positions=np.asarray([0.,0.9,0.95,1.0])
            try:
            # if True:
                for ind_pos,pos_curr in enumerate(positions):
                    for ind_ind_time,_ in enumerate(ind_time):
                        popt_curr=popt[ind_ind_time,:]
                        pcov_curr=pcov[ind_ind_time,:,:]
                        full_data_curr, full_data_error_curr = _calculate_mtanh_error(pos_curr,popt_curr,pcov_curr)
                        full_data[ind_ind_time,ind_pos]=full_data_curr
                        full_data_error[ind_ind_time,ind_pos]=full_data_error_curr
                        
            except Exception as e:
                print(f'Error in writing {param_string} in core')
                print(e)
                full_data[:,:]=np.nan
                full_data_error[:,:]=np.nan
                

            pedestal_db[param_string+' core']['data']=np.mean(full_data[:,0])
            pedestal_db[param_string+' core']['data error']=np.sqrt(np.sum(full_data_error[:,0]**2))/len(full_data_error[:,0])

            pedestal_db[param_string+' 90']['data']=np.mean(full_data[:,1])
            pedestal_db[param_string+' 90']['data error']=np.sqrt(np.sum(full_data_error[:,1]**2))/len(full_data_error[:,1])

            pedestal_db[param_string+' 95']['data']=np.mean(full_data[:,2])
            pedestal_db[param_string+' 95']['data error']=np.sqrt(np.sum(full_data_error[:,2]**2))/len(full_data_error[:,2])

            pedestal_db[param_string+' separatrix']['data']=np.mean(full_data[:,3])
            pedestal_db[param_string+' separatrix']['data error']=np.sqrt(np.sum(full_data_error[:,3]**2))/len(full_data_error[:,3])
            
            if plot_profile_fits and pdf_pages is not None:

                # print(params['Data'].shape,params['Flux r'].shape)
                # print(param_string, params['time_vec'].shape)
                for ind_ind_time in ind_time:
                    # print(ind_time)
                    # print(params['Error'].keys())
                    fix,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
                    try:
                        ax.errorbar(params['Flux r'][:,ind_ind_time],
                                    params['Data'][:,ind_ind_time],
                                    yerr=params['Error']['Data'][:,ind_ind_time])
                    except:
                        ax.scatter(params['Flux r'][:,ind_ind_time],
                                   params['Data'][:,ind_ind_time],)
                        
                    # print(params['Flux r'][:,ind_ind_time])
                    # print(params['Fit parameters'].shape)#[ind_ind_time,:])
                    
                    ax.plot(params['Flux r'][:,ind_ind_time],
                            mtanh_function(params['Flux r'][:,ind_ind_time],
                                           *params['Fit parameters'][ind_ind_time,:]),
                            color='tab:orange')
                    
                    ax.errorbar(np.asarray([0.,0.9,0.95,1.0]),
                               full_data[ind_ind_time-ind_time[0],:],
                               full_data_error[ind_ind_time-ind_time[0],:],
                               fmt='o',
                               color='tab:red')
                    ax.set_xlabel('$\\psi_{N}$')
                    ax.set_ylabel(param_string)
                    plt.tight_layout(pad=0.1)
                    pdf_pages.savefig()
                    #These are without the errorbars for better visibility of some erroneous data
                    fix,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54))

                    ax.scatter(params['Flux r'][:,ind_ind_time],
                               params['Data'][:,ind_ind_time],)
                        
                    # print(params['Flux r'][:,ind_ind_time])
                    # print(params['Fit parameters'].shape)#[ind_ind_time,:])
                    
                    ax.plot(params['Flux r'][:,ind_ind_time],
                            mtanh_function(params['Flux r'][:,ind_ind_time],
                                           *params['Fit parameters'][ind_ind_time,:]),
                            color='tab:orange')
                    
                    ax.scatter(np.asarray([0.,0.9,0.95,1.0]),
                               full_data[ind_ind_time-ind_time[0],:],
                               color='tab:red')
                    ax.set_xlabel('$\\psi_{N}$')
                    ax.set_ylabel(param_string)
                    plt.tight_layout(pad=0.1)
                    pdf_pages.savefig()
                    
        elif 'Effective charge state' or 'Velocity toroidal' in key_param:
            # try:
            if True:
                curr_data=params['Data'][:,ind_time]
                curr_data_error=params['Error']['Data'][:,ind_time]
                
                curr_psi_coord=params['Flux r'][:,ind_time]
                
                core_psi_range=[0,0.2]
                pedestal_psi_range=[0.85,0.95]

                ind_psi_core=np.where(np.logical_and(curr_psi_coord > core_psi_range[0],
                                                     curr_psi_coord < core_psi_range[1]))
                
                ind_psi_ped=np.where(np.logical_and(curr_psi_coord > pedestal_psi_range[0],
                                                     curr_psi_coord < pedestal_psi_range[1]))

                data_core=curr_data[ind_psi_core]
                data_core_error=curr_data_error[ind_psi_core]
                pedestal_db[param_string+' core']['data']=np.mean(data_core)
                pedestal_db[param_string+' core']['data error']=np.sqrt(np.sum(data_core_error**2))/len(data_core_error)
                
                data_ped=curr_data[ind_psi_ped]
                data_ped_error=curr_data_error[ind_psi_ped]
                pedestal_db[param_string+' pedestal']['data']=np.mean(data_ped)
                pedestal_db[param_string+' pedestal']['data error']=np.sqrt(np.sum(data_ped_error**2))/len(data_ped_error)
                
            # except Exception as e:
            #     print(f'Error in writing {param_string} at core or pedestal')
            #     print(e)
                
            #     pedestal_db[param_string+' pedestal']['data']=np.nan
            #     pedestal_db[param_string+' pedestal']['data error']=np.nan
            #     pedestal_db[param_string+' core']['data']=np.nan
            #     pedestal_db[param_string+' core']['data error']=np.nan
                

    return pedestal_db

def read_oak_jrt_db():
    df=pandas.read_csv(wd+'/db/2022_JRT_non-ELMING_database_NSTX_only.csv')
    return {'shot':np.asarray(df['shot']),
            'time':(np.asarray(df['start_time'])+np.asarray(df['end_time']))/2}
