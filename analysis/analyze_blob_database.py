#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:52:23 2023

@author: mlampert
"""
#Core modules
import os
import copy
import time as time_mod
import pickle
import warnings
warnings.filterwarnings("ignore")

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')

from flap_nstx.analysis import read_mean_blob_results, read_blob_data,read_all_plasma_data,read_blob_results
from flap_nstx.analysis import read_blob_database, read_blob_elm_database, read_blob_lh_mode_database
from flap_nstx.analysis import read_plasma_parameters, return_interesting

from flap_nstx.gpi import transform_frames_to_structures
from flap_nstx.gpi import read_analyzed_keys
from flap_nstx.tools import plot_pearson_matrix, calculate_corr_acceptance_levels
from flap_nstx.tools import correlation, mutual_information

import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas
import ppscore as pps
import seaborn as sns
from scipy.stats import linregress

#Plot settings for publications
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/plots'


def calculate_all_blob_results(time_range_around_peak=5e-3,
                               min_structure_lifetime=20,
                               str_finding_method='watershed',
                               plot=False,
                               pdf=False,
                               nocalc=False,
                               recalc_tracking=False,
                               test=False,
                               calculate_for_lh_study=False,
                               download_data_only=False,
                               ):
    
    if not calculate_for_lh_study:
        blob_database=read_blob_database(time_range_around_peak=time_range_around_peak)

        ncalc=len(blob_database['shot'])
        for ind in range(ncalc):
            start_time=time_mod.time()
            blob_time=blob_database['time'][ind]
            #if blob_database['shot'][ind] == 142270 or blob_database['shot'][ind] == 142279:
            read_blob_results(blob_database['shot'][ind],
                              [blob_time-time_range_around_peak,
                               blob_time+time_range_around_peak],
                              #calc_only=True,
                              nocalc=nocalc,
                              recalc_tracking=recalc_tracking,
                              min_structure_lifetime=min_structure_lifetime,
                              str_finding_method=str_finding_method,
                              )
    
            elapsed_time=time_mod.time()-start_time
            remaining_time=elapsed_time*(ncalc-ind-1)
            print('Remaining time from the calculation: '+str(remaining_time/3600.)+' hours.')
            flap.delete_data_object('*')
    else:
        l_mode_database=read_blob_lh_mode_database(l_mode=True, 
                                                   filter_lh_transition=True, 
                                                   filter_elms=False, #Filtering will be done during the analysis based on the ELM database established during the ELM work
                                                   filtered_blob_db=False,
                                                   time_range_around_peak=time_range_around_peak)
        
        h_mode_database=read_blob_lh_mode_database(h_mode=True, 
                                                   filter_lh_transition=True, 
                                                   filter_elms=False, 
                                                   filtered_blob_db=False,
                                                   time_range_around_peak=time_range_around_peak)
        
        ncalc=len(l_mode_database['shot'])+len(h_mode_database['shot'])
        
#        for database in [l_mode_database,h_mode_database]:
        for database in [h_mode_database]:
            for ind, shot in enumerate(database['shot']):
                
                if shot < 138113: continue
            
                start_time=time_mod.time()
                avg_time=np.mean(database['time'][:,ind])
                if avg_time > 10:
                    multiplier=1e-3
                else:
                    multiplier=1
                if download_data_only:
                    print(f'Downloading shot # {shot}')
                    try:
                        d=flap.get_data('NSTX_GPI',
                                        exp_id=int(shot),
                                        name='',
                                        object_name='GPI')
                    except:
                        print(f'Failed to download shot # {shot}')
                else:
                    print(f'Calculating shot # {shot} at time {avg_time}')
                    read_blob_results(int(shot),
                                    [database['time'][0,ind]*multiplier,
                                    database['time'][1,ind]*multiplier],
                                    #calc_only=True,
                                    nocalc=nocalc,
                                    recalc_tracking=recalc_tracking,
                                    min_structure_lifetime=min_structure_lifetime,
                                    str_finding_method=str_finding_method,
                                    calculate_only=True,
                                    )
            
                elapsed_time=time_mod.time()-start_time
                remaining_time=elapsed_time*(ncalc-ind-1)
                print('Remaining time from the calculation: '+str(remaining_time/3600.)+' hours.')
                flap.delete_data_object('*')
            
            


def calculate_blob_parameter_histograms(time_range_around_peak=5e-3,
                                        pdf=False,
                                        pdf_filename=None,
                                        plot=True,
                                        plot_for_publication=False,
                                        save_data_into_txt=False,
                                        calc_mean_distribution=False,
                                        nocalc=True,
                                        recalc_tracking=False,
                                        min_structure_lifetime=20,
                                        str_finding_method='watershed',
                                        analyze_h_mode_only=False,
                                        analyze_l_mode_only=False,
                                        filtered_blob_db=False, #Filters the database for ELMs based on the ELM db. Half of the shots are removed. Data should be filtered after the shot.
                                        plot_LH_diff=False,
                                        save_data_for_publication=False,
                                        ):
    import matplotlib
    if pdf:
        matplotlib.use('agg')
    else:
        matplotlib.use('qt5agg')

    if pdf_filename is None:
        if calc_mean_distribution:
            pdf_filename=wd+fig_dir+'/blob_database_parameter_histograms_mean_'+str_finding_method+'.pdf'
        else:
            pdf_filename=wd+fig_dir+'/blob_database_parameter_histograms_nomean_'+str_finding_method+'.pdf'

    if calc_mean_distribution:
        pickle_filename=wd+'/processed_data/blob_database_full_data_mean_'+str_finding_method
    else:
        pickle_filename=wd+'/processed_data/blob_database_full_data_nomean_'+str_finding_method
        
    if analyze_l_mode_only:
        pickle_filename += '_l_mode'
    elif analyze_h_mode_only:
        pickle_filename += '_h_mode'
        
    pickle_filename += '.pickle'
    
    if not analyze_h_mode_only and not analyze_l_mode_only:
        blob_database=read_blob_database(time_range_around_peak=time_range_around_peak)
    elif analyze_h_mode_only:
        blob_database=read_blob_lh_mode_database(h_mode=True,
                                                 time_range_around_peak=time_range_around_peak,
                                                 filtered_blob_db=filtered_blob_db)
    elif analyze_l_mode_only:
        blob_database=read_blob_lh_mode_database(l_mode=True,
                                                 time_range_around_peak=time_range_around_peak,
                                                 filtered_blob_db=filtered_blob_db)
        
    analyzed_keys=read_analyzed_keys()
    
    additional_diff_keys=['Convexity', 'Solidity', 'Roundness', 'Total curvature',
                          'Total bending energy','Area','Elongation']

    ncalc=len(blob_database['shot'])

    full_data={}

    for key in analyzed_keys:
        full_data[key]=[]
    for key in additional_diff_keys:
        full_data[key+' diff']=[]

    if not os.path.exists(pickle_filename) or not nocalc:
        n_str=0
        for ind in range(ncalc):
            blob_time=blob_database['time'][ind]
            start_time=time_mod.time()
            blob_results=read_blob_results(int(blob_database['shot'][ind]),
                                           [blob_time-time_range_around_peak,
                                            blob_time+time_range_around_peak],
                                           nocalc=True,
                                           recalc_tracking=recalc_tracking,
                                           min_structure_lifetime=min_structure_lifetime,
                                           str_finding_method=str_finding_method,
                                           )

            flap.delete_data_object('*')
            str_by_str=transform_frames_to_structures(blob_results)

            for ind_str, structure in enumerate(str_by_str):
                n_str+=1
                for key in analyzed_keys:

                    if key in          ['Velocity radial COG', 'Velocity poloidal COG',
                                       'Velocity radial centroid', 'Velocity poloidal centroid',
                                       'Velocity radial position', 'Velocity poloidal position',
                                       'Expansion fraction area', 'Expansion fraction axes',
                                       'Angular velocity angle', 'Angular velocity ALI']:
                        try:
                            full_data[key]=np.append(full_data[key],
                                                     structure[key])
                            print(structure[key])
                        except:
                            print(key)
                    else:

                        try:
                            full_data[key]=np.append(full_data[key],
                                                     structure[key][1:])
                            # if key == 'Angle of least inertia':
                            #     print(full_data[key])
                        except:
                            print(key)
                for key in additional_diff_keys:
                    try:
                        full_data[key+' diff']=np.append(full_data[key+' diff'],
                                                         (np.asarray(structure[key])[1:] -
                                                          np.asarray(structure[key])[0:-1]))
                    except:
                        print(key)
            remaining_time=(time_mod.time()-start_time)*(ncalc-ind-1)

            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            seconds = int(remaining_time % 60)

            print('Remaining time from the calculation: '+f"{hours}h {minutes:02}min {seconds:02}sec")
        pickle.dump(full_data,open(pickle_filename,'wb'))
     
        print('n_str:',n_str)            
    else:
        full_data=pickle.load(open(pickle_filename,'rb'))


    for key in additional_diff_keys:
        analyzed_keys.append(key+' diff')


    if plot:
        
        
        ranges={'Position radial':[1.4,1.6],
                'Position poloidal': [0.15, 0.35],
                'Area':[0,0.006],
                'Velocity poloidal position':[-10e3,10e3],
                'Velocity radial position':[-3e3,3e3],
                'Expansion fraction area':[0.75,1.25],
                'Expansion fraction axis':[0.75,1.25],
                'Convexity':[0.9,1.0],
                'Solidity':[0.75,1.0],
                'Total curvature':[0.9,1.0],
                'Total bending energy':[0e8,1.5e8],
                'Convexity diff':[-0.01,0.01],
                'Solidity diff':[-0.25,0.25],
                'Total curvature diff':[-0.05,0.05],
                'Total bending energy diff':[-0.3e8,0.3e8],
                'Area diff':[-0.0015,0.0015],
                'Elongation diff':[-0.075,0.075],
                'Angular velocity angle':[-250e3,250e3]
                }
        import scipy
        
        if plot_for_publication:
                      
            multiplier={'Area':1e4,
                        'Area diff':1e4,
                        'Angle':1,
                        'Angular velocity angle':1e-3,
                        'Roundness':1, 
                        'Roundness diff':1e3,
                        'Total curvature':1,
                        'Total curvature diff':1e3}
            
            xlabel={'Area':['Area','[$\\rm cm^2$]'],
                   'Area diff':['$\\rm\\Delta$Area','[$\\rm cm^2$]'],
                   'Angle':['Angle','[rad]'],
                   'Angular velocity angle':['$\\rm\\omega$','[krad/s]'],
                   'Roundness':['Roundness','[a.u.]'], 
                   'Roundness diff':['$\\rm\\Delta$Roundness','[a.u.]'],
                   'Total curvature': ['Curvature','[a.u.]'],
                   'Total curvature diff': ['$\\rm\\Delta$Curvature','[a.u.]']}
            
            if not plot_LH_diff:
                if analyze_l_mode_only:
                    pdf_page=PdfPages(wd+'/plots/8hist_blob_db_LT'+str(min_structure_lifetime)+'_'+str_finding_method+'_L_mode.pdf')
                elif analyze_h_mode_only:
                    pdf_page=PdfPages(wd+'/plots/8hist_blob_db_LT'+str(min_structure_lifetime)+'_'+str_finding_method+'_H_mode.pdf')
                else:
                    pdf_page=PdfPages(wd+'/plots/8hist_blob_db_LT'+str(min_structure_lifetime)+'_'+str_finding_method+'.pdf')
                    
                fig,axes=plt.subplots(4,2,figsize=(8.5/2.54,17/2.54))
                for ind, key in enumerate(['Area','Area diff',
                                           'Angle','Angular velocity angle',
                                           'Roundness', 'Roundness diff',
                                           'Total curvature','Total curvature diff',
                                           ]):
                    
                    labels=['a','b','c','d','e','f','g','h']
                    full_data[key]=full_data[key][~np.isnan(full_data[key])]*multiplier[key]
                    ax=axes[ind//2,np.mod(ind,2)]
                    # try:
                    print(key,'skewness',scipy.stats.skew(full_data[key]))
                    print(key,'kurtosis',scipy.stats.kurtosis(full_data[key]))
                    if key == 'Angle':
                        full_data[key]=np.mod(np.real(full_data[key]),
                                              np.pi)
                    if key in ranges.keys():
                        n, bins, patches=ax.hist(np.real(full_data[key]),
                                                 bins=51,
                                                 weights=np.ones_like(full_data[key])/len(full_data[key]),
                                                 range=np.asarray(ranges[key])*multiplier[key])
                    else:
                        n, bins, patches=ax.hist(np.real(full_data[key]),
                                                 bins=51,
                                                 weights=np.ones_like(full_data[key])/len(full_data[key]),)
                        
                    if save_data_for_publication:
                        filename=wd+'/'+labels[ind]+'_db_histogram_'+str(key)+'.txt'
                        file1=open(filename, 'w+')
                        for i in range(len(n)):
                            file1.write(str((bins[1:]+bins[:-1])[i]/2)+'\t'+str(n[i])+'\n')
                        file1.close()
                        
                    plt.locator_params(axis='y', nbins=5)
                    ax.set_xlabel(xlabel[key][0]+' '+xlabel[key][1])
                    ax.set_ylabel('Relative frequency')
                    ax.set_title('Histogram of \n '+xlabel[key][0])
                    ax.text(-0.4, 1.1, '('+labels[ind]+')', transform=ax.transAxes, size=9)
                    if np.mod(ind,2)==1:
                        ax.axvline(x=0,color='red')
                    # if key in ranges.keys():
                    #     ax.set_xlim(ranges[key])
                plt.tight_layout(pad=0.1)
                pdf_page.savefig()
                pdf_page.close()
            else:
                l_mode_filename=wd+'/processed_data/blob_database_full_data_nomean_'+str_finding_method+'_l_mode.pickle'
                full_data_l_mode=pickle.load(open(l_mode_filename, 'rb'))
                h_mode_filename=wd+'/processed_data/blob_database_full_data_nomean_'+str_finding_method+'_h_mode.pickle'
                full_data_h_mode=pickle.load(open(h_mode_filename, 'rb'))
                
                pdf_page=PdfPages(wd+'/plots/8hist_blob_db_LT'+str(min_structure_lifetime)+'_'+str_finding_method+'LH_diff.pdf')
                
                fig,axes=plt.subplots(4,2,figsize=(8.5/2.54,17/2.54))

                
                for ind, key in enumerate(['Area','Area diff',
                                           'Angle','Angular velocity angle',
                                           'Roundness', 'Roundness diff',
                                           'Total curvature','Total curvature diff',
                                           ]):
                    
                    labels=['a','b','c','d','e','f','g','h']
                    full_data_l_mode[key]=full_data_l_mode[key][~np.isnan(full_data_l_mode[key])]*multiplier[key]
                    full_data_h_mode[key]=full_data_h_mode[key][~np.isnan(full_data_h_mode[key])]*multiplier[key]
                    
                    ax=axes[ind//2,np.mod(ind,2)]
                    # try:
                    # print(key,'average lmode',np.mean(full_data_l_mode[key]))
                    # print(key,'sigma lmode', np.sqrt(np.var(full_data_l_mode[key])))
                    # print(key,'skewness lmode',scipy.stats.skew(full_data_l_mode[key]))
                    # print(key,'kurtosis lmode',scipy.stats.kurtosis(full_data_l_mode[key]))
                    
                    # print(key,'average hmode',np.mean(full_data_h_mode[key]))
                    # print(key,'sigma hmode', np.sqrt(np.var(full_data_h_mode[key])))
                    # print(key,'skewness hmode',scipy.stats.skew(full_data_h_mode[key]))
                    # print(key,'kurtosis hmode',scipy.stats.kurtosis(full_data_h_mode[key]))
                    
                    if key == 'Angle':
                        full_data_l_mode[key]=np.mod(np.real(full_data_l_mode[key]),
                                                     np.pi)
                        full_data_h_mode[key]=np.mod(np.real(full_data_h_mode[key]),
                                                     np.pi)
                    
                    l_data = np.array(full_data_l_mode[key])
                    h_data = np.array(full_data_h_mode[key])
                    
                    values = [
                        np.mean(l_data), 
                        np.mean(h_data),
                        np.sqrt(np.var(l_data)), 
                        np.sqrt(np.var(h_data)),
                        scipy.stats.skew(l_data),
                        scipy.stats.skew(h_data),
                        scipy.stats.kurtosis(l_data),
                        scipy.stats.kurtosis(h_data)
                    ]
                    
                    # Format with 2 decimal places and join with "&"
                    formatted = " & ".join(f"{v:.3f}" for v in np.real(values))
                    print(f"{xlabel[key][0]} {xlabel[key][1]} & {formatted} \\\\")
                    
                    
                        
                    if key in ranges.keys():
                        ax.hist(np.real(full_data_l_mode[key]),
                                bins=51,
                                weights=np.ones_like(full_data_l_mode[key])/len(full_data_l_mode[key]),
                                range=np.asarray(ranges[key])*multiplier[key],
                                alpha=0.5,
                                label='L mode')
                        ax.hist(np.real(full_data_h_mode[key]),
                                bins=51,
                                weights=np.ones_like(full_data_h_mode[key])/len(full_data_h_mode[key]),
                                range=np.asarray(ranges[key])*multiplier[key],
                                alpha=0.5,
                                label='H mode')
                    else:
                        ax.hist(np.real(full_data_l_mode[key]),
                                bins=51,
                                weights=np.ones_like(full_data_l_mode[key])/len(full_data_l_mode[key]),
                                alpha=0.5,
                                label='L mode')
                        ax.hist(np.real(full_data_h_mode[key]),
                                bins=51,
                                weights=np.ones_like(full_data_h_mode[key])/len(full_data_h_mode[key]),
                                alpha=0.5,
                                label='H mode')
                        
                    plt.locator_params(axis='y', nbins=5)
                    ax.set_xlabel(xlabel[key][0]+' '+xlabel[key][1])
                    ax.set_ylabel('Relative frequency')
                    ax.set_title('Histogram of \n '+xlabel[key][0])
                    ax.text(-0.4, 1.1, '('+labels[ind]+')', transform=ax.transAxes, size=9)
                    ax.legend(fontsize=5)
                    if np.mod(ind,2)==1:
                        ax.axvline(x=0,color='red')
                    # if key in ranges.keys():
                    #     ax.set_xlim(ranges[key])
                plt.tight_layout(pad=0.1)
                pdf_page.savefig()
                pdf_page.close()
                
        else:
            if pdf:
                pdf_page=PdfPages(pdf_filename)
            for key in analyzed_keys:
                full_data[key]=full_data[key][~np.isnan(full_data[key])]
                try:
                    fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
                    ax.hist(full_data[key],
                            bins=51)
                    ax.set_xlabel(key+' bins')
                    ax.set_ylabel('Relative frequency')
                    ax.set_title('Histogram of '+key)
                    if key in ranges.keys():
                        ax.set_xlim(ranges[key])
                except:
                    print('Failed to plot '+ key)
                if pdf:
                    pdf_page.savefig()
            if pdf:
                pdf_page.close()

    return full_data

def calculate_blob_parameter_histograms2(time_range_around_peak=5e-3,
                                        pdf=False,
                                        pdf_filename=None,
                                        plot=True,
                                        plot_for_publication=False,
                                        save_data_into_txt=False,
                                        calc_mean_distribution=False,
                                        nocalc=True,
                                        recalc_tracking=False,
                                        min_structure_lifetime=20,
                                        str_finding_method='watershed',
                                        analyze_h_mode_only=False,
                                        analyze_l_mode_only=False,
                                        save_data_for_publication=False,
                                        ):
    """
    Just for reading the data, should be merged with read data and the indices
    of differential and normal data would need to be handled properly.
    """
    
    import matplotlib
    if pdf:
        matplotlib.use('agg')
    else:
        matplotlib.use('qt5agg')

    if pdf_filename is None:
        if calc_mean_distribution:
            pdf_filename=wd+fig_dir+'/blob_database_parameter_histograms_mean_'+str_finding_method+'.pdf'
        else:
            pdf_filename=wd+fig_dir+'/blob_database_parameter_histograms_nomean_'+str_finding_method+'.pdf'

    if calc_mean_distribution:
        pickle_filename=wd+'/processed_data/blob_database_full_data_mean_'+str_finding_method+'.pickle'
    else:
        pickle_filename=wd+'/processed_data/blob_database_full_data_nomean_'+str_finding_method+'.pickle'

    if not analyze_h_mode_only and not analyze_l_mode_only:
        blob_database=read_blob_database(time_range_around_peak=time_range_around_peak)
    elif analyze_h_mode_only:
        blob_database=read_blob_lh_mode_database(h_mode=True,
                                                 time_range_around_peak=time_range_around_peak)
    elif analyze_l_mode_only:
        blob_database=read_blob_lh_mode_database(l_mode=True,
                                                 time_range_around_peak=time_range_around_peak)
        
    analyzed_keys=read_analyzed_keys()
    
    additional_diff_keys=['Convexity', 'Solidity', 'Roundness', 'Total curvature',
                          'Total bending energy','Area','Elongation']

    ncalc=len(blob_database['shot'])

    full_data={}

    for key in analyzed_keys:
        full_data[key]=[]
    for key in additional_diff_keys:
        full_data[key+' diff']=[]

    if not os.path.exists(pickle_filename) or not nocalc or analyze_l_mode_only or analyze_h_mode_only:
        n_str=0
        for ind in range(ncalc):
            blob_time=blob_database['time'][ind]
            start_time=time_mod.time()
            blob_results=read_blob_results(blob_database['shot'][ind],
                                           [blob_time-time_range_around_peak,
                                            blob_time+time_range_around_peak],
                                           nocalc=True,
                                           recalc_tracking=recalc_tracking,
                                           min_structure_lifetime=min_structure_lifetime,
                                           str_finding_method=str_finding_method,
                                           )

            flap.delete_data_object('*')
            str_by_str=transform_frames_to_structures(blob_results)

            for ind_str, structure in enumerate(str_by_str):
                n_str+=1
                for key in analyzed_keys:
                        if key in          ['Velocity radial COG', 'Velocity poloidal COG',
                                           'Velocity radial centroid', 'Velocity poloidal centroid',
                                           'Velocity radial position', 'Velocity poloidal position',
                                           'Expansion fraction area', 'Expansion fraction axes',
                                           'Angular velocity angle', 'Angular velocity ALI']:
                            full_data[key]=np.append(full_data[key],structure[key])
                        else:
                            full_data[key]=np.append(full_data[key],
                                                     structure[key][1:])

                for key in additional_diff_keys:
                    diff=(np.asarray(structure[key])[1:] - np.asarray(structure[key])[0:-1])
                    full_data[key+' diff']=np.append(full_data[key+' diff'],diff)
            remaining_time=(time_mod.time()-start_time)*(ncalc-ind-1)

            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            seconds = int(remaining_time % 60)

            print('Remaining time from the calculation: '+f"{hours}h {minutes:02}min {seconds:02}sec")
        print('n_str:',n_str)
        if not analyze_h_mode_only and not analyze_l_mode_only:
            pickle.dump(full_data,open(pickle_filename,'wb'))
    else:
        full_data=pickle.load(open(pickle_filename,'rb'))

    return full_data



def calculate_blob_blob_parameter_correlation_matrix(threshold_corr=False,
                                                     pdf=True,
                                                     pdf_filename=None,
                                                     calc_mean_distribution=False,
                                                     plot_interesting_only=False,
                                                     recalc_tracking=False,
                                                     str_finding_method='watershed',
                                                     nocalc=True,
                                                     averaging='no',
                                                     average=['avg','avg'],
                                                     fix_angle_for_correlation=True,
                                                     min_structure_lifetime=10,
                                                     analyze_h_mode_only=False,
                                                     analyze_l_mode_only=False,
                                                     analyze_lh_difference=False,
                                                     save_data_for_publication=False,
                                                     ):
    if analyze_h_mode_only:
        plasma_mode='h_mode'
    elif analyze_l_mode_only:
        plasma_mode='l_mode'
    elif analyze_lh_difference:
        plasma_mode='lh_diff'
    else:
        plasma_mode=''
        
    if pdf_filename is None:
        if averaging == 'no':
            pdf_filename=wd+'/plots/correlation_matrix_blob_blob_'+str_finding_method+'_full_'+plasma_mode+'.pdf'
        else:
            pdf_filename=wd+'/plots/correlation_matrix_blob_blob_'+str_finding_method+'_'+averaging+'_'+average[0]+'_'+average[1]+'_'+plasma_mode+'.pdf'

    if not analyze_lh_difference:
        full_data_1=calculate_blob_parameter_histograms2(calc_mean_distribution=calc_mean_distribution,
                                                         nocalc=nocalc,
                                                         plot=False,
                                                         recalc_tracking=recalc_tracking,
                                                         str_finding_method=str_finding_method,
                                                         analyze_h_mode_only=analyze_h_mode_only,
                                                         analyze_l_mode_only=analyze_l_mode_only
                                                         )
    
        full_data_2=full_data_1
    
        if not plot_interesting_only:
            #analyzed_keys=read_analyzed_keys()
            analyzed_keys=full_data_1.keys()
            # additional_diff_keys=['Convexity', 'Solidity', 'Roundness', 'Total curvature',
            #                       'Total bending energy','Area','Elongation']
    
            # for key in additional_diff_keys:
            #     analyzed_keys.append(key+' diff')
        else:
            interesting_key_pairs=[('Area','Convexity'),
                                    ('Size radial','Convexity'),
                                    ('Elongation','Roundness'),
                                    ('Position radial','Velocity radial position'),
                                    ('Axes length minor','Velocity radial position'),
                                    ('Axes length major','Velocity radial position'),
                                    #('Expansion fraction area','Roundness diff'),
                                    ('Area diff','Roundness diff'),
                                    ('Position radial','Axes length major'),
                                    ]
            analyzed_keys=list(np.unique(interesting_key_pairs))
            
            analyzed_keys=['Area',
                           'Area diff',
                           'Axes length major',
                           'Axes length minor',
                           'Convexity',
                           'Elongation',
                           'Position radial',
                           'Roundness',
                           'Roundness diff',
                           'Size radial',
                           'Velocity radial position']
    
        gpi_labels=analyzed_keys
    
        correlation_matrix=np.zeros([len(analyzed_keys),len(analyzed_keys)])
        
        for ind1,key1 in enumerate(analyzed_keys):
            ind_nan1=~np.isnan(full_data_1[key1])
            for ind2,key2 in enumerate(analyzed_keys):
                try:
    
                    if key1 == 'Angle' or key1 == 'Angle of least inertia':
                        full_data_1[key1]=np.mod(np.real(full_data_1[key1]), np.pi/2)
                        
                    if key2 == 'Angle' or key2 == 'Angle of least inertia':
                        full_data_2[key2]=np.mod(np.real(full_data_2[key2]), np.pi/2)
                        
                    ind_nan2 = ~np.isnan(full_data_2[key2])
                    ind_nan = np.logical_and(ind_nan1,ind_nan2)
                    
                    data1 = np.real(full_data_1[key1][ind_nan])
                    data2 = np.real(full_data_2[key2][ind_nan])
    
                    # ind_comp1=~np.iscomplex(full_data_1[key1][ind_nan])
                    # ind_comp2=~np.iscomplex(full_data_2[key2][ind_nan])
                    # ind_comp=np.logical_and(ind_comp1,ind_comp2)
    
                    # data1 = np.real(full_data_1[key1][ind_nan][ind_comp])
                    # data2 = np.real(full_data_2[key2][ind_nan][ind_comp])
                    
                    data1 -= np.mean(data1)
                    data2 -= np.mean(data2)
                    correlation_matrix[ind2,ind1] = np.sum(data1*data2)/(np.sqrt(np.sum(data1**2) * np.sum(data2**2)))
                    print(key1,';',key2,correlation_matrix[ind2,ind1])
                    
                except Exception as e:
                    print(key1, key2)
                    print(e)
    else:
        full_data_l_mode=calculate_blob_parameter_histograms2(calc_mean_distribution=calc_mean_distribution,
                                                         nocalc=nocalc,
                                                         plot=False,
                                                         recalc_tracking=recalc_tracking,
                                                         str_finding_method=str_finding_method,
                                                         analyze_h_mode_only=False,
                                                         analyze_l_mode_only=True
                                                         )
        full_data_h_mode=calculate_blob_parameter_histograms2(calc_mean_distribution=calc_mean_distribution,
                                                         nocalc=nocalc,
                                                         plot=False,
                                                         recalc_tracking=recalc_tracking,
                                                         str_finding_method=str_finding_method,
                                                         analyze_h_mode_only=True,
                                                         analyze_l_mode_only=False
                                                         )
        interesting_key_pairs=[('Area','Convexity'),
                                ('Size radial','Convexity'),
                                ('Elongation','Roundness'),
                                ('Position radial','Velocity radial position'),
                                ('Axes length minor','Velocity radial position'),
                                ('Axes length major','Velocity radial position'),
                                #('Expansion fraction area','Roundness diff'),
                                ('Area diff','Roundness diff'),
                                ('Position radial','Axes length major'),
                                ]
        analyzed_keys=list(np.unique(interesting_key_pairs))
        
        analyzed_keys=['Area',
                       'Area diff',
                       'Axes length major',
                       'Axes length minor',
                       'Convexity',
                       'Elongation',
                       'Position radial',
                       'Roundness',
                       'Roundness diff',
                       'Size radial',
                       'Velocity radial position']
    
        gpi_labels=analyzed_keys
    
        correlation_matrix_l_mode=np.zeros([len(analyzed_keys),len(analyzed_keys)])
        
        full_data_1=full_data_l_mode
        full_data_2=full_data_1
        
        for ind1,key1 in enumerate(analyzed_keys):
            ind_nan1=~np.isnan(full_data_l_mode[key1])
            for ind2,key2 in enumerate(analyzed_keys):
                try:
    
                    if key1 == 'Angle' or key1 == 'Angle of least inertia':
                        full_data_1[key1]=np.mod(np.real(full_data_1[key1]), np.pi/2)
                        
                    if key2 == 'Angle' or key2 == 'Angle of least inertia':
                        full_data_2[key2]=np.mod(np.real(full_data_2[key2]), np.pi/2)
                        
                    ind_nan2 = ~np.isnan(full_data_2[key2])
                    ind_nan = np.logical_and(ind_nan1,ind_nan2)
                    
                    data1 = np.real(full_data_1[key1][ind_nan])
                    data2 = np.real(full_data_2[key2][ind_nan])
                    
                    data1 -= np.mean(data1)
                    data2 -= np.mean(data2)
                    correlation_matrix_l_mode[ind2,ind1] = np.sum(data1*data2)/(np.sqrt(np.sum(data1**2) * np.sum(data2**2)))
                    print(key1,';',key2,correlation_matrix_l_mode[ind2,ind1])
                    
                except Exception as e:
                    print(key1, key2)
                    print(e)
        
        correlation_matrix_h_mode=np.zeros([len(analyzed_keys),len(analyzed_keys)])
        full_data_1=full_data_h_mode
        full_data_2=full_data_1
        
        for ind1,key1 in enumerate(analyzed_keys):
            ind_nan1=~np.isnan(full_data_1[key1])
            for ind2,key2 in enumerate(analyzed_keys):
                try:
    
                    if key1 == 'Angle' or key1 == 'Angle of least inertia':
                        full_data_1[key1]=np.mod(np.real(full_data_1[key1]), np.pi/2)
                        
                    if key2 == 'Angle' or key2 == 'Angle of least inertia':
                        full_data_2[key2]=np.mod(np.real(full_data_2[key2]), np.pi/2)
                        
                    ind_nan2 = ~np.isnan(full_data_2[key2])
                    ind_nan = np.logical_and(ind_nan1,ind_nan2)
                    
                    data1 = np.real(full_data_1[key1][ind_nan])
                    data2 = np.real(full_data_2[key2][ind_nan])
                    
                    data1 -= np.mean(data1)
                    data2 -= np.mean(data2)
                    correlation_matrix_h_mode[ind2,ind1] = np.sum(data1*data2)/(np.sqrt(np.sum(data1**2) * np.sum(data2**2)))
                    print(key1,';',key2,correlation_matrix_h_mode[ind2,ind1])
                    
                except Exception as e:
                    print(key1, key2)
                    print(e)
                    
        correlation_matrix=(correlation_matrix_h_mode - correlation_matrix_l_mode)
                    
    if pdf:
        pdf_page=PdfPages(pdf_filename)

    if plot_interesting_only:
        gpi_labels=['Area',
                    '$\\rm \\Delta$Area',
                    'Major semi-axis',
                    'Minor semi-axis',
                    'Convexity',
                    'Elongation',
                    '$\\rm R_{pos}$',
                    'Roundness',
                    '$\\rm \\Delta$Roundness',
                    '$\\rm d_{rad}$',
                    '$\\rm v_{rad}$',
                    ]
    if not analyze_lh_difference:
        colormap='seismic'
    else:
        colormap='twilight_shifted'
        plt.tight_layout()
        
        
    if save_data_for_publication:
        filename=wd+'/correlation_matrix_data.txt'
        file1=open(filename, 'w+')
        for ind_1 in range(len(correlation_matrix[:,0])):
            for ind_2 in range(len(correlation_matrix[0,:])):
                file1.write(str(correlation_matrix[ind_1,ind_2])+'\t')
            file1.write('\n')
        file1.close()
        
    plot_pearson_matrix(correlation_matrix,
                        xlabels=gpi_labels,
                        ylabels=gpi_labels,
                        colormap=colormap,
                        figsize=(17/2.54/(1+plot_interesting_only),
                                 17/2.54/(1+plot_interesting_only)), #(8.5/2.54,8.5/2.54*1.2)
                        #charsize=5 * (1+plot_interesting_only*0.66),
                        charsize=15,
                        plot_large=not plot_interesting_only,
                        plot_colorbar=not plot_interesting_only,
                        plot_values=True,
                        )

    if pdf:
        pdf_page.savefig()
        pdf_page.close()
        
    return correlation_matrix, gpi_labels


def plot_blob_blob_parameter_trends(pdf=True,
                                    pdf_filename=None,
                                    plot_if_correlation_is_higher_than=None,
                                    plot_if_pps_is_higher_than=None,
                                    nocalc=True,
                                    calc_mean_distribution=True,
                                    min_structure_lifetime=20,
                                    plot_for_publication=False,
                                    str_finding_method='watershed',
                                    recalc_tracking=False,
                                    averaging='no',
                                    analyze_l_mode_only=False,
                                    analyze_h_mode_only=False,
                                    analyze_lh_difference=False,
                                    save_data_for_publication=False,
                                    ):
    import pandas
    import ppscore as pps
    # from matplotlib.colors import LogNorm
    from scipy import stats
    
    if analyze_h_mode_only:
        plasma_mode='_h_mode'
    elif analyze_l_mode_only:
        plasma_mode='_l_mode'
    elif analyze_lh_difference:
        plasma_mode='_lh_diff'
    else:
        plasma_mode=''
    
    if not plot_for_publication:
        if pdf_filename is None and plot_if_correlation_is_higher_than is not None:
            pdf_filename=wd+'/plots/gpi_gpi_trends_corr_'+str(plot_if_correlation_is_higher_than)+'_'+str_finding_method+'.pdf'
        elif plot_if_correlation_is_higher_than is None:
            pdf_filename=wd+'/plots/gpi_gpi_trends_'+str_finding_method+plasma_mode+'.pdf'
    else:
        pdf_filename=wd+'/plots/gpi_gpi_trend_8plot_'+str_finding_method+plasma_mode+'.pdf'
    
    pickle_filename_l_mode=wd+'/processed_data/gpi_gpi_trends_'+str_finding_method+'_'+averaging+'l_mode.pickle'
    pickle_filename_h_mode=wd+'/processed_data/gpi_gpi_trends_'+str_finding_method+'_'+averaging+'h_mode.pickle'
    pickle_filename_full=wd+'/processed_data/gpi_gpi_trends_'+str_finding_method+'_'+averaging+'full.pickle'
        
    if analyze_lh_difference:
        if not os.path.exists(pickle_filename_l_mode):
            full_data=calculate_blob_parameter_histograms2(calc_mean_distribution=calc_mean_distribution,
                                                           nocalc=nocalc,
                                                           plot=False,
                                                           recalc_tracking=recalc_tracking,
                                                           str_finding_method=str_finding_method,
                                                           analyze_l_mode_only=True,
                                                           analyze_h_mode_only=False,
                                                           )
            pickle.dump(full_data, open(pickle_filename_l_mode,'wb'))
        else:
            full_data=pickle.load(open(pickle_filename_l_mode,'rb'))
    else:
        if analyze_l_mode_only:
            pickle_filename=pickle_filename_l_mode
        elif analyze_h_mode_only:
            pickle_filename=pickle_filename_h_mode
        else:
            pickle_filename=pickle_filename_full
            
        if not os.path.exists(pickle_filename):
            full_data=calculate_blob_parameter_histograms2(calc_mean_distribution=calc_mean_distribution,
                                                      nocalc=nocalc,
                                                      plot=False,
                                                      recalc_tracking=recalc_tracking,
                                                      str_finding_method=str_finding_method,
                                                      analyze_l_mode_only=analyze_l_mode_only,
                                                      analyze_h_mode_only=analyze_h_mode_only,
                                                      )
    
                
            pickle.dump(full_data, open(pickle_filename,'wb'))
        else:
            full_data=pickle.load(open(pickle_filename,'rb'))
    
    analyzed_keys=read_analyzed_keys()
    additional_diff_keys=['Convexity', 'Solidity', 'Roundness', 'Total curvature',
                          'Total bending energy','Area','Elongation']

    for key in additional_diff_keys:
        analyzed_keys.append(key+' diff')

    
    pickle_filename=wd+'/processed_data/blob_database_full_data_mean_pps_'+str_finding_method+'.pickle'
    
    if not nocalc or not os.path.exists(pickle_filename):
        df = pandas.DataFrame()

        for key in full_data.keys():
            df[key]=full_data[key]

        df.dropna(thresh=1)

        df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
        pickle.dump(matrix_df,open(pickle_filename,'wb'))
    else:
        matrix_df=pickle.load(open(pickle_filename,'rb'))

    ppscore_matrix=np.asarray(matrix_df).T
    xlabels=list(matrix_df.keys())
    
    if not plot_for_publication:
        pdf_page=PdfPages(pdf_filename)
        for ind1,key1 in enumerate(analyzed_keys):
            ind_nan1=~np.isnan(full_data[key1])

            for ind2,key2 in enumerate(analyzed_keys):
                if key1 != key2 and ind2>ind1:
                    ind_nan2=~np.isnan(full_data[key2])
                    ind_nan=np.logical_and(ind_nan1,ind_nan2)
                    # print(np.sum(ind_nan1),key1)
                    data1_4c=full_data[key1][ind_nan] - np.mean(full_data[key1][ind_nan])
                    data2_4c=full_data[key2][ind_nan] - np.mean(full_data[key2][ind_nan])
                    
                    fig,ax=plt.subplots(figsize=(8.5/2.54,
                                                 8.5/2.54))
                    
                    correlation=np.sum(data1_4c*data2_4c)/(np.sqrt(np.sum(data1_4c**2)*np.sum(data2_4c**2)))
                    
                    if (plot_if_correlation_is_higher_than is not None and
                        np.abs(correlation) > plot_if_correlation_is_higher_than):
                        plot=True
                        
                    elif (plot_if_correlation_is_higher_than is None and
                          plot_if_pps_is_higher_than is None):
                        plot=True
                        
                    else:
                        plot=False
                        
                    pps=ppscore_matrix[xlabels.index(key1),
                                       xlabels.index(key2)]

                    if (plot_if_pps_is_higher_than is not None and
                        pps > plot_if_pps_is_higher_than):
                            plot=True

                    if plot:
                        ax.scatter(full_data[key1][ind_nan],
                                   full_data[key2][ind_nan],
                                   s=0.5)
                        ax.set_xlabel(key1)
                        ax.set_ylabel(key2)
                        ax.set_title(key1+' vs '+key2)
                        pdf_page.savefig()

        if pdf:
            pdf_page.close()
    else:
        interesting_key_pairs=[('Area','Convexity'),
                               ('Size radial','Convexity'),
                               ('Elongation','Roundness'),
                               ('Position radial','Velocity radial position'),
                               ('Axes length minor','Velocity radial position'),
                               ('Axes length major','Velocity radial position'),
                               #('Expansion fraction area','Roundness diff'),
                               ('Area diff','Roundness diff'),
                               ('Position radial','Axes length major'),
                               ]

        ranges=[[[0,0.004],[0.92,1.0]],
                [[0.01,0.07],[0.92,1.0]],
                [[-0.75,0.5],[0.2,1.0]],
                [[1.42,1.6],[-2e3,2e3]],
                [[0,0.075],[-2e3,2e3]],
                [[0.0,0.03],[-2e3,2e3]],
                [[-0.05e-2,0.05e-2],[-0.15,0.15]],
#                [[0.85,1.2],[-0.1,0.1]],
                [[1.42,1.6],[0,0.03]],
                ]
        
        pdf_page=PdfPages(pdf_filename)
        
        fig,axes=plt.subplots(5,2,
                              figsize=(8.5/2.54, 17/2.54))
        
        multiplier={'Area':1e4,
                    'Area diff':1e4,
                    'Angle':1,
                    'Angular velocity angle':1e-3,
                    'Roundness':1, 
                    'Roundness diff':1e3,
                    'Convexity':1,
                    'Total curvature':1,
                    'Total curvature diff':1e3,
                    'Size radial':1e2,
                    'Elongation':1,
                    'Position radial':1,
                    'Velocity radial position':1e-3,
                    'Axes length minor':1e2,
                    'Axes length major':1e2,}
        
        xlabel={'Area':['Area','[$\\rm cm^2$]'],
               'Area diff':['$\\rm\\Delta$Area','[$\\rm cm^2$]'],
               'Angle':['Angle','[rad]'],
               'Angular velocity angle':['$\\rm\\omega$','[krad/s]'],
               'Roundness':['Roundness','[a.u.]'], 
               'Roundness diff':['$\\rm\\Delta$Roundness','[a.u.]'],
               'Convexity':['Convexity','[a.u.]'],
               'Total curvature': ['Curvature','[a.u.]'],
               'Total curvature diff': ['$\\rm\\Delta$Curvature','[a.u.]'],
               'Size radial':['$\\rm d_{rad}$','[cm]'],
               'Elongation':['Elongation','[a.u.]'],
               'Position radial':['$\\rm R_{pos}$','[m]'],
               'Velocity radial position':['$\\rm v_{rad}$','[km/s]'],
               'Axes length minor':['Minor semi-axis','[cm]'],
               'Axes length major':['Major semi-axis','[cm]']}
        labels=['a','b','c','d','e','f','g','h']
        
        
        if not analyze_lh_difference:

            for ind,(key1,key2) in enumerate(interesting_key_pairs):
                
                ind_nan1=~np.isnan(full_data[key1])
                ind_nan2=~np.isnan(full_data[key2])
                
                ind_nan=np.logical_and(ind_nan1,ind_nan2)
                
                data1=full_data[key1][ind_nan]*multiplier[key1]
                data2=full_data[key2][ind_nan]*multiplier[key2]
                
                
                data1_4c = data1 - np.mean(data1)
                data2_4c = data2 - np.mean(data2)
    
                correlation=np.sum(np.real(data1_4c)*np.real(data2_4c))/(np.sqrt(np.sum(np.real(data1_4c)**2)*np.sum(np.real(data2_4c)**2))) 
                
                ax=axes[ind//2,np.mod(ind,2)]
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
    
                # im=ax.hist2d(data1,
                #              data2,
                #              bins=[31,31],
                #              #weights=np.ones_like(np.real(data1))/len(np.real(data1)),
                #              range=[np.asarray(ranges[ind][0])*multiplier[key1],
                #                     np.asarray(ranges[ind][1])*multiplier[key2]],
                #              #s=0.5
                #              #norm=LogNorm()
                #              )
                counts, xedges, yedges = np.histogram2d(data1,
                                                        data2,
                                                        bins=[31,31],
                                                        range=[np.asarray(ranges[ind][0])*multiplier[key1],
                                                               np.asarray(ranges[ind][1])*multiplier[key2]])
                im=ax.imshow((counts/np.sum(counts)*1e2).T,
                             origin="lower",
                             extent=[xedges[0], xedges[-1], 
                                     yedges[0], yedges[-1]],
                             aspect="auto",
                             # cmap='seismic'
                             )
                if save_data_for_publication:
                    filename=wd+'/'+labels[ind]+'_db_2Dhistogram.txt'
                    file1=open(filename, 'w+')
                    file1.write('X bins\n')
                    file1.write(str((xedges[1:]+xedges[:-1])/2))
                    file1.write('\nY bins\n')
                    file1.write(str((yedges[1:]+yedges[:-1])/2))
                    file1.write('\nCounts\n')
                    file1.write(str(counts.T/np.sum(counts)*1e2))
                    file1.close()
                        
                print(key1,';',key2,correlation)
                fig.colorbar(im, cax=cax, orientation='vertical')
                ax.text(-0.45, 1.1, '('+labels[ind]+')', transform=ax.transAxes, size=9)
                ax.text(1.02, 1.05,'[%]', transform=ax.transAxes, size=6)
                ax.set_xlabel(xlabel[key1][0]+' '+xlabel[key1][1])
                ax.set_ylabel(xlabel[key2][0]+' '+xlabel[key2][1])
                ax.set_title('')
                # ax.set_xlim(ranges[ind][0:2])
                # ax.set_ylim(ranges[ind][2:])
        else:
            full_data_l_mode=full_data
            if not os.path.exists(pickle_filename_h_mode):
            
                full_data_h_mode=calculate_blob_parameter_histograms2(calc_mean_distribution=calc_mean_distribution,
                                                                      nocalc=nocalc,
                                                                      plot=False,
                                                                      recalc_tracking=recalc_tracking,
                                                                      str_finding_method=str_finding_method,
                                                                      analyze_l_mode_only=False,
                                                                      analyze_h_mode_only=True,
                                                                      )
                pickle.dump(full_data_h_mode, open(pickle_filename_h_mode,'wb'))
            else:
                full_data_h_mode=pickle.load(open(pickle_filename_h_mode, 'rb'))
            
            for ind,(key1,key2) in enumerate(interesting_key_pairs):
                
                ind_nan1=~np.isnan(full_data_l_mode[key1])
                ind_nan2=~np.isnan(full_data_l_mode[key2])
                
                ind_nan=np.logical_and(ind_nan1,ind_nan2)
                
                data1=full_data_l_mode[key1][ind_nan]*multiplier[key1]
                data2=full_data_l_mode[key2][ind_nan]*multiplier[key2]

                counts_l_mode, xedges, yedges = np.histogram2d(data1,
                                                               data2,
                                                               bins=[31,31],
                                                               range=[np.asarray(ranges[ind][0])*multiplier[key1],
                                                                      np.asarray(ranges[ind][1])*multiplier[key2]])
                
                ind_nan1=~np.isnan(full_data_h_mode[key1])
                ind_nan2=~np.isnan(full_data_h_mode[key2])
                
                ind_nan=np.logical_and(ind_nan1,ind_nan2)
                
                data1=full_data_h_mode[key1][ind_nan]*multiplier[key1]
                data2=full_data_h_mode[key2][ind_nan]*multiplier[key2]
    
                counts_h_mode, xedges, yedges = np.histogram2d(data1,
                                                               data2,
                                                               bins=[31,31],
                                                               range=[np.asarray(ranges[ind][0])*multiplier[key1],
                                                                      np.asarray(ranges[ind][1])*multiplier[key2]])
               
                ax=axes[ind//2,np.mod(ind,2)]
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                
                im=ax.imshow((counts_h_mode/np.sum(counts_h_mode)*1e2 - counts_l_mode/np.sum(counts_l_mode)*1e2).T,
                             origin="lower",
                             extent=[xedges[0], xedges[-1], 
                                     yedges[0], yedges[-1]],
                             vmin=-0.7,
                             vmax=0.7,
                             aspect="auto",
                             cmap='seismic')
                
                fig.colorbar(im, cax=cax, orientation='vertical')
                ax.text(-0.45, 1.1, '('+labels[ind]+')', transform=ax.transAxes, size=9)
                ax.text(1.02, 1.05,'[%]', transform=ax.transAxes, size=6)
                ax.set_xlabel(xlabel[key1][0]+' '+xlabel[key1][1])
                ax.set_ylabel(xlabel[key2][0]+' '+xlabel[key2][1])
                ax.set_title('')
                
        plt.tight_layout(pad=0.2)
        pdf_page.savefig()
        pdf_page.close()



def plot_blob_blob_parameter_predictive_power_score(threshold_corr=False,
                                                    pdf=True,
                                                    nocalc=True,
                                                    calc_mean_distribution=True
                                                    ):

    import pandas
    import ppscore as pps
    from scipy import stats


    if pdf:
        pdf_pages=PdfPages(wd+'/plots/predictive_power_score_blob_vs_blob.pdf')


    if calc_mean_distribution:
        pickle_filename=wd+'/processed_data/blob_database_full_data_mean.pickle'
    else:
        pickle_filename=wd+'/processed_data/blob_database_full_data_nomean.pickle'
        
    full_blob_data=pickle.load(open(pickle_filename,'rb'))

    pickle_filename_pps=wd+'/processed_data/blob_blob_predictive_power_score.pickle'
    if not nocalc or not os.path.exists(pickle_filename_pps):
        df = pandas.DataFrame()
        try:
            for key in full_blob_data.keys():
                df[key]=full_blob_data[key]
        except Exception as e:
            print(e)

        df.dropna(thresh=1)

        df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
        pickle.dump(matrix_df,open(pickle_filename_pps,'wb'))
    else:
        matrix_df=pickle.load(open(pickle_filename_pps,'rb'))

    ppscore_matrix_prelim=np.asarray(matrix_df).T
    xlabels=list(full_blob_data.keys())

    plot_pearson_matrix(ppscore_matrix_prelim,
                        xlabels=xlabels,
                        ylabels=xlabels,
                        title='Blob vs plasma parameter correlation map',
                        colormap='Blues',
                        figsize=(17/2.54,17/2.54), #(8.5/2.54,8.5/2.54*1.2)
                        charsize=6,
                        zrange=[0,1.0]
                        )
    if pdf:
        pdf_pages.savefig()
    pdf_pages.close()


def calculate_blob_plasma_parameter_correlation_matrix(threshold_corr=False,
                                                       threshold_multiplier=2,
                                                       pdf=True,
                                                       pdf_filename=None,
                                                       time_range_around_peak=5e-3,
                                                       str_finding_method='watershed',
                                                       fix_angle_for_correlation=True,
                                                       averaging='shot', #['no', 'blob', 'shot']: No averaging, every identified blob is represented by one value, every shot is represented by one value for each parameter
                                                       average='avg', #['avg', 'std', 'max']
                                                       
                                                       quantity='correlation', #['correlation','mutual_information', 'predictive_power']
                                                       
                                                       figsize=(17/2.54,17/2.54),
                                                       plot_for_publication=False,
                                                       colormap='seismic', #Blues
                                                       plot_colorbar=True,
                                                       plot_full=False,
                                                       linewidth=1.5,
                                                       ticksize=1,
                                                       charsize=9,
                                                       
                                                       nocalc=False,
                                                       nocalc_plasma_data=True,
                                                       nocalc_blob_data=True,
                                                       ):
    plt.close('all')
    if pdf:
        import matplotlib
        matplotlib.use('agg')
        
    if pdf_filename is None:
        
        if quantity == 'mutual_information':
            str_add='mutual_information'
        elif quantity == 'correlation':
            str_add='correlation'
        elif quantity == 'predictive_power':
            str_add='pps'
            
        if averaging == 'no':
            pdf_filename=wd+'/plots/'+str_add+'_matrix_gpi_plasma_'+str_finding_method+'_full'
        else:
            pdf_filename=wd+'/plots/'+str_add+'_matrix_gpi_plasma_'+str_finding_method+'_'+averaging+'_'+average
            
        if threshold_corr:
            pdf_filename+='_thres_'+str(int(threshold_multiplier))
        else:
            pdf_filename+='_nothres'
            
        pdf_filename+='.pdf'  
        
    if pdf:
        pdf_page=PdfPages(pdf_filename)

    full_plasma_data=read_all_plasma_data(nocalc=nocalc_plasma_data)

    full_blob_data=read_blob_data(nocalc=nocalc_blob_data, 
                                  str_finding_method=str_finding_method,
                                  fix_angle_for_correlation=fix_angle_for_correlation,
                                  averaging=averaging,
                                  average=average)
    
    scale_length=(full_plasma_data['Larmor radius sound']**0.8 * full_plasma_data['Connection length']**0.4 /
                      full_plasma_data['Pedestal radius']**0.2)
    
    full_plasma_data['Blob size dimensionless']=(np.sqrt(full_blob_data['Area']/np.pi)/scale_length)**2.5

    
    
    if plot_for_publication:
        
        interesting_key_pairs, units = return_interesting(with_plasma_frequency=True)
        gpi_labels=list(np.unique(interesting_key_pairs[:,0]))
        
        plasma_labels=np.unique(interesting_key_pairs[:,1])
        plasma_labels=list(plasma_labels[[2,1,4,6,0,3,5]])
        plot_full=True
    else:
        gpi_labels=list(full_blob_data.keys())
        plasma_labels=list(full_plasma_data.keys())
        
    if quantity == 'mutual_information':
        title='Blob vs plasma parameter mutual information map'
        colormap='Purples'
        zrange=[0,1]
        
    elif quantity == 'correlation':
        title='Blob vs plasma parameter correlation map'
        zrange=[-1,1]
        colormap='seismic'
        
    elif quantity == 'predictive_power':
        title='Blob vs plasma parameter predictive power map'
        zrange=[0,1]
        colormap='Blues'

    corr_accept=calculate_corr_acceptance_levels()
    if quantity in ['correlation','mutual_information']:
        if averaging == 'shot':
    
            if plot_full:
                full_blob_data.update(full_plasma_data)
                full_data_1=full_blob_data
                full_data_2=full_blob_data
                label_1=gpi_labels+plasma_labels
                label_2=gpi_labels+plasma_labels
                
            else:
                full_data_1=full_blob_data
                full_data_2=full_plasma_data
                label_1=gpi_labels
                label_2=plasma_labels
                
            correlation_matrix=np.zeros([len(label_2),
                                         len(label_1)
                                         ])
    
            for ind1,key1 in enumerate(label_1):
                
                ind_nan1=~np.isnan(full_data_1[key1])
                    
                for ind2,key2 in enumerate(label_2):
                    ind_nan2=~np.isnan(full_data_2[key2])
        
                    ind_nan=np.logical_and(ind_nan1,ind_nan2)
        
                    data1 = full_data_1[key1][ind_nan] 
                    data2 = full_data_2[key2][ind_nan]
     
                    if quantity == 'correlation':
                        correlation_matrix[ind2,ind1]=correlation(data1,data2,
                                                                  threshold_correlation=threshold_corr,
                                                                  correlation_accept=corr_accept,
                                                                  confidence_sigma=threshold_multiplier)
                    elif quantity == 'mutual_information':
                        data1 -=  np.mean(data1)
                        data2 -=  np.mean(data2)
                        correlation_matrix[ind2,ind1]=flap_nstx.tools.mutual_information(data1,data2)
    
        else:
            plasma_data={}
            blob_data={}
            
            for ind2,key2 in enumerate(plasma_labels):
                plasma_data[key2]={}
                for ind1,key1 in enumerate(gpi_labels):
                    plasma_data[key2][key1]=[]
                    for ind_shot in range(len(full_blob_data[key1])):
                        curr_blob_data=full_blob_data[key1][ind_shot]['data']
                        curr_plasma_data=copy.deepcopy(curr_blob_data)
                        curr_plasma_data[:]=full_plasma_data[key2][ind_shot]
                        for value in curr_plasma_data:
                            plasma_data[key2][key1].append(value)
                    plasma_data[key2][key1]=np.asarray(plasma_data[key2][key1])
    
            for ind1,key1 in enumerate(gpi_labels):
                blob_data[key1]=[]
                for ind_shot in range(len(full_blob_data[key1])):
                    curr_blob_data=full_blob_data[key1][ind_shot]['data']
                    for value in curr_blob_data:
                        blob_data[key1].append(value)
                        
                blob_data[key1]=np.asarray(blob_data[key1])
                
            label_1=gpi_labels
            label_2=plasma_labels
            
            if not plot_full:
                correlation_matrix=np.zeros([len(label_2),
                                             len(label_1)
                                             ])
                    
                for ind1, key1 in enumerate(label_1):
                    ind_nan1=~np.isnan(blob_data[key1])
                    for ind2, key2 in enumerate(label_2):
                        ind_nan2=~np.isnan(plasma_data[key2][key1])
                        
                        ind_nan=np.logical_and(ind_nan1,ind_nan2)
                        
                        data1=blob_data[key1][ind_nan]
                        data2=plasma_data[key2][key1][ind_nan]
    
                        if quantity == 'correlation':
                            correlation_matrix[ind2,ind1]=correlation(data1,data2,
                                                                      threshold_correlation=threshold_corr,
                                                                      correlation_accept=corr_accept,
                                                                      confidence_sigma=threshold_multiplier)
                        elif quantity == 'mutual_information':
                            data1 -=  np.mean(data1)
                            data2 -=  np.mean(data2)
                            correlation_matrix[ind2,ind1]=flap_nstx.tools.mutual_information(data1,data2)
            else:
                correlation_matrix=np.zeros([len(label_1+label_2),
                                             len(label_1+label_2)
                                             ])
                for ind1, key1 in enumerate(label_1):
                    ind_nan1=~np.isnan(blob_data[key1])
                    for ind2, key2 in enumerate(label_1):
                        ind_nan2 = ~np.isnan(blob_data[key2])    
                        ind_nan = np.logical_and(ind_nan1,ind_nan2)
                        
                        data1=blob_data[key1][ind_nan]
                        data2=blob_data[key2][ind_nan]
                        if quantity == 'correlation':
                            correlation_matrix[ind2,ind1]=correlation(data1,data2,
                                                                      threshold_correlation=threshold_corr,
                                                                      correlation_accept=corr_accept,
                                                                      confidence_sigma=threshold_multiplier)
                        elif quantity == 'mutual_information':
                            data1 -=  np.mean(data1)
                            data2 -=  np.mean(data2)
                            correlation_matrix[ind2,ind1]=flap_nstx.tools.mutual_information(data1,data2)
                            
                for ind1, key1 in enumerate(label_2):
                    
                    ind_nan1=~np.isnan(full_plasma_data[key1]) #Each label has an additional label for blob labels to correspond to the valid number of data points but each of them have the same data in.
                    for ind2, key2 in enumerate(label_2):
    
                        ind_nan2=~np.isnan(full_plasma_data[key2])
                        ind_nan=np.logical_and(ind_nan1,ind_nan2)
                        
                        data1=full_plasma_data[key1][ind_nan]
                        data2=full_plasma_data[key2][ind_nan]                     
                    
                        
                        if quantity == 'correlation':
                            correlation_matrix[len(label_1)+ind2,len(label_1)+ind1]=correlation(data1,data2,
                                                                                                threshold_correlation=threshold_corr,
                                                                                                correlation_accept=corr_accept,
                                                                                                confidence_sigma=threshold_multiplier)
                        elif quantity == 'mutual_information':
                            data1 -=  np.mean(data1)
                            data2 -=  np.mean(data2)
                            correlation_matrix[len(label_1)+ind2,len(label_1)+ind1]=mutual_information(data1,data2)
                            
                for ind1, key1 in enumerate(label_1):
                    ind_nan1=~np.isnan(blob_data[key1])
                    for ind2, key2 in enumerate(label_2):
                        ind_nan2=~np.isnan(plasma_data[key2][key1])
                        
                        ind_nan=np.logical_and(ind_nan1,ind_nan2)
                        
                        data1=blob_data[key1][ind_nan]
                        data2=plasma_data[key2][key1][ind_nan]
    
                        if quantity == 'correlation':
                            curr_corr=correlation(data1,data2,
                                                  threshold_correlation=threshold_corr,
                                                  correlation_accept=corr_accept,
                                                  confidence_sigma=threshold_multiplier)
    
                            correlation_matrix[len(label_1)+ind2,ind1]=curr_corr
                            correlation_matrix[ind1,len(label_1)+ind2]=curr_corr
                            
                        elif quantity == 'mutual_information':
                            data1 -=  np.mean(data1)
                            data2 -=  np.mean(data2)
                            curr_mi=mutual_information(data1,data2)
                            correlation_matrix[len(label_1)+ind2,ind1]=curr_mi
                            correlation_matrix[ind1,len(label_1)+ind2]=curr_mi
                            
                label_1=label_1+label_2
                label_2=label_1
                
    elif quantity == 'predictive_power':
        pickle_filename=wd+'/processed_data/blob_plasma_predictive_power_score_full_'+averaging+'.pickle'
        
        if averaging != 'shot':
            plasma_data={}
            blob_data={}
            for ind2,key2 in enumerate(plasma_labels):
                plasma_data[key2]={}
                for ind1,key1 in enumerate(gpi_labels):
                    plasma_data[key2]=[]
                    for ind_shot in range(len(full_blob_data[key1])):
                        curr_blob_data=full_blob_data[key1][ind_shot]['data']
                        curr_plasma_data=copy.deepcopy(curr_blob_data)
                        curr_plasma_data[:]=full_plasma_data[key2][ind_shot]
                        for value in curr_plasma_data:
                            plasma_data[key2].append(value)
                    plasma_data[key2]=np.asarray(plasma_data[key2])
    
            for ind1,key1 in enumerate(gpi_labels):
                blob_data[key1]=[]
                for ind_shot in range(len(full_blob_data[key1])):
                    curr_blob_data=full_blob_data[key1][ind_shot]['data']
                    for value in curr_blob_data:
                        blob_data[key1].append(value)
                        
                blob_data[key1]=np.asarray(blob_data[key1])
        else:
            blob_data=full_blob_data
            plasma_data=full_plasma_data
        
        if not nocalc or not os.path.exists(pickle_filename):
            df = pandas.DataFrame()
            
            for key in gpi_labels:
                df[key]=blob_data[key]
            for key in plasma_labels:
                df[key]=plasma_data[key]

            pps_matrix=pps.matrix(df)
            matrix_df = pps_matrix[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
            matrix_df = matrix_df.reindex(index=gpi_labels+plasma_labels, columns=gpi_labels+plasma_labels)
            pickle.dump(matrix_df,open(pickle_filename,'wb'))
        else:
            matrix_df=pickle.load(open(pickle_filename,'rb'))
            
        if averaging != 'shot':
            #Fixing the predictive power in the lower right corner where only the plasma parameters are compared
            df = pandas.DataFrame()

            for key in plasma_labels:
                df[key]=full_plasma_data[key]

            pps_matrix=pps.matrix(df)
            matrix_df_plasma = pps_matrix[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
            matrix_df_plasma = matrix_df_plasma.reindex(index=plasma_labels, 
                                                        columns=plasma_labels)
            
            # Get common rows and columns
            common_rows = matrix_df.index.intersection(matrix_df_plasma.index)
            common_cols = matrix_df.columns.intersection(matrix_df_plasma.columns)
            
            # Loop through the common index-column pairs and update df1 values
            for row in common_rows:
                for col in common_cols:
                    matrix_df.at[row, col] = matrix_df_plasma.at[row, col]
            
        correlation_matrix=np.asarray(matrix_df).T
        
        label_1=list(matrix_df.keys())
        label_2=list(matrix_df.keys())
        plot_full=True

    if not plot_full:
        plot_pearson_matrix(correlation_matrix,
                            xlabels=gpi_labels,
                            ylabels=plasma_labels,
                            title=title,
                            colormap=colormap,
                            zrange=zrange,
                            figsize=(17/2.54,17/2.54), #(8.5/2.54,8.5/2.54*1.2)
                            charsize=charsize,
                            linewidth=linewidth,
                            ticksize=ticksize,
                            minor_ticksize=0.001,
                            plot_colorbar=plot_colorbar,
                        )
    else:
        if not plot_for_publication:
            plot_pearson_matrix(correlation_matrix,
                                xlabels=label_1,
                                ylabels=label_2,
                                title=title,
                                colormap=colormap,
                                zrange=zrange,
                                figsize=(17/2.54,17/2.54), #(8.5/2.54,8.5/2.54*1.2)
                                charsize=3,
                                charsize_score=2,
                                ticksize=ticksize,
                                linewidth=linewidth,
                                minor_ticksize=0.001,
                                plot_colorbar=plot_colorbar,
                            )
        else:
            if quantity == 'mutual_information':
                title='Blob vs plasma parameter mutual information map'
                colormap='Purples'
                zrange=[0,1]
                
            elif quantity == 'correlation':
                title='Blob vs plasma parameter correlation map'
                colormap='seismic'
                zrange=[-1,1]
            
            if units is not None:
                for ind, label in enumerate(gpi_labels):
                    gpi_labels[ind] = units[label][0]
                for ind, label in enumerate(plasma_labels):
                    plasma_labels[ind] = units[label][0]  
            
            labels=gpi_labels+plasma_labels
            
            figsize=(8.5/2.54,8.5/2.54*1.2)
            fig,ax=plt.subplots(figsize=figsize)
            
            plot_pearson_matrix(correlation_matrix,
                                xlabels=labels,
                                ylabels=labels,
                                title=title,
                                colormap=colormap,
                                zrange=zrange,
                                fig_ax=(fig,ax),
                                charsize=charsize,
                                charsize_score=charsize*2/3,
                                ticksize=ticksize,
                                linewidth=linewidth,
                                minor_ticksize=0.001,
                                plot_colorbar=plot_colorbar,
                            )
            
            from matplotlib.patches import Rectangle
            
            ax.add_patch(Rectangle((-0.45, -0.45), 3.9, 3.9, facecolor='none', edgecolor='red', linewidth=2,zorder=0))
            ax.add_patch(Rectangle((-0.45, 3.55), 3.9, 6.9, facecolor='none', edgecolor='yellow', linewidth=2,zorder=0))
            ax.add_patch(Rectangle((3.55, -0.45), 6.9, 3.9, facecolor='none', edgecolor='cyan', linewidth=2,zorder=0))
            ax.add_patch(Rectangle((3.55, 3.55), 6.9, 6.9, facecolor='none', edgecolor='magenta', linewidth=2,zorder=0))
            
    plt.tight_layout(pad=0.1)

    if pdf:
        pdf_page.savefig()
        pdf_page.close()
        
        import matplotlib
        matplotlib.use('qt5agg')

def plot_all_cross_data_matrix():
    
    averaging=['no','blob','shot']
    quantity=['correlation','mutual_information', 'predictive_power']
    method=['watershed', 'contour']
    
    for avg in averaging:
        for qty in quantity:
            for mthd in method:
                calculate_blob_plasma_parameter_correlation_matrix(nocalc=False, 
                                                                   averaging=avg, 
                                                                   average='avg', 
                                                                   str_finding_method=mthd, 
                                                                   quantity=qty, 
                                                                   plot_for_publication=True, 
                                                                   plot_full=True, 
                                                                   threshold_corr=False, 
                                                                   linewidth=1, 
                                                                   ticksize=3, 
                                                                   charsize=9)
    
    


def plot_blob_plasma_parameter_trends(pdf_filename=None,
                                      nocalc=True,
                                      threshold_corr=False,
                                      threshold_multiplier=2,
                                      plot_for_publication=False,
                                      plot_2d_histogram=False,
                                      analyze_l_mode_only=False,
                                      analyze_h_mode_only=False,
                                      analyze_lh_difference=False,
                                      save_data_for_publication=False,
                                      ):

    import matplotlib
    matplotlib.use('agg')

    if pdf_filename is None and not plot_2d_histogram:
        pdf_filename=wd+'/plots/everything_vs_everything'
    else:
        pdf_filename=wd+'/plots/plasma_vs_blob_2d_histrogram'
        
    if threshold_corr:
        pdf_filename+='_thres_'+str(threshold_multiplier)
    if analyze_l_mode_only:
        str_add='_l_mode'
    elif analyze_h_mode_only:
        str_add='_h_mode'
    elif analyze_lh_difference:
        str_add='_lh_diff'
    else:
        str_add=''
    pdf_filename+=str_add+'.pdf'

    pdf_page=PdfPages(pdf_filename)
    
    pickle_filename_plasma_l_mode=wd+'/processed_data/plasma_vs_blob_plasma_data_l_mode.pickle'
    pickle_filename_blob_l_mode=wd+'/processed_data/plasma_vs_blob_blob_data_l_mode.pickle'    
    pickle_filename_plasma_h_mode=wd+'/processed_data/plasma_vs_blob_plasma_data_h_mode.pickle'
    pickle_filename_blob_h_mode=wd+'/processed_data/plasma_vs_blob_blob_data_h_mode.pickle'    
    
    if analyze_l_mode_only:
        pickle_filename_plasma=pickle_filename_plasma_l_mode
        pickle_filename_blob=pickle_filename_blob_l_mode
    elif analyze_h_mode_only:
        pickle_filename_plasma=pickle_filename_plasma_h_mode
        pickle_filename_blob=pickle_filename_blob_h_mode
    else:
        pickle_filename_plasma=wd+'/processed_data/plasma_vs_blob_plasma_data_full.pickle'
        pickle_filename_blob=wd+'/processed_data/plasma_vs_blob_blob_data_full.pickle'
        
    if not analyze_lh_difference:
        if not os.path.exists(pickle_filename_plasma):
            full_plasma_data=read_all_plasma_data(nocalc=nocalc,
                                                  read_l_mode_only=analyze_l_mode_only,
                                                  read_h_mode_only=analyze_h_mode_only)
            pickle.dump(full_plasma_data,open(pickle_filename_plasma,'wb'))
        else:
            full_plasma_data=pickle.load(open(pickle_filename_plasma,'rb'))
        #full_blob_data=read_mean_blob_results(nocalc=nocalc)
        if not os.path.exists(pickle_filename_blob):
            full_blob_data=read_blob_data(nocalc=nocalc, 
                                          str_finding_method='watershed',
                                          fix_angle_for_correlation=True,
                                          averaging='shot',
                                          average='avg',
                                          read_l_mode_only=analyze_l_mode_only,
                                          read_h_mode_only=analyze_h_mode_only)
            
            pickle.dump(full_blob_data,open(pickle_filename_blob,'wb'))
        else:
            full_blob_data=pickle.load(open(pickle_filename_blob,'rb'))
    else:
        #READ L-MODE DATA
        if not os.path.exists(pickle_filename_plasma_l_mode):
            full_plasma_data_l_mode=read_all_plasma_data(nocalc=nocalc,
                                                         read_l_mode_only=True,
                                                         read_h_mode_only=False)
            pickle.dump(full_plasma_data_l_mode,open(pickle_filename_plasma_l_mode,'wb'))
        else:
            full_plasma_data_l_mode=pickle.load(open(pickle_filename_plasma_l_mode,'rb'))
        #full_blob_data=read_mean_blob_results(nocalc=nocalc)
        
        if not os.path.exists(pickle_filename_blob_l_mode):
            full_blob_data_l_mode=read_blob_data(nocalc=nocalc, 
                                          str_finding_method='watershed',
                                          fix_angle_for_correlation=True,
                                          averaging='shot',
                                          average='avg',
                                          read_l_mode_only=True,
                                          read_h_mode_only=False)
            
            pickle.dump(full_blob_data_l_mode,open(pickle_filename_blob_l_mode,'wb'))
        else:
            full_blob_data_l_mode=pickle.load(open(pickle_filename_blob_l_mode,'rb'))
            
        #READ H-MODE DATA
        if not os.path.exists(pickle_filename_plasma_h_mode):
            full_plasma_data_h_mode=read_all_plasma_data(nocalc=nocalc,
                                                         read_l_mode_only=False,
                                                         read_h_mode_only=True)
            pickle.dump(full_plasma_data_h_mode,open(pickle_filename_plasma_h_mode,'wb'))
        else:
            full_plasma_data_h_mode=pickle.load(open(pickle_filename_plasma_h_mode,'rb'))
        
        if not os.path.exists(pickle_filename_blob_h_mode):
            full_blob_data_h_mode=read_blob_data(nocalc=nocalc, 
                                                 str_finding_method='watershed',
                                                 fix_angle_for_correlation=True,
                                                 averaging='shot',
                                                 average='avg',
                                                 read_l_mode_only=False,
                                                 read_h_mode_only=True)
            pickle.dump(full_blob_data_h_mode,open(pickle_filename_blob_h_mode,'wb'))
        else:
            full_blob_data_h_mode=pickle.load(open(pickle_filename_blob_h_mode,'rb'))
    
    if not analyze_h_mode_only and not analyze_l_mode_only:
        scale_length=(full_plasma_data['Larmor radius sound']**0.8 * full_plasma_data['Connection length']**0.4 /
                          full_plasma_data['Pedestal radius']**0.2)
    
        full_plasma_data['Blob size dimensionless']=(np.sqrt(full_blob_data['Area']/np.pi)/scale_length)**2.5
        full_plasma_data['Connection length'][np.where(full_plasma_data['Connection length']<1.5)]=np.nan
    
    if plot_for_publication and not plot_2d_histogram:
        flap_nstx.tools.set_matplotlib_for_publication(labelsize=6.,
                                                       linewidth=0.5,
                                                       major_ticksize=2.)
        
        interesting_key_pairs,units=return_interesting()

        ncol=3
        nrow=3
        fig,axs=plt.subplots(nrows=nrow,
                            ncols=ncol,
                            figsize=(17/2.54,10/2.54)
                            )
        if analyze_lh_difference:
            data_plasma_iterate=[full_plasma_data_l_mode,full_plasma_data_h_mode]
            data_blob_iterate=[full_blob_data_l_mode,full_blob_data_h_mode]
            colors=['tab:blue','tab:orange']
            labels=['L-mode','H-mode']
        else:
            data_plasma_iterate=[full_plasma_data]
            data_blob_iterate=[full_blob_data]
            colors=['tab:blue']
            labels=['']
        
        for ind_full_data, (full_plasma_data,full_blob_data) in enumerate(zip(data_plasma_iterate,data_blob_iterate)):
            ind_del_1=np.where(full_plasma_data['Pressure at max'] > 3.5)
            for key in full_plasma_data.keys():
                full_plasma_data[key]=np.delete(full_plasma_data[key], ind_del_1)
            for key in full_blob_data.keys():
                full_blob_data[key]=np.delete(full_blob_data[key], ind_del_1)
            
            ind_del_2=np.where(full_plasma_data['Temperature pedestal width'] < 0.005)
            for key in full_plasma_data.keys():
                full_plasma_data[key]=np.delete(full_plasma_data[key], ind_del_2)
            for key in full_blob_data.keys():
                full_blob_data[key]=np.delete(full_blob_data[key], ind_del_2)
            
            from string import ascii_lowercase as alc
            try:
                for ind_col in range(ncol):
                    for ind_row in range(nrow):
                        ax=axs[ind_row,ind_col]
                        
                        ind=ind_row*ncol+ind_col
                        key1=interesting_key_pairs[ind][0]
                        key2=interesting_key_pairs[ind][1]
                        
                        ind_nan1=~np.isnan(full_blob_data[key1])
                        ind_nan2=~np.isnan(full_plasma_data[key2])
            
                        ind_nan=np.logical_and(ind_nan1,ind_nan2)
            
                        data1=full_blob_data[key1][ind_nan]*units[key1][2]
                        data2=full_plasma_data[key2][ind_nan]*units[key2][2]
                    
                    
                        correlation = np.sum((data1 - np.mean(data1)) * (data2 - np.mean(data2))) / \
                                      (np.sqrt(np.sum((data1 - np.mean(data1))**2) * np.sum((data2 - np.mean(data2))**2)))
                        
                        ax.plot(data2,
                                data1,
                                linestyle='None',
                                marker='o',
                                ms=1,
                                label=labels[ind_full_data])

                            
                        # plot seaborn calculated interval (std interval, i.e. when ci=68.27) --- the orange one
                        sns.regplot(x=data2, 
                                    y=data1, 
                                    ci=68.27, 
                                    ax=ax, 
                                    color=colors[ind_full_data],
                                    scatter_kws={'s':1})
    
                        if ind_full_data == 0:
                            # ax.scatter(data2,data1,s=1)
                            xlabel=units[key2][0]+' ['+units[key2][1]+']'
                            ylabel=units[key1][0]+' ['+units[key1][1]+']'
                            
                            ax.set_xlabel(xlabel)
                            ax.set_ylabel(ylabel)
                            
                            #ax.yaxis.set_label_coords(-0., .5)
                            ax.text(-0.15, 1.02, 
                                    f"({alc[ind]})", 
                                    transform=ax.transAxes, 
                                    size=6, 
                                    verticalalignment='bottom', 
                                    horizontalalignment='left')
                            if not analyze_lh_difference:
                                slope, intercept, r_value, p_value, std_err = linregress(data2, data1)
                                r_squared = r_value ** 2
                                if correlation < 0:
                                    position_text=[0.05, 0.005]
                                    position_text_2=[0.05,0.125]
                                else:
                                    position_text=[0.65, 0.005]
                                    position_text_2=[0.65,0.125]
                                    
                                ax.text(position_text[0], position_text[1], 
                                        "$\\rho \\ =\\ $"+ f"{correlation:.2f}", 
                                        size=6, 
                                        verticalalignment='bottom', 
                                        horizontalalignment='left',
                                        transform=ax.transAxes)
                
                                ax.text(position_text_2[0], position_text_2[1], 
                                        "$R^2 \\ =\\ $"+ f"{r_squared:.2f}", 
                                        size=6, 
                                        verticalalignment='bottom', 
                                        horizontalalignment='left',
                                        transform=ax.transAxes)
                        
                        # print(key1,key2,correlation)
        
                        #ax.set_title(key1+' vs \n'+key2)
                        if save_data_for_publication:
                            file1=open(wd+f'/{alc[ind]}_blob_plasma_2dhist.txt','w+')
                            file1.write(key2+' data\n')
                            file1.write(str(data2))
                            
                            file1.write('\n'+key1+' data\n')
                            file1.write(str(data1))
                            
                            file1.write
                            file1.write(f'\nCorrelation: {correlation}\n')
                            file1.write(f'R^2 value: {r_squared}')
                            
                            file1.close()
            except Exception as e:
                print(e)
                continue
        if analyze_lh_difference:
            for ind_col in range(ncol):
                for ind_row in range(nrow):
                    ax=axs[ind_row,ind_col]
                    ax.legend(fontsize=6)
                    
        plt.tight_layout(pad=0.1)
        fig.canvas.draw()

        pdf_page.savefig()
        
        
    elif plot_2d_histogram:

        fig,axes=plt.subplots(4,3,
                              figsize=(8.5/2.54,
                                       17/2.54))
        interesting_key_pairs,units=return_interesting()
        
        for ind,(key1,key2) in enumerate(interesting_key_pairs):
            ind_nan1=~np.isnan(full_blob_data[key1])
            ind_nan2=~np.isnan(full_plasma_data[key2])
            ind_nan=np.logical_and(ind_nan1,ind_nan2)
            # print(np.sum(ind_nan1),key1)
            
            data1=np.real(full_blob_data[key1][ind_nan])
            data2=np.real(full_plasma_data[key2][ind_nan])
            
            data1_4c = data1 - np.mean(data1)
            data2_4c = data2 - np.mean(data2)

            correlation=np.sum(data1_4c*data2_4c)/(np.sqrt(np.sum(data1_4c**2)*np.sum(data2_4c**2)))
            ax=axes[ind//3,np.mod(ind,3)]
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            im=ax.hist2d(data1,
                         data2,
                         bins=[21,21],
                         #weights=np.ones_like(np.real(full_data[key1][ind_nan]))/len(np.real(full_data[key1][ind_nan])),
                         #range=ranges[ind],
                         #s=0.5
                         )
            
            print(key1,';',key2,correlation)
            fig.colorbar(im[3], cax=cax, orientation='vertical')
            #ax.text(0.1,0.9,str(correlation))
            ax.set_xlabel(key1)
            ax.set_ylabel(key2)
            ax.set_title('')
            # ax.set_xlim(ranges[ind][0:2])
            # ax.set_ylim(ranges[ind][2:])
        plt.tight_layout(pad=0.1)
        pdf_page.savefig()

    
    elif not plot_2d_histogram:
        corr_accept=calculate_corr_acceptance_levels()
        for ind1,key1 in enumerate(full_blob_data.keys()):
            ind_nan1=~np.isnan(full_blob_data[key1])
            #for ind2,key2 in enumerate(full_plasma_data.keys()):
            for ind2,key2 in enumerate(['Connection length', 'Collisionality dimensionless', 'Blob size dimensionless']):
                ind_nan2=~np.isnan(full_plasma_data[key2])
    
                ind_nan=np.logical_and(ind_nan1,ind_nan2)
    
                data1=full_blob_data[key1][ind_nan]
                data2=full_plasma_data[key2][ind_nan]
            
            
                correlation = np.sum((data1 - np.mean(data1)) * (data2 - np.mean(data2))) / \
                              (np.sqrt(np.sum((data1 - np.mean(data1))**2) * np.sum((data2 - np.mean(data2))**2)))
                if threshold_corr:
                    try:
                        if (np.abs(correlation) > (corr_accept['avg'][np.sum(ind_nan)] +
                                                    threshold_multiplier*corr_accept['stddev'][np.sum(ind_nan)])):
                            plot_page=True
                        else:
                            plot_page=False
                    except Exception as e:
                        print('Exception in analyze_blob_database line 1945',e)
                else:
                    plot_page=True
                    
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(data1, data2)
                    r_squared = r_value ** 2
                except Exception as e:
                    print(e)
                    r_squared = 0
                    
                if r_squared > 0.2:
                    plot_page=True
                else:
                    plot_page=False
                    
                if plot_page:
                    print(key1,key2,correlation)
                    fig,ax=plt.subplots(
                                        figsize=(8.5/2.54,8.5/2.54*1.2)
                                        )
                    ax.scatter(data1,data2)
                        # plot seaborn calculated interval (std interval, i.e. when ci=68.27) --- the orange one
                    sns.regplot(x=data1, 
                                y=data2, 
                                ci=68.27, 
                                ax=ax, 
                                scatter_kws={'s':1})
                    try:
                        if correlation < 0:
                            position_text=[0.05, 0.005]
                            position_text_2=[0.05,0.125]
                        else:
                            position_text=[0.65, 0.005]
                            position_text_2=[0.65,0.125]
                            
                        ax.text(position_text[0], position_text[1], 
                                "$\\rho \\ =\\ $"+ f"{correlation:.2f}", 
                                size=6, 
                                verticalalignment='bottom', 
                                horizontalalignment='left',
                                transform=ax.transAxes)
        
                        ax.text(position_text_2[0], position_text_2[1], 
                                "$R^2 \\ =\\ $"+ f"{r_squared:.2f}", 
                                size=6, 
                                verticalalignment='bottom', 
                                horizontalalignment='left',
                                transform=ax.transAxes)
                    except Exception as e:
                        print(e)
                    
                    
                    ax.set_xlabel(key1)
                    ax.set_ylabel(key2)
                    ax.set_title(key1+' vs \n'+key2)
                    plt.tight_layout(pad=0.1)
                    plt.show()
                    pdf_page.savefig()
    pdf_page.close()

def plot_blob_experiment_vs_theory_radial_velocity(pdf_filename=None,
                                                   nocalc=True,
                                                   ):
    
    import matplotlib
    matplotlib.use('agg')

    if pdf_filename is None:
        pdf_filename=wd+'/plots/experiment_vs_theory'
        
    pdf_filename+='.pdf'

    pdf_page=PdfPages(pdf_filename)

    full_plasma_data=read_all_plasma_data(nocalc=nocalc)
    #full_blob_data=read_mean_blob_results(nocalc=nocalc)
    full_blob_data=read_blob_data(nocalc=nocalc, 
                                  str_finding_method='watershed',
                                  fix_angle_for_correlation=True,
                                  averaging='shot',
                                  average='avg')
    
    experimental_vrad={}
    experimental_vrad['Position']=full_blob_data['Velocity radial position']
    experimental_vrad['COG']=full_blob_data['Velocity radial COG']
    experimental_vrad['Centroid']=full_blob_data['Velocity radial centroid']
    
    c_s=full_plasma_data['Sound speed']
    rho_s=full_plasma_data['Larmor radius']
    
    theoretical_vrad={}
    theoretical_vrad['Inertial']={}
    theoretical_vrad['Inertial']=c_s*rho_s/full_blob_data['Size radial']
    theoretical_vrad['Sheath limited']=c_s*(rho_s/full_blob_data['Size radial'])**2
    
    fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
    # ax.scatter(full_blob_data['Size radial'],
    #             full_blob_data['Velocity radial position'],s=5)
    # ax.set_xscale('log')
    
    ax.scatter(np.abs(experimental_vrad['Position']),
               theoretical_vrad['Inertial'],s=5)
    # ax.set_xlim([0,1.5e3])
    # ax.set_ylim([0,1.5e3])
    #ax.set_aspect(1.0)
    pdf_page.savefig()
    
    
    fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
    ax.scatter(np.abs(experimental_vrad['Position']),
               theoretical_vrad['Sheath limited'],s=5)
    # ax.set_xscale('log')
    #ax.set_yscale('log')
    # ax.set_xlim([0,1.5e3])
    # ax.set_ylim([0,1.5e3])
    #ax.set_aspect(1.0)
    pdf_page.savefig()
    pdf_page.close()
    
    matplotlib.use('qt5agg')
    
    
    
def plot_blob_regime_graph(pdf_filename=None,
                           nocalc=True,
                           save_data_for_publication=False,
                           ):
    
    import matplotlib
    matplotlib.use('agg')

    if pdf_filename is None:
        pdf_filename=wd+'/plots/blob_regimes'
        
    pdf_filename+='.pdf'

    pdf_page=PdfPages(pdf_filename)

    full_plasma_data=read_all_plasma_data(nocalc=nocalc,
                                          calculate_parameters_in_sol=True)
    
    full_blob_data=read_blob_data(nocalc=nocalc, 
                                  str_finding_method='watershed',
                                  fix_angle_for_correlation=True,
                                  averaging='shot',
                                  average='avg')
    
    scale_length=(full_plasma_data['Larmor radius sound']**0.8 * full_plasma_data['Connection length']**0.4 /
                  full_plasma_data['Pedestal radius']**0.2)

    x_data=(np.sqrt(full_blob_data['Area']/np.pi)/scale_length)**2.5

    y_data=full_plasma_data['Collisionality dimensionless']

    
    fig,ax=plt.subplots(figsize=(8.5/2.54,
                                 8.5/1.5/2.54))
    
    ax.scatter(x_data,
               y_data,
               s=5)
    
    if save_data_for_publication:
        file1=open(wd+'/blob_regime_data.txt','w+')
        file1.write('Dimensionless blob size\n')
        file1.write(str(x_data[np.logical_and(~np.isnan(x_data),~np.isnan(y_data))]))
        file1.write('\nDimensionless collisionality\n')
        file1.write(str(y_data[np.logical_and(~np.isnan(x_data),~np.isnan(y_data))]))
        file1.close()
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('$\\Theta$')
    ax.set_ylabel('$\\Lambda$')

    x_max=100
    y_min=0.01
    
    ax.set_xlim([0.1,x_max])
    ax.set_ylim([y_min,10])
    
    p1=[1e-3,1e-3]
    p2=[x_max,x_max]
    
    p3=[10,1]
    p4=[x_max,1]
    
    p5=[10,0.01]
    p6=p3
    
    p7=[0.1,y_min]
    p8=p3
    
    points=[[p1,p2],
            [p3,p4],
            [p5,p6],
            [p7,p8]
            ]
    
    for point in points:
        l = mlines.Line2D([point[0][0],point[1][0]], 
                          [point[0][1],point[1][1]],
                          color='black')
        ax.add_line(l)
    
    text_arr=[("RB",0.1,0.8),
              ("RX",0.8,0.8),
              ("$C_I$",0.4,0.15),
              ("$C_S$",0.85,0.05)]
    
    for (text,xpos,ypos) in text_arr:
        ax.text(xpos, ypos, 
                text, 
                transform=ax.transAxes, 
                size=9, 
                verticalalignment='bottom', 
                horizontalalignment='left')

    plt.tight_layout(pad=0.1)
    pdf_page.savefig()
    
    pdf_page.close()
    
    matplotlib.use('qt5agg')

    
def plot_well_known_parameter_dependences(pdf_filename=None,
                                          nocalc=True,
                                          save_data_for_publication=False,
                                          ):
    
    import matplotlib
    matplotlib.use('agg')

    if pdf_filename is None:
        pdf_filename=wd+'/plots/interesting_parameter_pairs_2'
        
    pdf_filename+='.pdf'

    pdf_page=PdfPages(pdf_filename)

    full_plasma_data=read_all_plasma_data(nocalc=nocalc,
                                          calculate_parameters_in_sol=True)
    #full_blob_data=read_mean_blob_results(nocalc=nocalc)
    full_blob_data=read_blob_data(nocalc=nocalc, 
                                  str_finding_method='watershed',
                                  fix_angle_for_correlation=True,
                                  averaging='shot',
                                  average='avg')
    
    
    interesting_key_pairs=[('Angle','Connection length'),
                           ('Angular velocity ALI', 'Connection length'),
                           ('Roundness','Connection length'),
                           ('Solidity','Connection length'),
                           # ('Velocity radial position', 'Connection length'),
                           # ('Velocity radial position', 'Collisionality dimensionless'),
                           ('Velocity radial dimensionless', 'Connection length'),
                           ('Velocity radial dimensionless', 'Collisionality dimensionless'),
                           #('Velocity radial dimensionless', 'Inverse A hat squared')
                           ]
    
    
    units={'Axes length minor':['$b_{ellipse}$','mm', 1e3],
           'Angle of least inertia':['$\\theta_{blob}$','rad', 1],
           'Angular velocity ALI':['$\omega_{blob}$','krad/s',1e-3],
           'Velocity poloidal centroid':['$v_{pol}$','km/s', 1e-3],
           'Area':['A', '$cm^2$', 1e4],
           'Area diff':['$\\Delta A$', '$cm^2/sample$', 1e4],
           'Angle':['$\\theta_{blob}$','rad', 1],
           'Solidity':['Solidity', '', 1],
           'Roundness':['Roundness', '', 1],
           'Velocity radial position':['$v_{rad}$','km/s', 1e-3],
           'Velocity radial centroid':['$v_{rad}$','km/s', 1e-3],
           'Line integrated density':['$n_{e,LID}$','$10^{19}\\ m^{-3}$',1e-19],
           'Pressure at max':['$p_{e,max\,\\nabla p}$','kPa',1],
           'Density at max':['$n_{e, max\,\\nabla p}$','$10^{19}\\ m^{-3}$', 1e-19],
           'Temperature pedestal width':['$\\Delta_{T_e,ped}$','mm',1e3],
           'Sound speed':['$c_{s,max\,\\nabla p}$','km/s', 1e-3],
           'Plasma frequency':['$\omega_{p,e,max\,\\nabla p}$','GHz', 1e-9],
           'Collisionality':['$\\nu_{ei,max\,\\nabla p}$','-',1],
           'Connection length':['$L_{||}$','m',1],
           'Collisionality dimensionless':['$\\Lambda$','',1],
           'Velocity radial dimensionless':['$\hat{v}$','',1],
           'Inverse A hat squared':['$\hat{a}^{-2}$','',1]
           }
    
    a_star=full_plasma_data['Larmor radius sound']**0.8 * full_plasma_data['Connection length']**0.4/full_blob_data['Velocity radial position']**0.2
    v_star=full_plasma_data['Sound speed']*(a_star/full_blob_data['Position radial'])**0.5
    
    full_blob_data['Velocity radial dimensionless']=full_blob_data['Velocity radial position']/v_star
    
    full_plasma_data['Inverse A hat squared']=1/(full_blob_data['Size radial']/a_star)**2
    
    ncol=2
    nrow=3
    fig,axs=plt.subplots(nrows=nrow,
                         ncols=ncol,
                         figsize=(8.5/2.54,8.5*1.5/2.54))
    
    from string import ascii_lowercase as alc
    
    for ind_col in range(ncol):
        for ind_row in range(nrow):
            ax=axs[ind_row,ind_col]
            
            ind=ind_row*ncol+ind_col
            
            key1=interesting_key_pairs[ind][0]
            key2=interesting_key_pairs[ind][1]
            
            ind_nan1=~np.isnan(full_blob_data[key1])
            ind_nan2=~np.isnan(full_plasma_data[key2])

            ind_nan=np.logical_and(ind_nan1,ind_nan2)

            data1=full_blob_data[key1][ind_nan]*units[key1][2]
            data2=full_plasma_data[key2][ind_nan]*units[key2][2]
        
        
            correlation = np.sum((data1 - np.mean(data1)) * (data2 - np.mean(data2))) / \
                          (np.sqrt(np.sum((data1 - np.mean(data1))**2) * np.sum((data2 - np.mean(data2))**2)))
            
            ax.plot(data2,
                    data1,
                    linestyle='None',
                    marker='o',
                    ms=1,)
            
                    
            # plot seaborn calculated interval (std interval, i.e. when ci=68.27) --- the orange one
            sns.regplot(x=data2, 
                        y=data1, 
                        ci=68.27, 
                        ax=ax, 
                        scatter_kws={'s':1})
                            # ax.scatter(data2,data1,s=1)
            if units[key2][1] == "":
                xlabel=units[key2][0]
            else:
                xlabel=units[key2][0]+' ['+units[key2][1]+']'
                
            if units[key1][1] == "":
                ylabel=units[key1][0]
            else:
                ylabel=units[key1][0]+' ['+units[key1][1]+']'
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            #ax.yaxis.set_label_coords(-0., .5)
            ax.text(-0.4, 1.02, 
                    f"({alc[ind]})", 
                    transform=ax.transAxes, 
                    size=9, 
                    verticalalignment='bottom', 
                    horizontalalignment='left')

            slope, intercept, r_value, p_value, std_err = linregress(data2, data1)
            r_squared = r_value ** 2
            
            if correlation < 0:
                position_text=[0.05, 0.005]
                position_text_2=[0.05,0.125]
            else:
                position_text=[0.65, 0.005]
                position_text_2=[0.6,0.125]
                
            if key1 == "Velocity radial dimensionless":
                position_text=[0.05, 0.755]
                position_text_2=[0.05,0.875]
                        
                
            ax.text(position_text[0], position_text[1], 
                    "$\\rho \\ =\\ $"+ f"{correlation:.2f}", 
                    size=6, 
                    verticalalignment='bottom', 
                    horizontalalignment='left',
                    transform=ax.transAxes)

            ax.text(position_text_2[0], position_text_2[1], 
                    "$R^2 \\ =\\ $"+ f"{r_squared:.2f}", 
                    size=6, 
                    verticalalignment='bottom', 
                    horizontalalignment='left',
                    transform=ax.transAxes)
            
            if save_data_for_publication:
                file1=open(wd+f'/{alc[ind]}_well_known_params.txt','w+')
                file1.write(key1+' data:\n')
                file1.write(str(data1))
                file1.write('\n'+key2+' data:\n')
                file1.write(str(data2))
                file1.write(f'\nR2 value: {r_squared}\n')
                file1.write(f'Correlation: {correlation}')
                file1.close()
                
    plt.tight_layout(pad=0.1)
    
    pdf_page.savefig()
    pdf_page.close()