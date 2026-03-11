#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 10:28:53 2025

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
from flap_nstx.chers import get_fit_nstx_chers_profiles
from flap_nstx.gpi import analyze_gpi_structures, transform_frames_to_structures
from flap_nstx.gpi import read_analyzed_keys
from flap_nstx.thomson import get_fit_nstx_thomson_profiles

from flap_nstx.pedestal_db import nstx_pedestal_database_header,nstx_pedestal_database_dictionary,read_csv_into_db_format

from flap_nstx.tools import mtanh_function, calculate_plasma_squareness

import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas

#Plot settings for publications
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir=wd+'/plots'

#TODO: look at the fit profiles one-by-one,

#TODO: Fix CHERS profile analysis and plot all fit profiles

#TODO: fill in the rest of the data pointspip

def compare_nstx_pedestal_db():
    pdf_pages=PdfPages(fig_dir+'/oak_mate_db_compare.pdf')
    
    oak_db_oak_calc=read_csv_into_db_format(csv_db_file_name='/Users/mlampert/work/NSTX_workspace/db/2022_JRT_non-ELMING_database_NSTX_only.csv',
                                            new_db_filename='pedestal_db_oak_db_oak_calc.pickle', 
                                            ind_mod=-1)
    
    oak_db_mlampert_calc=pickle.load(open('/Users/mlampert/work/NSTX_workspace/db/pedestal_db_oak.pickle','rb'))
    
    shots=np.asarray([oak_db_oak_calc[ind]['Shot data']['shot'] for ind,_ in enumerate(oak_db_oak_calc)])
    for key in list(oak_db_oak_calc[0].keys())[1:]:
        if key == 'Kinetic profile note':
            continue
        try:
            oak_data=[]
            for ind,_ in enumerate(oak_db_oak_calc):
                data=oak_db_oak_calc[ind][key]['data'] 
                if data is not None:
                    oak_data.append(data)
                else:
                    oak_data.append(np.nan)
            oak_data=np.asarray(oak_data)
    
        except:
            oak_data=np.full(shots.shape,np.nan)
            
        try:
            oak_error=[]
            for ind,_ in enumerate(oak_db_oak_calc):
                data_error=oak_db_oak_calc[ind][key]['data error'] 
                if data_error is not None:
                    oak_error.append(data_error)
                else:
                    oak_error.append(np.nan)
            oak_error=np.asarray(oak_error)
    
        except:
            oak_error=np.full(shots.shape,np.nan)
        
        try:
            mate_data=[]
            for ind,_ in enumerate(oak_db_mlampert_calc):
                data=oak_db_mlampert_calc[ind][key]['data'] 
                if data is not None:
                    mate_data.append(data)
                else:
                    mate_data.append(np.nan)
            mate_data=np.asarray(mate_data)
    
        except:
            mate_data=np.full(shots.shape,np.nan)
            
        try:
            mate_error=[]
            for ind,_ in enumerate(oak_db_mlampert_calc):
                data_error=oak_db_mlampert_calc[ind][key]['data error'] 
                if data_error is not None:
                    mate_error.append(data_error)
                else:
                    mate_error.append(np.nan)
            mate_error=np.asarray(mate_error)
    
        except:
            mate_error=np.full(shots.shape,np.nan)
            
        fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
        try:
            ax.scatter(mate_data,
                       oak_data,
                       color='tab:blue')
        except:
            pass
        
        label=oak_db_mlampert_calc[0][key]['label']
        unit=oak_db_mlampert_calc[0][key]['unit']
        if label == '$q_{\ast}$':
            label='$q_{\\ast}$'
        elif label == '$\beta_{t}$':
            label = '$\\beta_{t}$'
        elif label == '$\beta_{p}$':
            label = '$\\beta_{p}$'
        elif label == '$\beta_{N}$':
            label = '$\\beta_{N}$'
            
        if label is not None and unit is not None:
            ax.set_xlabel('Mate '+label+' ['+oak_db_mlampert_calc[0][key]['unit']+']')
            ax.set_ylabel('Oak '+label+' ['+oak_db_mlampert_calc[0][key]['unit']+']')

        try:
            ranges=[np.min([mate_data[~np.isnan(mate_data)],
                            oak_data[~np.isnan(oak_data)]]),
                    
                    np.max([mate_data[~np.isnan(mate_data)],
                            oak_data[~np.isnan(oak_data)]])]
            ax.set_xlim(ranges)
            ax.set_ylim(ranges)
            if ranges[0] != ranges[1]:
                ax.axline([ranges[0],ranges[0]],[ranges[1],ranges[1]], color='tab:red')
        except:
            pass
        plt.tight_layout(pad=0.1)
        
        pdf_pages.savefig()
        
    pdf_pages.close()