#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:34:44 2026

@author: mlampert
"""

import matplotlib.pyplot as plt
import MDSplus as mds
from matplotlib.backends.backend_pdf import PdfPages
import copy

wd='/Users/mlampert/work/NSTX_workspace'

def plot_basic_shot_data(shot=None,
                         time_range=None):
    signals=[{"tree":'EFIT02',
              'node':r'\IPMEAS',
              'title':'Plasma current',
              'label':'$I_p$',
              'unit':'kA',
              'multiplier':1e-3},
             
            {"tree":'WF',
             'node':r'\NEL',
             'title':'Line integrated ne',
             'label':'$n_e$',
             'unit':'$10^{19}m^{-2}$',
             'multiplier':1e-16},
            
            {"tree":'WF',
             'node':r'\PNB',
             'title':'NBI power',
             'label':'$p_{NBI}$',
             'unit':'$MW$',
             'multiplier':1},
            
            {"tree":'WF',
             'node':r'\DALPHA',
             'title':'$D_{\\alpha} signal',
             'label':'$D_{\\alpha}$',
             'unit':'a.u.',
             'multiplier':1},
            
            {"tree":'EFIT02',
             'node':r'\BETAN',
             'title':'Normalized beta',
             'label':'$\\beta_{N}$',
             'unit':'',
             'multiplier':1},
            
            {"tree":'EFIT02',
             'node':r'\Q95',
             'title':'Safety factor at psi_norm=0.95',
             'label':'$q_{95}$',
             'unit':'',
             'multiplier':1},
            
            ]
    
    fig,axs=plt.subplots(len(signals),1, figsize=(8.5/2.54, 14/2.54))
    
    conn=mds.Connection('skylark.pppl.gov:8501')
    
    pdf_page=PdfPages(f'{wd}/plots/shot_data_{shot}.pdf')
    for ind,signal in enumerate(signals):
        
        
        conn.openTree(signal['tree'], shot)    
        data=conn.get(signal['node']).data()
        time=conn.get(f"dim_of({signal['node']},0)")
        if ind==0: time_0=copy.deepcopy(time)
    
        axs[ind].plot(time, data * signal['multiplier'])
        if ind == len(signals)-1:
            axs[ind].set_xlabel('Time [s]')
        else:
            axs[ind].set_xticklabels([])
        axs[ind].set_ylabel(signal['label']+" ["+signal['unit']+"]")
        axs[ind].set_xlim([min(time_0),max(time_0)])
        if signal['label'] == '$\\beta_{N}$':
            axs[ind].set_ylim([0,8])
    fig.tight_layout(pad=0.1)
    pdf_page.savefig()
        
    pdf_page.close()
    
    try:
        conn.disconnect()
    except:
        print('Wrong command')
    
