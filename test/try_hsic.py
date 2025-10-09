#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 10:23:34 2025

@author: mlampert
"""


def try_hsic():
    import numpy as np
    import matplotlib.pyplot as plt
    from hyppo.independence import Hsic
    
    full_plasma_data=read_all_plasma_data(nocalc=True)
    #full_blob_data=read_mean_blob_results(nocalc=nocalc)
    full_blob_data=read_blob_data(nocalc=True, 
                                  str_finding_method='watershed',
                                  fix_angle_for_correlation=True,
                                  averaging='shot',
                                  average='avg')
    
    # interesting_key_pairs, units = return_interesting()
    # gpi_labels=np.unique(interesting_key_pairs[:,0])
    gpi_labels=np.unique(list(full_blob_data.keys()))
    plasma_labels=np.unique(list(full_plasma_data.keys()))
    
    figsize=(8.5/2.54,8.5/2.54*1.2)
    
    correlation_matrix=np.zeros([len(plasma_labels)+len(gpi_labels),
                                 len(plasma_labels)+len(gpi_labels)
                                 ])
    full_blob_data.update(full_plasma_data)
    data_matrix=full_blob_data
    

    gpi_labels=list(gpi_labels)
    plasma_labels=list(plasma_labels)
    hsic = Hsic()
    for ind1,key1 in enumerate(gpi_labels+plasma_labels):
        
        ind_nan1 = ~np.isnan(data_matrix[key1])
        # if key1 == 'Angle' or key1 == 'Angle of least inertia':
        #     full_blob_data[key1]=np.mod(np.real(full_blob_data[key1]), np.pi/2)
            
        for ind2,key2 in enumerate(gpi_labels+plasma_labels):
            ind_nan2 = ~np.isnan(data_matrix[key2])

            ind_nan = np.logical_and(ind_nan1,ind_nan2)

            data1 = data_matrix[key1][ind_nan] 
            data2 = data_matrix[key2][ind_nan]

            # Reshape for HSIC
            data1 = data1.reshape(-1, 1)
            data2 = data2.reshape(-1, 1)
            
            # Perform HSIC test

            stat, p_value = hsic.test(data1, data2)
            
            print(f"HSIC statistic: {stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            correlation_matrix[ind1,ind2]=p_value
            
            
    # if units is not None:
    #     for ind, label in enumerate(gpi_labels):
    #         gpi_labels[ind] = units[label][0]
    #     for ind, label in enumerate(plasma_labels):
    #         plasma_labels[ind] = units[label][0]  
    
    labels=gpi_labels+plasma_labels
    
    fig,ax=plt.subplots(figsize=figsize)
    
    plot_pearson_matrix(correlation_matrix,
                        xlabels=labels,
                        ylabels=labels,
                        title='RF',
                        colormap='Purples',
                        zrange=[np.min(correlation_matrix),np.max(correlation_matrix)],
                        figsize=(17/2.54, 17/2.54),
                        charsize=9,
                        charsize_score=9,
                        linewidth=1,
                        minor_ticksize=0.001,
                    )