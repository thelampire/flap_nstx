#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:52:23 2023

@author: mlampert
"""
#Core modules
import os
import copy
import cv2
import time as time_mod
import pickle
import warnings
warnings.filterwarnings("ignore")

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')

from flap_nstx.gpi import normalize_gpi, identify_structures
from flap_nstx.gpi import analyze_gpi_structures, transform_frames_to_structures
from flap_nstx.gpi import read_analyzed_keys

from flap_nstx.tools import detrend_multidim, plot_pearson_matrix
from flap_nstx.tools import calculate_corr_acceptance_levels

from flap_nstx.thomson import get_fit_nstx_thomson_profiles

import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.style as pltstyle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas

#Plot settings for publications
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/plots'


def calculate_all_blob_results(test=False,
                               plot=False,
                               pdf=False,
                               nocalc=False,
                               time_range_around_peak=5e-3,
                               ):

    blob_database=read_blob_database(time_range_around_peak=time_range_around_peak)

    ncalc=len(blob_database['shot'])
    start_time=time_mod.time()
    for ind in range(ncalc):
        blob_time=blob_database['time'][ind]

        read_blob_results(blob_database['shot'][ind],
                          [blob_time-time_range_around_peak,
                           blob_time+time_range_around_peak],
                          calc_only=True,
                          nocalc=nocalc,
                          )


        elapsed_time=time_mod.time()-start_time
        remaining_time=elapsed_time/(ind+1)*ncalc
        print('Remaining time from the calculation:'+str(remaining_time/3600.)+'hours.')
        flap.delete_data_object('*')


def calculate_blob_parameter_histograms(time_range_around_peak=5e-3,
                                        pdf=False,
                                        pdf_filename=None,
                                        plot=True,
                                        plot_for_publication=False,
                                        save_data_into_txt=False,
                                        calc_mean_distribution=False,
                                        nocalc=True):
    import matplotlib
    if pdf:
        matplotlib.use('agg')
    else:
        matplotlib.use('qt5agg')

    if pdf_filename is None:
        if calc_mean_distribution:
            pdf_filename=wd+fig_dir+'/blob_database_parameter_histograms_mean.pdf'
        else:
            pdf_filename=wd+fig_dir+'/blob_database_parameter_histograms_nomean.pdf'

    if calc_mean_distribution:
        pickle_filename=wd+'/processed_data/blob_database_full_data_mean.pickle'
    else:
        pickle_filename=wd+'/processed_data/blob_database_full_data_nomean.pickle'

    blob_database=read_blob_database(time_range_around_peak=time_range_around_peak)
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

            blob_results=read_blob_results(blob_database['shot'][ind],
                                           [blob_time-time_range_around_peak,
                                            blob_time+time_range_around_peak],
                                           nocalc=True,
                                           )
            flap.delete_data_object('*')
            str_by_str=transform_frames_to_structures(blob_results)

            for ind_str, structure in enumerate(str_by_str):
                n_str+=1
                for key in analyzed_keys:
                    if calc_mean_distribution:
                        full_data[key]=np.append(full_data[key],
                                                 np.mean(structure[key]))
                    else:
                        if key in          ['Velocity radial COG', 'Velocity poloidal COG',
                                           'Velocity radial centroid', 'Velocity poloidal centroid',
                                           'Velocity radial position', 'Velocity poloidal position',
                                           'Expansion fraction area', 'Expansion fraction axes',
                                           'Angular velocity angle', 'Angular velocity ALI']:
                            full_data[key]=np.append(full_data[key],
                                                     structure[key])
                        else:
                            full_data[key]=np.append(full_data[key],
                                                     structure[key][1:])
                for key in additional_diff_keys:
                    if calc_mean_distribution:
                        full_data[key+' diff']=np.append(full_data[key+' diff'],
                                                         np.mean((np.asarray(structure[key])[1:] -
                                                                  np.asarray(structure[key])[0:-1])))
                    else:
                        full_data[key+' diff']=np.append(full_data[key+' diff'],
                                                         (np.asarray(structure[key])[1:] -
                                                          np.asarray(structure[key])[0:-1]))

        pickle.dump(full_data,open(pickle_filename,'wb'))
        print('n_str:',n_str)
    else:
        full_data=pickle.load(open(pickle_filename,'rb'))


    for key in additional_diff_keys:
        analyzed_keys.append(key+' diff')

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

    if plot:
        if plot_for_publication:
            pdf_page=PdfPages(wd+'/plots/8hist_blob_db.pdf')
            fig,axes=plt.subplots(2,4,figsize=(17/2.54,8.5/2.54))
            for ind, key in enumerate(['Area','Area diff',
                                       'Angle','Angular velocity angle',
                                       'Roundness', 'Roundness diff',
                                       'Total curvature','Total curvature diff',
                                       ]):
                labels=['a','b','c','d','e','f','g','h']
                full_data[key]=full_data[key][~np.isnan(full_data[key])]
                ax=axes[np.mod(ind,2),ind//2]
                # try:
                print(key,'skewness',scipy.stats.skew(full_data[key]))
                print(key,'kurtosis',scipy.stats.kurtosis(full_data[key]))
                if key in ranges.keys():
                    ax.hist(full_data[key],
                            bins=51,
                            weights=np.ones_like(full_data[key])/len(full_data[key]),
                            range=ranges[key])
                else:
                    ax.hist(full_data[key],
                            bins=51,
                            weights=np.ones_like(full_data[key])/len(full_data[key]),)
                ax.set_xlabel(key)
                ax.set_ylabel('Relative frequency')
                ax.set_title('Histogram of \n '+key)
                ax.text(-0.4, 1.1, '('+labels[ind]+')', transform=ax.transAxes, size=9)
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

def calculate_blob_blob_parameter_correlation_matrix(threshold_corr=False,
                                                     pdf=True,
                                                     pdf_filename=None,
                                                     calc_mean_distribution=True):
    if pdf_filename is None:
        pdf_filename=wd+'/plots/correlation_matrix_gpi_gpi.pdf'

    if calc_mean_distribution:
        pickle_filename=wd+'/processed_data/blob_database_full_data_mean.pickle'
    else:
        pickle_filename=wd+'/processed_data/blob_database_full_data_nomean.pickle'
    full_data=pickle.load(open(pickle_filename,'rb'))

    analyzed_keys=read_analyzed_keys()
    additional_diff_keys=['Convexity', 'Solidity', 'Roundness', 'Total curvature',
                          'Total bending energy','Area','Elongation']

    for key in additional_diff_keys:
        analyzed_keys.append(key+' diff')
    gpi_labels=analyzed_keys
    gpi_label_num=len(gpi_labels)

    correlation_matrix=np.zeros([len(analyzed_keys),len(analyzed_keys)])


    for ind1,key1 in enumerate(analyzed_keys):
        ind_nan1=~np.isnan(full_data[key1])
        for ind2,key2 in enumerate(analyzed_keys):
            try:
                ind_nan2=~np.isnan(full_data[key2])
                ind_nan=np.logical_and(ind_nan1,ind_nan2)
                # print(np.sum(ind_nan1),key1)
                data1=full_data[key1][ind_nan] - np.mean(full_data[key1][ind_nan])
                data2=full_data[key2][ind_nan] - np.mean(full_data[key2][ind_nan])
                correlation_matrix[ind2,ind1]=np.sum(data1*data2)/(np.sqrt(np.sum(data1**2)*np.sum(data2**2)))
            except Exception as e:
                print(e)
    # if threshold_corr:
    #     for i_ts in range(ts_label_num):
    #         for j_gpi in range(gpi_label_num):
    #             if abs(correlation_matrix[i_ts,j_gpi]) < corr_thres_level_matrix[i_ts,j_gpi]:
    #                 correlation_matrix[i_ts,j_gpi]=0.
    if pdf:
        pdf_page=PdfPages(pdf_filename)

    plot_pearson_matrix(correlation_matrix,
                        xlabels=gpi_labels,
                        ylabels=gpi_labels,
                        title='Blob vs blob parameter correlation map',
                        colormap='seismic',
                        figsize=(17/2.54,17/2.54), #(8.5/2.54,8.5/2.54*1.2)
                        charsize=6,
                        )

    if pdf:
        pdf_page.savefig()
        pdf_page.close()

def plot_blob_blob_parameter_trends(pdf=True,
                                    pdf_filename=None,
                                    plot_if_correlation_is_higher_than=None,
                                    plot_if_pps_is_higher_than=None,
                                    nocalc=True,
                                    calc_mean_distribution=True,
                                    plot_for_publication=False,
                                    ):
    import pandas
    import ppscore as pps

    from scipy import stats
    if not plot_for_publication:
        if pdf_filename is None and plot_if_correlation_is_higher_than is not None:
            pdf_filename=wd+'/plots/gpi_gpi_trends_corr_'+str(plot_if_correlation_is_higher_than)+'.pdf'
        elif plot_if_correlation_is_higher_than is None:
            pdf_filename=wd+'/plots/gpi_gpi_trends.pdf'
    else:
        pdf_filename=wd+'/plots/gpi_gpi_trend_8plot.pdf'

    if calc_mean_distribution:
        pickle_filename=wd+'/processed_data/blob_database_full_data_mean.pickle'
    else:
        pickle_filename=wd+'/processed_data/blob_database_full_data_nomean.pickle'
    full_data=pickle.load(open(pickle_filename,'rb'))

    analyzed_keys=read_analyzed_keys()
    additional_diff_keys=['Convexity', 'Solidity', 'Roundness', 'Total curvature',
                          'Total bending energy','Area','Elongation']

    for key in additional_diff_keys:
        analyzed_keys.append(key+' diff')

    pdf_page=PdfPages(pdf_filename)
    pickle_filename=wd+'/processed_data/blob_database_full_data_mean_pps.pickle'
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
        for ind1,key1 in enumerate(analyzed_keys):
            ind_nan1=~np.isnan(full_data[key1])

            for ind2,key2 in enumerate(analyzed_keys):
                if key1 != key2 and ind2>ind1:
                    ind_nan2=~np.isnan(full_data[key2])
                    ind_nan=np.logical_and(ind_nan1,ind_nan2)
                    # print(np.sum(ind_nan1),key1)
                    data1_4c=full_data[key1][ind_nan] - np.mean(full_data[key1][ind_nan])
                    data2_4c=full_data[key2][ind_nan] - np.mean(full_data[key2][ind_nan])
                    fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
                    correlation=np.sum(data1_4c*data2_4c)/(np.sqrt(np.sum(data1_4c**2)*np.sum(data2_4c**2)))

                    if (plot_if_correlation_is_higher_than is not None and
                        np.abs(correlation) > plot_if_correlation_is_higher_than):
                        plot=True
                    elif plot_if_correlation_is_higher_than is None and plot_if_pps_is_higher_than is None:
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
                               ('Expansion fraction area','Roundness diff'),
                               ('Position radial','Axes length major'),
                               ]
        ranges=[[[0,0.004],[0.92,1.0]],
                [[0.01,0.07],[0.92,1.0]],
                [[-0.75,0.5],[0.2,1.0]],
                [[1.42,1.6],[-2e3,2e3]],
                [[0,0.075],[-2e3,2e3]],
                [[0.0,0.03],[-2e3,2e3]],
                [[0.85,1.2],[-0.1,0.1]],
                [[1.42,1.6],[0,0.03]],
                ]
        fig,axes=plt.subplots(2,4,
                            figsize=(17/2.54,8.5/2.54))
        for ind,(key1,key2) in enumerate(interesting_key_pairs):
            ind_nan1=~np.isnan(full_data[key1])
            ind_nan2=~np.isnan(full_data[key2])
            ind_nan=np.logical_and(ind_nan1,ind_nan2)
            # print(np.sum(ind_nan1),key1)
            data1_4c=np.real(full_data[key1][ind_nan]) - np.real(np.mean(full_data[key1][ind_nan]))
            data2_4c=np.real(full_data[key2][ind_nan]) - np.real(np.mean(full_data[key2][ind_nan]))

            correlation=np.sum(data1_4c*data2_4c)/(np.sqrt(np.sum(data1_4c**2)*np.sum(data2_4c**2)))
            ax=axes[ind//4,np.mod(ind,4)]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            im=ax.hist2d(np.real(full_data[key1][ind_nan]),
                         np.real(full_data[key2][ind_nan]),
                         bins=[31,31],
                         #weights=np.ones_like(np.real(full_data[key1][ind_nan]))/len(np.real(full_data[key1][ind_nan])),
                         range=ranges[ind]
                         #s=0.5
                         )
            print(key1,key2,correlation)
            fig.colorbar(im[3], cax=cax, orientation='vertical')
            # ax.text(0.1,0.9,str(correlation))
            ax.set_xlabel(key1)
            ax.set_ylabel(key2)
            ax.set_title('')
            # ax.set_xlim(ranges[ind][0:2])
            # ax.set_ylim(ranges[ind][2:])
        plt.tight_layout(pad=0.1)
        pdf_page.savefig()
        pdf_page.close()


def calculate_blob_plasma_parameter_correlation_matrix(threshold_corr=False,
                                                       pdf=True,
                                                       pdf_filename=None,
                                                       time_range_around_peak=5e-3,
                                                       nocalc=False):

    if pdf:
        import matplotlib
        matplotlib.use('agg')

    if pdf_filename is None:
        pdf_filename=wd+fig_dir+'/blob_database_parameter_histograms_mean.pdf'

    if pdf_filename is None:
        pdf_filename=wd+'/plots/correlation_matrix_gpi_plasma.pdf'

    if pdf:
        pdf_page=PdfPages(pdf_filename)

    full_plasma_data=read_all_plasma_data(nocalc=nocalc)
    full_blob_data=read_mean_blob_results(nocalc=nocalc)

    correlation_matrix=np.zeros([len(full_plasma_data.keys()),
                                 len(full_blob_data.keys())
                                 ])
    corr_accept=calculate_corr_acceptance_levels()

    for ind1,key1 in enumerate(full_blob_data.keys()):
        ind_nan1=~np.isnan(full_blob_data[key1])
        for ind2,key2 in enumerate(full_plasma_data.keys()):
            ind_nan2=~np.isnan(full_plasma_data[key2])

            ind_nan=np.logical_and(ind_nan1,ind_nan2)

            data1=full_blob_data[key1][ind_nan] - np.mean(full_blob_data[key1][ind_nan])
            data2=full_plasma_data[key2][ind_nan] - np.mean(full_plasma_data[key2][ind_nan])

            correlation=np.sum(data1*data2)/(np.sqrt(np.sum(data1**2)*np.sum(data2**2)))

            if (np.abs(correlation) > (corr_accept['avg'][np.sum(ind_nan)] +
                               2*corr_accept['stddev'][np.sum(ind_nan)])):
                correlation_matrix[ind2,ind1]=correlation
            else:
                correlation_matrix[ind2,ind1]=np.nan
    corr_accept=calculate_corr_acceptance_levels()

    plot_pearson_matrix(correlation_matrix,
                        xlabels=full_blob_data.keys(),
                        ylabels=full_plasma_data.keys(),
                        title='Blob vs plasma parameter correlation map',
                        colormap='seismic',
                        figsize=(17/2.54,17/2.54), #(8.5/2.54,8.5/2.54*1.2)
                        charsize=6
                        )

    if pdf:
        pdf_page.savefig()
        pdf_page.close()

def plot_blob_plasma_parameter_predictive_power_score(threshold_corr=False,
                                                      pdf=True,
                                                      nocalc=True):

    import pandas
    import ppscore as pps
    import seaborn as sns
    from scipy import stats


    if pdf:
        pdf_pages=PdfPages(wd+'/plots/predictive_power_score_blob_vs_plasma.pdf')

    pickle_filename=wd+'/processed_data/blob_plasma_predictive_power_score.pickle'
    full_plasma_data=read_all_plasma_data(nocalc=nocalc)
    full_blob_data=read_mean_blob_results(nocalc=nocalc)
    if not nocalc or not os.path.exists(pickle_filename):
        df = pandas.DataFrame()

        for key in full_plasma_data.keys():
            df[key]=full_plasma_data[key]

        for key in full_blob_data.keys():
            df[key]=full_blob_data[key]

        df.dropna(thresh=1)

        df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
        pickle.dump(matrix_df,open(pickle_filename,'wb'))
    else:
        matrix_df=pickle.load(open(pickle_filename,'rb'))

    ppscore_matrix_prelim=np.asarray(matrix_df).T
    labels=list(matrix_df.keys())
    xlabels=list(full_blob_data.keys())
    ylabels=list(full_plasma_data.keys())
    ppscore_matrix_blob_to_plasma=np.zeros([len(ylabels),len(xlabels)])
    ppscore_matrix_plasma_to_blob=np.zeros([len(xlabels),len(ylabels)])

    for indx,xlab in enumerate(xlabels):
        for indy, ylab in enumerate(ylabels):
            ind1=labels.index(ylab)
            ind2=labels.index(xlab)
            ppscore_matrix_blob_to_plasma[indy,indx] = ppscore_matrix_prelim[ind1,ind2]
            ppscore_matrix_plasma_to_blob[indx,indy] = ppscore_matrix_prelim[ind2,ind1]

    plot_pearson_matrix(ppscore_matrix_blob_to_plasma,
                        xlabels=xlabels,
                        ylabels=ylabels,
                        title='Blob vs plasma parameter correlation map',
                        colormap='Blues',
                        figsize=(17/2.54,17/2.54), #(8.5/2.54,8.5/2.54*1.2)
                        charsize=6,
                        zrange=[0,1.0]
                        )
    if pdf:
        pdf_pages.savefig()

    plot_pearson_matrix(ppscore_matrix_plasma_to_blob,
                        xlabels=ylabels,
                        ylabels=xlabels,
                        title='Plasma vs blob parameter correlation map',
                        colormap='Blues',
                        figsize=(17/2.54,17/2.54), #(8.5/2.54,8.5/2.54*1.2)
                        charsize=6,
                        zrange=[0,1.0]
                        )
    if pdf:
        pdf_pages.savefig()
    pdf_pages.close()

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
    labels=list(matrix_df.keys())
    xlabels=list(full_blob_data.keys())
    ppscore_matrix_blob_to_plasma=np.zeros([len(xlabels),len(xlabels)])

    plot_pearson_matrix(ppscore_matrix_blob_to_plasma,
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


def plot_blob_plasma_parameter_trends(pdf_filename=None,
                                      nocalc=True):

    import matplotlib
    matplotlib.use('agg')

    if pdf_filename is None:
        pdf_filename=wd+'/plots/everything_vs_everything_beware.pdf'

        pdf_page=PdfPages(pdf_filename)

    full_plasma_data=read_all_plasma_data(nocalc=nocalc)
    full_blob_data=read_mean_blob_results(nocalc=nocalc)

    for ind1,key1 in enumerate(full_blob_data.keys()):
        ind_nan1=~np.isnan(full_blob_data[key1])
        for ind2,key2 in enumerate(full_plasma_data.keys()):
            ind_nan2=~np.isnan(full_plasma_data[key2])

            ind_nan=np.logical_and(ind_nan1,ind_nan2)

            data1=full_blob_data[key1][ind_nan]
            data2=full_plasma_data[key2][ind_nan]
            fig,ax=plt.subplots(
                                figsize=(8.5/2.54,8.5/2.54*1.2)
                                )
            ax.scatter(data1,data2)
            ax.set_xlabel(key1)
            ax.set_ylabel(key2)
            ax.set_title(key1+' vs '+key2)
            plt.tight_layout(pad=0.1)
            plt.show()
            pdf_page.savefig()
    pdf_page.close()


def read_mean_blob_results(time_range_around_peak=5e-3,
                           nocalc=False):

    pickle_filename=wd+'/processed_data/blob_database_shot_by_shot_blob.pickle'

    blob_database=read_blob_database(time_range_around_peak=time_range_around_peak)
    analyzed_keys=read_analyzed_keys()
    additional_diff_keys=['Convexity', 'Solidity', 'Roundness', 'Total curvature',
                          'Total bending energy','Area','Elongation']

    ncalc=len(blob_database['shot'])

    full_blob_data={}

    for key in analyzed_keys:
        full_blob_data[key]=[]
    for key in additional_diff_keys:
        full_blob_data[key+' diff']=[]
    full_blob_error=copy.deepcopy(full_blob_data)

    curr_blob_data_ref=copy.deepcopy(full_blob_data)
    curr_blob_error_ref=copy.deepcopy(full_blob_data)

    if not os.path.exists(pickle_filename) or not nocalc:
        for ind in range(ncalc):
            curr_blob_data=copy.deepcopy(curr_blob_data_ref)
            curr_blob_error=copy.deepcopy(curr_blob_error_ref)
            blob_time=blob_database['time'][ind]
            shot=blob_database['shot'][ind]

            blob_results=read_blob_results(shot,
                                           [blob_time-time_range_around_peak,
                                            blob_time+time_range_around_peak],
                                           nocalc=True,
                                           )
            flap.delete_data_object('*')
            str_by_str=transform_frames_to_structures(blob_results)

            for structure in str_by_str:
                for key in analyzed_keys:
                    mean_data=[]
                    for data in structure[key]:
                        if np.isreal(data) and ~np.isnan(data): #There are a bunch of complex and nan data which are not handled.
                            mean_data=np.append(mean_data,
                                                np.real(data))
                    curr_blob_data[key]=np.append(curr_blob_data[key],
                                                  np.mean(mean_data))
                    curr_blob_error[key]=np.append(curr_blob_error[key],
                                                   np.sqrt(np.var(mean_data)))

                for key in additional_diff_keys:
                    mean_data=[]
                    for ind_data in range(len(structure[key])-1):
                        if (np.isreal(structure[key][ind_data+1]-structure[key][ind_data]) and
                           ~np.isnan(structure[key][ind_data+1]-structure[key][ind_data])):
                            mean_data=np.append(mean_data,
                                                np.real(structure[key][ind_data+1]-structure[key][ind_data]))

                    curr_blob_data[key+' diff']=np.append(curr_blob_data[key+' diff'],
                                                          np.mean(mean_data))

                    curr_blob_error[key+' diff']=np.append(curr_blob_error[key+' diff'],
                                                           np.sqrt(np.var(mean_data)))
                    if key == 'Elongation':
                        print(mean_data)
            for key in full_blob_data.keys():
                full_blob_data[key]=np.append(full_blob_data[key],
                                              np.mean(curr_blob_data[key]))
                full_blob_error[key]=np.append(full_blob_error[key],
                                               np.mean(curr_blob_error[key]) /
                                               np.sqrt(len(curr_blob_error[key])))

        pickle.dump(full_blob_data,open(pickle_filename,'wb'))
    else:
        full_blob_data=pickle.load(open(pickle_filename,'rb'))
    return full_blob_data

def read_all_plasma_data(time_range_around_peak=5e-3,
                         nocalc=False):
    pickle_filename=wd+'/processed_data/blob_database_shot_by_shot_plasma.pickle'
    blob_database=read_blob_database(time_range_around_peak=time_range_around_peak)

    if not os.path.exists(pickle_filename) or not nocalc:
        ncalc=len(blob_database['shot'])

        curr_plasma_data=read_plasma_parameters(exp_id=blob_database['shot'][0],
                                                time=blob_database['time'][0])
        full_plasma_data={}
        for key in curr_plasma_data:
            full_plasma_data[key]=[]

        for ind in range(ncalc):
            blob_time=blob_database['time'][ind]
            shot=blob_database['shot'][ind]

            curr_plasma_data=read_plasma_parameters(exp_id=shot,
                                                    time=blob_time)
            for key in curr_plasma_data.keys():
                full_plasma_data[key]=np.append(full_plasma_data[key],
                                                curr_plasma_data[key])

        pickle.dump(full_plasma_data,open(pickle_filename,'wb'))
    else:
        full_plasma_data=pickle.load(open(pickle_filename,'rb'))
    return full_plasma_data


def read_blob_results(shot,
                      time_range,
                      calculate_only=False,
                      nocalc=True):
    # try:
    if True:
        blob_results=analyze_gpi_structures(exp_id=shot,
                                            time_range=time_range,
                                            normalize='simple',
                                            str_finding_method='watershed',
                                            threshold_bg_multiplier=2.,
                                            ellipse_method='linalg',
                                            fit_shape='ellipse',
                                            smooth_contours=5,

                                            tracking='weighted',
                                            matrix_weight={'iou':1,'cccf':0},
                                            ignore_side_structures=True,
                                            remove_orphans=True,
                                            min_structure_lifetime=10,
                                            tracking_assignment='max_score',      #Method of assigning the correspondence, 'hungarian' or 'max_score'
                                            score_threshold=0.7,

                                            nocalc=nocalc,
                                            structure_pixel_calc=False,

                                            test_structures=False,
                                            return_results=not calculate_only,

                                            plot=False,
                                            levels=51,
                                            plot_str_by_str=True,
                                            plot_scatter=True,
                                            plot_tracking=True,
                                            calculate_rough_diff_velocities=False,
                                            plot_for_publication=True,
                                            pdf=False,
                                            structure_pdf_save=False,
                                            structure_video_save=False,

                                            test=False,
                                            )
        if not calculate_only:
            return blob_results

    # except Exception as e:
    #     print('Exception in read_plasa_parameter_db_blob.py line 86.')
    #     print(e)
    #     if not calculate_only:
    #         return None


def read_blob_database(time_range_around_peak=5e-3,
                       blob_db_file='/Users/mlampert/work/NSTX_workspace/db/2010.csv',
                       elm_db_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv',
                       nofilter=False
                       ):

    database=np.asarray(pandas.read_csv(blob_db_file))
    ind_shots=np.where(database[:,2]==0)
    blob_shots=database[ind_shots,0][0,:]
    peak_times=database[ind_shots,1][0,:]/1000.
    blob_database={'shot':blob_shots,
                   'time':peak_times}

    db=pandas.read_csv(elm_db_file, index_col=0)
    elm_shots=np.asarray(db)[:,1]
    elm_times=np.asarray(db)[:,3]
    _elm_database={'shot':elm_shots,
                        'time':elm_times}
    if not nofilter:
        for ind_blob,shot_blob in enumerate(blob_database['shot']):
            ind_overlap=np.where(_elm_database['shot'] == shot_blob)[0]
            blob_time=blob_database['time'][ind_blob]
            if len(ind_overlap) > 0:
                elm_times=_elm_database['time'][ind_overlap]
                min_time=np.min(elm_times)
                max_time=np.max(elm_times)
                if np.logical_and(blob_time < max_time,
                                  blob_time > min_time):
                    if abs(blob_time-min_time) < abs(blob_time-max_time):
                        blob_database['time'][ind_blob] = min_time - 2*time_range_around_peak
                    else:
                        blob_database['time'][ind_blob] = max_time + 2*time_range_around_peak
                if (blob_database['time'][ind_blob] > blob_time+50e3 or
                    blob_database['time'][ind_blob] < blob_time-50e3):

                    blob_database['time'].pop(ind_blob)
                    blob_database['shot'].pop(ind_blob)

        ind=np.where(blob_database['shot'] > 138127)
        blob_database['shot']=blob_database['shot'][ind]
        blob_database['time']=blob_database['time'][ind]

    return blob_database

def read_blob_elm_database(time_range_around_peak=5e-3,
                           blob_db_file='/Users/mlampert/work/NSTX_workspace/db/2010.csv',
                           elm_db_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good_ne.csv',
                           nofilter=False
                           ):

    database=np.asarray(pandas.read_csv(blob_db_file))
    ind_shots=np.where(database[:,2]==0)
    blob_shots=database[ind_shots,0][0,:]
    peak_times=database[ind_shots,1][0,:]/1000.
    blob_database={'shot':blob_shots,
                   'time':peak_times}

    db=pandas.read_csv(elm_db_file, index_col=0)
    elm_shots=np.asarray(db)[:,1]
    elm_times=np.asarray(db)[:,3]
    elm_database={'shot':elm_shots,
                  'time':elm_times}
    database={}
    for key in ['shot','time']:
        database[key]=np.append(blob_database[key],elm_database[key])

    return database

def read_plasma_parameters_for_db(database=None,
                                  print_ranges=False):
    if database is None:
        database=read_blob_database()

    density=[]
    current=[]
    btoroidal=[]
    greenwald=[]
    collisionality=[]
    q95=[]
    pdf_pages_density=PdfPages(wd+'/plots/blob_database_density_fits.pdf')
    pdf_pages_temperature=PdfPages(wd+'/plots/blob_database_temperature_fits.pdf')

    for ind_shot in range(len(database['shot'])):
        print(ind_shot/len(database['shot'])*100,'% done from the calculation.')
        time_curr=database['time'][ind_shot]
        shot=database['shot'][ind_shot]

        start_time=time_mod.time()

        plasma_parameters=read_plasma_parameters(exp_id=shot,time=time_curr,
                                                 pdf_pages_density=pdf_pages_density,
                                                 pdf_pages_temperature=pdf_pages_temperature)

        greenwald.append(plasma_parameters['Greenwald fraction'])
        density.append(plasma_parameters['Line integrated density'])
        q95.append(plasma_parameters['q95'])
        current.append(plasma_parameters['Current'])

        btoroidal.append(plasma_parameters['Toroidal field'])
        collisionality.append(plasma_parameters['Collisionality'])
        print(str((ind_shot+1)/len(database['shot'])*100.)+'% done')
        print('Finished in: ',time_mod.time()-start_time,'s')

    if print_ranges:
        print('Collisionality range: ',min(collisionality),max(collisionality))
        print('Density range: ',min(density), max(density))
        print('Greenwald range: ',min(greenwald),max(greenwald))
        print('BT range: ',min(btoroidal),max(btoroidal))
        print('current range: ',min(current),max(current))

    pdf_pages_density.close()
    pdf_pages_temperature.close()

    return collisionality, q95, greenwald, current, btoroidal, density

def read_plasma_parameters(exp_id=None,
                           time=None,
                           pdf_pages_density=None,
                           pdf_pages_temperature=None
                           ):

    gamma=5/3.
    Z=1.
    k=1.38e-23                                                      #Boltzmann constant
    m_i=2.014*1.66e-27                                               # Deuterium mass
    m_e=9.1093835e-31
    q_e=1.6e-19

    ln_LAMBDA=17
    c=3e8

    Z=1.
    k_B=1.38e-23                                                      #Boltzmann constant

    mu0=4*np.pi*1e-7
    epsilon_0=8.854e-12

    #THESE READ THE ENTIRE SHOT"S PROFILES AND FIT THEM
    ne_params=get_fit_nstx_thomson_profiles(exp_id=exp_id,
                                            density=True,
                                            spline_data=True,
                                            modified_tanh=False,
                                            outboard_only=False,

                                            #flux_coordinates=True,
                                            # flux_range=[0.7,1.1],
                                            device_coordinates=True,
                                            radial_range=[1.3,1.55],
                                            pdf_object=pdf_pages_density,
                                            plot_time_vec=time
                                            )

    te_params=get_fit_nstx_thomson_profiles(exp_id=exp_id,
                                            temperature=True,
                                            spline_data=True,
                                            modified_tanh=False,
                                            outboard_only=False,

                                            #flux_coordinates=True,
                                            # flux_range=[0.7,1.1],
                                            device_coordinates=True,
                                            radial_range=[1.3,1.55],
                                            pdf_object=pdf_pages_temperature,
                                            plot_time_vec=time
                                            )

    ind=np.argmin(np.abs(ne_params['time_vec']-time))
    n_e=ne_params['Value at max'][ind]
    T_e=te_params['Value at max'][ind]

    """Line integrated density"""
    try:
        d_ne=flap.get_data('NSTX_THOMSON',
                        exp_id=exp_id,
                        name='',
                        object_name='THOMSON_DATA',
                        options={'pressure':False,
                                 'temperature':False,
                                 'density':True,
                                 'spline_data':False,
                                 'add_flux_coordinates':False,
                                 'force_mdsplus':False})

        ind=np.argmin(np.abs(d_ne.coordinate('Time')[0][1,:]-time))

        #goodind=np.where(np.logical_and(d_ne.coordinate('Flux r')[0][:,elm_index] < 1.0, d_ne.coordinate('Flux r')[0][:,elm_index] > 0))
        # dR = (d_ne.coordinate('Device R')[0][:,:]-
        #       np.insert(d_ne.coordinate('Device R')[0][0:-1,:],0,0,axis=0))
        norm_factor=(np.max(d_ne.coordinate('Device R')[0][:,:],axis=0)-
                     np.min(d_ne.coordinate('Device R')[0][:,:],axis=0))

        density=(np.trapz(d_ne.data[:,:],
                          d_ne.coordinate('Device R')[0][:,:],
                          axis=0)/norm_factor)[ind]
        #LID=np.sum(((d_ne.data[:,:])[:,:])*dR,axis=0)/np.sum(dR)
    except Exception as e:
        print(e)
        print('Failed to read LID for shot ',exp_id)
        density=np.nan

    """Plasma current"""
    try:
        current=np.mean(flap.get_data('NSTX_MDSPlus',
                                      name='\EFIT02::\IPMEAS',
                                      exp_id=exp_id,
                                      ).slice_data(slicing={'Time':time}).data)
    except Exception as e:
        print(e)
        print('Failed to read current for shot ',exp_id)
        current=np.nan

    """Toroidal field"""
    try:
        b_toroidal=np.mean(flap.get_data('NSTX_MDSPlus',
                                      name='\EFIT02::\BT0',
                                      exp_id=exp_id,
                                      ).slice_data(slicing={'Time':time}).data)
    except Exception as e:
        print(e)
        print('Failed to read Bt for shot ',exp_id)
        b_toroidal=np.nan

    """Minor radius"""
    try:
        a_minor=flap.get_data('NSTX_MDSPlus',
                              name='\EFIT02::\AMINOR',
                              exp_id=exp_id,
                              ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read current for shot ',exp_id)
        a_minor=np.nan

    """Pedestal radius"""
    try:
        R_ped=flap.get_data('NSTX_MDSPlus',
                         name='\EFIT02::\RMIDOUT',
                         exp_id=exp_id,
                         ).slice_data(slicing={'Time':time}).data-0.02
    except Exception as e:
        print(e)
        print('Failed to read RMIDOUT for shot ',exp_id)
        R_ped=np.nan

    """Safety factor"""
    try:
        q95=flap.get_data('NSTX_MDSPlus',
                         name='\EFIT02::\Q95',
                         exp_id=exp_id,
                         ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read Q95 for shot ',exp_id)
        q95=np.nan

    """Lower triangularity"""
    try:
        lower_triang=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\TRIBOT',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read TRIBOT for shot ',exp_id)
        lower_triang=np.nan

    """Upper triangularity"""
    try:
        upper_triang=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\TRITOP',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read TRITOP for shot ',exp_id)
        upper_triang=np.nan

    """Elongation"""
    try:
        elongation=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\KAPPA',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read KAPPA for shot ',exp_id)
        elongation=np.nan

    """Inner gap"""
    try:
        inner_gap=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\GAPIN',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read GAPIN for shot ',exp_id)
        inner_gap=np.nan

    """Outer gap"""
    try:
        outer_gap=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\GAPOUT',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read GAPOUT for shot ',exp_id)
        outer_gap=np.nan

    """Current density at psi_norm=0.95"""
    try:
        cdens_95=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\J95N',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read J95N for shot ',exp_id)
        cdens_95=np.nan

    """Current density at psi_norm=0.99"""
    try:
        cdens_99=flap.get_data('NSTX_MDSPlus',
                             name='\EFIT02::\J99N',
                             exp_id=exp_id,
                             ).slice_data(slicing={'Time':time}).data
    except Exception as e:
        print(e)
        print('Failed to read J99N for shot ',exp_id)
        cdens_99=np.nan


    omega_pe=np.sqrt(n_e*q_e**2/m_e/epsilon_0)

    #delta_e=c/omega_pe #It's a good question what this is.

    c_s=np.sqrt(gamma*Z*k*(T_e)/m_i)

    n_i=n_e
    tau_ei= (12 * np.pi**(1.5) / np.sqrt(2) * np.sqrt(m_e) * T_e**(1.5) *
             epsilon_0**2 / (n_i * Z**2 * q_e**4 * np.log(ln_LAMBDA)))
    ei_collision_rate=1/tau_ei

    collisionality=ei_collision_rate/k_B*T_e/(q95*R_ped)
    greenwald_fraction=density/(current.copy()/(np.pi*a_minor**2)*1e14)

    return {'Line integrated density':density,
            'Current':current,
            'Greenwald fraction':greenwald_fraction,
            'Toroidal field':b_toroidal,
            'Collisionality':collisionality,
            'q95':q95,
            'Sound speed':c_s,
            'Plasma frequency':omega_pe,
            'Plasma elongation':elongation,
            'Plasma triangularity upper':upper_triang,
            'Plasma triangularity lower':lower_triang,
            'Plasma triangularity':(upper_triang+lower_triang)/2,
            'Inner gap':inner_gap,
            'Outer gap':outer_gap,
            'Current density at 95':cdens_95,
            'Current density at 99':cdens_99,
            }