#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:03:12 2022

@author: mlampert
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import copy

import numpy as np
import pickle

import flap
import flap_nstx
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages

import pyuda

pyuda.Client.server='uda2.mast.l'


def calculate_magnetics_spectrogram_mast(exp_id=None,
                                         time_range=None,
                                         channel=1,
                                         time_res=1e-3,
                                         freq_res=None,
                                         frange=[1e3,100e3],
                                         recalc=False,
                                         plot=True,
                                         pdf=False,
                                         pdfobject=None,
                                         ):

    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    filename=flap_nstx.tools.filename(exp_id=exp_id,
                                      working_directory=wd+'/processed_data',
                                      time_range=time_range,
                                      comment='magnetic_spectrogram_hf_ch'+str(channel)+'_tr_'+str(time_res)+'_frange_'+str(frange[0])+'_'+str(frange[1]),
                                      extension='pickle')

    if not recalc and not os.path.exists(filename):
        print('File doesn\'t exist, needs to be calculated!')
        recalc=True
    if recalc or not os.path.exists(filename):
        if freq_res is None:
            freq_res=2/time_res
        client=pyuda.Client()
        data_obj=client.get('/XMC/ACQ216_202/CH13', str(exp_id))

        data=data_obj.data
        time=data_obj.time.data
        coord=[]
        coord.append(copy.deepcopy(flap.Coordinate(name  = 'Time',
                                                   unit  = 's',
                                                   mode  = flap.CoordinateMode(equidistant=True),
                                                   shape = [],
                                                   start = time[0],
                                                   step  = time[1]-time[0],
                                                   dimension_list=[0])))

        coord.append(copy.deepcopy(flap.Coordinate(name  = 'Sample',
                                                   unit  = '',
                                                   mode  = flap.CoordinateMode(equidistant=True),
                                                   shape = [],
                                                   start = 0,
                                                   step  = 1,
                                                   dimension_list=[0])))

        magnetics = flap.DataObject(data_array=data,
                                    data_unit=flap.Unit(name='',unit='V'),
                                    coordinates=coord,
                                    exp_id=exp_id,
                                    data_title='MAST_OMAHA_data',
                                    data_source="MAST_OMAHA")


        magnetics.coordinates.append(copy.deepcopy(flap.Coordinate(name='Time equi',
                                                   unit='s',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   shape = [],
                                                   start=magnetics.coordinate('Time')[0][0],
                                                   step=magnetics.coordinate('Time')[0][1]-magnetics.coordinate('Time')[0][0],
                                                   dimension_list=[0])))
        flap.add_data_object(magnetics, 'MIRNOV')
        n_time=int((time_range[1]-time_range[0])/time_res)


        spectrum=[]
        for i in range(n_time-1):
            spectrum.append(flap.apsd('MIRNOV',
                                      coordinate='Time equi',
                                      intervals={'Time equi':flap.Intervals(time_range[0]+(i-0.5)*time_res,
                                                                            time_range[0]+(i+1.5)*time_res)},
                                      options={'Res':freq_res,
                                               'Range':frange,
                                               'Interval':1,
                                               'Trend':None,
                                               'Logarithmic':False,
                                               'Hanning':True},
                                      output_name='MIRNOV_TWIN_APSD').data)
        time=np.arange(n_time-1)*time_res+time_range[0]
        freq=flap.get_data_object_ref('MIRNOV_TWIN_APSD').coordinate('Frequency')[0]
        data=np.asarray(spectrum).T
        pickle.dump((time,freq,data), open(filename, 'wb'))
    else:
        time, freq, data = pickle.load(open(filename, 'rb'))
        print(f"{time.shape},{freq.shape},{data.shape}")
    if plot:
        import matplotlib
        matplotlib.use('QT5Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
    if pdf:
        filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                         working_directory=wd+'/plots',
                                         time_range=time_range,
                                         comment='magnetic_spectrogram_hf_ch'+str(channel)+'_tr_'+str(time_res)+'_frange_'+str(frange[0])+'_'+str(frange[1]),
                                         extension='pdf')
        spectrogram_pdf=PdfPages(filename)
    plt.figure()
    plt.contourf(time,
                 freq/1000.,
                 data,
                 locator=ticker.LogLocator(),
                 cmap='jet',
                 levels=101)
    plt.title('BDOT_L1DMIVVHF'+str(channel)+' spectrogram for '+str(exp_id)+' with fres '+str(1/time_res/1000.)+'kHz')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [kHz]')
    plt.pause(0.001)

    if pdf:
        spectrogram_pdf.savefig()
        spectrogram_pdf.close()