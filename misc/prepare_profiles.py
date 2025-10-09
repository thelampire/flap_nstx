#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:49:00 2024

@author: mlampert
"""

import os
import numpy as np
import MDSplus as mds

def prepare_profiles(exp_id=139901,
                     time=0.325):
    data={}
    conn=mds.Connection('skylark.pppl.gov')
    conn.openTree('ACTIVESPEC', exp_id)
    time_vec_ts=conn.get('\\TS_BEST:TS_TIMES').data()
    rad_coord_ts=conn.get('\\TS_BEST:FIT_RADII').data()/100.
    
    data['time_ts'] = time_vec_ts
    data['rad_coord_ts'] = rad_coord_ts
    
    e_temp=conn.get('\\TS_BEST:FIT_TE').data()
    e_temp_err=conn.get('\\TS_BEST:FIT_TE_ERR').data()
    
    data['Te']=e_temp
    data['Te error']=e_temp_err
    
    e_dens=conn.get('\\TS_BEST:FIT_NE').data()
    e_dens_err=conn.get('\\TS_BEST:FIT_NE_ERR').data()
    
    data['ne']=e_dens
    data['ne error']=e_dens_err
    
    e_press=conn.get('\\TS_BEST:FIT_PE').data()
    e_press_err=conn.get('\\TS_BEST:FIT_PE_ERR').data()
    
    data['pe']=e_press
    data['pe error']=e_press_err
    
    
    time_vec_cx=conn.get('dim_of(\\TOP.CHERS.ANALYSIS.CT1:TI,1)').data()
    rad_coord_cx=conn.get('dim_of(\\TOP.CHERS.ANALYSIS.CT1:TI,0)').data()/100.
    
    data['time_cx'] = time_vec_cx
    data['rad_coord_cx'] = rad_coord_cx
    
    i_temp=conn.get('\\TOP.CHERS.ANALYSIS.CT1:TI').data()
    i_temp_err=conn.get('\\TOP.CHERS.ANALYSIS.CT1:DTI').data()
    
    data['Ti'] = i_temp
    data['Ti error'] = i_temp_err
    
    i_dens=conn.get('\\TOP.CHERS.ANALYSIS.CT1:ND').data()
    i_dens_err=conn.get('\\TOP.CHERS.ANALYSIS.CT1:DND').data()
    
    data['ni'] = i_dens
    data['ni error'] = i_dens_err
    
    v_tor=conn.get('\\TOP.CHERS.ANALYSIS.CT1:VT').data()
    v_tor_err=conn.get('\\TOP.CHERS.ANALYSIS.CT1:DVT').data()
    
    data['v toroidal']=v_tor
    data['v toroidal error']=v_tor_err
    
    c_dens=conn.get('\\TOP.CHERS.ANALYSIS.CT1:NC').data()
    c_dens_err=conn.get('\\TOP.CHERS.ANALYSIS.CT1:DNC').data()
    
    data['n_C6']=c_dens
    data['n_C6 error']=c_dens_err
    
    z_eff=conn.get('\\TOP.CHERS.ANALYSIS.CT1:ZEFF').data()
    z_eff_err=conn.get('\\TOP.CHERS.ANALYSIS.CT1:DZEFF').data()
    
    data['Zeff']=z_eff
    data['Zeff error']=z_eff_err

    try:
        conn.openTree('EFIT02',exp_id)    
        
        data_psirz=conn.get('\PSIRZ').data()
        time_psirz=conn.get('dim_of(\PSIRZ,0)').data()
        rad_coord_psirz=conn.get('dim_of(\PSIRZ,1)').data()

        data_ssimag=conn.get('\SSIMAG').data()

        data_ssibry=conn.get('\SSIBRY').data()

        
        # R_data=flap.get_data('NSTX_MDSPlus',
        #                      name='\EFIT02::\R',
        #                      exp_id=exp_id,
        #                      object_name='R_FOR_COORD')
        # PSI_norm_data=flap.get_data('NSTX_MDSPlus',
        #                          name='\EFIT02::\PSIN',
        #                          exp_id=exp_id,
        #                          object_name='PSIN')
        
    except:
        raise ValueError("The PSIRZ MDSPlus node is missing.")
   
    """#thomson_mappsing:"""
    
    psi_n=(data_psirz-data_ssimag[:,None,None])/(data_ssibry-data_ssimag)[:,None,None]
    psi_n[np.isnan(psi_n)]=0.
               
    psi_n=psi_n[:,32,:] #psi_n is transposed originally
    psi_t_coord=time_psirz
    psi_r_coord=rad_coord_psirz
    
    #Do the interpolation
    #psi_values_spat_interpol=np.zeros([thomson_r_coord.shape[0],
    #                                   psi_t_coord.shape[0]])
    psi_values_ts=np.zeros([e_temp.shape[0], time_vec_ts.shape[0]])
    
    for index_t in range(len(time_vec_ts)):
        ind_t_efit=np.argmin(np.abs(psi_t_coord-time_vec_ts[index_t]))
        psi_values_ts[:,index_t]=np.interp(rad_coord_ts,
                                           psi_r_coord[ind_t_efit,:],
                                           psi_n[ind_t_efit,:])
        
    psi_values_ts[np.isnan(psi_values_ts)]=0.
    
    psi_values_cx=np.zeros([i_temp.shape[1], time_vec_cx.shape[0]])
    
    for index_t in range(len(time_vec_cx)):
        ind_t_efit=np.argmin(np.abs(psi_t_coord - time_vec_cx[index_t]))
        psi_values_cx[:,index_t]=np.interp(rad_coord_cx,
                                           psi_r_coord[ind_t_efit,:],
                                           psi_n[ind_t_efit,:])
        
    psi_values_cx[np.isnan(psi_values_cx)]=0.
    
    data['Psi_norm TS']=psi_values_ts
    data['Psi_norm CX']=psi_values_cx
    
    import pickle
    pickle.dump(data,open('/Users/mlampert/work/NSTX_workspace/'+str(exp_id)+'_profiles.pickle','wb'))


    # text_file=open('profiles_NSTX' + str(exp_id) + '_' + str(int(time*1e3)) + '.txt', 'wt')
    # for i in range(len(psi_norm)):
    #     text_file.write(str(psi_norm[i])+'\t\t'+str(d_temp_slice.data[i])+'\t\t'+str(d_dens_slice.data[i])+'\t\t'+str(Z_eff[i])+'\t\t'+str(q_eff[i])+'\n')
    # text_file.close()