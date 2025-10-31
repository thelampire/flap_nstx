#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:11:35 2025

@author: mlampert
"""
import os
import copy
import time as time_module
import pickle

import MDSplus as mds
import numpy as np

def read_nstx_mdsplus_data(shot=None,
                           server='skylark.pppl.gov:8505',
                           tree=None,
                           node=None,
                           read_local_data=True,
                           return_xarray=False,
                           quantity_name=None,
                           description=None,
                           spatial_coordinate_names=['radial','vertical'],
                           ):
    if return_xarray:
        import xarray as xr
        import pint_xarray
    if shot is None or tree is None or node is None:
        raise ValueError('shot, tree and node needs to be set.')
    
    time_units=['us','usec','ms','msec','s','sec', 'seconds']
    spatial_units=['mm','cm','m']
    
    conn=mds.Connection(server)
    conn.openTree(tree,shot)
    
    data=conn.get(node).data()
    n_dim=len(data.shape)
    
    data_unit=conn.get(f'units_of({node})').data()
    dim_unit=[]
    
    spat_dim=[]
    spat_unit=[]
    
    for ind_dim in range(n_dim):
        dim_unit.append(str(conn.get(f'units_of(dim_of({node},{ind_dim}))')).lower())
        if dim_unit[-1] in time_units:
            time_dim=ind_dim
            time_unit=dim_unit[-1]
            
        elif dim_unit[-1] in spatial_units:
            spat_dim.append(ind_dim)
            spat_unit.append(dim_unit[-1])
            
    
    print(dim_unit)
    if 'EFIT' in tree:
        time_dim=0
        dim_unit=['sec']
        if n_dim == 2:
            spat_dim=np.arange(n_dim-1)+1
            for ind_dim in range(1,n_dim):
                coord_data=conn.get(f'dim_of({node},{ind_dim})')
                if len(coord_data.shape) == 1 and np.abs(coord_data[1]-coord_data[0]-1) < 1e-6:
                    dim_unit.append('sample')
                else:
                    dim_unit.append('m')
        else:
            spat_dim=None
        print(dim_unit)
        
        time_dim_unit=dim_unit[0]
        spat_dim_unit=dim_unit[1:]

    time_data=conn.get(f'dim_of({node},{time_dim})').data()
    
    if data.shape[time_dim] != time_data.shape[0]:
        print('Size of the array along the time dimension does not match the size of the time array. Setting time dim to fix the mistake.')
        time_dim=np.where(np.asarray(data.shape) == time_data.shape[0])[0]
        spat_dim=np.where(np.asarray(data.shape) != time_data.shape[0])[0]
        
        if len(time_dim) > 1:
            raise ValueError('There are two dimensions having the same size as the time vector. The temporal dimension cannot be deducted unambigously.')
        else:
            time_dim=int(time_dim)
        
    if spat_dim is not None:
        coord_data=[]
        for ind_spat_dim in spat_dim:
            coord_data_curr=conn.get(f'dim_of({node},{ind_spat_dim})').data()
            coord_data.append(coord_data_curr)
        if time_dim != 0:
            data=np.transpose(data,axes=tuple(list([time_dim])+list(spat_dim)))
            dim_unit=[dim_unit[time_dim]]+[dim_unit[ind_dim] for ind_dim in spat_dim]
            time_dim=0
            spat_dim=np.asarray(spat_dim)+1

    conn.disconnect()
    print(time_dim, spat_dim)
    print(time_data, data, dim_unit)
    if spat_dim is not None: print(coord_data)
    
    if not return_xarray:
        
        return {'data':data,
                'data unit':data_unit,
                
                'time':time_data,
                'time dim':time_dim,
                'time unit':time_dim_unit,
                
                'spatial coord':coord_data,
                'spatial unit':spat_dim_unit,
                'spatial dim':spat_dim,
                
                'source':'mdsplus',
                'tree':tree,
                'node':node,
                }
    else:
        if quantity_name == None: quantity_name = node
        coords=None
        dim_names=['time']
        coords={'time':time_data}
        if spat_dim is not None:
            for ind_spat_dim,_ in enumerate(spat_dim):
                
                if dim_unit[ind_spat_dim] in spatial_units:
                    dim_names.append(spatial_coordinate_names[ind_spat_dim])
                    curr_coord_name=spatial_coordinate_names[ind_spat_dim]
                    
                elif dim_unit[ind_spat_dim] == 'sample':
                    dim_names.append('sample')
                    curr_coord_name='sample'
                    
                else:
                    print(dim_unit[ind_spat_dim])
                    dim_names.append(f'unknown dimension #{ind_spat_dim}')
                    curr_coord_name=f'unknown dimension #{ind_spat_dim}'
                    
                coords[curr_coord_name]=coord_data[ind_spat_dim]
                
        print(spatial_coordinate_names)
        print(spat_dim)
        # print(spat_coord)
        print(data.shape, dim_names)
        print(coords)
        print(dim_names)
        print(data.shape)
        print(dim_unit)
        print(data_unit)
        #pint cannot handle m3 like units
        
        if set(np.arange(10).astype(str)) & set(data_unit) and "^" not in dim_unit[ind_dim]:
            for ind_str,string in enumerate(data_unit):
                if string in 'qwertyuiopasdfghjklzxcvbnm' and data_unit[ind_str+1] in '1234567890-':
                    data_unit=data_unit[0:ind_str+1]+'^'+data_unit[ind_str+1:]
                    break
                
        ds=xr.DataArray(data, 
                        coords=coords,
                        dims=dim_names,
                        name=quantity_name,
                        attrs={"units":data_unit, 
                               "description":description})

        for ind_dim,dim_name in enumerate(dim_names):
            if set(np.arange(10).astype(str)) & set(dim_unit[ind_dim]) and "^" not in dim_unit[ind_dim]:
                for ind_str,_ in enumerate(dim_unit[ind_dim]):
                    if string in 'qwertyuiopasdfghjklzxcvbnm' and dim_unit[ind_dim][ind_str+1] in '1234567890-':
                        dim_unit[ind_dim]=dim_unit[ind_dim][0:ind_str+1]+'^'+dim_unit[ind_dim][ind_str+1:]
                        break
                    
            ds.coords[dim_name].attrs['units']=dim_unit[ind_dim]
            
        try:
        # if True:
            ds.pint.quantify()
        except:
            pass
        return ds