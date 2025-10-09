#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:18:08 2025

@author: mlampert
"""

def nstx_pedestal_database_header():
    return ['tokamak',
            'shot','regime','wall_condition','start_time','end_time','gas_a',
            'gas_z','gas_minority_a','gas_minority_z','author','eq_notes',
            'i_p','b_t0','r_0','z_0','r_geo','a_minor','kappa','tri_lower',
            'tri_upper','lambda_upper','lambda_lower','volume','cx_area',
            'surf_area','drsep','q95','qmin','qstar','w_mhd','dwdt_mhd',
            'w_dia','dwdt_dia','pres','beta_t','beta_p','beta_n','v_surf','li',
            'p_aux_notes','p_rad','dp_rad_dt','p_rad_div','p_rf_ech','p_rf_lh',
            'p_rf_icrh','p_rf_hhfw','p_nbi_inj','p_nbi','t_nbi','cd_rf',
            'cd_nbi','kinetic_notes','p_ped','p_ped_location','ped_width',
            'ne_width','te_width','ped_location','ne_ped','ne_sep','te_sep',
            'ti_sep','te_ped','ti_ped','z_eff_ped','ne_90','te_90','pe_90',
            'ne_95','te_95','pe_95','ti_90','ti_95','ti_core','te_core',
            'v_core','zeff_core','ne','gas_meff','fbs','pres_fast','pres_th',
            'w_fast','w_th','dw_th','dt','kappa_areal','n_gr','i_norm','reff',
            'q95in','p_oh','p_rf','p_aux','p_loss','taue_th','taue',
            'aspect_ratio','h98','h89','dr_dpsin','drho_dpsin','circ',
            'p_rf_icrh_abs']
    
def nstx_pedestal_database_dictionary():
    
    data_dict={'mds_tree':'',
               'mds_node':'',
               'label':'',
               'unit':'',
               'multiplier':1.,
               'description':'',
               'source':'',
               'data':None,
               'data error':None,
               'db column #': -1,
               }
    
    """
    THESE ARE EQUILIBRIUM RECONSTRUCTION RELATED DATA
    """
    full_plasma_dict={}
    full_plasma_dict['Shot data']={'shot':None,
                                   'time_range':None,
                                   'regime':None,
                                   'wall conditioning':'C+Li',
                                   'gas_a':2,
                                   'gas_z':1,
                                   'gas_minority_a':12,
                                   'gas_minority_b':6,
                                   'author':'mlampert',
                                   'equilibrium':'EFIT02'
                                    }
    
    """Read measured plasma current"""
    full_plasma_dict['Plasma current']=         {'mds_tree':     'EFIT02',
                                                 'mds_node':     'IPMEAS',
                                                 'label':        '$I_p$',
                                                 'unit':         'A',                    #Maybe this should be read
                                                 'multiplier':   1.,                        #Along with this one
                                                 'description':  'plasma current',
                                                 'source':       'mdsplus',
                                                 'db column #': 1,
                                                 }
    
    """Read plasma toroidal field"""
    full_plasma_dict['Toroidal field']=         {'mds_tree':     'EFIT02',
                                                 'mds_node':     'BT0',
                                                 'label':        '$B_t$',
                                                 'unit':         'T',                    #Maybe this should be read
                                                 'multiplier':   1.,                     #Along with this one
                                                 'description':  'Toroidal field',
                                                 'source':       'mdsplus',
                                                 'db column #': 2,
                                                 }
    
    """Read magnetic axis R"""
    full_plasma_dict['Magnetic axis R']=        {'mds_tree':     'EFIT02',
                                                 'mds_node':     'RMAXIS',
                                                 'label':        '$R_0$',
                                                 'unit':         'm',                    #Maybe this should be read
                                                 'multiplier':   1.,                        #Along with this one
                                                 'description':  'Radial position of the magnetic axis',
                                                 'source':       'mdsplus',
                                                 'db column #': 3,
                                                 }       
    """Read magnetic axis z"""
    full_plasma_dict['Magnetic axis z']=        {'mds_tree':     'EFIT02',
                                                 'mds_node':     'RMAXIS',
                                                 'label':        '$z_0$',
                                                 'unit':         'm',                    #Maybe this should be read
                                                 'multiplier':   1.,                        #Along with this one
                                                 'description':  'Vertical position of the magnetic axis',
                                                 'source':       'mdsplus',
                                                 'db column #': 4,
                                                 }   
        
    """Read geometrical radius"""
    full_plasma_dict['Geometrical radius']=           {'mds_tree':     'EFIT02',
                                                       'mds_node':     'RSURF',
                                                       'label':        '$R_{geo}$',
                                                       'unit':         'm',                    #Maybe this should be read
                                                       'multiplier':   1.,                        #Along with this one
                                                       'description':  'Geometric radius',
                                                       'source':       'mdsplus',
                                                       'db column #': 5,
                                                       }
    
    """Read minor radius"""
    full_plasma_dict['Minor radius']=           {'mds_tree':     'EFIT02',
                                                 'mds_node':     'AMINOR',
                                                 'label':        'a',
                                                 'unit':         'm',                    #Maybe this should be read
                                                 'multiplier':   1.,                        #Along with this one
                                                 'description':  'Minor radius',
                                                 'source':       'mdsplus',
                                                 'db column #': 6,
                                                 }
    
    """Read elongation (kappa)"""
    full_plasma_dict["Elongation"]=            {'mds_tree':     'EFIT02',
                                                'mds_node':     'KAPPA',
                                                'label':        '',
                                                'unit':         '',               
                                                'multiplier':   1.,               
                                                'description':  'Plasma elongation',
                                                'source':       'mdsplus',
                                                 'db column #': 7,
                                                }
    
    """Read lower triangularity"""
    full_plasma_dict['Lower triangularity']=    {'mds_tree':     'EFIT02',
                                                 'mds_node':     'TRIBOT',
                                                 'label':        '$\delta_{low}$',
                                                 'unit':         '',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'Lower triangularity',
                                                 'source':       'mdsplus',
                                                 'db column #':  8,
                                                 }
    """Read upper triangularity"""
    full_plasma_dict['Upper triangularity']=    {'mds_tree':     'EFIT02',
                                                 'mds_node':     'TRITOP',
                                                 'label':        '$\delta_{low}$',
                                                 'unit':         '',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'Lower triangularity',
                                                 'source':       'mdsplus',
                                                 'db column #':  9,
                                                 }


    """Read lambda upper"""
    full_plasma_dict['Upper squareness']=    {'mds_tree':        '',
                                              'mds_node':     '',
                                              'label':        '$\lambda_{upper}$',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  'Upper squareness',
                                              'source':       'Squareness calculation',
                                              'db column #':  10,
                                              }
    """Read lambda lower"""
    full_plasma_dict['Lower squareness']=    {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$\lambda_{lower}$',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  'Lower squareness',
                                              'source':       'Squareness calculation',
                                              'db column #':  11,
                                              }
    """Read plasma volume"""
    full_plasma_dict['Volume']=              {'mds_tree':     'EFIT02',
                                                 'mds_node':     'VOLUME',
                                                 'label':        '$V$',
                                                 'unit':         'm3',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'Plasma volume',
                                                 'source':       'mdsplus',
                                                 'db column #':  12,
                                                 }
    
    """Read cx_area"""
    full_plasma_dict['Area crossectional']=     {'mds_tree':     'EFIT02',
                                                 'mds_node':     'AREA',
                                                 'label':        '$A_{CX}$',
                                                 'unit':         '$m^2$',    
                                                 'multiplier':   1.,                        
                                                 'description':  'Crossectional area',
                                                 'source':       'mdsplus',
                                                 'db column #':  13,
                                                 }
    """Read surface area"""
    full_plasma_dict['Plasma surface']=         {'mds_tree':     'EFIT02',
                                                 'mds_node':     'PSURFA',
                                                 'label':        '$A_{plasma}$',
                                                 'unit':         '$m^2$',                    
                                                 'multiplier':    1.,                        
                                                 'description':  'Surface area of the plasma',
                                                 'source':       'mdsplus',
                                                 'db column #':  14,
                                                 }
    """Read dr sep"""
    full_plasma_dict['DRSEP']=                  {'mds_tree':     'EFIT02',
                                                 'mds_node':     'DRSEP',
                                                 'label':        '$dR_{sep}$',
                                                 'unit':         'm',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'DR_sep',
                                                 'source':       'mdsplus',
                                                 'db column #':  15,
                                                 }
    """Read q95"""
    full_plasma_dict['q95']=                    {'mds_tree':     'EFIT02',
                                                 'mds_node':     'Q95',
                                                 'label':        '$q_{95}$',
                                                 'unit':         '',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'Safety factor at psi_norm=0.95',
                                                 'source':       'mdsplus',
                                                 'db column #':  16,
                                                 
                                                 }
    
    """Read qmin"""
    full_plasma_dict['QMIN']=                   {'mds_tree':     'EFIT02',
                                                 'mds_node':     'QMIN',
                                                 'label':        '$q_{min}$',
                                                 'unit':         '',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'Minimum safety factor',
                                                 'source':       'mdsplus',
                                                 'db column #':  17,
                                                 }
    
    """Read qstar"""
    full_plasma_dict['QSTAR']=                  {'mds_tree':     'EFIT02',
                                                 'mds_node':     'QSTAR',
                                                 'label':        '$q_{\\ast}$',
                                                 'unit':         '',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'Equivalent (kink) safety factor',
                                                 'source':       'mdsplus',
                                                 'db column #':  18,
                                                 }
    
    """Read w_mhd"""
    full_plasma_dict['Plasma energy']=          {'mds_tree':     'EFIT02',
                                                 'mds_node':     'WMHD',
                                                 'label':        '$W_{MHD}$',
                                                 'unit':         'J',                    
                                                 'multiplier':    1.,                        
                                                 'description':  'Plasma energy',
                                                 'source':       'mdsplus',
                                                 'db column #':  19,
                                                 }
    """Read dwdt_mhd"""
    full_plasma_dict['Plasma energy dt']=       {'mds_tree':     'EFIT02',
                                                 'mds_node':     'WPDOT',
                                                 'label':        '$dW_{MHD}/dt$',
                                                 'unit':         'J/s', 
                                                 'multiplier':    1.,                        
                                                 'description':  'Plasma energy time derivative',
                                                 'source':       'mdsplus',
                                                 'db column #':  20,
                                                 }
    
    """Read w_dia"""
    full_plasma_dict['Plasma diamagnetic energy']=      {'mds_tree':     'EFIT02',
                                                         'mds_node':     'WDIA',
                                                         'label':        '$W_{diam}$',
                                                         'unit':         'J', 
                                                         'multiplier':    1.,                        
                                                         'description':  'Diamagnetic plasma energy',
                                                         'source':       'mdsplus',
                                                         'db column #':  21,
                                                         }
    """Read dwdt_dia"""
    full_plasma_dict['Plasma diamagnetic energy time derivative']=      {'mds_tree':     'EFIT02',
                                                                         'mds_node':     'WDIA',
                                                                         'label':        '$dWdt_{diam}$',
                                                                         'unit':         'J', 
                                                                         'multiplier':    1.,                        
                                                                         'description':  'Diamagnetic plasma energy time derivative',
                                                                         'source':       'mdsplus time derivative',
                                                                         'db column #':  22,
                                                                         }
    
    """Read pressure"""
    full_plasma_dict['Total pressure']=          {'mds_tree':     'EFIT02',
                                                 'mds_node':     'PRES',
                                                 'label':        '$p_{avg}$',
                                                 'unit':         '',    
                                                 'multiplier':   1.,                        
                                                 'description':  'Average pressur volume averaged',
                                                 'source':       'mdsplus',
                                                 'db column #':  23,
                                                 }
    """Read beta toroidal"""
    full_plasma_dict['Beta toroidal']=          {'mds_tree':     'EFIT02',
                                                 'mds_node':     'BETAT',
                                                 'label':        '$\\beta_{t}$',
                                                 'unit':         '',    
                                                 'multiplier':   1.,                        
                                                 'description':  'Toroidal beta',
                                                 'source':       'mdsplus',
                                                 'db column #':  24,
                                                 }
    """Read beta poloidal"""
    full_plasma_dict['Beta poloidal']=          {'mds_tree':     'EFIT02',
                                                 'mds_node':     'BETAP',
                                                 'label':        '$\\beta_{p}$',
                                                 'unit':         '',    
                                                 'multiplier':   1.,                        
                                                 'description':  'Poloidal beta',
                                                 'source':       'mdsplus',
                                                 'db column #':  25,
                                                 }
    """Read beta normalized"""
    full_plasma_dict['Beta normalized']=        {'mds_tree':     'EFIT02',
                                                 'mds_node':     'BETAN',
                                                 'label':        '$\\beta_{N}$',
                                                 'unit':         '',    
                                                 'multiplier':   1.,                        
                                                 'description':  'Normalized beta',
                                                 'source':       'mdsplus',
                                                 'db column #':  26,
                                                 }
    """Read v_surf""" #???
    full_plasma_dict['Voltage loop']=            {'mds_tree':     'EFIT02',
                                                 'mds_node':     'VSURF',
                                                 'label':        '$U_{surf}$',
                                                 'unit':         '',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'Loop voltage',
                                                 'source':       'mdsplus',
                                                 'db column #':  27,
                                                 }
    
    """Read li internal inductance"""
    full_plasma_dict['Internal inductance']=    {'mds_tree':     'EFIT02',
                                                 'mds_node':     'LI',
                                                 'label':        '$l_{i}$',
                                                 'unit':         '',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'Internal inductance',
                                                 'source':       'mdsplus',
                                                 'db column #':  28,
                                                 }
    
    """
    THESE ARE HEATING RELATED DATA FROM HERE
    """
    
    """Read p_aux_notes"""
    full_plasma_dict['Heating power note']=     {'mds_tree':      '',
                                                 'mds_node':     '',
                                                 'label':        '',
                                                 'unit':         '',                    
                                                 'multiplier':   None,                        
                                                 'description':  'Note for auxiliary heating',
                                                 'data':          '',
                                                 'source':        '',
                                                 'db column #':  29,
                                                  }
    
    """Read p_rad"""
    full_plasma_dict['Radiated power']=          {'mds_tree':      'WF',
                                                  'mds_node':     'PRAD',
                                                  'label':        '$P_{rad}$',
                                                  'unit':         'W',                    
                                                  'multiplier':   1.,                        
                                                  'description':  'Radiated power',
                                                  'source':       'mdsplus',
                                                  'db column #':  30,
                                                  }
    """Read dp_rad_dt"""
    full_plasma_dict['Radiated power time derivative']=      {'mds_tree':     'EFIT02',
                                                              'mds_node':     'WDIA',
                                                              'label':        '$dWdt_{diam}$',
                                                              'unit':         'J', 
                                                              'multiplier':    1.,                        
                                                              'description':  'Diamagnetic plasma energy time derivative',
                                                              'source':       'mdsplus time derivative',
                                                              'db column #':  31,
                                                              }
    
    """Read p_rad_div"""
    full_plasma_dict['Radiated power divertor']= {'mds_tree':      '',
                                                  'mds_node':     '',
                                                  'label':        '$P_{rad, div}$',
                                                  'unit':         'W',                    
                                                  'multiplier':   1.,                        
                                                  'description':  'Radiated power in the divertor',
                                                  'source':       'mdsplus',
                                                  'db column #':  32,
                                                  }
    """Read p_rf_ech"""
    full_plasma_dict['Heating power ECH']=       {'mds_tree':      'RF',
                                                  'mds_node':     'ECHPOWER',
                                                  'label':        '$p_{ECH}$',
                                                  'unit':         'W',                    
                                                  'multiplier':   1.,                        
                                                  'description':  'ECH heating power',
                                                  'source':       'mdsplus',
                                                  'db column #':  33,
                                                  }
    """Read p_rf_lh"""
    full_plasma_dict['Heating power LH']=       {'mds_tree':     '',
                                                 'mds_node':     '',
                                                 'label':        '$p_{LH}$',
                                                 'unit':         'W',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'Lower hybrid heating power',
                                                 'source':       'mdsplus',
                                                 'db column #':  34,
                                                 }
    """Read p_rf_icrh"""
    full_plasma_dict['Heating power ICRH']=     {'mds_tree':     '',
                                                 'mds_node':     '',
                                                 'label':        '$p_{ICRH}$',
                                                 'unit':         'W',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'ICRH heating power',
                                                 'source':       'mdsplus',
                                                 'db column #':  35,
                                                 }
    """Read p_rf_hhfw"""
    full_plasma_dict['Heating power HHFW']=     {'mds_tree':     'RF',
                                                 'mds_node':     'HHFW_POWER',
                                                 'label':        '$p_{HHFW}$',
                                                 'unit':         'W',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'HHFW heating power',
                                                 'source':       'mdsplus',
                                                 'db column #':  36,
                                                 }
    """Read p_nbi_inj"""
    full_plasma_dict['Heating power NBI injected']= {'mds_tree':     'WF',
                                                     'mds_node':     'PNB',
                                                     'label':        '$P_{NBI}$',
                                                     'unit':         'W',                    
                                                     'multiplier':   1.,                        
                                                     'description':  'NBI heating power',
                                                     'source':       'mdsplus',
                                                     'db column #':  37,
                                                     }
    """Read p_nbi"""
    full_plasma_dict['NBI power']=                 {'mds_tree':     'WF',
                                                    'mds_node':     'HHFW_POWER',
                                                    'label':        '$P_{HHFW}$',
                                                    'unit':         'W',                    
                                                    'multiplier':   1.,                        
                                                    'description':  'NBI heating power',
                                                    'source':       'mdsplus',
                                                    'db column #':  38,
                                                    }
    """Read t_nbi"""
    full_plasma_dict['NBI start time']=             {'mds_tree':     '',
                                                     'mds_node':     '',
                                                     'label':        '$t_{0,NBI}$',
                                                     'unit':         's',                    
                                                     'multiplier':   1.,                        
                                                     'description':  'NBI heating power',
                                                     'source':       'mdsplus threshold time',
                                                     'db column #':  39,
                                                     }
    
    """Read cd_rf"""
    full_plasma_dict['Current drive RF']=       {'mds_tree':     '',
                                                 'mds_node':     '',
                                                 'label':        '$I_{CD,RF}$',
                                                 'unit':         'A',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'RF current drive',
                                                 'source':       'mdsplus',
                                                 'db column #':  40,
                                                 }
    """Read cd_nbi (NBI current drive)"""
    full_plasma_dict['Current drive NBI']=      {'mds_tree':     '',
                                                 'mds_node':     '',
                                                 'label':        '$I_{CD,NBI}$',
                                                 'unit':         'A',                    
                                                 'multiplier':   1.,                        
                                                 'description':  'NBI current drive',
                                                 'source':       'mdsplus',
                                                 'db column #':  41,
                                                 }
    
    """
    THESE ARE FITTED PEDESTAL DATA FROM HERE
    """
    
    """Read kinetic notes"""
    full_plasma_dict['Kinetic profile note']=   {'mds_tree':      '',
                                                 'mds_node':     '',
                                                 'label':        '',
                                                 'unit':         '',                    
                                                 'multiplier':   None,                        
                                                 'description':  'Note for kinetic fitting',
                                                 'data':         'Fitted by get_fit_nstx_thomson_profiles.py in flap_nstx',
                                                 'source':      None,
                                                 'db column #':  42,
                                                 }
    """
    Electron pressure
    """
        
    """Read p_ped_location"""
    full_plasma_dict['Pressure pedestal location']=  {'mds_tree':     '',
                                                      'mds_node':     '',
                                                      'label':        '$\\psi_{p,ped}$',
                                                      'unit':         '',               
                                                      'multiplier':   1.,                        
                                                      'description':  'Pressure pedestal location',
                                                      'source':       'TS profile fitting',
                                                      'db column #':  44,
                                                      }
    """Read p_ped"""
    full_plasma_dict['Pressure pedestal height']=      {'mds_tree':     '',
                                                        'mds_node':     '',
                                                        'label':        '$p_{ped}$',
                                                        'unit':         'Pa',                    
                                                        'multiplier':   1.,                        
                                                        'description':  'Pressure pedestal height',
                                                        'source':       'TS profile fitting',
                                                        'db column #':  43,
                                                        }
    
    """Read ped_width"""
    full_plasma_dict['Pressure pedestal width']=  {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$w_{p,ped}$',
                                                   'unit':         '',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Pressure pedestal width',
                                                   'source':       'TS profile fitting',
                                                   'db column #':  45,
                                                   }
    """Read pe separatrix"""
    full_plasma_dict['Pressure core']=        {'mds_tree':     '',
                                                  'mds_node':     '',
                                                  'label':        '$p_{e,core}$',
                                                  'unit':         '',                    
                                                  'multiplier':   1.,                        
                                                  'description':  'Pressure in core',
                                                  'source':       'TS profile fitting',
                                                  'db column #':  None,
                                                   }
    
    """Read pe_90"""
    full_plasma_dict['Pressure 90']=  {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$p_{e,90}$',
                                                   'unit':         '',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Pressure at 90',
                                                   'source':       'TS profile fitting',
                                                   'db column #':  58,
                                                   }
    
    """Read pe_95"""
    full_plasma_dict['Pressure 95']=              {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$p_{e,95}$',
                                                   'unit':         '',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Pressure at 95',
                                                   'source':       'TS profile fitting',
                                                   'db column #':  60,
                                                   }
    """Read pe separatrix"""
    full_plasma_dict['Pressure separatrix']=      {'mds_tree':     '',
                                                  'mds_node':     '',
                                                  'label':        '$p_{e,sep}$',
                                                  'unit':         '',                    
                                                  'multiplier':   1.,                        
                                                  'description':  'Pressure at separatrix',
                                                  'source':       'TS profile fitting',
                                                  'db column #':  None,
                                                   }
    
    """Read pe SOL"""
    full_plasma_dict['Pressure SOL']=            {'mds_tree':     '',
                                                  'mds_node':     '',
                                                  'label':        '$p_{e,SOL}$',
                                                  'unit':         '',                    
                                                  'multiplier':   1.,                        
                                                  'description':  'Pressure at SOL',
                                                  'source':       'TS profile fitting',
                                                  'db column #':  None,
                                                   }
    
    """
    Electron density
    """
    
    """Read ped_location""" #same as pressure pedestal location
    full_plasma_dict['Density pedestal location']=  {'mds_tree':        '',
                                                     'mds_node':     '',
                                                     'label':        '$\\psi_{n_{e},ped}$',
                                                     'unit':         '',                    
                                                     'multiplier':   1.,                        
                                                     'description':  'Density pedestal location',
                                                     'source':       'TS profile fitting',
                                                     'db column #':  48,
                                                     }
    
    """Read ne_ped"""
    full_plasma_dict['Density pedestal height']=  {'mds_tree':     '',
                                                   'mds_node':     '',
                                                   'label':        '$n_{e,ped}$',
                                                   'unit':         '$m^{-3}$',
                                                   'multiplier':   1.,                        
                                                   'description':  'Density pedestal height',
                                                   'source':       'TS profile fitting',
                                                   'db column #':  49,
                                                   }
    """Read ne_width"""
    full_plasma_dict['Density pedestal width']=  {'mds_tree':        '',
                                                  'mds_node':     '',
                                                  'label':        '$w_{n_{e},ped}$',
                                                  'unit':         '',                 
                                                  'multiplier':   1.,                        
                                                  'description':  'Electron density pedestal width',
                                                  'source':       'TS profile fitting',
                                                  'db column #':  46,
                                                  }
    
    
    """Read ne_core"""
    full_plasma_dict['Density core']=           {'mds_tree':        '',
                                                 'mds_node':     '',
                                                 'label':        '$n_{e,core}$',
                                                 'unit':         '$m^{-3}$',                   
                                                 'multiplier':   1.,                        
                                                 'description':  'Pressure pedestal width',
                                                 'source':       'TS profile fitting',
                                                 'db column #':  None,
                                                 }
    
    """Read ne_90"""
    full_plasma_dict['Density 90']=              {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$n_{e,90}$',
                                                   'unit':         '$m^{-3}$',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Electron density at 90',
                                                   'source':       'TS profile fitting',
                                                   'db column #':  56,
                                                   }


    
    """Read ne_95"""
    full_plasma_dict['Density 95']=               {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$n_{e,95}$',
                                                   'unit':         '$m^{-3}$',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Electron density at 95',
                                                   'source':       'TS profile fitting',
                                                   'db column #':  59,
                                                   }

    
    """Read ne_sep"""
    full_plasma_dict['Density separatrix']=       {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$n_{e,sep}$',
                                                   'unit':         '$m^{-3}$',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Separatrix density',
                                                   'source':       'TS profile fitting',
                                                   'db column #':  50,
                                                   }
    """Read pe SOL"""
    full_plasma_dict['Density SOL']=             {'mds_tree':     '',
                                                  'mds_node':     '',
                                                  'label':        '$n_{e,SOL}$',
                                                  'unit':         '',                    
                                                  'multiplier':   1.,                        
                                                  'description':  'Density at SOL',
                                                  'source':       'TS profile fitting',
                                                  'db column #':  None,
                                                   }
    
    """
    Electron temperature
    """
    
    """Read te_ped location"""
    full_plasma_dict['Temperature pedestal location']=  {'mds_tree':        '',
                                                         'mds_node':     '',
                                                         'label':        '$\\psi_{T_{e},ped}$',
                                                         'unit':         '',                    
                                                         'multiplier':   1.,                        
                                                         'description':  'Temperature pedestal location',
                                                         'source':       'TS profile fitting',
                                                         'db column #':  None,
                                                         }
    """Read te_ped"""
    full_plasma_dict['Temperature pedestal height']=  {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$T_{e,ped}$',
                                                   'unit':         '',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Temperature pedestal height',
                                                   'source':       'TS profile fitting',
                                                   'db column #':  53,
                                                   }
    
    """Read te_width"""
    full_plasma_dict['Temperature pedestal width']=  {'mds_tree':        '',
                                                      'mds_node':     '',
                                                      'label':        '$w_{T_{e},ped}$',
                                                      'unit':         '',                    
                                                      'multiplier':   1.,                        
                                                      'description':  'Temperature pedestal width',
                                                      'source':       'TS profile fitting',
                                                      'db column #':  47,
                                                      }
    

    
        
    """Read te_core"""
    full_plasma_dict['Temperature core']=           {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$T_{e,core}$',
                                                   'unit':         'eV',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Temperature in core',
                                                   'source':     'TS profile fitting',
                                                   'db column #':  65,
                                                   }
    
    """Read te_90"""
    full_plasma_dict['Temperature 90']=          {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$T_{e,90}$',
                                                   'unit':         'eV',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Temperature at 90',
                                                   'source':     'TS profile fitting',
                                                   'db column #':  57,
                                                   }
    
    """Read te_95"""
    full_plasma_dict['Temperature 95']=          {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$T_{e,95}$',
                                                   'unit':         'eV',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Temperature at 95',
                                                   'source':     'TS profile fitting',
                                                   'db column #':  61,
                                                   }
    
    
    """Read te_sep"""
    full_plasma_dict['Temperature separatrix']=  {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$T_{e,sep}$',
                                                   'unit':         'eV',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Temperature separatrix ',
                                                   'source':     'TS profile fitting',
                                                   'db column #':  51,
                                                   }
    
    full_plasma_dict['Temperature SOL']=         {'mds_tree':     '',
                                                  'mds_node':     '',
                                                  'label':        '$T_{e,SOL}$',
                                                  'unit':         'eV',                    
                                                  'multiplier':   1.,                        
                                                  'description':  'Temperature at SOL',
                                                  'source':       'TS profile fitting',
                                                  'db column #':  None,
                                                   }
    
    
    
    """
    Ion temperature
    """

    """Read T_i ped location"""
    full_plasma_dict['Temperature ion pedestal location']=  {'mds_tree':        '',
                                                           'mds_node':     '',
                                                           'label':        '$\\psi_{T_i,ped}$',
                                                           'unit':         '',                    
                                                           'multiplier':   1.,                        
                                                           'description':  'Ion temperature pedestal location',
                                                           'source':       'CHERS profile fitting',
                                                           'db column #':  None,
                                                           }    
    
    
    """Read ti_ped"""
    full_plasma_dict['Temperature ion pedestal height']=  {'mds_tree':        '',
                                                           'mds_node':     '',
                                                           'label':        '$T_{i,ped}$',
                                                           'unit':         '',                    
                                                           'multiplier':   1.,                        
                                                           'description':  'Ion temperature pedestal height',
                                                           'source':     'CHERS profile fitting',
                                                           'db column #':  54,
                                                           }
    
    """Read ti_sep"""
    full_plasma_dict['Temperature ion pedestal width']=  {'mds_tree':        '',
                                                          'mds_node':     '',
                                                          'label':        '$w_{i,ped}$',
                                                          'unit':         '',                    
                                                          'multiplier':   1.,                        
                                                          'description':  'Ion temperature pedestal width',
                                                          'source':     'CHERS profile fitting',
                                                          'db column #':  52,
                                                          }    

    """Read ti_core"""
    full_plasma_dict['Temperature ion core']=           {'mds_tree':        '',
                                                         'mds_node':     '',
                                                         'label':        '$T_{i,core}$',
                                                         'unit':         '',                    
                                                         'multiplier':   1.,                        
                                                         'description':  'Ion temperature core',
                                                         'source':     'CHERS profile fitting',
                                                         'db column #':  64,
                                                         }
    
    """Read ti_90"""
    full_plasma_dict['Temperature ion 90']=             {'mds_tree':        '',
                                                         'mds_node':     '',
                                                         'label':        '$T_{i,90}$',
                                                         'unit':         '',                    
                                                         'multiplier':   1.,                        
                                                         'description':  'Ion temperature at 90',
                                                         'source':     'CHERS profile fitting',
                                                         'db column #':  62,
                                                         }
    """Read ti_95"""
    full_plasma_dict['Temperature ion 95']=             {'mds_tree':        '',
                                                         'mds_node':     '',
                                                         'label':        '$T_{i,95}$',
                                                         'unit':         '',                    
                                                         'multiplier':   1.,                        
                                                         'description':  'Ion temperature at 95',
                                                         'source':     'CHERS profile fitting',
                                                         'db column #':  63,
                                                         }
    
    """Read ti_sep"""
    full_plasma_dict['Temperature ion separatrix']=  {'mds_tree':     '',
                                                         'mds_node':     '',
                                                         'label':        '$T_{i,sep}$',
                                                         'unit':         '',                    
                                                         'multiplier':   1.,                        
                                                         'description':  'Ion temperature at separatrix',
                                                         'source':       'CHERS profile fitting',
                                                         'db column #':  None,
                                                         }
    
    """Read ti_SOL"""
    full_plasma_dict['Temperature ion SOL']=            {'mds_tree':     '',
                                                         'mds_node':     '',
                                                         'label':        '$T_{i,SOL}$',
                                                         'unit':         '',                    
                                                         'multiplier':   1.,                        
                                                         'description':  'Ion temperature at SOL',
                                                         'source':       'CHERS profile fitting',
                                                         'db column #':  None,
                                                         }
    
    """
    Other CHERS parameters
    """
#TODO: THe following are not coming from fit profiles and should be handled differently
    """Read v_core"""
    full_plasma_dict['Velocity toroidal pedestal']=    {'mds_tree':        '',
                                                        'mds_node':     '',
                                                        'label':        '$v_{tor,core}$',
                                                        'unit':         'm/s',
                                                        'multiplier':   1.,                        
                                                        'description':  'Toroidal rotation in the pedestal',
                                                        'source':        'CHERS profile fitting',
                                                        'db column #':  None,
                                                        }

    """Read v_core"""
    full_plasma_dict['Velocity toroidal core']=    {'mds_tree':        '',
                                                   'mds_node':     '',
                                                   'label':        '$v_{tor,core}$',
                                                   'unit':         'm/s',
                                                   'multiplier':   1.,                        
                                                   'description':  'Toroidal rotation in the core',
                                                   'source':        'CHERS profile fitting',
                                                   'db column #':  66,
                                                   }
    
    """Read z_eff_ped"""
    full_plasma_dict['Effective charge state pedestal']=  {'mds_tree':     '',
                                                           'mds_node':     '',
                                                           'label':        '$Z_{eff,ped}$',
                                                           'unit':         '',                    
                                                           'multiplier':   1.,                        
                                                           'description':  'Effective charge state in the pedestal',
                                                           'source':       'CHERS profile fitting',
                                                           'db column #':  55,
                                                           }
    
    """Read zeff_core"""
    full_plasma_dict['Effective charge state core']=  {'mds_tree':     '',
                                                       'mds_node':     '',
                                                       'label':        '$Z_{eff,core}$',
                                                       'unit':         '',                    
                                                       'multiplier':   1.,                        
                                                       'description':  'Effective charge state in the core',
                                                       'source':       'CHERS profile fitting',
                                                       'db column #':  67,
                                                       }
    
    
    """
    Other plasma parameters
    """
    
    """Read ne"""
    full_plasma_dict['Line integrated density']=  {'mds_tree':     'WF',
                                                   'mds_node':     'NEL',
                                                   'label':        '$n_{e,l}$',
                                                   'unit':         '$m^2$',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Line integrated density',
                                                   'source':'mdsplus',
                                                   'db column #':  68,
                                                   }
    
    """Read gas_meff""" #???
    full_plasma_dict['Gas effective mass']=  {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$m_{gas,eff}$',
                                              'unit':         'amu',           
                                              'multiplier':   1.,                        
                                              'description':  'Effective atomic mass of fuel',
                                              'source':       None,
                                              'db column #':  69,
                                              }
    """Read fbs""" #???
    full_plasma_dict['Bootstrap fraction']=  {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$f_{BS}$',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  'Bootstrap fraction',
                                              'source':       None,
                                              'db column #':  70,
                                              }
    
    """
    THESE ARE FAST ION RELATED DATA
    """
    
    """Read pressure fast ion pres_fast"""
    full_plasma_dict['Pressure fast ion']=   {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$p_{fast}$',
                                              'unit':         'Pa',                    
                                              'multiplier':   1.,                        
                                              'description':  'Fast ion pressure',
                                              'source':       None,
                                              'db column #':  71,
                                              }
    
    """Read thermal pressure pres_th"""
    full_plasma_dict['Pressure thermal']=    {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$p_{th}$',
                                              'unit':         'Pa',                    
                                              'multiplier':   1.,                        
                                              'description':  'Thermal pressure',
                                              'source':       None,
                                              'db column #':  72,
                                              }
    """Read w_fast""" #???
    full_plasma_dict['Energy fast ion']=    {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        'W_{fast}',
                                              'unit':         'J',                    
                                              'multiplier':   1.,                        
                                              'description':  'Fast ion energy',
                                              'source':       None,
                                              'db column #':  73,
                                              }
    """read w_thermal""" #???
    full_plasma_dict['Energy thermal']=      {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        'W_{thermal}',
                                              'unit':         'J',          
                                              'multiplier':   1.,                        
                                              'description':  'Thermal energy',
                                              'source':       None,
                                              'db column #':  74,
                                              }
    """Read dw_th""" #???
    full_plasma_dict['']=                    {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  None,
                                              'source':       None,
                                              'db column #':  75,
                                              }
    """Read dt""" #???
    full_plasma_dict['']=                    {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  None,
                                              'source':       None,
                                              'db column #':  76,
                                              }
    """Read kappa_areal""" #???
    full_plasma_dict['']=                    {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  None,
                                              'source':       None,
                                              'db column #':  77,
                                              }
    """Read n_gr"""
    full_plasma_dict['Density Greenwald']=   {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$n_{e,GW}$',
                                              'unit':         '$m^3$',                    
                                              'multiplier':   1.,                        
                                              'description':  'Greenwald density',
                                              'source':       'Greenwald density calculation',
                                              'db column #':  78,
                                              }
    """Read i_norm""" #???
    full_plasma_dict['']=  {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  None,
                                              'source':       None,
                                              'db column #':  79,
                                              }
    """Read reff""" #???
    full_plasma_dict['']=                     {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  None,
                                              'source':       None,
                                              'db column #':  80,
                                              }
    """Read q95in""" #??? q95 inside?
    full_plasma_dict['q95 in']=              {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  None,
                                              'source':       None,
                                              'db column #':  81,
                                              }
    
    """Read p_oh"""
    full_plasma_dict['Heating power OH']=  {'mds_tree':     'EFIT02',
                                            'mds_node':     'POH',
                                            'label':        '$P_{OH}$',
                                            'unit':         '$W$',                    
                                            'multiplier':   1.,                        
                                            'description':  'Ohmic heating power',
                                            'source':       'mdsplus',
                                            'db column #':  82,
                                            }
    """Read p_rf"""
    full_plasma_dict['Power RF']=            {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$P_{RF}$',
                                              'unit':         'W', 
                                              'multiplier':   1.,                        
                                              'description':  'Radio frequency power',
                                              'source':       None,
                                              'db column #':  83,
                                              }
    """Read p_aux"""
    full_plasma_dict['Power auxiliary']=     {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$P_{aux}$',
                                              'unit':         'W',                    
                                              'multiplier':   1.,                        
                                              'description':  'Auxiliary heating power',
                                              'source':       None,
                                              'db column #':  84,
                                              }
    """Read p_loss""" #???
    full_plasma_dict['Power loss']=          {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$P_{loss}$',
                                              'unit':         'W',                    
                                              'multiplier':   1.,                        
                                              'description':  'Power loss',
                                              'source':       None,
                                              'db column #':  85,
                                              }
    """Read taue_th"""
    full_plasma_dict['tau_e thermal']=       {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$\\tau_{e,th}$',
                                              'unit':         's',                    
                                              'multiplier':   1.,                        
                                              'description':  '',
                                              'source':       None,
                                              'db column #':  86,
                                              }
    """Read taue"""
    full_plasma_dict['Tau_e']=                    {'mds_tree':     'EFIT02',
                                                   'mds_node':     'TAUMHD',
                                                   'label':        '$\\tau_e$',
                                                   'unit':         's',                    
                                                   'multiplier':   1.,                        
                                                   'description':  'Energy confinement time',
                                                   'source':       'mdsplus',
                                                   'db column #':  87,
                                                   }
    """Read aspect_ratio"""
    full_plasma_dict['Aspect ratio']=        {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        'A',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  'R/a',
                                              'source':       'Aspect ratio calculation',
                                              'db column #':  88,
                                              }
    """Read H98"""
    full_plasma_dict['H98']=                 {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$H_{98}$',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  'H98',
                                              'source':       'H98 calculation',
                                              'db column #':  89,
                                              }
    """Read H89"""
    full_plasma_dict['H89']=                  {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '$H_{89}$',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  'H89',
                                              'source':       'H89 calculation',
                                              'db column #':  90,
                                              }
    
    """
    THESE ARE ALSO EQUILIBRIUM DATA
    """
    
    """Read dr_dpsin"""
    #empty
    full_plasma_dict['dR per dPsi_n']=       {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  'Derivative of R-psi mapping',
                                              'source':       None,
                                              'db column #':  91,
                                              }
    """Read drho_dpsin"""
    #empty
    full_plasma_dict['drho per dPsi_n']=  {'mds_tree':     '',
                                          'mds_node':     '',
                                          'label':        '',
                                          'unit':         '',                    
                                          'multiplier':   1.,                        
                                          'description':  'Derivative of rho-psi mapping',
                                          'source':       None,
                                          'db column #':  92,
                                              }
    """Read circ""" #???
    full_plasma_dict['circ']=               {'mds_tree':     '',
                                              'mds_node':     '',
                                              'label':        '',
                                              'unit':         '',                    
                                              'multiplier':   1.,                        
                                              'description':  None,
                                              'source':       None,
                                              'db column #':  93,
                                              }
    
    """Read p_rf_icrh_abs"""
    full_plasma_dict['Heating power RF']=  {'mds_tree':     'WF',
                                            'mds_node':     'PRF',
                                            'label':        '$P_{RF}$',
                                            'unit':         'W',                    
                                            'multiplier':   1.,                        
                                            'description':  'Total RF heating opwer',
                                            'source':       'mdsplus',
                                            'db column #':  94,
                                            }
    
    return full_plasma_dict