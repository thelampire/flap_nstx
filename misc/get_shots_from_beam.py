#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 22:42:36 2023

@author: mlampert
"""

import os
import flap
import flap_nstx
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

def get_shots_from_beam(get_gpi=False,
                        get_magnetics=False):
    res=[
    137582, 138113, 138114, 138115, 138116, 138117, 138118, 138119,
          138120, 138121, 138122, 138123, 138124, 138125, 138126, 138614,
          138617, 138620, 138748, 138844, 138846, 138847, 138848, 138854,
          138855, 139044, 139047, 139048, 139053,

          139056, 139057, 139286,

           139288, 139289, 139292, 139296, 139297,
           139298, 139299, 139301,
            139432, 139433, 139436, 139437,
           139438, 139441, 139442, 139443,
            139448, 139499, 139506, 139507, 139508, 139509, 139510, 139895,
            139900, 139903, 139954, 139957, 139962, 140380, 140383, 140385,
            140519, 140520, 140521, 140525, 140526, 140616, 140617, 140622,
            140626, 141256, 141270, 141271, 141277, 141283, 141309, 141315,
            141318, 141319, 141321, 141324, 141328, 141740, 141741, 141742,
            141745, 141746, 141747, 141749, 141751, 141754, 141755, 141756,
          142220, 142230, 142234, 142269, 142270
         ]


    for shot in res:
        try:
            if get_gpi:
                if not os.path.exists('/Users/mlampert/data/'+str(shot)+'/nstx_5_'+str(shot)+'.cin'):
                    os.system('mkdir /Users/mlampert/data/'+str(shot))
                    print(os.popen('scp -r mlampert@beam.fusion.energia.mta.hu:./NSTX_TEMPORARY/nstx_5_'+str(shot)+'.cin /Users/mlampert/data/'+str(shot)).read())
            elif get_magnetics:

                R_sep=flap.get_data('NSTX_MDSPlus',
                                    name='\EFIT02::\RBDRY',
                                    exp_id=shot,
                                    object_name='SEP R OBJ'
                                    ).data

                z_sep=flap.get_data('NSTX_MDSPlus',
                                    name='\EFIT02::\ZBDRY',
                                    exp_id=shot,
                                    object_name='SEP Z OBJ'
                                    ).data

                flux=flap.get_data('NSTX_MDSPlus',
                                   name='\EFIT02::\PSIRZ',
                                   exp_id=shot,
                                   object_name='PSI RZ OBJ'
                                   ).data
                print(str(shot)+ '\'s magnetic boundary is downloaded')
        except Exception as e:
            print(e)
            print(str(shot)+' failed to be downloaded')