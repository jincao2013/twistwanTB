# Copyright (C) 2020  Jin Cao
#
# This file is distributed as part of the twistwanTB code and
# under the terms of the GNU General Public License. See the
# file `LICENSE' in the root directory of the twistwanTB
# distribution, or http://www.gnu.org/licenses/gpl-3.0.txt
#
# The twistwanTB code is hosted on GitHub:
#
# https://github.com/jincao2013/twistwanTB

__author__ = 'Jin Cao'
__email__ = "caojin.phy@gmail.com"
__date__ = "July. 24, 2019"

import os
import numpy as np
from twistwanTB import PYGUI, ROOT_WDIR
from twistwanTB.core.wanTB_TMG import WFs_TMG
from twistwanTB.core.coulomb_parameters import cal_columb_interaction, cal_assisted_columb_interaction


wdir = os.getcwd()
input_dir = os.path.join(ROOT_WDIR, r'libhtbFLG')


if __name__ == "__main__":
    os.chdir(wdir)
    '''
      * Job = 
      ** cal_columb_interaction_selected 
      ** cal_columb_interaction_all 
      ** cal_assisted_columb 
    '''
    Job = 'cal_assisted_columb'

    fname_wfs = r'wfs_mTBG_m30n31_nnsuper3.npz'
    # fname_wfs = r'wfs_mTDBG_m26n27_UD10meV_nnsuper3.npz'
    # fname_wfs = r'wfs_mTTG_m26n27_UD_hBN_nnsuper3.npz'

    WFs = WFs_TMG(ns=2, latt=np.eye(3))
    WFs.load_npz(fname_wfs)
    rr = WFs.rr
    gridrr = WFs.grid
    grid_ucell = WFs.grid_ucell
    nnsuper = WFs.nnsuper
    wcc = WFs.wcc
    wfs = WFs.wfs
    wfs_symm = WFs.wfs_symm
    latt = WFs.latt

    # WFs.plot_wan_WFs_all(valley=0)
    # WFs.plot_wan_WFs_nonzero(valley=0)


if __name__ == "__main__" and Job == 'cal_columb_interaction_all':
    seedname = r'uJ_mTTG'
    '''
      * auto cal all
    '''
    worbit_distance_index, u, J, u_xy, J_xy = \
        cal_columb_interaction(rr, wfs_symm, grid_ucell, latt, nnsuper=5, savefname=seedname+'.dat')

    np.savez_compressed(seedname+'.npz',
                        worbit_distance_index=worbit_distance_index,
                        u=u,
                        J=J,
                        u_xy=u_xy,
                        J_xy=J_xy,
                        )

    if os.environ.get('PYGUI') == 'True':
        npdata = np.load(seedname+'.npz', allow_pickle=True)
        worbit_distance_index = npdata['worbit_distance_index']
        u = npdata['u']
        u_xy = npdata['u_xy']
        J = npdata['J']
        J_xy = npdata['J_xy']

        print('=======================')
        print('  Summary')
        print('=======================')
        print('         U       J     ')
        for i in range(len(worbit_distance_index)):
            nn = worbit_distance_index[i][0]
            munu = worbit_distance_index[i][1]
            print('U{}({})  {:<8.4f}{:<8.4f}'.format(nn, munu, u[i], J[i]))
        print('=======================')

if __name__ == "__main__" and Job == 'cal_assisted_columb':

    # next-nearest
    worbit_distance_index = [
        # obi, obj, objR, obtri, obtriR

        [1, 1, np.array([0, 1, 0]), 2, np.array([0, 0, 0])],  # charging at 1
        [1, 3, np.array([0, 1, 0]), 2, np.array([0, 0, 0])],  # charging at 1
        [1, 1, np.array([0, 1, 0]), 4, np.array([0, 0, 0])],  # charging at 1
        [1, 3, np.array([0, 1, 0]), 4, np.array([0, 0, 0])],  # charging at 1

        [1, 1, np.array([0, 1, 0]), 2, np.array([-1, 1, 0])], # charging at 2
        [1, 3, np.array([0, 1, 0]), 2, np.array([-1, 1, 0])], # charging at 2
        [1, 1, np.array([0, 1, 0]), 4, np.array([-1, 1, 0])], # charging at 2
        [1, 3, np.array([0, 1, 0]), 4, np.array([-1, 1, 0])], # charging at 2

        [1, 1, np.array([0, 1, 0]), 2, np.array([-1, 0, 0])], # charging at 3
        [1, 3, np.array([0, 1, 0]), 2, np.array([-1, 0, 0])], # charging at 3
        [1, 1, np.array([0, 1, 0]), 4, np.array([-1, 0, 0])], # charging at 3
        [1, 3, np.array([0, 1, 0]), 4, np.array([-1, 0, 0])], # charging at 3

        [1, 1, np.array([0, 1, 0]), 1, np.array([-1, 1, 0])], # charging at 4
        [1, 3, np.array([0, 1, 0]), 1, np.array([-1, 1, 0])], # charging at 4
        [1, 1, np.array([0, 1, 0]), 3, np.array([-1, 1, 0])], # charging at 4
        [1, 3, np.array([0, 1, 0]), 3, np.array([-1, 1, 0])], # charging at 4

    ]

    # # next-nearest
    # worbit_distance_index = [
    #     # obi, obj, objR, obtri, obtriR
    #     [1, 1, np.array([0, 1, 0]), 2, np.array([0, 0, 0])],  # charging at 1
    #     [1, 3, np.array([0, 1, 0]), 2, np.array([0, 0, 0])],  # charging at 1
    #     [3, 1, np.array([0, 1, 0]), 2, np.array([0, 0, 0])],  # charging at 1
    #     [3, 3, np.array([0, 1, 0]), 2, np.array([0, 0, 0])],  # charging at 1
    #     [1, 1, np.array([0, 1, 0]), 4, np.array([0, 0, 0])],  # charging at 1
    #     [1, 3, np.array([0, 1, 0]), 4, np.array([0, 0, 0])],  # charging at 1
    #     [3, 1, np.array([0, 1, 0]), 4, np.array([0, 0, 0])],  # charging at 1
    #     [3, 3, np.array([0, 1, 0]), 4, np.array([0, 0, 0])],  # charging at 1
    # ]

    # # nearest
    # worbit_distance_index = [
    #     # obi, obj, objR, obtri, obtriR
    #     [1, 2, np.array([0, 0, 0]), 1, np.array([0, 1, 0])],  # charging at 1
    #     [1, 4, np.array([0, 0, 0]), 1, np.array([0, 1, 0])],  # charging at 1
    #     [3, 2, np.array([0, 0, 0]), 1, np.array([0, 1, 0])],  # charging at 1
    #     [3, 4, np.array([0, 0, 0]), 1, np.array([0, 1, 0])],  # charging at 1
    #     [1, 2, np.array([0, 0, 0]), 3, np.array([0, 1, 0])],  # charging at 1
    #     [1, 4, np.array([0, 0, 0]), 3, np.array([0, 1, 0])],  # charging at 1
    #     [3, 2, np.array([0, 0, 0]), 3, np.array([0, 1, 0])],  # charging at 1
    #     [3, 4, np.array([0, 0, 0]), 3, np.array([0, 1, 0])],  # charging at 1
    # ]
    # worbit_distance_index = [
    #     # obi, obj, objR, obtri, obtriR
    #     [1, 2, np.array([0, 0, 0]), 1, np.array([0, 0, 0])],
    #     [1, 2, np.array([0, 0, 0]), 2, np.array([0, 0, 0])],
    #     [1, 2, np.array([0, 0, 0]), 3, np.array([0, 0, 0])],
    #     [1, 2, np.array([0, 0, 0]), 4, np.array([0, 0, 0])],
    #     [1, 2, np.array([0, 0, 0]), 1, np.array([0, 1, 0])],
    #     [1, 2, np.array([0, 0, 0]), 3, np.array([0, 1, 0])],
    #     [1, 2, np.array([0, 0, 0]), 2, np.array([-1, 0, 0])],
    #     [1, 2, np.array([0, 0, 0]), 4, np.array([-1, 0, 0])],
    #     [1, 2, np.array([0, 0, 0]), 1, np.array([-1, 1, 0])],
    #     [1, 2, np.array([0, 0, 0]), 2, np.array([-1, 1, 0])],
    #     [1, 2, np.array([0, 0, 0]), 3, np.array([-1, 1, 0])],
    #     [1, 2, np.array([0, 0, 0]), 4, np.array([-1, 1, 0])],
    # ]

    u = []
    u_xy = []
    for i, j, jR, tri, triR in worbit_distance_index:
        print('cal VH2 ...')
        _u, _u_xy = cal_assisted_columb_interaction(rr, wfs_symm, grid_ucell, latt, i-1, j-1, jR, tri-1, triR)
        u.append(_u)
        u_xy.append(_u_xy)
        print(_u)
    u = np.array(u)
    u_xy = np.array(u_xy)

    if PYGUI:
        with np.printoptions(precision=3, suppress=True):
            print(u_xy)
            print(u)
