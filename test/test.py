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
import numpy.linalg as LA
from wanpy.env import PYGUI, ROOT_WDIR
from wanpy.core.mesh import make_kpath
from wanpy.core.structure import Htb, BandstructureHSP
from twistwanTB.core.wanTB_TMG import simu_eigh
from twistwanTB.core.wanTB_TMG import WanTB_TMG
from twistwanTB.core.wanTB_TMG import cal_band_Dk, cal_band
from twistwanTB.core.wanTB_TMG import plot_ham_decay, plot_ham, plot_2band_compare


if PYGUI:
    wdir = os.path.join(ROOT_WDIR, r'TMG_w90/debug')
    input_dir = os.path.join(ROOT_WDIR, r'TMG_w90/hamlib')
else:
    wdir = os.getcwd()
    input_dir = r'./libhtbFLG'


if __name__ == '__main__':
    os.chdir(input_dir)
    '''
      * Input
    '''
    m = 30
    n = m + 1

    NG = 3
    w1 = 1*0.0797
    w2 = 1*0.0975
    Umax = 0.0

    enable_hBN = False
    # for others
    U_BN_u = None
    U_BN_d = None

    # # for mTBG
    # U_BN_u = np.array([0])
    # U_BN_d = np.array([0.03])

    # # for mTDBG
    # U_BN_u = None
    # U_BN_d = None

    # # for mTTG
    # U_BN_u = np.array([-0.02, -0.02])
    # U_BN_d = np.array([0.08])

    htbfname_u = r'htb_SL_DFT.h5'
    htbfname_d = r'htb_SL_DFT.h5'

    # htbfname_u = r'htb_AB_SCAN.h5'
    # htbfname_d = r'htb_AB_SCAN.h5'

    # htbfname_u = r'htb_AB_SCAN.h5'
    # htbfname_d = r'htb_SL_DFT.h5'

    # htbfname_u = r'htb_ABCA_SCAN.h5'
    # htbfname_d = r'htb_AB_SCAN.h5'

    # htbfname_u = r'htb_ABC_SCAN.h5'
    # htbfname_d = r'htb_ABC_SCAN.h5'

    # htbfname_u = r'htb_ABCAC_SCAN.h5'
    # htbfname_d = r'htb_ABCA_SCAN.h5'

    mp_grid = np.array([6, 6, 1])
    froz_i1 = None # 442
    froz_i2 = None # 446

    if_cal_wfs = True
    fname_wfs = r'wfs.npz'


    htb = Htb()
    htb.load_htb(htbfname_u)
