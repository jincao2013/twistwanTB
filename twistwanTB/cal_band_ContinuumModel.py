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
from twistwanTB import PYGUI, ROOT_WDIR
from twistwanTB.wanpy.mesh import make_kpath
from twistwanTB.wanpy.structure import Htb
from twistwanTB.core.continuum_model import MacDonald_TMG_wan
from twistwanTB.core.wanTB_TMG import cal_band, plot_band_BM


wdir = os.getcwd()
input_dir = os.path.join(ROOT_WDIR, r'libhtbFLG')


if __name__ == "__main__":
    os.chdir(input_dir)

    m = 30
    Umax = 0.0
    enable_hBN = False
    # # for others
    U_BN_u = None
    U_BN_d = None
    # for mTBG
    U_BN_u = np.array([0.0])
    U_BN_d = np.array([0.0])
    # for mTTG
    # U_BN_u = np.array([-0.02, -0.02])
    # U_BN_d = np.array([0.08])
    # for 3+3
    # U_BN_u = np.array([0, 0, 0, 0])
    # U_BN_d = np.array([0.01, 0.01, 0.0, 0.0])

    # htbfname_u = r'htb_SLG_SK.h5'
    # htbfname_d = r'htb_SLG_SK.h5'

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

    # htbfname_u = r'htb_ABCA_SCAN.h5'
    # htbfname_d = r'htb_ABCA_SCAN.h5'

    # htbfname_u = r'htb_ABCAC_SCAN.h5'
    # htbfname_d = r'htb_ABCA_SCAN.h5'

    htb_u = Htb()
    htb_d = Htb()
    htb_u.load_htb(htbfname_u)
    htb_d.load_htb(htbfname_d)

    # htb_u = remove_warping_for_tdbg(htb_u)
    # htb_d = remove_warping_for_tdbg(htb_d)

    ham = MacDonald_TMG_wan(htb_u, htb_d, m=m, n=m+1, N=3, w1=1*0.0797, w2=1*0.0975, tLu=0, tLd=-1, vac=300, rotk=False, Umax=Umax)
    ham.hBN = ham.get_hBN(U_BN_u, U_BN_d, enable=enable_hBN)
    os.chdir(wdir)

    nk1 = 51
    kpath_HSP = np.array([
        [-1/3, 1/3, 0.0], #  K
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 1/2, 0.0], #  M
        [ 1/3, 2/3, 0.0], # -K
    ])
    xlabel = ['K', 'G', 'M', '-K']
    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = LA.multi_dot([ham.lattG, kpath.T]).T

    bandE = cal_band(ham, kpath_car, ham.nb, simudiag=True)
    # fb_index = ham.get_fb_index()  # (valley-,VB); (-,CB); (+,VB); (+,CB);

    if PYGUI:
        plot_band_BM(bandE, xlabel, kpath_car, eemin=-0.04, eemax=0.04)