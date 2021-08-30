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
from twistwanTB.wanpy.structure import Htb, BandstructureHSP
from twistwanTB.core.wanTB_TMG import simu_eigh
from twistwanTB.core.wanTB_TMG import WanTB_TMG
from twistwanTB.core.wanTB_TMG import cal_band_Dk, cal_band
from twistwanTB.core.wanTB_TMG import plot_ham_decay, plot_ham, plot_2band_compare


# if PYGUI:
#     wdir = os.path.join(ROOT_WDIR, r'TMG_w90/debug')
#     input_dir = os.path.join(os.path.split(__file__)[0], r'libhtbFLG')
# else:
#     wdir = os.getcwd()
#     input_dir = r'./libhtbFLG'

wdir = os.getcwd()
input_dir = os.path.join(ROOT_WDIR, r'libhtbFLG')


if __name__ == '__main__':
    os.chdir(wdir)
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

    mp_grid = np.array([12, 12, 1])
    froz_i1 = None # 442
    froz_i2 = None # 446

    if_cal_wfs = True
    fname_wfs = r'wfs.npz'

    '''
      * Plot 
    '''
    plot_wfs = False
    plot_band = True
    plot_valley_eign = False
    plot_band_valley_expectation_vaule = False
    show_ham = False

    '''
      * Main
    '''
    os.chdir(input_dir)
    htb_u, htb_d = [Htb(), Htb()]
    htb_u.load_htb(htbfname_u)
    htb_d.load_htb(htbfname_d)
    os.chdir(wdir)

    wanTB =                 WanTB_TMG(m, n, NG, w1, w2, mp_grid, Umax, U_BN_u, U_BN_d, enable_hBN, htb_u, htb_d)
    ham =                   wanTB.ham_BM
    wcc, AMN, EIG, EIG_D =  wanTB.wanTB_tmg_w90_setup(ham, wanTB.NNKP)
    htb, w90ip =            wanTB.wanTB_build_tmg_wanTB(ham, wcc, AMN, EIG, EIG_D, froz_i1, froz_i2)
    WFs =                   wanTB.wanTB_plot_wfs(ham, wcc, w90ip.vmat, nnsuper=3, plot_rr_dense=2, cal_wfs=if_cal_wfs)

    # save Wannier tight-binding model in h5 format
    # htb.save_h5(r'htb.mttg.m30n31.k6.h5')
    htb.save_h5(r'htb.twistTMG.h5')

    # save final WFs
    if if_cal_wfs:
        WFs.save_npz(fname_wfs)
    htb.save_wannier90_hr_dat()

if __name__ == '__main__' and plot_wfs:
    '''
      * read from wfs.npz
    '''
    # fname_wfs = r'wfs_mTBG_dft_m30n31_nnsuper3.npz'
    # fname_wfs = r'wfs_MTDBG_dft_m26n27_nnsuper3.npz'
    # WFs.load_npz(fname_wfs)
    # WFs.plot_wan_WFs_all()

    '''
      * direct plot
    '''
    # WFs.plot_wan_WFs_all(valley=0)
    # WFs.plot_wan_WFs_all(valley=1)
    WFs.plot_wan_WFs_nonzero(valley=0)

if __name__ == '__main__' and plot_band:
    nk1 = 31
    kpath_HSP = np.array([
        [-1/3, 1/3, 0.0], #  K
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 1/2, 0.0], #  M
        [ 1/3, 2/3, 0.0], # -K
    ])
    xlabel = ['K', '$\Gamma$', 'M', '-K']

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = LA.multi_dot([ham.lattG, kpath.T]).T
    nk = kpath.shape[0]

    # bandE = cal_band(token_htb, kpath, nw)
    # bandE = cal_band(w90ip, kpath, nw)
    # bandE2 = cal_band(ham, kpath_car, ham.nb)

    nw = 4
    bandE = np.zeros([nk, nw], dtype='float64')
    bandE_ref = np.zeros([nk, ham.nb], dtype='float64')
    valley_eign_symm = np.zeros([nk, nw], dtype='float64')
    for ik, k, kc in zip(range(nk), kpath, kpath_car):
        print('cal k {}/{}'.format(ik+1, nk))
        hk = w90ip.get_hk(k)
        Dk = w90ip.get_Dk(k, 2)
        # hk = token_htb.get_hk(k)
        E, U = LA.eigh(hk)
        E_symm, U_symm = simu_eigh(hk, Dk)

        bandE[ik] = E_symm - w90ip.fermi
        bandE_ref[ik], U = ham.get_eigh(ham.get_hk(kc))

        valley_eign_symm[ik] = np.real(np.diag(LA.multi_dot([U_symm.T.conj(), Dk, U_symm])))

    plot_2band_compare(bandE*1e3, bandE_ref*1e3, kpath_HSP, xlabel, kpath_car, eemin=-0.010*1e3, eemax=0.010*1e3)

if __name__ == '__main__' and plot_valley_eign:
    nk1 = 51
    kpath_HSP = np.array([
        [-1/3, 1/3, 0.0], #  K
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 1/2, 0.0], #  M
        [ 1/3, 2/3, 0.0], # -K
    ])
    xlabel = ['K', '$\Gamma$', 'M', '-K']

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = LA.multi_dot([ham.lattG, kpath.T]).T

    bandW = cal_band_Dk(w90ip, kpath, nw, isymmOP=2)
    bandE, bandU = cal_band(w90ip, kpath, nw, returnU=True)

    bandstructure_hsp = BandstructureHSP()
    bandstructure_hsp.eig = bandW
    bandstructure_hsp.HSP_list = kpath_HSP
    bandstructure_hsp.HSP_path_frac = kpath
    bandstructure_hsp.HSP_path_car = kpath_car
    bandstructure_hsp.HSP_name = xlabel
    bandstructure_hsp.nk, bandstructure_hsp.nb = bandE.shape

    bandstructure_hsp.plot_band(eemin=-1.5, eemax=1.5, unit='C')

if __name__ == '__main__' and plot_band_valley_expectation_vaule:
    '''
      Cal band valley expectation vaule 
      valley_eign_symm = <psi_fb, ik|valley operator|psi_fb, ik>
    '''
    nw = 4
    nk1 = 31
    kpath_HSP = np.array([
        [-1/3, 1/3, 0.0], #  K
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 1/2, 0.0], #  M
        [ 1/3, 2/3, 0.0], # -K
    ])
    xlabel = ['K', '$\Gamma$', 'M', '-K']

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = LA.multi_dot([ham.lattG, kpath.T]).T
    nk = kpath.shape[0]

    bandE = np.zeros([nk, nw], dtype='float64')
    bandE_symm = np.zeros([nk, nw], dtype='float64')
    valley_eign = np.zeros([nk, nw], dtype='float64')
    valley_eign_symm = np.zeros([nk, nw], dtype='float64')
    for ik, k in zip(range(nk), kpath):
        Dk = w90ip.get_Dk(k, 2)
        hk = htb.get_hk(k)
        # hk = token_htb.get_hk(k)
        E, U = LA.eigh(hk)
        E_symm, U_symm = simu_eigh(hk, Dk)

        bandE[ik] = E
        bandE_symm[ik] = E_symm
        valley_eign[ik] = np.real(np.diag(LA.multi_dot([U.T.conj(), Dk, U])))
        valley_eign_symm[ik] = np.real(np.diag(LA.multi_dot([U_symm.T.conj(), Dk, U_symm])))

    # for ik in range(nk):
    #     inline = valley_eign_symm[ik]
    #     sum = np.sum(inline)
    #     print('{:>4}  {:>+3.2f} {:>+3.2f} {:>+3.2f} {:>+3.2f} {:>+2.1f}'.format(ik+1, inline[0], inline[1], inline[2], inline[3], sum))

    bandstructure_hsp = BandstructureHSP()
    bandstructure_hsp.eig = bandE
    bandstructure_hsp.HSP_list = kpath_HSP
    bandstructure_hsp.HSP_path_frac = kpath
    bandstructure_hsp.HSP_path_car = kpath_car
    bandstructure_hsp.HSP_name = xlabel
    bandstructure_hsp.nk, bandstructure_hsp.nb = bandE.shape

    bandstructure_hsp.plot_band(eemin=-0.02, eemax=0.02, unit='C')
    bandstructure_hsp.plot_band(eemin=-1.2, eemax=1.2, unit='C')

if __name__ == '__main__' and show_ham:
    htb = Htb()
    htb.load_htb(r'htb.h5')

    plot_ham_decay(htb)

    # get tmax
    mask = np.ones_like(htb.hr_Rmn)
    mask[htb.nR//2] = 1 - np.identity(htb.nw)
    tmax = np.max(np.abs(np.real(htb.hr_Rmn*mask)))
    # hr_Rmn = htb.hr_Rmn * np.kron(np.eye(2, 2, 1) + np.eye(2, 2, -1) , np.ones([2,2]))
    # plot_ham(htb.Rc, hr_Rmn, s=50)
    # plot_ham(htb.Rc, htb.hr_Rmn, tmax=tmax, LM=LA.norm(htb.latt.T[0]), ticks=[-4, -2, 0, 2, 4], s=60)  # for 12 12 1
    plot_ham(htb.Rc, htb.hr_Rmn, tmax=tmax, LM=LA.norm(htb.latt.T[0]), ticks=[-4, -2, 0, 2, 4], s=60, axis=[-8,8])

    os.chdir(wdir)
