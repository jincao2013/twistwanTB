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
from scipy.spatial import distance
import numpy as np
import numpy.linalg as LA
from twistwanTB import PYGUI, ROOT_WDIR
from twistwanTB.wanpy.units import *
from twistwanTB.wanpy.mesh import make_mesh
from twistwanTB.core.wanTB_TMG import WFs_TMG


'''
  * Calalte columb interaction
'''
def cal_columb_interaction(rr, wfs_symm, grid_ucell, latt, nnsuper=5, savefname=r'uJ.dat'):
    worbit_distance_index = [
        # nn, (mu,nu), obi, obj, objR
        [0, '1,1', 1, 1, np.array([0, 0, 0])], # 0
        [0, '1,2', 1, 3, np.array([0, 0, 0])], # 0
        [1, '1,1', 1, 2, np.array([0, 0, 0])], # 1
        [1, '1,2', 1, 4, np.array([0, 0, 0])], # 1
        [2, '1,1', 1, 1, np.array([1, 0, 0])], # 2
        [2, '1,2', 1, 3, np.array([1, 0, 0])], # 2
        [3, '1,1', 1, 2, np.array([1,-1, 0])], # 3
        [3, '1,2', 1, 4, np.array([1,-1, 0])], # 3
        [4, '1,1', 1, 2, np.array([1, 0, 0])], # 4
        [4, '1,2', 1, 4, np.array([1, 0, 0])], # 4
        [5, '1,1', 1, 1, np.array([1, 1, 0])], # 5
        [5, '1,2', 1, 3, np.array([1, 1, 0])], # 5
    ]

    nw, ns = wfs_symm.shape[:2]
    nVJ = len(worbit_distance_index)
    u = np.zeros([nVJ], dtype='float64')
    u_xy = np.zeros([nVJ, ns, ns], dtype='float64')
    J = np.zeros([nVJ], dtype='float64')
    J_xy = np.zeros([nVJ, ns, ns], dtype='float64')

    tmg_wfs_mask = get_tmg_nonzero_mask(wfs)
    wfs_symm_with_mask = (tmg_wfs_mask.T * wfs_symm.T).T
    wfs_symm_with_mask = normalize_wfs(wfs_symm_with_mask, grid_ucell, latt)
    # plot_wan_WFs(rr, wfs_symm_with_mask, valley=0, s=10, vmin=-1, vmax=1, latt=latt)

    f = open(savefname, 'w+')
    print('wannier orbitals read from: {}\n'.format(fname_wfs), file=f)
    print('Coulomb interaction matrix elements <..|Vee|..>', file=f)
    print('in unit of e^2/(4*pi*epsilon*Lm) ~ 102 meV * theta(deg)/epsilon_r\n', file=f)
    for i in range(nVJ):
        nn = worbit_distance_index[i][0]
        munu = worbit_distance_index[i][1]
        wobit_i = worbit_distance_index[i][2] - 1
        wobit_j = worbit_distance_index[i][3] - 1
        wobit_j_R = worbit_distance_index[i][4]

        print('cal U{}({}) & J{}({})'.format(nn, munu, nn, munu))
        u[i], J[i], u_xy[i], J_xy[i] = cal_columb_interaction_ijR(rr, wfs_symm_with_mask, grid_ucell, latt, wobit_i, wobit_j,
                                                                  wobit_j_R, nnsuper=nnsuper)

    # f = open(savefname, 'w+')
    # print('wannier orbitals read from: {}\n'.format(fname_wfs), file=f)
    # print('Columb interaction matrix elements 0.5 * <..|Vee|..>', file=f)
    # print('in unit of e^2/(4*pi*epsilon*Lm) ~ 102 meV * theta(deg)/epsilon_r\n', file=f)
    # for i in range(nVJ):
    #     nn = worbit_distance_index[i][0]
    #     munu = worbit_distance_index[i][1]
    #     wobit_i = worbit_distance_index[i][2] - 1
    #     wobit_j = worbit_distance_index[i][3] - 1
    #     wobit_j_R = worbit_distance_index[i][4]
        with np.printoptions(precision=5, suppress=True):
            print('===== U{}({}) & J{}({}) =====\n'.format(nn, munu, nn, munu), file=f)
            print('i={}, j={}, R={}\n'.format(wobit_i, wobit_j, wobit_j_R), file=f)
            print('U{}({})={:<9.5f}'.format(nn, munu, u[i]), file=f)
            print('J{}({})={:<9.5f}\n'.format(nn, munu, J[i]), file=f)
            print('sublattice & layer component of U:', '\n', u_xy[i], '\n', file=f)
            print('sublattice & layer component of J:', '\n', J_xy[i], '\n', file=f)
            print('\n', file=f)

    print('=======================', file=f)
    print('  Summary', file=f)
    print('=======================', file=f)
    print('         U       J     ', file=f)
    for i in range(nVJ):
        nn = worbit_distance_index[i][0]
        munu = worbit_distance_index[i][1]
        print('U{}({})  {:<8.4f}{:<8.4f}'.format(nn, munu, u[i], J[i]), file=f)
    print('=======================', file=f)
    f.close()
    return worbit_distance_index, u, J, u_xy, J_xy

def cal_columb_interaction_ijR(rr, wfs_symm, grid_ucell, latt, wobit_i, wobit_j, wobit_j_R, nnsuper=5, mask=None, cal_J=True):
    '''
      * Used for calculating electron-electron interaction parameters U & J

      * The wfs in a 3 * 3 mcell are needed,
      * and the columb interactions are calculated in 5*5 mcell

    :param rr: grid of wfs_symm
    :param wfs_symm: wannier functions in shape of (nw=4, n_{sublattice basis}, nrr)
    :param grid_ucell: shape of unit cell grid in rr (rr is a larger grid of super cell)
    :param latt: lattice of TMG
    :param wobit_i: wannier orbital i, seted at R=(0, 0, 0) lattice
    :param wobit_j: wannier orbital j, seted at R=wobit_j_R lattice
    :param wobit_j_R: for instance = np.array([1, 0, 0])
    :param nnsuper: nnsuper larger of griducell will be used to calculate U & J
    :return u, J: U and J
    :return u_sc, J_sx: sublattice component of U and J

    in unit of e^2/(4*pi*epsilon*Lm) = 102 meV * theta(deg)/epsilon_r

    '''
    if wobit_i == wobit_j and (wobit_j_R == 0).all():
        cal_J = False
    wobit_j_Rc = LA.multi_dot([latt, wobit_j_R])

    # get larger grid rr2 to compute U and J
    nmeshR_plot = grid_ucell * np.array([nnsuper, nnsuper, 1])
    rr2 = make_mesh(nmeshR_plot, type='continuous', centersym=False) * nnsuper
    rr2 -= np.array([nnsuper // 2, nnsuper // 2, 0], dtype='float64')
    rr2 = LA.multi_dot([latt, rr2.T]).T
    nrr2 = rr2.shape[0]
    deltaS = LA.det(latt[:2,:2]) / grid_ucell[0] / grid_ucell[1]

    # get mapping between rr and rr2
    distance_rr_rr2 = distance.cdist(rr, rr2, 'euclidean')
    index_for_i = np.argmin(distance_rr_rr2, 0)
    in_Rc_for_i = np.array(np.min(distance_rr_rr2, 0) < np.max(distance_rr_rr2) * 1e-6, dtype='int')

    distance_rr_rr2 = distance.cdist(rr + wobit_j_Rc, rr2, 'euclidean')
    index_for_j = np.argmin(distance_rr_rr2, 0)
    in_Rc_for_j = np.array(np.min(distance_rr_rr2, 0) < np.max(distance_rr_rr2) * 1e-6, dtype='int')


    # get wfs on rr2 from wfs on rr
    wfs_symm2_i = in_Rc_for_i * wfs_symm.T[index_for_i].T
    wfs_symm2_j = in_Rc_for_j * wfs_symm.T[index_for_j].T
    # normalization
    norm_of_wfs = np.einsum('nxvr,nxvr->n', wfs_symm2_i.conj(), wfs_symm2_i).real * deltaS
    wfs_symm2_i = (wfs_symm2_i.T / norm_of_wfs ** 0.5).T
    norm_of_wfs = np.einsum('nxvr,nxvr->n', wfs_symm2_j.conj(), wfs_symm2_j).real * deltaS
    wfs_symm2_j = (wfs_symm2_j.T / norm_of_wfs ** 0.5).T

    # # times mask
    # if mask is not None:
    #     wfs_symm2_i = (mask.T * wfs_symm2_i.T).T
    #     wfs_symm2_j = (mask.T * wfs_symm2_j.T).T

    # # debug
    # plot_wan_WFs(rr, wfs_symm, valley=0, s=10, vmin=-1, vmax=1, latt=htb.latt)
    # plot_wan_WFs(rr, wfs_symm, valley=1, s=10, vmin=-1, vmax=1, latt=htb.latt)
    # plot_grid2(rr, wfs_symm[0,2], s=30, vmin=-1.2, vmax=1.2)
    # plot_grid2(rr2, wfs_symm2_i[0,2], s=10, vmin=-1.2, vmax=1.2)
    # plot_grid2(rr2, wfs_symm2_j[0,2], s=10, vmin=-1.2, vmax=1.2)

    # cal U and J
    lattA = LA.norm(latt.T[0])
    eps = LA.norm(rr2[0] - rr2[1]) / 50

    norm_rr2 = distance.cdist(rr2, rr2, 'euclidean')
    vrr = np.real(1 / (norm_rr2 + 1j * eps))

    wfs_i = wfs_symm2_i[wobit_i]
    wfs_j = wfs_symm2_j[wobit_j]
    u_xy = 1.0 * np.einsum('xvr,ywp,vw,rp->xy', np.abs(wfs_i) ** 2, np.abs(wfs_j) ** 2, np.ones([2,2]), vrr) * deltaS**2 * lattA
    u = np.sum(u_xy)
    if cal_J:
        J_xy = 1.0 * np.einsum('xvr,ywp,ywp,xvr,vw,rp->xy', wfs_i.conj(), wfs_j.conj(), wfs_i, wfs_j, np.ones([2,2]), vrr) * deltaS**2 * lattA
    else:
        J_xy = np.zeros_like(u_xy)
    J = np.sum(J_xy)

    # print(u, '\n', u_xy)
    # print(J, '\n', J_xy)
    return u, J, u_xy, J_xy

def cal_assisted_columb_interaction(rr, wfs_symm, grid_ucell, latt,
                                    wobit_i, wobit_j, wobit_j_R,
                                    wobit_tri, wobit_tri_R,
                                    nnsuper=5, mask=None):
    '''
      * Used for calculating assisted hopping in [PNAS 2018 115 (52) 13174]

      * The wfs in a 3 * 3 mcell are needed,
      * and the columb interactions are calculated in 5*5 mcell

    :param rr: grid of wfs_symm
    :param wfs_symm: wannier functions in shape of (nw=4, n_{sublattice basis}, nrr)
    :param grid_ucell: shape of unit cell grid in rr (rr is a larger grid of super cell)
    :param latt: lattice of TMG
    :param wobit_i: wannier orbital i, seted at R=(0, 0, 0) lattice
    :param wobit_j: wannier orbital j, seted at R=wobit_j_R lattice
    :param wobit_j_R: for instance = np.array([1, 0, 0])
    :param wobit_tri: trigonal charging site
    :param nnsuper: nnsuper larger of griducell will be used to calculate U & J
    :return u, J: U and J
    :return u_sc, J_sx: sublattice component of U and J

    in unit of e^2/(4*pi*epsilon*Lm) = 102 meV * theta(deg)/epsilon_r

    '''
    # if wobit_i != wobit_j:
    #     print('wobit_i != wobit_j')
    #     sys.exit(1)
    wobit_j_Rc = LA.multi_dot([latt, wobit_j_R])
    wobit_tri_Rc = LA.multi_dot([latt, wobit_tri_R])

    # get larger grid rr2 to compute U and J
    nmeshR_plot = grid_ucell * np.array([nnsuper, nnsuper, 1])
    rr2 = make_mesh(nmeshR_plot, type='continuous', centersym=False) * nnsuper
    rr2 -= np.array([nnsuper // 2, nnsuper // 2, 0], dtype='float64')
    rr2 = LA.multi_dot([latt, rr2.T]).T
    nrr2 = rr2.shape[0]
    deltaS = LA.det(latt[:2,:2]) / grid_ucell[0] / grid_ucell[1]

    # get mapping between rr and rr2
    distance_rr_rr2 = distance.cdist(rr, rr2, 'euclidean')
    index_for_i = np.argmin(distance_rr_rr2, 0)
    in_Rc_for_i = np.array(np.min(distance_rr_rr2, 0) < np.max(distance_rr_rr2) * 1e-6, dtype='int')

    distance_rr_rr2 = distance.cdist(rr + wobit_j_Rc, rr2, 'euclidean')
    index_for_j = np.argmin(distance_rr_rr2, 0)
    in_Rc_for_j = np.array(np.min(distance_rr_rr2, 0) < np.max(distance_rr_rr2) * 1e-6, dtype='int')

    distance_rr_rr2 = distance.cdist(rr + wobit_tri_Rc, rr2, 'euclidean')
    index_for_tri = np.argmin(distance_rr_rr2, 0)
    in_Rc_for_tri = np.array(np.min(distance_rr_rr2, 0) < np.max(distance_rr_rr2) * 1e-6, dtype='int')


    # get wfs on rr2 from wfs on rr
    wfs_symm2_i = in_Rc_for_i * wfs_symm.T[index_for_i].T
    wfs_symm2_j = in_Rc_for_j * wfs_symm.T[index_for_j].T
    wfs_symm2_tri = in_Rc_for_tri * wfs_symm.T[index_for_tri].T
    # normalization
    norm_of_wfs = np.einsum('nxvr,nxvr->n', wfs_symm2_i.conj(), wfs_symm2_i).real * deltaS
    wfs_symm2_i = (wfs_symm2_i.T / norm_of_wfs ** 0.5).T
    norm_of_wfs = np.einsum('nxvr,nxvr->n', wfs_symm2_j.conj(), wfs_symm2_j).real * deltaS
    wfs_symm2_j = (wfs_symm2_j.T / norm_of_wfs ** 0.5).T
    norm_of_wfs = np.einsum('nxvr,nxvr->n', wfs_symm2_tri.conj(), wfs_symm2_tri).real * deltaS
    wfs_symm2_tri = (wfs_symm2_tri.T / norm_of_wfs ** 0.5).T

    # # times mask
    # if mask is not None:
    #     wfs_symm2_i = (mask.T * wfs_symm2_i.T).T
    #     wfs_symm2_j = (mask.T * wfs_symm2_j.T).T

    # # debug
    # plot_wan_WFs(rr, wfs_symm, valley=0, s=10, vmin=-1, vmax=1, latt=htb.latt)
    # plot_wan_WFs(rr, wfs_symm, valley=1, s=10, vmin=-1, vmax=1, latt=htb.latt)
    # plot_grid2(rr, wfs_symm[0,2], s=30, vmin=-1.2, vmax=1.2)
    # plot_grid2(rr2, wfs_symm2_i[0,2], s=10, vmin=-1.2, vmax=1.2)
    # plot_grid2(rr2, wfs_symm2_j[0,2], s=10, vmin=-1.2, vmax=1.2)

    # cal U and J
    lattA = LA.norm(latt.T[0])
    eps = LA.norm(rr2[0] - rr2[1]) / 50

    norm_rr2 = distance.cdist(rr2, rr2, 'euclidean')
    vrr = np.real(1 / (norm_rr2 + 1j * eps))

    wfs_i = wfs_symm2_i[wobit_i]
    wfs_j = wfs_symm2_j[wobit_j]
    wfs_tri = wfs_symm2_tri[wobit_tri]

    u_xy = np.einsum('yvr,yvr,xwp,xwp,vw,rp->xy', wfs_tri.conj(), wfs_tri, wfs_i.conj(), wfs_j, np.ones([2,2]), vrr) * deltaS**2 * lattA
    u = np.sum(u_xy)
    # print(u, '\n', u_xy)
    # print(J, '\n', J_xy)
    return u, u_xy

def check_inner_product(wfs, grid_ucell, latt):
    deltaS = LA.det(latt[:2,:2]) / grid_ucell[0] / grid_ucell[1]
    inner_product = np.einsum('nxvr,mxvr->nm', wfs.conj(), wfs).real * deltaS
    return inner_product

def normalize_wfs(wfs, grid_ucell, latt):
    deltaS = LA.det(latt[:2,:2]) / grid_ucell[0] / grid_ucell[1]
    norm = np.einsum('nxvr,nxvr->n', wfs.conj(), wfs).real * deltaS
    wfs_1 = (wfs.T / norm**0.5).T
    return wfs_1

def get_tmg_nonzero_mask(wfs):
    nw, ns, nvalley, nrr = wfs.shape
    tmg_wfs_mask = np.zeros([nw, ns]).reshape([nw, ns//2, 2])
    tmg_wfs_mask[0, -1, 0] = 1
    tmg_wfs_mask[0, 0, 1] = 1
    tmg_wfs_mask[1, -1, 1] = 1
    tmg_wfs_mask[1, 0, 0] = 1
    tmg_wfs_mask[2, 0, 1] = 1
    tmg_wfs_mask[2, -1, 0] = 1
    tmg_wfs_mask[3, 0, 0] = 1
    tmg_wfs_mask[3, -1, 1] = 1
    tmg_wfs_mask = tmg_wfs_mask.reshape([nw, ns])
    return tmg_wfs_mask

def unit_to_meV(x, theta, relative_epsilon=1):
    dim = EV * 1e3 / (2*np.pi) / (2.46*1e-10) / Epsilon0
    y = dim * x * np.sin(theta/2) / relative_epsilon
    return y

if __name__ == "__main__":
    if PYGUI:
        wdir = os.path.join(ROOT_WDIR, r'TMG_w90/tmg')
        input_dir = os.path.join(ROOT_WDIR, r'TMG_w90/hamlib')
    else:
        wdir = os.getcwd()
        input_dir = r'./'

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


if __name__ == "__main__" and Job == 'cal_columb_interaction_selected':
    # wfs_1 = normalize_wfs(wfs, grid_ucell, latt)
    # wfs_symm_1 = normalize_wfs(wfs_symm, grid_ucell, latt)
    # check_inner_product(wfs, grid_ucell, latt)
    # check_inner_product(wfs_1, grid_ucell, latt)
    # check_inner_product(wfs_symm, grid_ucell, latt)
    # check_inner_product(wfs_symm_1, grid_ucell, latt)

    '''
      * manu cal one
    '''
    wobit_i = 1 - 1
    wobit_j = 3 - 1
    wobit_j_R = np.array([1, 0, 0])
    tmg_wfs_mask = get_tmg_nonzero_mask(wfs_symm)
    wfs_symm_0 = (tmg_wfs_mask.T * wfs_symm.T).T
    wfs_symm_1 = normalize_wfs(wfs_symm_0, grid_ucell, latt)
    u, J, u_xy, J_xy = cal_columb_interaction_ijR(rr, wfs_symm_1, grid_ucell, latt, wobit_i, wobit_j, wobit_j_R, nnsuper=5)
    with np.printoptions(precision=3, suppress=True):
        print('i={}, j={}, R={}'.format(wobit_i, wobit_j, wobit_j_R))
        print('U={:<9.3f}'.format(u))
        print('J={:<9.3f}'.format(J))
        print('sublattice component of U:', '\n', u_xy)
        print('sublattice component of J:', '\n', J_xy)

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

    if PYGUI:
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

    worbit_distance_index = [
        # obi, obj, obtri, objR, obtriR
        [1, 1, 2, np.array([0, 1, 0]), np.array([0, 0, 0])], # 0
        [1, 1, 2, np.array([0, 1, 0]), np.array([-1, 1, 0])], # 0
        [1, 1, 2, np.array([0, 1, 0]), np.array([-1, 0, 0])], # 0
        [1, 1, 1, np.array([0, 1, 0]), np.array([-1, 1, 0])], # 0
    ]

    u = []
    u_xy = []
    for i, j, tri, jR, triR in worbit_distance_index:
        print('cal VH2 ...')
        _u, _u_xy = cal_assisted_columb_interaction(rr, wfs_symm, grid_ucell, latt, i-1, j-1, jR, tri-1, triR)
        u.append(_u)
        u_xy.append(_u_xy)
    u = np.array(u)
    u_xy = np.array(u_xy)

    if PYGUI:
        print(u)
        print(u_xy)
