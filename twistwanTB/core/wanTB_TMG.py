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

import time
import os
import numpy as np
import numpy.linalg as LA
from scipy.spatial import distance
from twistwanTB import PYGUI, ROOT_WDIR
from twistwanTB.wanpy.units import *
from twistwanTB.wanpy.mesh import make_kpath, make_mesh
from twistwanTB.wanpy.structure import Cell, Htb, BandstructureHSP, Bandstructure
from twistwanTB.wanpy.wannier90 import W90_nnkp, Wannier_setup, W90_interpolation
from twistwanTB.core.continuum_model import MacDonald_TMG_wan, return_twist_par

import matplotlib.pyplot as plt

# from mpi4py import MPI
#
# MPI_COMM = MPI.COMM_WORLD
# MPI_RANK = MPI_COMM.Get_rank()
# MPI_NCORE = MPI_COMM.Get_size()

'''
  * WFs
'''
class WFs_TMG(object):
    """
      * Info
        nw=4: num of WFs
        ns: nsublattice * nlayer
        nv=2: num of valleys
        nrr: num of rr grid
             for example, nrr=36*36 if grid=(36,36,1)

        latt: lattice of moire cell
        nnsuper: old number,
                 nnsuper * nnsuper large of super-moire lattice
                 is used to show WFs
        grid_ucell(3): (12,12,1) for example
        grid(3): (36,36,1) for nnsuper=3 and grid=(12,12,1)
        rr(nrr, 3): real-space grid
        wfs_symm: c3z symmtric WFs
        wcc: Wannier charge center
    """
    def __init__(self, ns, latt, nnsuper=5, plot_rr_dense=2):
        self.nw = 4
        self.ns = ns
        self.nv = 2

        self.latt = latt
        self.nnsuper = nnsuper

        self.grid_ucell = np.array([6 * plot_rr_dense, 6 * plot_rr_dense, 1]) # 18=6*3
        self.grid = self.grid_ucell * np.array([nnsuper, nnsuper, 1])
        self.rr = make_mesh(self.grid, type='continuous', centersym=False) * nnsuper
        self.rr -= np.array([nnsuper//2, nnsuper//2, 0], dtype='float64')
        self.rr = LA.multi_dot([self.latt, self.rr.T]).T
        self.nrr = self.rr.shape[0]

        self.wfs = np.zeros([self.nw, self.ns, self.nv, self.nrr], dtype='complex128')
        self.wfs_symm = np.zeros([self.nw, self.ns, self.nv, self.nrr], dtype='float64')
        self.wcc = np.zeros([self.nw, 3], dtype='float64')

    def save_npz(self, fname='wfs.npz'):
        np.savez_compressed(fname,
                            description='wfs in shape of (norbi, nsublattice*nlayer, nrr)',
                            nw=self.nw,
                            ns=self.ns,
                            nv=self.nv,
                            latt=self.latt,
                            nnsuper=self.nnsuper,
                            grid_ucell=self.grid_ucell,
                            grid=self.grid,
                            rr=self.rr,
                            wfs=self.wfs,
                            wfs_symm=self.wfs_symm,
                            wcc=self.wcc,
                            )

    def load_npz(self, fname='wfs.npz'):
        npdata = np.load(fname)
        description = npdata['description'].item()
        self.nw = npdata.get('nw')
        self.ns = npdata.get('ns')
        self.nv = npdata.get('nv')
        self.latt = npdata.get('latt')
        self.nnsuper = npdata.get('nnsuper')
        self.grid_ucell = npdata.get('grid_ucell')
        self.grid = npdata.get('grid')
        self.rr = npdata.get('rr')
        self.wfs = npdata.get('wfs')
        self.wfs_symm = npdata.get('wfs_symm')
        self.wcc = npdata.get('wcc')
        npdata.close()

    def plot_wan_WFs_all(self, valley=0, s=10, vmin=-1, vmax=1):
        rr = self.rr
        wfs = self.wfs_symm
        latt = self.latt

        cmap = 'seismic'
        wfs_rel = wfs / np.max(np.abs(wfs))
        wfs_rel = wfs_rel[:,:,valley,:]
        nw, nbasis = wfs.shape[:2]

        fig, axs = plt.subplots(nw, nbasis, sharex=True, sharey=True, figsize=(7, 10))
        for i in range(nw):
            for j in range(nbasis):
                ax = axs[i, j]
                # ax.set_aspect('equal')
                ax.scatter(rr[:,0], rr[:,1], c=wfs_rel[i,j], s=s*np.abs(wfs_rel[i,j]), cmap=cmap, alpha=1, vmax=vmax, vmin=vmin)

                # plot latt
                xx = np.arange(-1, 2)
                yy = np.arange(-1, 2)
                latt2 = latt[:2, :2]
                for iy in yy:
                    for ix in xx:
                        line1 = np.array([[ix, iy], [ix + 1, iy]]).T
                        line2 = np.array([[ix, iy], [ix, iy + 1]]).T
                        line1 = LA.multi_dot([latt2, line1])
                        line2 = LA.multi_dot([latt2, line2])
                        ax.plot(line1[0], line1[1], color='black', linewidth=0.5, zorder=10)
                        ax.plot(line2[0], line2[1], color='black', linewidth=0.5, zorder=10)

        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

    def plot_wan_WFs_nonzero(self, valley=0, s=3, vmin=-1, vmax=1):
        rr = self.rr
        wfs = self.wfs_symm
        latt = self.latt

        cmap = 'seismic'
        nw, ns, nvalley, nrr = wfs.shape
        wfs_rel = wfs / np.max(np.abs(wfs))
        wfs_rel = wfs_rel[:, :, valley, :].reshape(nw, ns//2, 2, nrr)

        u1 = 0
        d1 = -1
        fdict = [
            [[d1, 0], [u1, 1]],
            [[d1, 1], [u1, 0]],
            [[u1, 1], [d1, 0]],
            [[u1, 0], [d1, 1]],
        ]

        fig, axs = plt.subplots(nw, 2, sharex=True, sharey=True, figsize=(2, 6.5), dpi=150)
        for i in range(nw):
            for j in range(2):
                ax = axs[i, j]
                _l, _s = fdict[i][j]
                _wfs = wfs_rel[i, _l, _s]
                ax.scatter(rr[:, 0], rr[:, 1], c=_wfs, s=s*np.abs(_wfs), cmap=cmap, alpha=1, vmax=vmax, vmin=vmin)
                ax.set_xticks([])
                ax.set_yticks([])
                for ii in range(4):
                    line1 = np.array([[-1+ii,-1], [-1+ii,2]]).T
                    line2 = np.array([[-1,-1+ii], [2,-1+ii]]).T
                    line1 = LA.multi_dot([latt[:2, :2], line1])
                    line2 = LA.multi_dot([latt[:2, :2], line2])
                    ax.plot(line1[0], line1[1], color='black', linewidth=0.5, linestyle='-', zorder=10)
                    ax.plot(line2[0], line2[1], color='black', linewidth=0.5, linestyle='-', zorder=10)

        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.1)


'''
  * W90 interface
'''
class Wannier_setup_TMG(Wannier_setup):

    def __init__(self, *args, **kwargs):
        Wannier_setup.__init__(self, *args, **kwargs)

        self.nsuper = self.Ham.nsuper
        self.gxx = np.identity(self.nb)
        # self.meshRc = LA.multi_dot([self.latt, self.meshR.T]).T

        self.uX = None
        self.eibr = None

        # self.cell = cell
        # self.Rcs = cell.ions_car

        self.latt_wan = np.copy(self.latt)
        self.latt_wan[2, 2] = self.Ham.vac
        self.lattG_wan = 2 * np.pi * LA.inv(self.latt_wan.T)

        '''
          * note:
            
        '''
        self.gnr = None  # nAR (nw, nw, nrr), n = index of WFs i.e. (0,1,2,3),  A = (u/d(from top to bottom), layer(from bottom to top), sublattice)
        self.gnk = None  # knb (nk, nw, nb), n = WFs
        self.gnk_bloch = None  # nb (nw, nb), n = pz (d/u(from bottom to top), layer(from bottom to top), sublattice)

    def run(self):
        self.gnk_bloch = self.init_gnk_bloch()
        self.printer()

        self.cal_bandstructure()
        self.get_valley_erep()

        EIG_D = [
            self.op_proj_v1_erep,
            self.op_proj_v2_erep,
            self.op_valley_erep
        ]

        # w90_setup.bandE[:, w90_setup.nb // 2 + 2:] += 0.01
        # w90_setup.bandE[:, :w90_setup.nb // 2 - 2] -= 0.02
        AMN, MMN, EIG = self.main(calAMN=True, calMMN=False, saveW90=False)
        return AMN, MMN, EIG, EIG_D

    def cal_bandstructure(self):
        kkc = self.meshkc

        self.bandE = np.zeros([self.NNKP.nk, self.nb], dtype='float64')
        self.bandU = np.zeros([self.NNKP.nk, self.nb, self.nb], dtype='complex128')

        for i in range(self.NNKP.nk):
            print('>>> cal band u(k) at {}/{}'.format(i + 1, self.NNKP.nk))
            hk = self.w90gethk(kkc[i])
            self.bandE[i], self.bandU[i] = self.Ham.get_eigh(hk)
            # self.bandE[i], self.bandU[i] = LA.eigh(self.w90gethk(kkc[i]))

        self.bandE -= self.fermi

    def get_valley_erep(self):
        bandU = self.bandU
        self.op_proj_v1_erep = np.real(np.einsum('kim,ij,kjm->km', bandU.conj(), self.Ham.op_v1_proj, bandU, optimize=True))
        self.op_proj_v2_erep = np.real(np.einsum('kim,ij,kjm->km', bandU.conj(), self.Ham.op_v2_proj, bandU, optimize=True))
        self.op_valley_erep = np.real(np.einsum('kim,ij,kjm->km', bandU.conj(), self.Ham.op_valley, bandU, optimize=True))

    def get_gn(self, saveWF=False):
        # gn in bloch sum basis
        return self.get_gn_4band_TMG(saveWF)

    def get_gn_4band_TMG(self, saveWF=False):
        #
        # In auto WFs, only frist four WFs are presented.
        #

        radial_part_n = 0
        nk = self.meshk.shape[0]
        rr = self.meshRc
        borden = LA.norm(self.latt.T[0]) / 1.5

        gnr = np.zeros([self.nw, self.Ham.nw_d+self.Ham.nw_u, rr.shape[0]], dtype='complex128') # nAR, A=(d, u)

        wcc = LA.multi_dot([self.latt,
            np.array([

                [1 / 3, 1 / 3, 0],
                [2 / 3, 2 / 3, 0],
                [1 / 3, 1 / 3, 0],
                [2 / 3, 2 / 3, 0],

                # [1 / 3, 1 / 3, 0],
                # [0, 0, 0],
                # [1 / 3, 1 / 3, 0],
                # [0, 0, 0],

            ]).T
        ]).T

        i = -1

        n_d_A = self.Ham.nw_d - 2
        n_d_B = self.Ham.nw_d - 1
        n_u_A = self.Ham.nw_d
        n_u_B = self.Ham.nw_d + 1
        '''
          * bottom layer (AB)
        '''
        i += 1
        gnr[i, n_d_A] = 1
        gnr[i] *= self.get_moire_orbi(rr - wcc[i], radial_part_n, 's', borden)

        i += 1
        gnr[i, n_d_B] = 1
        gnr[i] *= self.get_moire_orbi(rr - wcc[i], radial_part_n, 's', borden)

        '''
          * upper layer (AB)
        '''
        i += 1
        gnr[i, n_u_B] = -1
        gnr[i] *= self.get_moire_orbi(rr - wcc[i], radial_part_n, 's', borden)

        i += 1
        gnr[i, n_u_A] = -1
        gnr[i] *= self.get_moire_orbi(rr - wcc[i], radial_part_n, 's', borden)

        gnk = np.zeros([nk, self.nw, self.nb], dtype='complex128')
        for i in range(nk):
            print('>>> cal band gnk(k) at {}/{}'.format(i + 1, nk))
            gnk[i] = self.get_gnk_from_gnr(self.meshkc[i], gnr)

        self.gnr = gnr
        self.gnk = gnk
        return gnk

    def get_gnk_from_gnr(self, kc, gnr):
        #
        # get |gn(r)> in bloch sum basis |gnk>
        #

        # gnr = np.random.random([5, self.Ham.nw_u+self.Ham.nw_d, nrr])
        ham_nw_u = self.Ham.nw_u
        ham_nw_d = self.Ham.nw_d

        # gnr: nAR=n(d+u)R
        gnr_d = gnr[:, :ham_nw_d, :]  # nuR
        gnr_u = gnr[:, ham_nw_d:, :]  # ndR

        rr = self.meshRc
        K1uc = self.Ham.K1uc
        K2uc = self.Ham.K2uc
        wn_u = gnr_u * np.cos(LA.multi_dot([K1uc, rr.T])).T
        wn_d = gnr_d * np.cos(LA.multi_dot([K1uc, rr.T])).T

        # The following lines should keep in line with kkc of TMG
        gridGc = LA.multi_dot([self.lattG, self.gridG.T]).T
        gridGc_u = np.array([
            gridGc - self.Ham.Ku1 + K1uc,
            -gridGc - self.Ham.Ku2 + K2uc,
        ])
        gridGc_d = np.array([
            gridGc - self.Ham.Kd1 + K1uc,
            -gridGc - self.Ham.Kd2 + K2uc,
        ])
        eikR = np.exp(-1j * np.einsum('a,Ra->R', kc, rr))
        eiGR_u = np.exp(-1j * np.einsum('VGa,Ra->VGR', gridGc_u, rr))
        eiGR_d = np.exp(-1j * np.einsum('VGa,Ra->VGR', gridGc_d, rr))

        gnk = np.hstack([
            np.einsum('nuR,R,GR->nGu', wn_u, eikR, eiGR_u[0], optimize=True).reshape(self.nw, self.Ham.nb_u),
            np.einsum('ndR,R,GR->nGd', wn_d, eikR, eiGR_d[0], optimize=True).reshape(self.nw, self.Ham.nb_d),
            np.einsum('nuR,R,GR->nGu', wn_u, eikR, eiGR_u[1], optimize=True).reshape(self.nw, self.Ham.nb_u),
            np.einsum('ndR,R,GR->nGd', wn_d, eikR, eiGR_d[1], optimize=True).reshape(self.nw, self.Ham.nb_d),
        ])

        gnorm = np.sqrt(np.real(np.einsum('nb,nb->n', gnk.conj(), gnk)))
        gnk = np.einsum('nb,n->nb', gnk, np.real(1/(gnorm+0.0001j)))
        return gnk

    def init_gnk_bloch(self):
        # * get |gn(r)> in bloch sum basis |gnk>
        #   used to project bloch band onto atomic pz orbitals
        #   * eikR is smooth function in mucell thus can be
        #     neglected. This has been numerically checked.

        # gnr = np.random.random([5, self.Ham.nw_u+self.Ham.nw_d, nrr])
        ham_nw_u = self.Ham.nw_u
        ham_nw_d = self.Ham.nw_d

        meshR = self.make_mesh(self.nmeshR, type='continuous', centersym=False)
        rr = LA.multi_dot([self.latt, meshR.T]).T

        gnr = np.zeros([self.nw, ham_nw_d + ham_nw_u, rr.shape[0]], dtype='complex128')  # nAR, A=(d, u)
        for i in range(ham_nw_d + ham_nw_u):
            gnr[i, i] = 1
        # gnr: nAR=n(d+u)R
        gnr_d = gnr[:, :ham_nw_d, :]  # nuR
        gnr_u = gnr[:, ham_nw_d:, :]  # ndR

        K1uc = self.Ham.K1uc
        K2uc = self.Ham.K2uc
        wn_u = gnr_u #* np.cos(LA.multi_dot([K1uc, rr.T])).T
        wn_d = gnr_d #* np.cos(LA.multi_dot([K1uc, rr.T])).T

        gridGc = LA.multi_dot([self.lattG, self.gridG.T]).T
        gridGc_u = np.array([
            gridGc - self.Ham.Ku1 + K1uc,
            -gridGc - self.Ham.Ku2 + K2uc,
        ])
        gridGc_d = np.array([
            gridGc - self.Ham.Kd1 + K1uc,
            -gridGc - self.Ham.Kd2 + K2uc,
        ])
        eiGR_u = np.abs(np.exp(-1j * np.einsum('VGa,Ra->VGR', gridGc_u, rr)))
        eiGR_d = np.abs(np.exp(-1j * np.einsum('VGa,Ra->VGR', gridGc_d, rr)))

        # kc = np.array([0, 0, 0])
        # eikR = np.exp(-1j * np.einsum('a,Ra->R', kc, rr))
        # gnk = np.hstack([
        #     np.einsum('nuR,R,GR->nGu', wn_u, eikR, eiGR_u[0], optimize=True).reshape(self.nw, self.Ham.nb_u),
        #     np.einsum('ndR,R,GR->nGd', wn_d, eikR, eiGR_d[0], optimize=True).reshape(self.nw, self.Ham.nb_d),
        #     np.einsum('nuR,R,GR->nGu', wn_u, eikR, eiGR_u[1], optimize=True).reshape(self.nw, self.Ham.nb_u),
        #     np.einsum('ndR,R,GR->nGd', wn_d, eikR, eiGR_d[1], optimize=True).reshape(self.nw, self.Ham.nb_d),
        # ])
        gnk = np.hstack([
            np.einsum('nuR,GR->nGu', wn_u, eiGR_u[0], optimize=True).reshape(self.nw, self.Ham.nb_u),
            np.einsum('ndR,GR->nGd', wn_d, eiGR_d[0], optimize=True).reshape(self.nw, self.Ham.nb_d),
            np.einsum('nuR,GR->nGu', wn_u, eiGR_u[1], optimize=True).reshape(self.nw, self.Ham.nb_u),
            np.einsum('ndR,GR->nGd', wn_d, eiGR_d[1], optimize=True).reshape(self.nw, self.Ham.nb_d),
        ])

        gnorm = np.sqrt(np.real(np.einsum('nb,nb->n', gnk.conj(), gnk)))
        gnk = np.einsum('nb,n->nb', gnk, np.real(1/(gnorm+0.0001j)))
        return gnk

    def get_moire_orbi(self, rr, n=1, l='s', sigma=1.0):
        """
        get evenlop function
        :param rr:
        :param n: n=0 is gauss function, n>0 is hydrogenic-atom-radial-part-function
        :param l:
        :param sigma: smearing for the orbital
        :return:
        """
        a = sigma

        xx, yy, zz = rr.T[0], rr.T[1], rr.T[2]
        r = LA.norm(rr, axis=1)
        invr = np.real(1 / (r - 0.1j))
        rho = r/a
        rho2 = (r/a) ** 2

        if n == 0:
            Rn = 2 * a**-1.5 * np.exp(-rho2)
        elif n == 1:
            Rn = 2 * a**-1.5 * np.exp(-rho)
        elif n == 2:
            Rn = 2*-0.5 * a**-1.5 * (1-0.5*rho) * np.exp(-rho/2)
        elif n == 21:
            Rn = 24**-0.5 * a**-1.5 * rho * np.exp(-rho/2)
        elif n == 3:
            Rn = 2*27**-0.5 * a**-1.5 * (1-2*rho/3+2*rho**2/27) * np.exp(-rho/3)
        else:
            Rn = None

        if l == 's' or l == 'pz':
            Ylm = np.sqrt(1/2/np.pi)
        elif l == 'px':
            Ylm = np.sqrt(1/2/np.pi) * invr * xx
        elif l == 'py':
            Ylm = np.sqrt(1/2/np.pi) * invr * yy
        elif l == 'p+':
            Ylm = np.sqrt(1/2/np.pi) * invr * (xx + 1j * yy)
        elif l == 'p-':
            Ylm = np.sqrt(1/2/np.pi) * invr * (xx - 1j * yy)
        else:
            Ylm = None

        g = Rn * Ylm
        gnorm = np.sqrt(np.sum(np.abs(g) ** 2) / rr.shape[0])
        g /= gnorm
        return g

class WanTB_TMG(object):

    def __init__(self,
                 m, n, NG, w1, w2, mp_grid, Umax,
                 U_BN_u, U_BN_d, enable_hBN,
                 htb_u, htb_d,
                 ):
        # os.chdir(input_dir)

        print('nsuper, natom, theta = ', return_twist_par(m))

        self.m = m
        self.n = n
        self.NG = NG
        self.w1 = w1
        self.w2 = w2
        self.Umax = Umax

        self.htb_u = htb_u
        self.htb_d = htb_d

        # self.htb_u = Htb()
        # self.htb_d = Htb()
        # self.htb_u.load_htb(htbfname_u)
        # self.htb_d.load_htb(htbfname_d)

        # self.cell_u = Cell()
        # self.cell_d = Cell()
        # self.cell_u.load_poscar(r'graphene_SL.vasp')
        # self.cell_d.load_poscar(r'graphene_SL.vasp')

        self.mp_grid = mp_grid

        self.bandU = None

        # os.chdir(wdir)

        # self.cell_real_tmg = Cell_twisted_hex(self.cell_u, self.cell_d, self.m, self.n)
        self.ham_BM = MacDonald_TMG_wan(self.htb_u, self.htb_d, m=self.m, n=self.n, N=self.NG, w1=self.w1, w2=self.w2, tLu=0, tLd=-1, vac=300, Umax=self.Umax, rotk=True)
        self.ham_BM.hBN = self.ham_BM.get_hBN(U_BN_u, U_BN_d, enable=enable_hBN)

        self.NNKP = W90_nnkp()
        # self.NNKP.load_from_w90(fname=nnkp_fname)
        self.NNKP.init_NNKP(self.ham_BM.latt, self.mp_grid)
        self.vmat_symm_index = self.symmetric_kmesh(self.NNKP)

        self.fb_index = None

    # def run(self, froz_i1=None, froz_i2=None):
    #     wcc, AMN, EIG, EIG_D = self.wanTB_tmg_w90_setup(self.ham_BM, self.NNKP)
    #     w90ip = self.wanTB_build_tmg_wanTB(self.ham_BM, wcc, AMN, EIG, EIG_D, froz_i1, froz_i2)
    #     rr, wfs = self.wanTB_plot_wfs(self.ham_BM, w90ip.vmat, nmeshR_plot, plot_wfs, fname=fname_wfs)

    def wanTB_tmg_w90_setup(self, ham, NNKP):
        wccf = np.array([
            [1 / 3, 1 / 3, 2.600269],  # (L1, hexA)
            [2 / 3, 2 / 3, 2.600269],  # (L1, hexB)
            [1 / 3, 1 / 3, 5.762960],  # (L2, hexA)
            [2 / 3, 2 / 3, 5.762960],  # (L2, hexB)
        ], dtype='float64')

        wccf.T[2] /= ham.vac
        wcc = LA.multi_dot([ham.latt, wccf.T]).T

        nmeshR = np.array([60, 60, 1], dtype='int')
        nnWSR = 1
        gridGsym = ham.gridG

        ngridG = np.array([12, 12, 1]) # this parameter does not matter
        w90_setup = Wannier_setup_TMG(
            Ham=ham, NNKP=NNKP, latt=ham.latt,
            nb=ham.nb, nw=ham.nw, wcc=wcc,
            nmeshR=nmeshR, ngridG=ngridG, nnWSR=nnWSR,
            seedname=r'wannier90',
            bloch_outer_win=None,
            )
        w90_setup.gridG = gridGsym

        AMN, MMN, EIG, EIG_D = w90_setup.run()
        self.bandU = w90_setup.bandU
        self.w90_setup = w90_setup
        return wcc, AMN, EIG, EIG_D

    def wanTB_build_tmg_wanTB(self, ham, wcc, AMN, EIG, EIG_D, froz_i1=None, froz_i2=None, applysymmetry=True, set_zero_onsite=True):
        # NNKP = wanTB_init.NNKP
        # mp_grid = wanTB_init.mp_grid
        latt = ham.latt
        mp_grid = self.mp_grid
        NNKP = self.NNKP
        nb = ham.nb
        vmat_symm_index = self.vmat_symm_index

        if froz_i1 is None:
            i1 = nb // 4
        else:
            i1 = froz_i1

        if froz_i2 is None:
            i2 = nb // 4 - 1
        else:
            i2 = froz_i2
        fb_index = np.array([i1, i2, i1+nb//2, i2+nb//2])
        self.fb_index = fb_index
        self.fb_index_vb = fb_index.reshape(2, 2)

        '''
          * Interpolation
        '''
        # step 1
        nw = 4

        # EIG.eig = EIG.eig#[vmat_symm_index]
        amn = AMN.amn

        w90ip = W90_interpolation(nw, mp_grid, latt, NNKP, EIG, wcc, load_u=False)
        w90ip.vmat = w90ip.svd_init_guess(amn)
        w90ip.init_symmOP(nD=3)

        # step 2
        eig_froz = np.zeros_like(w90ip.eig)
        eig_froz[:, fb_index] = 1
        # eig_froz = Rect(EIG.eig, -0.08, 0.08, T=100)

        # plt.plot(EIG.eig[0], Rect(EIG.eig, -0.5, 0.5, T=50)[0])

        vmat_selec = np.zeros([NNKP.nk, nb, 4], dtype='complex128')
        vmat_selec[:, :, 0] = w90ip.vmat[:, :, 0]
        vmat_selec[:, :, 1] = w90ip.vmat[:, :, 1]
        vmat_selec[:, :, 2] = w90ip.vmat[:, :, 2]
        vmat_selec[:, :, 3] = w90ip.vmat[:, :, 3]
        vmat = np.einsum('km,kmn->kmn', eig_froz, vmat_selec)

        vmat = w90ip.svd_init_guess(vmat)
        # vmat = (vmat.T * np.exp(-1j * np.angle(vmat).T[0])).T  # fix gauge, added Nov.14,2020
        w90ip.vmat = vmat

        # def cal_chern_latt(bandindex):
        #     nk1, nk2 = (12,12)
        #     UU = np.einsum('kqmn->kqnm', vmat[:, fb_index, :].reshape(nk1,nk2,4,4))[:,:,:,bandindex]
        #     C = 0
        #     for i in range(nk1 - 1):
        #         print('cal at {}/{}'.format(i + 1, nk1))
        #
        #         for j in range(nk2 - 1):
        #             F = LA.det(UU[i, j].T.conj() @ UU[i + 1, j]) * \
        #                 LA.det(UU[i + 1, j].T.conj() @ UU[i + 1, j + 1]) * \
        #                 LA.det(UU[i + 1, j + 1].T.conj() @ UU[i, j + 1]) * \
        #                 LA.det(UU[i, j + 1].T.conj() @ UU[i, j])
        #             F = -np.imag(np.log(F))
        #             C += F
        #     C = C / np.pi / 2
        #     print(C)
        #
        # cal_chern_latt(np.array([0]))
        # cal_chern_latt(np.array([1]))
        # cal_chern_latt(np.array([0,1]))
        # cal_chern_latt(np.array([2]))
        # cal_chern_latt(np.array([3]))
        # cal_chern_latt(np.array([2,3]))
        # sys.exit()

        '''
          * interpolate
        '''
        w90ip.interpolate_hR()
        w90ip.interpolate_DR(EIG_D[0], i=0, name='proj_v1')
        w90ip.interpolate_DR(EIG_D[1], i=1, name='proj_v2')
        w90ip.interpolate_DR(EIG_D[2], i=2, name='op_valley')

        '''
          * apply symmetry
        '''
        if applysymmetry:
            w90ip.h_Rmn = self.apply_symmetry(w90ip.h_Rmn, w90ip.gridRc, w90ip.wcc)

        # set zero onsite energy
        if set_zero_onsite:
            onsiet_energy = np.sum(np.diag(np.real(w90ip.h_Rmn[w90ip.nR//2]))) / w90ip.nw
            w90ip.h_Rmn[w90ip.nR//2] -= onsiet_energy * np.identity(w90ip.nw)
            w90ip.fermi = -1 * onsiet_energy

        # build htb
        cell_tmg_wcc = self.wanTB_get_cell_tmg_wcc(ham.latt, wcc)
        htb = w90ip.build_htb(cell_tmg_wcc)

        # '''
        #   * recal vmat
        # '''
        # for i in range(NNKP.nk):
        #     V, E = LA.eigh(htb.get_hk(w90ip.meshk[i]))
        #     w90ip.vmat[i, fb_index] = V.T.conj()

        return htb, w90ip

    def wanTB_plot_wfs(self, ham, wcc, vmat, nnsuper=3, plot_rr_dense=2, cal_wfs=False, applysymmetry=True):
        ns = self.htb_u.nw + self.htb_d.nw
        WFs = WFs_TMG(ns, ham.latt, nnsuper=nnsuper, plot_rr_dense=plot_rr_dense)
        WFs.wcc = wcc

        if not cal_wfs:
            return WFs

        print('[INFO] calculating wfs in real space')
        meshk = self.NNKP.kk
        meshkc = LA.multi_dot([ham.lattG, meshk.T]).T
        nk = meshkc.shape[0]

        # kc = np.array([0, 0, 0])
        # bloch_sum_k_rr = ham.get_bloch_sum_in_Rspace(kc, rr)
        # psi_k_rr = ham.get_phi_in_Rspace(kc, rr)
        # psi_k_rr = np.real(psi_k_rr * psi_k_rr.conj())
        #
        # un = psi_k_rr[ham.nb//2, 0]
        # plot_orbital(rr, un, nmeshR, latt=ham.latt)

        psi_ks_rr = np.zeros([nk, ham.nb, ham.nw, WFs.nrr], dtype='complex128')
        for i in range(nk):
            kc = meshkc[i]
            print('[{:>3d}/{:<3d}] {} Cal psi_k_rr at ({:.3f} {:.3f} {:.3f})'.format(
                    i+1, nk,
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    kc[0], kc[1], kc[2]
                    )
                  )
            U = self.bandU[i]
            bloch_sum_k_rr = ham.get_bloch_sum_in_Rspace(kc, WFs.rr)
            psi_ks_rr[i] = np.einsum('nb,bsR->nsR', U.T, bloch_sum_k_rr)
        print('F.T. FROM psi_k_rr to wfs_R0_rr ...')
        wfs = np.einsum('kvbn,kvbsR->nsvR', vmat[:,self.fb_index_vb,:], psi_ks_rr[:, self.fb_index_vb, :, :]) / nk
        WFs.wfs = wfs

        if applysymmetry:
            wfs_symm = np.zeros_like(wfs, dtype='complex128')

            # apply c3 symmetry
            nw, nbasis = WFs.wfs.shape[:2]
            for i in range(nw):
                c3_index, c3c3_index = self.find_Rspace_c3_symm_index(WFs.rr, wcc[i])
                for j in range(nbasis):
                    for v in range(2):
                        wfs_symm[i, j, v] = (wfs[i, j, v, :] + wfs[i, j, v, c3_index] + wfs[i, j, v, c3c3_index]) / 3
                        # WFs.wfs_symm[i, j, v] = (wfs[i, j, v, :] + wfs[i, j, v, c3_index] + wfs[i, j, v, c3c3_index]).imag / 3 # imag part is zero

            # apply time-reversal symmetry
            wfs_symm = np.einsum('nsvR,vw->nswR', wfs_symm, 0.5*np.ones([2, 2], dtype='complex128'))

        WFs.wfs_symm = wfs_symm.real
        # un = wfs[1, 3]
        # un = np.real(un * un.conj())
        # plot_orbital(rr, un, nmeshR, latt=ham.latt)
        # WFs.save_npz(fname)
        return WFs

    # def wanTB_cal_columb_interaction(self, rr, wfs_symm, grid_ucell, latt, wobit_i, wobit_j, wobit_j_R, nnsuper=5):
    #     '''
    #       * Used for calculating electron-electron interaction parameters U & J
    #       * Notice this function are not dependent on any varierbles of WanTB.
    #     :param rr: grid of wfs_symm
    #     :param wfs_symm: wannier functions in shape of (nw=4, n_{sublattice basis}, nrr)
    #     :param griducell: shape of unit cell grid in rr (rr is a larger grid of super cell)
    #     :param latt: lattice of TMG
    #     :param wobit_i: wannier orbital i, seted at R=(0, 0, 0) lattice
    #     :param wobit_j: wannier orbital j, seted at R=wobit_j_R lattice
    #     :param wobit_j_R: for instance = np.array([1, 0, 0])
    #     :param nnsuper: nnsuper larger of griducell will be used to calculate U & J
    #     :return u, J: U and J
    #     :return u_sc, J_sx: sublattice component of U and J
    #
    #     '''
    #     # '''
    #     #   * main
    #     #     this is a simple calculator only used for cal U
    #     # '''
    #     # fname_wfs = r'wfs_MTBG_m30n31_dft_nnsuper3.npz'
    #     # npdata = np.load(fname_wfs)
    #     # description = npdata['description'].item()
    #     # rr = npdata['rr']
    #     # gridrr = npdata['grid']
    #     # wcc = npdata['wcc']
    #     # wfs = npdata['wfs']
    #     # wfs_symm = npdata['wfs_symm']
    #     # latt = npdata['latt']
    #     # npdata.close()
    #     #
    #     # lattA = LA.norm(latt.T[0])
    #     # eps = LA.norm(rr[0]-rr[1]) / 50
    #     #
    #     # wobit_j_Rc = LA.multi_dot([latt, wobit_j_R])
    #     # norm_rr = distance.cdist(rr-wcc[wobit_i], rr-wcc[wobit_j]-wobit_j_Rc, 'euclidean')
    #     # vrr = np.real(1 / (norm_rr + 1j * eps))
    #     #
    #     # wfs_i = wfs_symm[wobit_i]
    #     # wfs_j = wfs_symm[wobit_j]
    #     # u = np.einsum('sr,xp,rp->sx', np.abs(wfs_i)**2, np.abs(wfs_j)**2, vrr) / lattA
    #     # u_exact = np.sum(u)
    #     # J = np.einsum('sr,xp,xp,sr,rp->sx', wfs_i.conj(), wfs_i.conj(), wfs_j, wfs_j, vrr) / lattA
    #     # J_exact = np.sum(J)
    #     #
    #     # print(u, '\n', u_exact)
    #     # print(J, '\n', J_exact)
    #
    #     '''
    #       * main
    #     '''
    #     # griducell = np.array([12, 12, 1])  # remove in fruture
    #
    #     wobit_j_Rc = LA.multi_dot([latt, wobit_j_R])
    #
    #     # get larger grid rr2 to compute U and J
    #     # nnsuper = 5
    #     nmeshR_plot = grid_ucell * np.array([nnsuper, nnsuper, 1])
    #     rr2 = make_mesh(nmeshR_plot, type='continuous', centersym=False) * nnsuper
    #     rr2 -= np.array([nnsuper // 2, nnsuper // 2, 0], dtype='float64')
    #     rr2 = LA.multi_dot([ham.latt, rr2.T]).T
    #     nrr2 = rr2.shape[0]
    #
    #     # get mapping between rr and rr2
    #     distance_rr_rr2 = distance.cdist(rr, rr2, 'euclidean')
    #     index_for_i = np.argmin(distance_rr_rr2, 0)
    #     in_Rc_for_i = np.array(np.min(distance_rr_rr2, 0) < np.max(distance_rr_rr2) * 1e-6, dtype='int')
    #
    #     distance_rr_rr2 = distance.cdist(rr + wobit_j_Rc, rr2, 'euclidean')
    #     index_for_j = np.argmin(distance_rr_rr2, 0)
    #     in_Rc_for_j = np.array(np.min(distance_rr_rr2, 0) < np.max(distance_rr_rr2) * 1e-6, dtype='int')
    #
    #     # get wfs on rr2 from wfs on rr
    #     wfs_symm2_i = in_Rc_for_i * wfs_symm[:, :, index_for_i]
    #     wfs_symm2_j = in_Rc_for_j * wfs_symm[:, :, index_for_j]
    #
    #     # # debug
    #     # plot_grid2(rr, wfs_symm[0,2], s=30, vmin=-1.2, vmax=1.2)
    #     # plot_grid2(rr2, wfs_symm2_i[0,2], s=10, vmin=-1.2, vmax=1.2)
    #     # plot_grid2(rr2, wfs_symm2_j[0,2], s=10, vmin=-1.2, vmax=1.2)
    #
    #     # cal U and J
    #     lattA = LA.norm(latt.T[0])
    #     eps = LA.norm(rr2[0] - rr2[1]) / 50
    #
    #     norm_rr2 = distance.cdist(rr2, rr2, 'euclidean')
    #     vrr = np.real(1 / (norm_rr2 + 1j * eps))
    #
    #     wfs_i = wfs_symm2_i[wobit_i]
    #     wfs_j = wfs_symm2_j[wobit_j]
    #     u_xy = np.einsum('xr,yp,rp->xy', np.abs(wfs_i) ** 2, np.abs(wfs_j) ** 2, vrr) / lattA
    #     u = np.sum(u_xy)
    #     J_xy = np.einsum('xr,yp,yp,xr,rp->xy', wfs_i.conj(), wfs_j.conj(), wfs_i, wfs_j, vrr) / lattA
    #     J = np.sum(J_xy)
    #
    #     # print(u, '\n', u_xy)
    #     # print(J, '\n', J_xy)
    #     return u, J, u_xy, J_xy

    def wanTB_get_cell_tmg_wcc(self, latt, wcc):
        # get cell obj with wcc of tmg
        cell = Cell()
        cell.name = 'TMG wannierTB model'
        cell.lattice = latt
        cell.latticeG = 2 * np.pi * LA.inv(latt.T)
        cell.ions_car = wcc
        cell.ions = cell.get_ions()
        cell.N = wcc.shape[0]
        cell.spec = ['C' for i in range(cell.N)]
        return cell

    '''
      * object build-in tools
    '''
    def symmetric_kmesh(self, NNKP):
        """
          * C3 symmetry
        """
        latt = NNKP.latt
        lattG = NNKP.lattG
        kkc = LA.multi_dot([lattG, NNKP.kk.T]).T

        theta = 2 * np.pi / 3
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 0]
        ])

        c3_kkc = LA.multi_dot([rot, kkc.T]).T
        c3c3_kkc = LA.multi_dot([rot, rot, kkc.T]).T

        c3_kk = LA.multi_dot([LA.inv(lattG), c3_kkc.T]).T
        c3_kk = np.remainder(c3_kk + 1e-10, 1.0)

        c3c3_kk = LA.multi_dot([LA.inv(lattG), c3c3_kkc.T]).T
        c3c3_kk = np.remainder(c3c3_kk + 1e-10, 1.0)

        c3_kkc = LA.multi_dot([lattG, c3_kk.T]).T
        c3c3_kkc = LA.multi_dot([lattG, c3c3_kk.T]).T

        # index in kkc after applied c3 and c3c3
        c3_index = np.argmin(distance.cdist(kkc, c3_kkc, 'euclidean'), 0)
        c3c3_index = np.argmin(distance.cdist(kkc, c3c3_kkc, 'euclidean'), 0)

        k_star_degen = 2 * np.array(c3_index == c3c3_index, dtype='int') + 1
        self.k_star_index = np.argwhere(k_star_degen==3)
        vmat_symm_index = np.zeros(kkc.shape[0], dtype='int') - 1
        vmat_symm_Dc3k = np.zeros([kkc.shape[0], 4], dtype='complex128')

        self.Dc3k = np.exp(-1j * np.einsum('ka,na->kn', c3_kkc, np.array([-latt.T[0], -2*latt.T[0], -latt.T[0], -2*latt.T[0]])))
        self.Dc3c3k = np.exp(-1j * np.einsum('ka,na->kn', c3c3_kkc, np.array([-latt.T[1], -2*latt.T[1], -latt.T[1], -2*latt.T[1]])))

        # x = _D_c3 ** 0
        # (x[k_star_index] + _D_c3[k_star_index] + _D_c3c3[k_star_index]) / 3

        nk = kkc.shape[0]
        for i in range(nk):
            if vmat_symm_index[i] == -1:
                for j in range(i):
                    # c3
                    if c3_index[j] == i:
                        vmat_symm_index[i] = j
                        vmat_symm_Dc3k[i] = self.Dc3k[i]
                        break
                    # c3c3
                    elif c3c3_index[j] == i:
                        vmat_symm_index[i] = j
                        vmat_symm_Dc3k[i] = self.Dc3c3k[i]
                        break
            if vmat_symm_index[i] == -1:
                vmat_symm_index[i] = i
                vmat_symm_Dc3k[i] = 1

        self.vmat_symm_Dc3k = vmat_symm_Dc3k

        return vmat_symm_index

    def apply_symmetry(self, hr_Rmn, Rc, wcc):
        hr_Rmn_symm = np.zeros_like(hr_Rmn, dtype='complex128')
        for i in range(4):
            for j in range(4):
                c3_index, c3c3_index = self.find_Rspace_c3_symm_index(Rc, wcc[i] - wcc[j])
                hr_Rmn_symm[:, i, j] = np.real(
                    (hr_Rmn[:, i, j] + hr_Rmn[c3_index, i, j] + hr_Rmn[c3c3_index, i, j]) / 3
                )
        return hr_Rmn_symm

    def find_Rspace_c3_symm_index(self, grid, R0=0):
        """
          * find c3 index of gridR - R0
          How to use:
          Rc = gridR - R0
          Rc[c3_index] = c3*Rc
          Rc[c3c3_index] = c3*c3*Rc
        """
        theta = 2 * np.pi / 3
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 0]
        ])
        Rc = grid - R0
        Rc.T[2] = 0
        nrr = Rc.shape[0]
        c3_Rc = LA.multi_dot([rot, Rc.T]).T
        c3c3_Rc = LA.multi_dot([rot, rot, Rc.T]).T

        e_index = np.arange(nrr)

        distance_Rc_c3Rc = distance.cdist(Rc, c3_Rc, 'euclidean')
        distance_Rc_c3c3Rc = distance.cdist(Rc, c3c3_Rc, 'euclidean')

        in_Rc = np.array(np.min(distance_Rc_c3Rc, 0) < np.max(distance_Rc_c3Rc) * 1e-6, dtype='int')
        c3_index = (1 - in_Rc) * e_index + in_Rc * np.argmin(distance_Rc_c3Rc, 0)

        in_Rc = np.array(np.min(distance_Rc_c3c3Rc, 0) < np.max(distance_Rc_c3c3Rc) * 1e-6, dtype='int')
        c3c3_index = (1 - in_Rc) * e_index + in_Rc * np.argmin(distance_Rc_c3c3Rc, 0)
        return c3_index, c3c3_index

    def find_Rspace_c2z_symm_index(self, grid, R0=0):
        Rc = grid - R0
        Rc.T[2] = 0.
        nrr = Rc.shape[0]
        e_index = np.arange(nrr)
        distance_Rc_c2zRc = distance.cdist(Rc, -Rc, 'euclidean')
        in_Rc = np.array(np.min(distance_Rc_c2zRc, 0) < np.max(distance_Rc_c2zRc) * 1e-6, dtype='int')
        c2z_index = (1 - in_Rc) * e_index + in_Rc * np.argmin(distance_Rc_c2zRc, 0)
        return c2z_index

    def find_Rspace_c2x_symm_index(self, grid, R0=0):
        theta = np.pi / 3
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 0]
        ])
        Rc = grid - R0
        Rc.T[2] = 0
        Rc = LA.multi_dot([rot, Rc.T]).T
        nrr = Rc.shape[0]
        e_index = np.arange(nrr)
        distance_Rc_c2xRc = distance.cdist(Rc, Rc*np.array([-1,1,0]), 'euclidean')
        in_Rc = np.array(np.min(distance_Rc_c2xRc, 0) < np.max(distance_Rc_c2xRc) * 1e-6, dtype='int')
        c2x_index = (1 - in_Rc) * e_index + in_Rc * np.argmin(distance_Rc_c2xRc, 0)
        return c2x_index


'''
  * Cal bandprojection
'''
def cal_bandprojection_morie(w90_setup, ham, kpath_car):
    nk = kpath_car.shape[0]
    nw = 4
    nb = ham.nb
    bandE = np.zeros([nk, nb], dtype='float64')
    bandprojection = np.zeros([nw, nk, nb], dtype='complex128')

    for i, kc in zip(range(nk), kpath_car):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                kc[0], kc[1], kc[2]
                )
              )
        hk = ham.get_hk(kc)
        E, U = LA.eigh(hk)
        bandE[i] = E - ham.fermi

        gnk = w90_setup.get_gnk_from_gnr(kc, w90_setup.gnr)[:nw] # used for projects on Morie orbitals
        # gnk = w90_setup.gnk_bloch # used for projects on real pz orbitals
        bandprojection[:, i, :] = np.einsum('nb,bm->nm', gnk.conj(), U)

    return bandE, bandprojection

def cal_bandprojection_pz(w90_setup, ham, kpath_car):
    nk = kpath_car.shape[0]
    nw = w90_setup.nw
    nb = ham.nb
    bandE = np.zeros([nk, nb], dtype='float64')
    bandprojection = np.zeros([nw, nk, nb], dtype='complex128')

    gnk = w90_setup.gnk_bloch  # used for projects on real pz orbitals

    for i, kc in zip(range(nk), kpath_car):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                kc[0], kc[1], kc[2]
                )
              )
        hk = ham.get_hk(kc)
        E, U = LA.eigh(hk)
        bandE[i] = E - ham.fermi

        bandprojection[:, i, :] = np.einsum('nb,bm->nm', gnk.conj(), U)

    return bandE, bandprojection

'''
  * Calculators
'''
def cal_band(ham, kpath, nb, returnU=False, simudiag=False):
    nk = kpath.shape[0]
    bandE = np.zeros([nk, nb], dtype='float64')
    bandU = np.zeros([nk, nb, nb], dtype='float64')

    print('calculating band ... ')
    for i, kc in zip(range(nk), kpath):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                kc[0], kc[1], kc[2]
                )
              )
        hk = ham.get_hk(kc)
        if simudiag:
            E, U = ham.get_eigh(hk)
        else:
            E, U = LA.eigh(hk)
        bandE[i] = E - ham.fermi
        if returnU:
            bandU[i] = U

    if returnU:
        return bandE, bandU
    else:
        return bandE

def cal_band_Dk(ham, kpath, nb, isymmOP):
    nk = kpath.shape[0]
    bandW = np.zeros([nk, nb], dtype='float64')

    print('calculating band Dk ... ')
    for i, k in zip(range(nk), kpath):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                k[0], k[1], k[2]
                )
              )
        hk = ham.get_Dk(k, isymmOP)
        W, V = LA.eigh(hk)
        bandW[i] = W

    return bandW

def cal_berry_curvature(ham, kpath, nb, ewide=0.0001):
    nk = kpath.shape[0]
    bandE = np.zeros([nk, nb], dtype='float64')
    BC = np.zeros([3, nk, nb], dtype='float64')

    for i, kc in zip(range(nk), kpath):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                kc[0], kc[1], kc[2]
                )
              )
        bandE[i], BC[:, i, :] = ham.cal_berry_curvature(kc, ewide)

    return bandE, BC

def cal_slab_band(ham, kpath, nb):
    nk = kpath.shape[0]
    bandE = np.zeros([nk, nb], dtype='float64')

    for i, k in zip(range(nk), kpath):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                k[0], k[1], k[2]
                )
              )
        hk = ham.get_hk_slab(k, Ny)
        E, U = LA.eigh(hk)
        bandE[i] = E

    return bandE

def cal_surfDOS(ham, kpath, ne):
    nk = kpath.shape[0]
    surfDOS_L = np.zeros([nk, ne], dtype='float64')
    surfDOS_R = np.zeros([nk, ne], dtype='float64')

    for i, kc in zip(range(nk), kpath):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                kc[0], kc[1], kc[2]
                )
              )
        surfDOS_L[i], surfDOS_R[i] = ham.get_slab_green(ee, kc, Ny, eps)

    return surfDOS_L, surfDOS_R

# @MPI_Gather(MPI, iterprint=100000)
# def cal_wilson_loop(ik, dim):
#     print('cal wcc at ({}/{})'.format(ik + 1, nk2))
#     return ham.cal_wilson_loop_k(ik, i1, i2, nk1, nk2)


'''
  * Tools
'''
# class Token_htb(object):
#     def __init__(self, latt, lattG, R, ndegen, hr_Rmn, wcc, wccf):
#         self.latt = latt
#         self.lattG = lattG
#         self.R = R
#         self.ndegen = ndegen
#         self.hr_Rmn = hr_Rmn
#         self.wcc = wcc
#         self.wccf = wccf
#         self.fermi = 0
#         nR = R.shape[0]
#
#     def get_hk(self, k, tbgauge=False):
#         eikR = np.exp(2j * np.pi * np.einsum('a,Ra->R', k, self.R)) / self.ndegen
#         if tbgauge:
#             eiktau = np.exp(2j * np.pi * np.einsum('a,na', k, self.wccf))
#             hk = np.einsum('R,m,Rmn,n->mn', eikR, eiktau.conj(), self.h_Rmn, eiktau, optimize=True)
#         else:
#             hk = np.einsum('R,Rmn->mn', eikR, self.hr_Rmn, optimize=True)
#         return hk

def find_Gspace_c3_symm_index(kkc, lattG):

    theta = 2 * np.pi / 3
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 0]
    ])

    c3_kkc = LA.multi_dot([rot, kkc.T]).T
    c3c3_kkc = LA.multi_dot([rot, rot, kkc.T]).T

    c3_kk = LA.multi_dot([LA.inv(lattG), c3_kkc.T]).T
    c3_kk = np.remainder(c3_kk + 1e-10, 1.0)

    c3c3_kk = LA.multi_dot([LA.inv(lattG), c3c3_kkc.T]).T
    c3c3_kk = np.remainder(c3c3_kk + 1e-10, 1.0)

    c3_kkc = LA.multi_dot([lattG, c3_kk.T]).T
    c3c3_kkc = LA.multi_dot([lattG, c3c3_kk.T]).T

    # index in kkc after applied c3 and c3c3
    c3_index = np.argmin(distance.cdist(kkc, c3_kkc, 'euclidean'), 0)
    c3c3_index = np.argmin(distance.cdist(kkc, c3c3_kkc, 'euclidean'), 0)

    k_star_degen = 2 * np.array(c3_index == c3c3_index, dtype='int') + 1
    k_star_index = np.argwhere(k_star_degen == 3)
    c3_symm_index = np.zeros(kkc.shape[0], dtype='int') - 1

    nk = kkc.shape[0]
    for i in range(nk):
        if c3_symm_index[i] == -1:
            for j in range(i):
                if c3_index[j] == i:
                    c3_symm_index[i] = j
                    break
                elif c3c3_index[j] == i:
                    c3_symm_index[i] = j
                    break
        if c3_symm_index[i] == -1:
            c3_symm_index[i] = i
    return c3_symm_index

def find_Rspace_c3_symm_index(grid, R0=0):
    theta = 2 * np.pi / 3
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 0]
    ])
    Rc = grid - R0
    Rc.T[2] = 0
    nrr = Rc.shape[0]
    c3_Rc = LA.multi_dot([rot, Rc.T]).T
    c3c3_Rc = LA.multi_dot([rot, rot, Rc.T]).T

    e_index = np.arange(nrr)

    distance_Rc_c3Rc = distance.cdist(Rc, c3_Rc, 'euclidean')
    distance_Rc_c3c3Rc = distance.cdist(Rc, c3c3_Rc, 'euclidean')

    in_Rc = np.array(np.min(distance_Rc_c3Rc, 0) < np.max(distance_Rc_c3Rc) * 1e-6, dtype='int')
    c3_index = (1 - in_Rc) * e_index + in_Rc * np.argmin(distance_Rc_c3Rc, 0)

    in_Rc = np.array(np.min(distance_Rc_c3c3Rc, 0) < np.max(distance_Rc_c3c3Rc) * 1e-6, dtype='int')
    c3c3_index = (1 - in_Rc) * e_index + in_Rc * np.argmin(distance_Rc_c3c3Rc, 0)
    return c3_index, c3c3_index


'''
  * remove warping
'''
def remove_warping_for_tdbg(htb_u):
    """
      * use only gamma0 (intralayer hopping) and
        gamma1 (interlayer hopping) parameters
        for bilayer graphene
    """
    htb_mask = np.zeros([587, 4, 4])
    # htb_mask[:] = np.kron(np.eye(2), np.ones([2,2]))
    htb_mask[587//2] = np.kron(np.eye(2), sigmax.real)
    htb_mask[587//2, 1, 2] = 1
    htb_mask[587//2, 2, 1] = 1
    htb_mask[587//2+1] = np.kron(np.eye(2), np.array([
        [0, 0],
        [1, 0]
    ]))  # [0, 1, 0]
    htb_mask[587//2-1] = np.kron(np.eye(2), np.array([
        [0, 1],
        [0, 0]
    ]))  # [0, -1, 0]
    htb_mask[587//2+25] = np.kron(np.eye(2), np.array([
        [0, 0],
        [1, 0]
    ]))  # [1, 0, 0]
    htb_mask[587//2-25] = np.kron(np.eye(2), np.array([
        [0, 1],
        [0, 0]
    ]))  # [-1, 0, 0]
    htb_u.hr_Rmn *= htb_mask
    return htb_u

'''
  * func
'''
def simu_eigh(H, S):
    """
      * used for Simultaneous Diagonalize 4 by 4 matrix H
        with constrain valley symmetry S. 
        [H, S] = 0
        The valley symmetry for retured U == w
    """  #
    # S = 0.5 * (S + LA.multi_dot([H, S, LA.inv(H)]))
    # S = LA.multi_dot([H, S, LA.inv(H)])
    # print(LA.eigvals(S))
    S = 0.5 * (S + S.T.conj())
    ES, V1 = LA.eigh(S)  # w = [-1, -1, 1, 1]
    H1_blockdiag = LA.multi_dot([V1.T.conj(), H, V1])

    h1 = H1_blockdiag[:2, :2]
    h2 = H1_blockdiag[2:, 2:]
    h1 = 0.5 * (h1 + h1.T.conj())
    h2 = 0.5 * (h2 + h2.T.conj())
    e1, v1 = LA.eigh(h1)
    e2, v2 = LA.eigh(h2)
    E = np.hstack([e1, e2])
    V2 = np.block([
        [v1, np.zeros([2, 2])],
        [np.zeros([2, 2]), v2],
    ])
    U = LA.multi_dot([V1, V2])
    U *= np.exp(-1j * np.angle(U)[0])
    return E, U

'''
  * Plot
'''
def plot_band_BM(eig, xlabel, HSP_path_car, eemin=-0.08, eemax=0.08, yticks=None):
    import matplotlib.pyplot as plt
    from twistwanTB.wanpy.toolkits import kmold

    nline = len(xlabel) - 1
    kpath = kmold(HSP_path_car)

    nk, nb = eig.shape

    eig_v_p, eig_v_n = np.einsum('kvn->vkn', eig.reshape(nk, 2, nb//2))

    '''
      * plot band
    '''
    fig = plt.figure('compare', figsize=[4, 3], dpi=150)
    # fig = plt.figure('compare')
    fig.clf()
    ax = fig.add_subplot(111)

    ax.axis([kpath.min(), kpath.max(), eemin, eemax])
    ax.axhline(0, color='k', linewidth=1, zorder=101, linestyle='--')

    # ax.plot(kpath, eig, linewidth=6, linestyle="-", color='#ff1744', alpha=0.3, zorder=12)
    # ax.plot(kpath, eig_v_n, linewidth=1., linestyle="-", color='k', alpha=1, zorder=12)  # valley -, wanTB model
    ax.plot(kpath, eig_v_n, linewidth=1., linestyle="-", color='blue', alpha=1, zorder=12)  # valley -, wanTB model
    ax.plot(kpath, eig_v_p, linewidth=1., linestyle="-", color='red', alpha=1, zorder=12) # valley +, wanTB model

    for i in range(1, nline):
        ax.axvline(x=kpath[i * nk // nline], linestyle='--', color='k', linewidth=1, alpha=1, zorder=101)

    if xlabel is not None:
        # xlabel = ['K', 'G', '-K', '-M', 'G', 'M', 'K']
        num_xlabel = len(xlabel)
        plt.xticks(kpath[np.arange(num_xlabel) * (nk // nline)], xlabel)
    ax.set_ylabel('Energy (eV)')
    if yticks is not None:
        ax.set_yticks(yticks)
    else:
        ax.set_yticks(np.linspace(eemin, eemax, 5))

    fig.tight_layout()
    fig.show()

def plot_2band_compare(eig, eig_ref, HSP_list, xlabel, HSP_path_car, eemin=-3.0, eemax=3.0, yticks=None, save=False, savefname='compareband.png'):
    import matplotlib.pyplot as plt
    from twistwanTB.wanpy.toolkits import kmold

    nline = HSP_list.shape[0] - 1
    kpath = kmold(HSP_path_car)

    nk, nw = eig.shape

    eig_v1, eig_v2 = np.einsum('kvn->vkn', eig.reshape(nk, 2, 2))

    '''
      * plot band
    '''
    fig = plt.figure('compare', figsize=[4, 3], dpi=150)
    # fig = plt.figure('compare')
    fig.clf()
    ax = fig.add_subplot(111)

    ax.axis([kpath.min(), kpath.max(), eemin, eemax])
    ax.axhline(0, color='k', linewidth=0.5, zorder=101)

    # ax.plot(kpath, eig, linewidth=6, linestyle="-", color='#ff1744', alpha=0.3, zorder=12)
    ax.plot(kpath, eig_v1, linewidth=3, linestyle="-", color='blue', alpha=0.7, zorder=12)  # valley -, wanTB model
    ax.plot(kpath, eig_v2, linewidth=3, linestyle="-", color='red', alpha=0.7, zorder=12) # valley +, wanTB model
    ax.plot(kpath, eig_ref, linewidth=1, linestyle="--", color='k', alpha=1, zorder=11)

    for i in range(1, nline):
        ax.axvline(x=kpath[i * nk // nline], linestyle='-', color='k', linewidth=0.5, alpha=1, zorder=101)

    if xlabel is not None:
        # xlabel = ['K', 'G', '-K', '-M', 'G', 'M', 'K']
        num_xlabel = len(xlabel)
        plt.xticks(kpath[np.arange(num_xlabel) * (nk // nline)], xlabel)
    ax.set_ylabel('Energy / (meV)')
    if yticks is not None:
        ax.set_yticks(yticks)
    else:
        ax.set_yticks(np.linspace(eemin, eemax, 5))

    fig.tight_layout()
    fig.show()
    if save:
        fig.savefig(savefname)

def plot_bandprojection(eig, bandproj, kpath_car, xlabel, eemin=-0.1, eemax=0.1, s=3, yticks=None, plotaxis=True):
    import matplotlib.pyplot as plt
    from twistwanTB.wanpy.toolkits import kmold

    nk, nb = bandproj.shape
    proj = np.abs(bandproj)  # (nk, nb)

    vmax = np.max(proj)
    vmin = np.min(proj)
    print('vmin={}, vmax={}'.format(vmin, vmax))

    # cmap = 'Reds'
    cmap = 'seismic'
    nline = len(xlabel) - 1
    kpath = kmold(kpath_car)

    '''
      * plot band
    '''
    # fig = plt.figure('proj band', figsize=[3, 3], dpi=150) # for FIG.S1&2
    # fig = plt.figure('proj band', figsize=[3.5, 4.5], dpi=150) # for FIG.2
    fig = plt.figure('proj band', figsize=[3.5, 3.5], dpi=150)  # for FIG.3
    fig.clf()
    ax = fig.add_subplot(111)
    ax.axis([kpath.min(), kpath.max(), eemin, eemax])

    if plotaxis:
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=1, zorder=101)
        for i in range(1, nline):
            ax.axvline(x=kpath[i * nk // nline], linestyle='-', color='black', linewidth=0.8, alpha=1, zorder=101)

    if xlabel is not None:
        num_xlabel = len(xlabel)
        plt.xticks(kpath[np.arange(num_xlabel) * (nk // nline)], xlabel)

    # ax.plot(kpath, eig, linewidth=1, linestyle="-", color='k', alpha=1)
    cs = ax.scatter(np.kron(kpath, np.ones([nb])).T, eig, cmap=cmap, c=proj, s=s, alpha=1, vmin=vmin, vmax=vmax)

    if yticks is not None:
        ax.set_yticks(yticks)
    else:
        ax.set_yticks(np.linspace(eemin, eemax, 3))

    cbar = fig.colorbar(cs, ticks=np.linspace(vmin, vmax, 2))
    cbar.ax.set_yticklabels(['Low', 'High'])

    ax.set_ylabel('Energy (eV)')

    fig.tight_layout()
    fig.show()

def plot_bc_distribution(meshkc, nmesh, dist, vmax=None):
    import matplotlib.pyplot as plt

    # dist = np.sum(meshkc, axis=1)

    # meshkc = self.meshkc
    nk1, nk2, nk3 = nmesh
    cmap = 'seismic'

    # nk, nw = bandE.shape
    # dos = np.log(dos)

    fig = plt.figure('bc_dist', figsize=[6, 2.8], dpi=150)
    fig.clf()
    ax = fig.add_subplot(111)

    XX_MIN = meshkc.T[0].min()
    XX_MAX = meshkc.T[0].max()
    YY_MIN = meshkc.T[1].min()
    YY_MAX = meshkc.T[1].max()

    ax.axis([XX_MIN, XX_MAX, YY_MIN, YY_MAX])
    ax.axhline(0, color='k', linewidth=0.5, zorder=101)

    ax.set_xlim(XX_MIN, XX_MAX)
    ax.set_ylim(YY_MIN, YY_MAX)

    meshkc_2D = meshkc.reshape(nk1, nk2, 3)  # XY or ac face
    dist_2D = dist.reshape(nk1, nk2)
    # meshkc_2D = np.einsum('YXa->XYa', meshkc_2D)
    # dist_2D = np.einsum('YX->XY', dist_2D)

    if vmax is None:
        vmax = np.max(np.abs(dist))
    levels = np.linspace(-vmax, vmax, 500)

    cs = ax.contourf(meshkc_2D[:, :, 0], meshkc_2D[:, :, 1], dist_2D, levels, vmax=vmax, vmin=-vmax, cmap=cmap)
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    # plt.title('Fermi={:.4f} eV')

    cbar = plt.colorbar(cs)
    cbar.set_label(r'$\Omega_{n}^{z}\left(k\right)$')
    cbar.set_ticks(np.linspace(-vmax, vmax, 5))

    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()

    fig.show()

def plot_ham_decay(htb):
    lattA = LA.norm(htb.latt.T[0])
    d = np.zeros(htb.nR)
    h = np.zeros(htb.nR)
    D = np.zeros(htb.nR)
    for i in range(htb.nR):
        d[i] = LA.norm(htb.Rc[i])
        h[i] = LA.norm(htb.hr_Rmn[i])
        D[i] = LA.norm(htb.D_iRmn[2, i])
    d /= lattA
    h /= h.max()
    D /= D.max()

    # plt.plot(d)
    # plt.plot(h)

    fig = plt.figure('hopping decay in real space', figsize=[3.5, 3.5], dpi=150)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.axis([-0.5, d.max()+0.5, -0.1, 1.1])

    ax.scatter(d, h, c='black', s=3, alpha=1, label='ham')
    ax.scatter(d, D, c='red', s=3, alpha=1, label='op_valley')

    ax.legend(loc='upper right')
    ax.set_xlabel('$d_{0-R}/a$')
    ax.set_ylabel('$t/t_{max}$')

    fig.tight_layout()
    fig.show()

def plot_ham(grid, hr_Rmn, tmax=None, LM=1, ticks=None, s=50, axis=None):
    # LM = lattice contance of morie cell
    # grid = htb.Rc
    # hr_Rmn = htb.hr_Rmn

    tt = np.real(hr_Rmn)
    if tmax is not None:
        tt = tt / tmax
    else:
        tt = tt / np.max(np.abs(tt))

    cmap = 'seismic'

    vmin = -1
    vmax = 1
    levels = np.linspace(vmin, vmax, 500)

    fig, axs = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(10,10))
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            # ax.set_aspect('equal')
            ax.scatter(grid[:,0]/LM, grid[:,1]/LM, c=tt[:,i,j], s=s, cmap=cmap, alpha=1, vmax=vmax, vmin=vmin)
            ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3, zorder=101)
            ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3, zorder=101)
            if axis is not None:
                ax.axis(axis*2)
            if ticks is not None:
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)

    # cbar = plt.colorbar(cs)
    # cbar.set_ticks(np.linspace(vmin, vmax, 5))
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

def plot_orbital(rr, un, grid, latt=None):
    # un = np.log(np.abs(un))
    un = np.abs(un)

    rr = rr.reshape(grid[0], grid[1], grid[2], 3)[:, :, 0, :2]
    un = un.reshape(grid[0], grid[1], grid[2])[:, :, 0]
    # vmax = max_value(un)
    vmax = np.max(un)
    vmin = 0 # np.min(un)

    fig = plt.figure('Orbital in real space')
    fig.clf()
    ax = fig.add_subplot(111)

    # if latt is not None:
    #     xx = np.arange(-1, 2)
    #     yy = np.arange(-1, 2)
    #     latt2 = latt[:2, :2]
    #     for iy in yy:
    #         for ix in xx:
    #             line1 = np.array([[ix, iy], [ix+1, iy]]).T
    #             line2 = np.array([[ix, iy], [ix, iy+1]]).T
    #             line1 = LA.multi_dot([latt2, line1])
    #             line2 = LA.multi_dot([latt2, line2])
    #             ax.plot(line1[0], line1[1], color='black', zorder=10)
    #             ax.plot(line2[0], line2[1], color='black', zorder=10)


    # params = {
    #     'figure.figsize': '6, 8'  # set figure size
    # }
    # pylab.rcParams.update(params)

    cmap = 'Reds'
    # cmap = sns.diverging_palette(127, 255, s=99, l=57, n=100, as_cmap=True)
    levels = np.linspace(vmin, vmax, 500)
    CS = plt.contourf(rr[:,:,0], rr[:,:,1], un, levels, vmax=vmax, vmin=vmin, cmap=cmap)

    plt.xlabel('$X (\mathring{A})$')
    plt.ylabel('$Y (\mathring{A})$')

    cbar = plt.colorbar(CS)
    cbar.set_label('$|g_n>$')
    cbar.set_ticks(np.linspace(-vmax, vmax, 5))

    plt.tight_layout()
    ax.axis('equal')

def plot_wan_WFs(rr, wfs, valley=0, s=10, vmin=-1, vmax=1, latt=None):
    cmap = 'seismic'
    wfs_rel = wfs / np.max(np.abs(wfs))
    wfs_rel = wfs_rel[:,:,valley,:]
    nw, nbasis = wfs.shape[:2]

    fig, axs = plt.subplots(nw, nbasis, sharex=True, sharey=True, figsize=(7, 10))
    for i in range(nw):
        for j in range(nbasis):
            ax = axs[i, j]
            # ax.set_aspect('equal')
            ax.scatter(rr[:,0], rr[:,1], c=wfs_rel[i,j], s=s*np.abs(wfs_rel[i,j]), cmap=cmap, alpha=1, vmax=vmax, vmin=vmin)

            if latt is not None:
                xx = np.arange(-1, 2)
                yy = np.arange(-1, 2)
                latt2 = latt[:2, :2]
                for iy in yy:
                    for ix in xx:
                        line1 = np.array([[ix, iy], [ix + 1, iy]]).T
                        line2 = np.array([[ix, iy], [ix, iy + 1]]).T
                        line1 = LA.multi_dot([latt2, line1])
                        line2 = LA.multi_dot([latt2, line2])
                        ax.plot(line1[0], line1[1], color='black', linewidth=0.5, zorder=10)
                        ax.plot(line2[0], line2[1], color='black', linewidth=0.5, zorder=10)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

if __name__ == '__main__':
    '''
      * Job = 
      ** band
      ** surband
      
      ** wloop
      ** bc_dist_in_bz
      ** chern_latt_method
      
      ** wanTB
      ** bandprojection
      ** bandprojection_pz
      
    '''
    Job = 'wanTB'

    if PYGUI:
        wdir = os.path.join(ROOT_WDIR, r'TMG_w90/tmg')
        input_dir = os.path.join(ROOT_WDIR, r'TMG_w90/hamlib')
    else:
        wdir = os.getcwd()
        input_dir = r'/home/jincao/2_Works/2_twist_graphene/hamlib'

'''
  * Input
'''
if __name__ == '__main__' and Job not in ['wanTB', 'bandprojection_pz']:
    os.chdir(input_dir)

    m = 26
    Umax = 0.0
    enable_hBN = False
    # # for others
    U_BN_u = None
    U_BN_d = None
    # fot mTBG
    # U_BN_u = np.array([0.0])
    # U_BN_d = np.array([0.03])
    # fot mTTG
    # U_BN_u = np.array([-0.0, -0.0])
    # U_BN_d = np.array([0.05])
    # U_BN_u = np.array([-0.02, -0.02])
    # U_BN_d = np.array([0.08])
    # U_BN_u = np.array([-0.1, -0.])
    # U_BN_d = np.array([0.1])
    # for 3+3
    # U_BN_u = np.array([0, 0, 0, 0])
    # U_BN_d = np.array([0.01, 0.01, 0.0, 0.0])

    # htbfname_u = r'htb_SLG_SK.h5'
    # htbfname_d = r'htb_SLG_SK.h5'

    # htbfname_u = r'htb_SL_DFT.h5'
    # htbfname_d = r'htb_SL_DFT.h5'

    htbfname_u = r'htb_AB_SCAN.h5'
    htbfname_d = r'htb_AB_SCAN.h5'

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
    # FLG = Cell()
    # FLG.load_poscar(r'graphene_AB.vasp')
    # cell = Cell_twisted_hex(FLG, FLG, m=2, n=3)
    # cell.save_poscar()
    os.chdir(wdir)

if __name__ == '__main__' and Job == 'band':
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
    # bandE, BC = cal_berry_curvature(ham, kpath_car, ham.nb)

    # i1 = ham.nb // 4 - 1
    # i2 = ham.nb // 4
    # fb_index = np.array([i1 + ham.nb // 2, i2 + ham.nb // 2, i1, i2])  # (valley-,VB); (-,CB); (+,VB); (+,CB);

    fb_index = ham.get_fb_index()  # (valley-,VB); (-,CB); (+,VB); (+,CB);

    # bandE[:, fb_index[:2]] += 0.03

    if PYGUI:
        plot_band_BM(bandE, xlabel, kpath_car, eemin=-0.04, eemax=0.04)

        # vmax = np.max(np.abs(BC[2])) / 30
        # bandstructure_hsp.plot_distribution_in_band(BC[2], eemin=-0.05, eemax=0.05, unit='C', S=30, vmax=vmax)


        # i1 = ham.nb // 4
        # i2 = ham.nb // 4 - 1
        # fb_index = np.array([i1, i2, i1+ham.nb//2, i2+ham.nb//2])
        # plt.plot(BC[2, :, fb_index[0]], linestyle='-', color='blue', linewidth=1, alpha=1)
        # plt.plot(BC[2, :, fb_index[1]], linestyle='-', color='blue', linewidth=2, alpha=1)
        # plt.plot(BC[2, :, fb_index[2]], linestyle='-', color='red', linewidth=1, alpha=1)
        # plt.plot(BC[2, :, fb_index[3]], linestyle='-', color='red', linewidth=2, alpha=1)

if __name__ == '__main__' and Job == 'surband':
    os.chdir(wdir)

    eps = 0.01
    ne = 100
    ee = np.linspace(-6, 6, ne)

    Ny = 10
    nk1 = 11
    kpath_HSP = np.array([
        [-1/2, 0.0, 0.0], #  X
        [ 0.0, 0.0, 0.0], #  G
        [ 1/2, 0.0, 0.0], #  X
    ])
    xlabel = ['-X', 'G', 'X']

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = LA.multi_dot([ham.lattG, kpath.T]).T

    # bandE = cal_band(ham, kpath, ham.nb)
    bandE = cal_slab_band(ham, kpath, Ny*ham.nb)
    # surfDOS_L, surfDOS_R = cal_surfDOS(ham, kpath, ne)

    bandstructure_hsp = BandstructureHSP()
    bandstructure_hsp.HSP_list = kpath_HSP
    bandstructure_hsp.HSP_path_frac = kpath
    bandstructure_hsp.HSP_path_car = kpath_car
    bandstructure_hsp.HSP_name = xlabel
    bandstructure_hsp.ee = ee

    # bandstructure_hsp.surfDOS1 = surfDOS_L
    # bandstructure_hsp.surfDOS2 = surfDOS_R

    bandstructure_hsp.eig = bandE
    bandstructure_hsp.nk, bandstructure_hsp.nb = bandE.shape

    if PYGUI:
        bandstructure_hsp.plot_band(eemin=-0.1, eemax=0.1, unit='C')
        # bandstructure_hsp.plot_surfDOS(ee, surfDOS_L, eemin=-6, eemax=6, unit='C')
        # bandstructure_hsp.plot_surfDOS(ee, surfDOS_R, eemin=-6, eemax=6, unit='C')

        # np.savez_compressed(r'RobbonBand.npz',
        #                     bandstructure_hsp=bandstructure_hsp,
        #                     )

        # npdata = np.load(r'RobbonBand.npz')
        # bandstructure_hsp = npdata['bandstructure_hsp'].item()

if __name__ == '__main__' and Job == 'bc_dist_in_bz':
    os.chdir(wdir)

    nmesh = np.array([101, 101, 1])
    ewide = 0.1 / 1000

    meshk = make_mesh(nmesh, type='continuous')
    meshkc = LA.multi_dot([ham.lattG, meshk.T]).T
    vcell2d = LA.det(ham.latt[:2,:2])

    bandE, BC = cal_berry_curvature(ham, meshkc, ham.nb, ewide=ewide)

    nk, nb = bandE.shape
    bandstructure = Bandstructure(nb=nb, nk=nk, latt=ham.latt, nmesh=nmesh)
    bandstructure.eig = bandE
    bandstructure.BC = BC

    fb_index = ham.get_fb_index()  # (valley-,VB); (-,CB); (+,VB); (+,CB);
    chern_numbers = (2 * np.pi) * np.sum(BC[2, :, fb_index], axis=1) / meshk.shape[0] / vcell2d

    print(chern_numbers)

    np.savez_compressed(r'bc_dist_in_bz.npz',
                        meshk=meshk,
                        meshkc=meshkc,
                        nmesh=nmesh,
                        latt=ham.latt,
                        lattG=ham.lattG,
                        vcell2d=vcell2d,
                        fb_index=fb_index,
                        chern_numbers=chern_numbers,
                        BC=BC
                        )

    if PYGUI:
        npdata = np.load(r'bc_dist_in_bz.npz')
        chern_numbers = npdata['chern_numbers']
        fb_index = npdata['fb_index']
        meshkc = npdata['meshkc']
        nmesh = npdata['nmesh']
        BC = npdata['BC']
        plot_bc_distribution(meshkc, nmesh, BC[2, :, fb_index[0]], vmax=None)

if __name__ == '__main__' and Job == 'wloop':
    os.chdir(wdir)

    fb_index = ham.get_fb_index() # (valley-,VB); (-,CB); (+,VB); (+,CB);

    bandindex = fb_index[[1,0]]
    bandindex = np.arange(ham.nb)[:fb_index[2]]

    # bandindex = fb_index[[2]]
    # kk1, theta2 = ham.cal_wilson_loop(bandindex, e1=0, e2=1, e3=2, k3=0, nk1=30, nk2=20)
    # bandindex = fb_index[[3]]
    # kk1, theta3 = ham.cal_wilson_loop(bandindex, e1=0, e2=1, e3=2, k3=0, nk1=30, nk2=20)
    # bandindex = fb_index[[2,3]]
    # kk1, theta23 = ham.cal_wilson_loop(bandindex, e1=0, e2=1, e3=2, k3=0, nk1=30, nk2=20)


    bandindex = np.arange(ham.nb)[:fb_index[2]]
    kk1, theta2 = ham.cal_wilson_loop(bandindex, e1=0, e2=1, e3=2, k3=0, nk1=100, nk2=50)
    bandindex = np.arange(ham.nb)[:fb_index[2]+1]
    kk1, theta3 = ham.cal_wilson_loop(bandindex, e1=0, e2=1, e3=2, k3=0, nk1=100, nk2=50)
    bandindex = np.arange(ham.nb)[:fb_index[2]+2]
    kk1, theta4 = ham.cal_wilson_loop(bandindex, e1=0, e2=1, e3=2, k3=0, nk1=100, nk2=50)

    if PYGUI:
        pass
        # from wanpy.response.response_plot import plot_wloop
        # plot_wloop(kk1, theta3, ymin=-0.8, ymax=0.8, s=5)

    '''
      * par 
    '''
    dim = [1, 4]
    # ikk2 = np.arange(nk2)
    # theta = cal_wilson_loop(ikk2, dim)
    #
    # if MPI_RANK == 0:
    #     kk2 = np.linspace(0, 1, nk2 + 1)[:-1]
    #     theta = theta[0]
    #
    #     np.savez_compressed(r'wloop.npz',
    #                         kk2=kk2,
    #                         twist_ang=ham.theta_deg,
    #                         theta=theta,
    #                         )
    #
    #     if PYGUI:
    #         from wanpy.response.response_plot import plot_wloop
    #         data = np.load(r'wloop.npz')
    #         kk2 = data['kk2']
    #         theta = data['theta']
    #
    #         plot_wloop(kk2, theta, ymin=-0.8, ymax=0.8)

    # '''
    #   * multi twisted angles, par
    # '''
    # ikk2 = np.arange(nk2)
    # mm = np.arange(10, 33, 1)
    # # mm = np.arange(10, 18, 1)
    # nmm = mm.shape[0]
    # if MPI_RANK == 0:
    #     thetas = np.zeros([nmm, nk2, 4])
    #
    # for i in range(nmm):
    #     # ham = MacDonald_TMG_wan(m=mm[i], n=mm[i]+1, N=3, w1=0.0797, w2=0.0975, tLu=0, tLd=-1, vac=300, htbfname_u=r'htb_AB_SCAN.npz', htbfname_d=r'htb_AB_SCAN.npz')
    #     ham = MacDonald_TMG_wan(m=mm[i], n=mm[i]+1, N=3, w1=0.0797, w2=0.0975, tLu=0, tLd=-1, vac=300, htbfname_u=r'htb_SL_DFT.npz', htbfname_d=r'htb_SL_DFT.npz', rotk=True)
    #
    #     theta = cal_wilson_loop(ikk2, dim)
    #     if MPI_RANK == 0:
    #         thetas[i] = theta[0]
    #
    # if MPI_RANK == 0:
    #     kk2 = np.linspace(0, 1, nk2 + 1)[:-1]
    #     np.savez_compressed(r'wloops.npz',
    #                         kk2=kk2,
    #                         mm=mm,
    #                         thetas=thetas,
    #                         )
    #
    #     if PYGUI:
    #         from wanpy.response.response_plot import plot_wloop
    #
    #         data = np.load(r'wloops.npz')
    #         kk2 = data['kk2']
    #         mm = data['mm']
    #         thetas = data['thetas']
    #         nsuper, natom, twist_ang = return_twist_par(mm)
    #         data.close()
    #
    #         i = 4
    #         print('m{}n{}, theta={:5.3f} deg'.format(mm[i], mm[i]+1, twist_ang[i]))
    #         plot_wloop(kk2, thetas[i], ymin=-0.8, ymax=0.8, save=False, savefname='wloop.pdf')

if __name__ == '__main__' and Job == 'chern_latt_method':
    os.chdir(wdir)
    nk1 = 30
    nk2 = 30

    fb_index = ham.get_fb_index() # (valley-,VB); (-,CB); (+,VB); (+,CB);

    chern_number_sep = []
    chern_number_sep.append(ham.cal_chern_latt_method(fb_index[[2]], nk1, nk2))
    chern_number_sep.append(ham.cal_chern_latt_method(fb_index[[3]], nk1, nk2))
    chern_number_sep.append(ham.cal_chern_latt_method(fb_index[[2,3]], nk1, nk2))

    chern_number_int = []
    chern_number_int.append(ham.cal_chern_latt_method(np.arange(ham.nb)[:fb_index[2]], nk1, nk2))
    chern_number_int.append(ham.cal_chern_latt_method(np.arange(ham.nb)[:fb_index[2]+1], nk1, nk2))
    chern_number_int.append(ham.cal_chern_latt_method(np.arange(ham.nb)[:fb_index[2]+2], nk1, nk2))

    with np.printoptions(precision=5, suppress=True):
        print(np.array(chern_number_sep))
        print(np.array(chern_number_int))

if __name__ == '__main__' and Job == 'wanTB':
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
    # # for others
    U_BN_u = None
    U_BN_d = None
    # # for mTBG
    # U_BN_u = np.array([0])
    # U_BN_d = np.array([0.03])
    # fot mTTG
    # U_BN_u = np.array([-0.0, -0.0])
    # U_BN_d = np.array([0.05])
    # U_BN_u = np.array([-0.02, -0.02])
    # U_BN_d = np.array([0.08])
    # U_BN_u = np.array([-0.1, -0.0])
    # U_BN_d = np.array([0.1])
    # U_BN_u = np.array([0.0, -0.])
    # U_BN_d = np.array([0.02])

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

    plot_wfs = True
    plot_band = False
    plot_valley_eign = False
    plot_band_valley_expectation_vaule = False
    show_ham = False
    load_savepoints = False
    save_htb = False

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
    htb, w90ip =            wanTB.wanTB_build_tmg_wanTB(ham, wcc, AMN, EIG, EIG_D, froz_i1, froz_i2, applysymmetry=True, set_zero_onsite=True)
    WFs =                   wanTB.wanTB_plot_wfs(ham, wcc, w90ip.vmat, nnsuper=3, plot_rr_dense=2, cal_wfs=if_cal_wfs, applysymmetry=True)

    if save_htb:
        htb.save_h5(r'htb.mttg.m26n27.k12.debug.h5')

        # htb.save_h5(r'htb.mtbg.m30n31.hBN.k18.h5')
        # htb.save_h5(r'htb.tdbg.m16n17.k18.h5')
        # htb.save_h5(r'htb.mtdbg.m26n27.UD.k18.h5')
        # htb.save_h5(r'htb.mttg.m26n27.UD.hBN.k18.h5')
        # htb.save_h5(r'htb.tmg4+4.m12n13.k18.h5')
        # htb.save_wannier90_hr_dat()
        # htb.save_D_dat()

    if PYGUI:
        np.savez_compressed('savepoints_wanTB.npz',
                            wanTB=wanTB,
                            ham=ham,
                            wcc=wcc,
                            AMN=AMN,
                            EIG=EIG,
                            EIG_D=EIG_D,
                            w90ip=w90ip,
                            )
        WFs.save_npz('wfs.npz')

    if PYGUI and load_savepoints:
        savepoints_wanTB = r'savepoints_wanTB.npz'
        npdata = np.load(savepoints_wanTB, allow_pickle=True)
        wanTB_init = npdata['wanTB_init'].item()
        ham = npdata['ham'].item()
        wcc = npdata['wcc']
        AMN = npdata['AMN'].item()
        EIG = npdata['EIG'].item()
        EIG_D = npdata['EIG_D']
        w90ip = npdata['w90ip'].item()
        rr = npdata['rr']
        wfs = npdata['wfs']
        wfs_symm = npdata['wfs_symm']
        npdata.close()

    if PYGUI and Job == 'debug_symm':
        self = wanTB
        latt = ham.latt
        lattG = ham.lattG
        NNKP = self.NNKP
        nk = NNKP.nk
        kk = wanTB.NNKP.kk
        kkc = LA.multi_dot([lattG, kk.T]).T
        eig = EIG.eig
        fb_index = self.fb_index
        vmat_symm_index = self.vmat_symm_index
        vmat = w90ip.vmat[:, fb_index, :]


        # vmat_symm = np.einsum('kmn,kn->kmn', vmat[vmat_symm_index], self.vmat_symm_Dc3k.conj())
        #
        # plot_grid2_complex(kkc, vmat[:, 0, 0], s=200, vmax=None)
        # plot_grid2_complex(kkc, vmat_symm[:, 0, 0], s=200)
        # plot_grid2_complex(kkc, vmat[vmat_symm_index, 0, 0], s=200)
        #
        # amn = AMN.amn[:, fb_index, :]
        # amn = (amn + amn[c3_index] + amn[c3c3_index]) / 3
        # amn = w90ip.svd_init_guess(amn)
        # plot_grid2_complex(kkc, amn[:, 3, 0], s=200, vmax=0.4)
        #
        # vmatvmat = np.einsum('kmi,kmj->kij', vmat.conj(), vmat).real

        # plot_grid2(kkc, np.abs(vmat[:, 0, 0]), s=200)
        # plot_grid(np.vstack([kkc, c3_kkc, c3c3_kkc]))
        # plot_grid(kkc)


        self = w90ip
        ek = np.array([
            np.diag(eig[ik, fb_index])
            for ik in range(nk)
        ], dtype='float64')
        hwk = np.einsum('kim,kij,kjn->kmn', vmat.conj(), ek, vmat, optimize=True)
        hwk = 0.5 * (hwk + np.einsum('kmn->knm', hwk).conj())

        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', self.meshk, self.gridR))

        hr_Rmn = np.einsum('kR,kmn->Rmn', eikR.conj(), hwk) / self.nk
        hr_Rmn_symm = np.zeros_like(hr_Rmn, dtype='complex128')

        for i in range(4):
            for j in range(4):
                c3_index, c3c3_index = find_Rspace_c3_symm_index(htb.Rc, tau_ij=wcc[i]-wcc[j])
                hr_Rmn_symm[:, i, j] = np.real((hr_Rmn[:, i, j] + hr_Rmn[c3_index, i, j] + hr_Rmn[c3c3_index, i, j]) / 3)

        mask_minitb = np.real(hr_Rmn_symm)
        mask_minitb = np.abs(mask_minitb / np.max(np.abs(mask_minitb)))
        mask_minitb = np.array(mask_minitb>0.1, dtype='float')
        plot_ham(htb.Rc, mask_minitb, s=50)


        tt = hr_Rmn[c3c3_index, 1, 2].real
        tt = tt / np.max(np.abs(tt))
        # plot_grid2(htb.Rc, tt, s=200, vmin=-1.0, vmax=1.0)

        plot_ham(htb.Rc, htb.hr_Rmn, s=50)
        plot_ham(htb.Rc, hr_Rmn_symm, s=50)


        # token_htb = Token_htb(htb.latt, htb.lattG, htb.R, htb.ndegen, htb.hr_Rmn, htb.wcc, htb.wccf)
        # token_htb = Token_htb(htb.latt, htb.lattG, htb.R, htb.ndegen, mask_minitb*hr_Rmn_symm, htb.wcc, htb.wccf)

    '''
      Plot M.B. band vs Wannier band
    '''
    if PYGUI and plot_band:
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

        # bandE = cal_band(token_htb, kpath, nw)
        # bandE = cal_band(w90ip, kpath, nw)
        # bandE2 = cal_band(ham, kpath_car, ham.nb)

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
            # bandE[ik] = E_symm - w90ip.fermi
            bandE_ref[ik], U = ham.get_eigh(ham.get_hk(kc))

            valley_eign_symm[ik] = np.real(np.diag(LA.multi_dot([U_symm.T.conj(), Dk, U_symm])))

        bandE *= 1000
        bandE_ref *= 1000

        # plot_2band_compare(bandE, bandE_ref, kpath_HSP, xlabel, kpath_car, eemin=-0.05*1000, eemax=0.05*1000, save=False, savefname='compareband.pdf')
        plot_2band_compare(bandE, bandE_ref, kpath_HSP, xlabel, kpath_car, eemin=-0.080*1000, eemax=0.080*1000, save=False, savefname='compareband.pdf')

    '''
      Plot wfs
    '''
    if PYGUI and plot_wfs:
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
        # WFs.plot_wan_WFs_nonzero()
        # plot_wan_WFs(WFs.rr, WFs.wfs, valley=0, s=10, vmin=-1, vmax=1, latt=WFs.latt)
        # plot_wan_WFs(rr, wfs, valley=0, s=10, vmin=-1, vmax=1, latt=ham.latt)
        plot_wan_WFs(WFs.rr, WFs.wfs_symm, valley=0, s=10, vmin=-1, vmax=1, latt=WFs.latt)
        # plot_wan_WFs(WFs.rr, WFs.wfs_symm.imag, valley=0, s=10, vmin=-1, vmax=1, latt=WFs.latt)
        # plot_wan_WFs(WFs.rr, WFs.wfs.imag, valley=1, s=10, vmin=-1, vmax=1, latt=WFs.latt)
        # plot_wan_WFs(WFs.rr, WFs.wfs.real, valley=0, s=10, vmin=-1, vmax=1, latt=WFs.latt)

    '''
      Plot valley eignvalues
    '''
    if os.environ.get('PYGUI') == 'True' and plot_valley_eign:
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

    if PYGUI and plot_band_valley_expectation_vaule:
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

    '''
      Plot Ham
    '''
    if PYGUI and show_ham:
        os.chdir(os.path.join(ROOT_WDIR, r'TMG_w90/tmg'))
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

if __name__ == '__main__' and Job == 'bandprojection':
    os.chdir(wdir)

    nk1 = 31
    kpath_HSP = np.array([
        [-1/3, 1/3, 0.0], #  K
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 1/2, 0.0], #  M
        [ 1/3, 2/3, 0.0], # -K
    ])
    xlabel = ['K', 'G', 'M', '-K']

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = LA.multi_dot([ham.lattG, kpath.T]).T

    bandE, bandprojection = cal_bandprojection_morie(wanTB.w90_setup, ham, kpath_car)

    # eig_froz = np.copy(bandE)
    # eig_froz[:, nb // 2 + 2:] += 20
    # eig_froz[:, :nb // 2 - 2] -= 20
    # vmat = np.einsum('kb,wkb->kbw', Rect(eig_froz, -0.4, 0.4, T=50), bandprojection)
    # for i in range(nk):
    #     u, s, vh = LA.svd(vmat[i])
    #     vmat[i] = LA.multi_dot([u, np.matlib.eye(nb, nw), vh])
    # vmat = np.einsum('kbw->wkb', vmat)

    # plot cal_bandprojection_morie
    if PYGUI:
        S = np.sum(np.abs(bandprojection), axis=0)

        plot_bandprojection(bandE, S, kpath_car, xlabel, eemin=-0.1, eemax=0.1, s=3, plotaxis=True)

        # bandstructure_hsp.plot_bandprojection(S, eemin=-0.7, eemax=0.7, unit='C', savefname='proj_TBG.png') # for TBG
        # bandstructure_hsp.plot_bandprojection(S, eemin=-0.5, eemax=0.5, unit='C', savefname='proj_TDBG.pdf') # for TDBG
        # bandstructure_hsp.plot_bandprojection(S, eemin=-0.4, eemax=0.4, unit='C', savefname='proj_TMG.pdf') # for TMG

        # bandstructure_hsp.plot_bandprojection(S, eemin=-0.4, eemax=0.4, unit='C')

if __name__ == '__main__' and Job == 'bandprojection_pz':
    os.chdir(wdir)

    '''
      * Input
    '''
    m = 12
    n = m + 1

    NG = 3
    w1 = 1*0.0797
    w2 = 1*0.0975

    htbfname_u = r'htb_ABCAC_SCAN.h5'
    htbfname_d = r'htb_ABCA_SCAN.h5'

    # htbfname_u = r'htb_ABA_SCAN.h5'
    # htbfname_d = r'htb_AB_SCAN.h5'

    mp_grid = np.array([18, 18, 1])

    nk1 = 101
    kpath_HSP = np.array([
        [-1/3, 1/3, 0.0], #  K
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 1/2, 0.0], #  M
        [ 1/3, 2/3, 0.0], # -K
    ])
    xlabel = ['K', 'G', 'M', '-K']

    '''
      * Main
    '''
    wanTB =                 WanTB_TMG(m, n, NG, w1, w2, mp_grid, Umax=0, U_BN_u=None, U_BN_d=None, enable_hBN=False, htbfname_u=htbfname_u, htbfname_d=htbfname_d)
    ham =                   wanTB.ham_BM
    wcc, AMN, EIG, EIG_D =  wanTB.wanTB_tmg_w90_setup(ham, wanTB.NNKP)

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = LA.multi_dot([ham.lattG, kpath.T]).T
    bandE, bandprojection = cal_bandprojection_pz(wanTB.w90_setup, ham, kpath_car)

    # plot cal_bandprojection_pz
    if PYGUI:
        # S = np.sum(np.abs(bandprojection[np.array([0,1,2,3])]), axis=0)
        # S = np.sum(np.abs(bandprojection[np.array([2,3,4,5])]), axis=0)
        # S = np.sum(np.abs(bandprojection[np.array([6,7,8,9])]), axis=0)
        # S = np.sum(np.abs(bandprojection[np.array([0,6,7,8,9,11])]), axis=0)
        # S = np.sum(np.abs(bandprojection[np.array([6,7,8,9])]), axis=0)
        #
        # # S = np.sum(np.abs(bandprojection[np.array([2,3,4,5])]), axis=0)
        # # S = np.sum(np.abs(bandprojection[np.array([0,3,4,7])]), axis=0)
        # S = np.sum(np.abs(bandprojection[np.array([0,1,2,3])]), axis=0)
        # # bandstructure_hsp.plot_band(eemin=-0.03, eemax=0.03, unit='C')
        # bandstructure_hsp.plot_bandprojection(S, eemin=-0.7, eemax=0.7, unit='C', savefname='proj_TBG.png') # for TBG
        # bandstructure_hsp.plot_bandprojection(S, eemin=-0.5, eemax=0.5, unit='C', savefname='proj_TDBG.pdf') # for TDBG
        # bandstructure_hsp.plot_bandprojection(S, eemin=-0.4, eemax=0.4, unit='C', savefname='proj_TMG.pdf') # for TMG

        # S = np.sum(np.abs(bandprojection[np.arange(14)[:-2]]), axis=0)
        # bandstructure_hsp.plot_bandprojection(S, eemin=-0.3, eemax=0.3, unit='C')

        pz_index = np.arange(wanTB.w90_setup.nw)[:-2]
        # pz_index = np.arange(wanTB.w90_setup.nw)[-2:]
        S = np.sum(np.abs(bandprojection[pz_index]), axis=0)

        plot_bandprojection(bandE, S, kpath_car, xlabel, eemin=-0.3, eemax=0.3, s=1, plotaxis=True)
