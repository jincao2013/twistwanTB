# Copyright (C) 2020 Jin Cao
#
# This file is distributed as part of the wanpy code and
# under the terms of the GNU General Public License. See the
# file LICENSE in the root directory of the wanpy
# distribution, or http://www.gnu.org/licenses/gpl-3.0.txt
#
# The wanpy code is hosted on GitHub:
#
# https://github.com/jincao2013/wanpy

__date__ = "Aug. 23, 2019"

__all__ = [
    'W90',
    'Wannier_setup',
    'W90_interpolation',
]


import os
import sys
import re
import numpy as np
import numpy.linalg as LA

from twistwanTB.wanpy.structure import Cell, Htb
from twistwanTB.wanpy.mesh import make_mesh
from twistwanTB.wanpy.toolkits import kmold


class W90(object):
    '''
    Wannier90 object
    '''
    def __init__(self):

        self.nb = None
        self.nw = None


class W90_win(W90):

    def __init__(self):
        W90.__init__(self)


class W90_wout(W90):

    def __init__(self):
        W90.__init__(self)
        self._container = ['lattice', 'wcc', 'wccf', 'wborden']
        self.lattice = None
        self.wcc = None
        self.wccf = None
        self.wborden = None

    def load_wc(self, fname, shiftincell=True):

        lattice = np.zeros((3, 3), dtype='float64')
        with open(fname, 'r') as f:
            inline = f.readline()
            while 'Lattice Vectors' not in inline:
                inline = f.readline()
            lattice[:, 0] = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
            lattice[:, 1] = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
            lattice[:, 2] = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
            while 'Number of Wannier Functions' not in inline:
                inline = f.readline()
            nw = int(re.findall(r'\d+', inline)[0])
            wcc = np.zeros((nw, 3), dtype='float64')
            wborden = np.zeros(nw, dtype='float64')
            while inline != ' Final State\n':
                inline = f.readline()
            for i in range(nw):
                inline = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
                wcc[i] = inline[:3]
                wborden[i] = inline[-1]
        f.close()

        wccf = LA.multi_dot([LA.inv(lattice), wcc.T]).T
        if shiftincell:
            wccf = np.remainder(wccf, np.array([1, 1, 1]))
            wcc = LA.multi_dot([lattice, wccf.T]).T

        self.lattice = lattice
        self.wcc = wcc
        self.wccf = wccf
        self.wborden = wborden
        self.nw = wcc.shape[0]

    def save_wcc(self, fname=r'wcc.vasp', unit='C'):
        if os.path.exists(fname):
            os.remove(fname)

        if unit.upper() == 'C':
            wcc_unit = 'C'
            wcc = self.wcc
        elif unit.upper() == 'D':
            wcc_unit = 'C'
            wcc = self.wccf
        else:
            wcc_unit = 'C'
            wcc = self.wcc

        with open(fname, 'a') as poscar:
            poscar.write('writen by wanpy\n')
            poscar.write('   1.0\n')
            for i in self.lattice.T:
                poscar.write('   {: 2.16f}    {: 2.16f}    {: 2.16f}\n'.format(i[0], i[1], i[2]))

            poscar.write('H   \n')
            poscar.write('{}    \n'.format(self.nw))

            poscar.write('{}\n'.format(wcc_unit))

            for i in wcc:
                poscar.write('  {: 2.16f}  {: 2.16f}  {: 2.16f}\n'.format(i[0], i[1], i[2]))
            poscar.write('\n')
        poscar.close()


# class W90_wout(W90):
#
#     def __init__(self):
#         W90.__init__(self)
#         self._container = ['lattice', 'wcc', 'wccf', 'wborden']
#         self.lattice = None
#         self.wcc = None
#         self.wccf = None
#         self.wborden = None
#
#     def load_from_w90(self, fname=r'wannier90.wout'):
#
#         lattice = np.zeros((3, 3), dtype='float64')
#         with open(fname, 'r') as f:
#             inline = f.readline()
#             while 'Lattice Vectors' not in inline:
#                 inline = f.readline()
#             lattice[:, 0] = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
#             lattice[:, 1] = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
#             lattice[:, 2] = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
#             while 'Number of Wannier Functions' not in inline:
#                 inline = f.readline()
#             nw = int(re.findall(r'\d+', inline)[0])
#             wcc = np.zeros((nw, 3), dtype='float64')
#             wborden = np.zeros(nw, dtype='float64')
#             while inline != ' Final State\n':
#                 inline = f.readline()
#             for i in range(nw):
#                 inline = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
#                 wcc[i] = inline[:3]
#                 wborden[i] = inline[-1]
#         f.close()
#
#         wccf = LA.multi_dot([LA.inv(lattice), wcc.T]).T
#         if shiftincell:
#             wccf = np.remainder(wccf, np.array([1, 1, 1]))
#             wcc = LA.multi_dot([lattice, wccf.T]).T
#
#         self.lattice = lattice
#         self.wcc = wcc
#         self.wccf = wccf
#         self.wborden = wborden


class W90_nnkp(W90):

    def __init__(self):
        W90.__init__(self)

        self.title = None
        self.nk = None
        self.nnk = None

        self.latt = np.zeros([3, 3], dtype='float64')
        self.lattG = np.zeros([3, 3], dtype='float64')

        self.kk =None
        self.nnkpts = None
        self.nnkptsG = None

        self.nnbpts = None
        self.nnbpts_car = None
        self.bvecs = None
        self.bvecs_index = None


    def load_from_w90(self, fname=r'wannier90.nnkp'):
        with open(fname, 'r') as f:

            self.title = f.readline()

            inline = f.readline()
            while 'begin real_lattice' not in inline:
                inline = f.readline()

            self.latt.T[0] = np.array(f.readline().split(), dtype='float64')
            self.latt.T[1] = np.array(f.readline().split(), dtype='float64')
            self.latt.T[2] = np.array(f.readline().split(), dtype='float64')

            inline = f.readline()
            while 'begin recip_lattice' not in inline:
                inline = f.readline()

            self.lattG.T[0] = np.array(f.readline().split(), dtype='float64')
            self.lattG.T[1] = np.array(f.readline().split(), dtype='float64')
            self.lattG.T[2] = np.array(f.readline().split(), dtype='float64')

            inline = f.readline()
            while 'begin kpoints' not in inline:
                inline = f.readline()

            nk = int(f.readline())
            kk = np.zeros([nk, 3], dtype='float64')
            for i in range(nk):
                kk[i] = np.array(f.readline().split(), dtype='float64')

            inline = f.readline()
            while 'begin nnkpts' not in inline:
                inline = f.readline()

            nnk = int(f.readline())
            nnkpts = np.zeros([nk, nnk], dtype='int64')
            nnkptsG = np.zeros([nk, nnk, 3], dtype='int64')
            for i in range(nk):
                for j in range(nnk):
                    inline = np.array(f.readline().split()[1:], dtype='int64')
                    nnkpts[i, j] = inline[0]
                    nnkptsG[i, j] = inline[1:]
        f.close()

        self.nk = nk
        self.nnk = nnk
        self.kk = kk
        self.nnkpts = nnkpts
        self.nnkptsG = nnkptsG

        self.nnbpts, self.nnbpts_car, self.bvecs, self.bvecs_index = self.get_nnbpts()

        return kk, nnkpts, nnkptsG

    def get_nnbpts(self):
        nnbpts = np.zeros([self.nk, self.nnk, 3], dtype='float64')
        for ik in range(self.nk):
            for ib in range(self.nnk):
                jk = self.nnkpts[ik, ib] - 1
                nnbpts[ik, ib] = self.kk[jk] + self.nnkptsG[ik, ib] - self.kk[ik]
        nnbpts_car = LA.multi_dot([self.lattG, nnbpts.reshape(self.nk * self.nnk, 3).T]).T.reshape(self.nk, self.nnk, 3)

        bvecs = nnbpts_car[0]
        bvecs = bvecs[np.lexsort(bvecs.T)]
        bvecs_index = np.zeros([self.nk, self.nnk], dtype='int64')
        for i in range(self.nk):
            index = np.lexsort(nnbpts_car[i].T)
            for j in range(self.nnk):
                bvecs_index[i, index[j]] = j

        # for i in range(self.nk):
        #     for j in range(self.nnk):
        #         y = nnbpts_car[i, j] - bvecs[bvecs_index[i, j]]
        #         print(y)

        return nnbpts, nnbpts_car, bvecs, bvecs_index

    def init_NNKP(self, latt, mp_grid):
        self.title = 'NNKP generated by wanpy'
        self.latt = latt
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

        self.kk = make_mesh(mp_grid, type='continuous', centersym=False, info=True)
        self.nk = self.kk.shape[0]


class W90_amn(W90):

    def __init__(self):
        W90.__init__(self)
        self.nw = None
        self.nb = None
        self.nk = None
        self.amn = None

    def reduce_into_outer_window(self, bloch_outer_win=None):
        if bloch_outer_win != None:
            self.nb = bloch_outer_win[1] - bloch_outer_win[0]
            self.amn = self.amn[:, bloch_outer_win[0]:bloch_outer_win[1], :]

    def load_from_w90(self, fname=r'wannier90.amn'):
        with open(fname, 'r') as f:
            f.readline()
            self.nb, self.nk, self.nw = np.array(f.readline().split(), dtype='int64')
            self.amn = np.zeros([self.nk, self.nb, self.nw], dtype='complex128')
            for i in range(self.nk):
                for n in range(self.nw):
                    for m in range(self.nb):
                        inline = f.readline().split()[-2:]
                        self.amn[i, m, n] = complex(float(inline[0]), float(inline[1]))
        f.close()

    def save_w90(self, fname=r'wannier90.amn'):
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'a') as f:
            f.write('AMN Created by Wanpy\n')
            f.write('{:>6}{:>6}{:>6}\n'.format(self.nb, self.nk, self.nw))
            for i in range(self.nk):
                for n in range(self.nw):
                    for m in range(self.nb):
                        inline = self.amn[i, m, n]
                        f.write('{:>6}{:>6}{:>6}{:>22.12f}{:>22.12f}\n'.format(m+1, n+1, i+1, np.real(inline), np.imag(inline)))
        f.close()


class W90_mmn(W90):

    def __init__(self):
        W90.__init__(self)

        self.title = None
        self.nk = None
        self.nnk = None  # num of nearist k
        self.nb = None

        self.mmn = None
        self.nnkpts = None
        self.nnkptsG = None

    def reduce_into_outer_window(self, bloch_outer_win=None):
        if bloch_outer_win != None:
            self.nb = bloch_outer_win[1] - bloch_outer_win[0]
            self.mmn = self.mmn[:, :, bloch_outer_win[0]:bloch_outer_win[1], bloch_outer_win[0]:bloch_outer_win[1]]

    def load_from_w90(self, fname=r'wannier90.mmn'):
        with open(fname, 'r') as f:

            self.title = f.readline()
            nb, nk, nnk = np.array(f.readline().split(), dtype='int64')
            self.nb, self.nk, self.nnk = nb, nk, nnk
            mmn = np.zeros([nk, nnk, nb, nb], dtype='complex128')
            nnkpts = np.zeros([nk, nnk], dtype='int64')
            nnkptsG = np.zeros([nk, nnk, 3], dtype='int64')

            for i in range(nk):
                for j in range(nnk):
                    inline = np.array(f.readline().split(), dtype='int64')
                    nnkpts[i, j] = inline[1]
                    nnkptsG[i, j] = inline[2:]
                    for n in range(nb):
                        for m in range(nb): # the first index m is fastest
                            inline = f.readline().split()
                            mmn[i, j, m, n] = complex(float(inline[0]), float(inline[1]))
        f.close()
        self.mmn = mmn
        self.nnkpts = nnkpts
        self.nnkptsG = nnkptsG
        return mmn

    def save_w90(self, fname=r'wannier90.mmn'):
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'a') as f:
            f.write('MMN Created by Wanpy\n')
            f.write('{:>6}{:>6}{:>6}\n'.format(self.nb, self.nk, self.nnk))

            for i in range(self.nk):
                for j in range(self.nnk):
                    nnki = self.nnkpts[i, j]
                    nnkiG = self.nnkptsG[i, j]
                    f.write('{:>6}{:>6}{:>6}{:>6}{:>6}\n'.format(i+1, nnki, nnkiG[0], nnkiG[1], nnkiG[2]))
                    for n in range(self.nb):
                        for m in range(self.nb): #  the first index m is fastest
                            inline = self.mmn[i, j, m, n]
                            f.write('{:>18.12f}{:>18.12f}\n'.format(np.real(inline), np.imag(inline)))
            f.write(r'')
        f.close()


class W90_eig(W90):

    def __init__(self, nb, nk):
        W90.__init__(self)

        self.nb = nb
        self.nk = nk
        self.eig = np.zeros([nk, nb], dtype='float64')

    def reduce_into_outer_window(self, bloch_outer_win=None):
        if bloch_outer_win != None:
            self.nb = bloch_outer_win[1] - bloch_outer_win[0]
            self.eig = self.eig[:, bloch_outer_win[0]:bloch_outer_win[1]]

    def load_from_w90(self, fname=r'wannier90.eig'):
        with open(fname, 'r') as f:
            for i in range(self.nk):
                for j in range(self.nb):
                    # print(i, j)
                    self.eig[i, j] = float(f.readline().split()[-1])
        f.close()

    def save_w90(self, fname=r'wannier90.eig'):
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'a') as f:
            for i in range(self.nk):
                for j in range(self.nb):
                    f.write('{:>12}{:>12}{:>22.14f}\n'.format(j+1, i+1, self.eig[i, j]))
            f.write(r'')
        f.close()


class W90_umat(W90):

    def __init__(self):
        W90.__init__(self)

        self.nk = None
        self.nw = None
        self.umat = None
        self.meshkf = None

    def load_from_w90(self, fname=r'wannier90_u.mat'):
        with open(fname, 'r') as f:
            f.readline()
            nk, nw = np.array(f.readline().split(), dtype='int64')[:2]
            self.nk = nk
            self.nw = nw
            self.umat = np.zeros([nk, nw, nw], dtype='complex128')
            self.meshkf = np.zeros([nk, 3], dtype='float64')

            for ik in range(nk):
                f.readline()
                self.meshkf[ik] = np.array(f.readline().split(), dtype='float64')
                for nwj in range(nw): # col
                    for nwi in range(nw): # row
                        inline = np.array(f.readline().split(), dtype='float64')
                        self.umat[ik, nwi, nwj] = complex(inline[0], inline[1])
        f.close()

    def save_w90(self, fname=r'wannier90_u.mat'):
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'a') as f:
            f.write(' written by wanpy \n')
            f.write('{:>12}{:>12}{:>12}\n'.format(self.nk, self.nw, self.nw))
            for ik in range(self.nk):
                f.write('\n')
                kf = self.meshkf[ik]
                f.write('{:>15.10f}{:>15.10f}{:>15.10f}\n'.format(kf[0], kf[1], kf[2]))
                for nwj in range(self.nw):
                    for nwi in range(self.nw):
                        inline = self.umat[ik, nwi, nwj]
                        f.write('{:>+15.10f}{:>+15.10f}\n'.format(np.real(inline), np.imag(inline)))
        f.close()


class W90_umat_dis(W90):

    def __init__(self):
        W90.__init__(self)

        self.nk = None
        self.nw = None
        self.nb = None
        self.umat_dis = None
        self.meshkf = None

    def load_from_w90(self, fname=r'wannier90_u_dis.mat'):
        with open(fname, 'r') as f:
            f.readline()
            nk, nw, nb = np.array(f.readline().split(), dtype='int64')
            self.nk = nk
            self.nw = nw
            self.nb = nb
            self.umat_dis = np.zeros([nk, nb, nw], dtype='complex128')
            self.meshkf = np.zeros([nk, 3], dtype='float64')

            for ik in range(nk):
                f.readline()
                self.meshkf[ik] = np.array(f.readline().split(), dtype='float64')
                for nwi in range(nw):
                    for nbi in range(nb):
                        inline = np.array(f.readline().split(), dtype='float64')
                        self.umat_dis[ik, nbi, nwi] = complex(inline[0], inline[1])
        f.close()

    def save_w90(self, fname=r'wannier90_u_dis.mat'):
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'a') as f:
            f.write(' written by wanpy \n')
            f.write('{:>12}{:>12}{:>12}\n'.format(self.nk, self.nw, self.nb))
            for ik in range(self.nk):
                f.write('\n')
                kf = self.meshkf[ik]
                f.write('{:>15.10f}{:>15.10f}{:>15.10f}\n'.format(kf[0], kf[1], kf[2]))
                for nwi in range(self.nw):
                    for nbi in range(self.nb):
                        inline = self.umat_dis[ik, nbi, nwi]
                        f.write('{:>+15.10f}{:>+15.10f}\n'.format(inline.real, inline.imag))
        f.close()


class W90_band(W90):

    def __init__(self, latt=np.identity(3)):
        W90.__init__(self)

        self.nw = None
        self.nk = None
        self.latt = latt
        self.lattG = 2 * np.pi * LA.inv(latt.T)
        self.kpt = None
        self.kpath = None
        # self.kpaths = None
        self.eig = None

    def load_from_w90(self, seedname='wannier90'):
        self.load_kpt(seedname+'_band.kpt')
        self.load_band_dat(seedname+'_band.dat')

    def load_kpt(self, fname):
        with open(fname, 'r') as f:
            self.nk = int(f.readline())
            self.kpt = np.zeros([self.nk, 3], dtype='float64')
            for i in range(self.nk):
                self.kpt[i] = np.array(f.readline().split(), dtype='float64')[:-1]
        f.close()
        kkc = LA.multi_dot([self.lattG, self.kpt.T]).T
        self.kpath = kmold(kkc)

    def load_band_dat(self, fname):
        f = open(fname, 'r')
        self.nw = len(f.readlines()) // (self.nk + 1)
        f.close()

        self.eig = np.zeros([self.nk, self.nw], dtype='float64')
        # self.kpaths = np.zeros([self.nk, self.nw], dtype='float64')
        with open(fname, 'r') as f:
            for iw in range(self.nw):
                for ik in range(self.nk):
                    inline = np.array(f.readline().split(), dtype='float64')
                    # self.kpaths[ik, iw] = inline[0]
                    self.eig[ik, iw] = inline[1]
                f.readline()
        f.close()


class Wannier_setup(object):

    def __init__(self, Ham, NNKP, latt,
                 nb, nw, wcc,
                 nmeshR, ngridG, nnWSR,
                 bloch_outer_win=None,
                 seedname=r'wannier90',
                 ):
        '''
          * Wannier_setup
            Input: H(k)
            Output: AMN, MMN, EIG used to build Wannier based Hamiltonian
          ** Ham(object): should consist of w90gethk(k) method to get hk matrix
          ** NNKP(object): NNKP object read from wannier90.nnkpt
          ** latt(np.ndarray dim(3,3)):  crystal parameters
          ** nb(int): num of band
          ** nw(int): num of wannier orbitals
          ** grid(np.ndarray int dim(3)): Used to sample real space 
          ** gridG(np.ndarray float dim(nG,3)): Plane wave basis
          ** projections: np.array[
                [n, l, m]
                ...
             dtype='int64']
          ** wcc: np.array[
                [x, y, z]
                ...
             dtype='float64'] in axis of car.
        '''  #
        # self.Ham = Tbg_MacDonald()
        # self.NNKP = W90_nnkp()
        self.Ham = Ham
        self.NNKP = NNKP
        self.w90gethk = Ham.w90gethk
        self.fermi = Ham.fermi

        self.latt = latt
        self.lattG = 2 * np.pi * LA.inv(latt.T)
        self.latt_wan = None
        self.lattG_wan = None
        self.meshk = self.NNKP.kk
        self.meshkc = LA.multi_dot([self.lattG, self.NNKP.kk.T]).T

        self.nw = nw
        self.nb = nb
        self.nb_outwin = nb
        self.nmeshR = nmeshR
        # self.ngridG = ngridG
        self.nnWSR = nnWSR
        self.wcc = wcc
        # self.projections = projections
        self.bloch_outer_win = bloch_outer_win
        self.seedname = seedname

        if bloch_outer_win != None:
            self.nb_outwin = bloch_outer_win[1] - bloch_outer_win[0]

        # self.meshR, self.gridG, self.WSR, self.meshRws = self.init_grid(nmeshR, ngridG, nnWSR)
        self.meshR, self.WSR = self.init_grid(nx=nmeshR[0], ny=nmeshR[1], nws=nnWSR)
        self.meshRc = LA.multi_dot([self.latt, self.meshR.T]).T
        self.gridG = None

        self.bandE, self.bandU = None, None

        # self.gxx = np.identity(self.nb)

    def printer(self):
        print('=====================================================')
        print('Wannier setup procedrue')
        print('=====================================================')
        print('nmeshR = {:>4}{:>4}{:>4}'.format(self.nmeshR[0], self.nmeshR[1], self.nmeshR[2]))
        # print('Number of gridG: {:>4}     Number of meshR: {:>4}'.format(self.gridG.shape[0], self.meshR.shape[0]))
        print('Number of gridk: {:>4}'.format(self.NNKP.nk))
        # print('Number of nnk:   {:>4}'.format(self.NNKP.nnk))
        # print('Number of nnWSR:  {:>4}'.format(self.meshRws.shape[0]))
        print('Number of projector: {:>4} Number of Bloch bands: {:>4}'.format(self.nw, self.nb))
        if self.bloch_outer_win != None:
            print('Outer window was used.')
            print('Number of Bloch bands in outer window: {:>4}({}:{})'.format(
                self.nb_outwin, self.bloch_outer_win[0], self.bloch_outer_win[1])
            )
        print('=====================================================')

    def write_W90win(self):
        print('num_bands         =  {:>5}       # Number of input Bloch states'.format(self.nb_outwin))
        print('num_wann          =  {:>5}       # Number of Wannier Functions'.format(self.nw))
        print('')
        print('begin unit_cell_cart')
        print('{:>20.12f}{:>20.12f}{:>20.12f}'.format(self.latt.T[0, 0], self.latt.T[0, 1], self.latt.T[0, 2]))
        print('{:>20.12f}{:>20.12f}{:>20.12f}'.format(self.latt.T[1, 0], self.latt.T[1, 1], self.latt.T[1, 2]))
        print('{:>20.12f}{:>20.12f}{:>20.12f}'.format(self.latt.T[2, 0], self.latt.T[2, 1], self.latt.T[2, 2]))
        print('end unit_cell_cart')
        print('')
        print('begin atoms_cart')
        for _wcc in self.wcc:
            print('C {:>20.12f}{:>20.12f}{:>20.12f}'.format(_wcc[0], _wcc[1], _wcc[2]))
        print('end atoms_cart')

    def cal_bandstructure(self):
        kkc = self.meshkc

        self.bandE = np.zeros([self.NNKP.nk, self.nb], dtype='float64')
        self.bandU = np.zeros([self.NNKP.nk, self.nb, self.nb], dtype='complex128')

        for i in range(self.NNKP.nk):
            print('>>> cal band u(k) at {}/{}'.format(i + 1, self.NNKP.nk))
            self.bandE[i], self.bandU[i] = LA.eigh(self.w90gethk(kkc[i]))

        self.bandE -= self.fermi

    def main(self, calAMN=True, calMMN=False, saveW90=False):
        NNKP = self.NNKP
        # Ham = self.Ham
        # nb = self.nb
        # nw = self.nw
        # kkc = self.meshkc

        # self.bandE = np.zeros([NNKP.nk, nb], dtype='float64')
        # self.bandU = np.zeros([NNKP.nk, nb, nb], dtype='complex128')
        #
        # # for i in range(NNKP.nk):
        # #     print('>>> cal band u(k) at {}/{}'.format(i + 1, NNKP.nk))
        # #     self.bandE[i], self.bandU[i] = LA.eigh(self.w90gethk(kkc[i]))
        # #     self.bandE[i] -= self.fermi
        # #
        # # self.bandE[:, self.nb // 2 + 2:] += 0.05
        # # self.bandE[:, :self.nb // 2 - 2] -= 0.05

        print('[INFO] calculating EIG')
        EIG = self.cal_EIG(self.nb)
        EIG.reduce_into_outer_window(self.bloch_outer_win)

        if calAMN:
            print('[INFO] calculating AMN')
            AMN = self.cal_AMN(self.nb, self.nw)
            AMN.reduce_into_outer_window(self.bloch_outer_win)
        else:
            AMN = None

        if calMMN:
            print('[INFO] calculating MMN')
            MMN = self.cal_MMN(self.nb, self.nw)
            MMN.reduce_into_outer_window(self.bloch_outer_win)
        else:
            MMN = None

        if saveW90:
            print('save eig ...')
            EIG.save_w90(self.seedname + '.eig')

            if calAMN:
                print('save amn ...')
                AMN.save_w90(self.seedname + '.amn')
            if calMMN:
                print('save mmn ...')
                MMN.save_w90(self.seedname + '.mmn')

        return AMN, MMN, EIG

    def cal_EIG(self, nb):
        EIG = W90_eig(nb=nb, nk=self.NNKP.nk)

        for i in range(self.NNKP.nk):
            EIG.eig[i] = self.bandE[i]
        return EIG

    def cal_MMN(self, nb, nw):
        cross_bz = True
        print('[INFO] gxx is taken into account')
        if cross_bz:
            print('[INFO] crossing BZ is taken into account')

        bandU = self.bandU
        NNKP = self.NNKP
        kkc = self.meshkc
        MMN = W90_mmn()
        MMN.nk = NNKP.nk
        MMN.nw = nw
        MMN.nb = nb
        MMN.nnk = NNKP.nnk
        MMN.nnkpts = NNKP.nnkpts
        MMN.nnkptsG = NNKP.nnkptsG
        MMN.mmn = np.zeros([NNKP.nk, NNKP.nnk, MMN.nb, MMN.nb], dtype='complex128')

        for ik in range(NNKP.nk):
            print('>>> cal overlap matrix M(k,k+b) at {}/{}'.format(ik + 1, NNKP.nk))
            for ib in range(MMN.nnk):
                jk = MMN.nnkpts[ik, ib] - 1

                if cross_bz and not (NNKP.nnkptsG[ik, ib] == 0).all():
                    kc2 = kkc[jk] + LA.multi_dot([self.lattG, NNKP.nnkptsG[ik, ib]])
                    bandEj, bandUj = LA.eigh(self.w90gethk(kc2))
                    MMN.mmn[ik, ib] = LA.multi_dot([bandU[ik].conj().T, self.get_gxx(ik, ib), bandUj])
                else:
                    MMN.mmn[ik, ib] = LA.multi_dot([bandU[ik].conj().T, self.get_gxx(ik, ib), bandU[jk]])

        return MMN

    def cal_AMN(self, nb, nw):
        bandU = self.bandU
        AMN = W90_amn()
        AMN.nw = nw
        AMN.nb = nb
        AMN.nk = self.NNKP.nk
        AMN.amn = np.zeros([AMN.nk, AMN.nb, AMN.nw], dtype='complex128')

        # rrws = LA.multi_dot([self.latt, self.meshRws.T]).T
        gnk = self.get_gn()
        wkm = self.get_wkm()
        # AMN.amn = np.einsum('km,kbm,nkb->kmn', wkm, bandU.conj(), gn)
        AMN.amn = np.einsum('km,kbm,knb->kmn', wkm, bandU.conj(), gnk)

        return AMN

    def get_gxx(self, ik, ib):
        '''
          * metric matrix g_mn(k,k+b)
          ** jk = self.NNKP.nnkpts[ik, ib] - 1
          ** kc1 = self.meshkc[ik]
          ** kc2 = self.meshkc[jk] + LA.multi_dot([self.lattG, self.nnkptsG[ik, ib]])
          ** bc_index = self.NNKP.bvecs_index[ik, ib]
          bc can be get:
          ** bc = self.NNKP.bvecs[bc_index]
          ** bc = self.NNKP.nnbpts_car[ik, ib]
          ** bc = kc2 - kc1
        ''' #
        return np.identity(self.nb)

    def get_wkm(self):
        return np.ones([self.NNKP.nk, self.nb], dtype='float64')

    '''
      * Build [Grid] used to cal projection Amn(k)=<psi_mk|gn>
      * it can be done in either real space or reciprocal space
      ** meshR 
      ** gridG 
      ** FT 
    '''
    def make_mesh(self, nmesh, type='continuous', centersym=False):
        '''
        * type = continuous, discrete
        * centersym = False, True
        ''' #
        N1, N2, N3 = nmesh
        N = N1 * N2 * N3
        n2, n1, n3 = np.meshgrid(np.arange(N2), np.arange(N1), np.arange(N3))
        mesh = np.array([n1.reshape(N), n2.reshape(N), n3.reshape(N)], dtype='float64').T

        if centersym:
            if not (np.mod(nmesh, 2) == 1).all():
                print('centersym mesh need odd number of [nmesh]')
                sys.exit(1)
            else:
                mesh -= mesh[N // 2]
        if type[0].lower() == 'c':
            mesh /= nmesh
        return mesh

    # def init_grid_old(self, nmeshR, ngridG, nnWSR):
    #     '''
    #       * meshR = meshRws.reshape(nnr, nrr, 3)[nnr//2]
    #     ''' #
    #     meshR = self.make_mesh(nmeshR, type='continuous', centersym=False)
    #     gridG = self.make_mesh(ngridG, type='discrete', centersym=True)
    #     WSR = self.make_mesh(nnWSR, type='discrete', centersym=True)
    #
    #     nnr = WSR.shape[0]
    #     nrr = meshR.shape[0]
    #     meshRws = np.kron(np.ones([nnr, 1]), meshR) + np.kron(WSR, np.ones([nrr, 1]))
    #
    #     return meshR, gridG, WSR, meshRws

    def init_grid(self, nx=60, ny=60, nws=1):
        # nx = 60 #int(self.nsuper**0.5 * 2)
        # ny = 60 #int(self.nsuper**0.5 * 2)
        # nws = 1
        nmeshR = np.array([(2*nws+1)*nx, (2*nws+1)*ny, 1])
        meshR = self.make_mesh(nmeshR, type='continuous', centersym=False) * (2*nws+1)
        meshR -= np.array([1, 1, 0], dtype='float64')
        WSR = self.make_mesh([(2*nws+1), (2*nws+1), 1], type='discrete', centersym=True)
        return meshR, WSR

    def get_eiGR(self, gridG, meshR, signal=-1):
        return np.exp(signal * 2j * np.pi * np.einsum('Ga,Ra->GR', gridG, meshR))

    '''
      * psi_mk in <psi_mk|gn>
    '''
    def get_gn(self):
        pass
    # def get_psi_RSpace(self, ki):
    #     psi = np.zeros([nk, nb, nrr], dtype='complex128')
    #
    #     return psi
    #
    # def get_gn_RSpace(self, n):
    #     gn = np.zeros([nproj, nrr], dtype='complex128')
    #     return gn
    #
    # def get_psi_GSpace(self, ki):
    #     psi = np.zeros([nk, nb, nrr], dtype='complex128')
    #     return psi
    #
    # def get_gn_GSpace(self, n):
    #     gn = np.zeros([nproj, nrr], dtype='complex128')
    #     return gn

    '''
      * Atomic Orbitals
    '''
    def get_2pz(self, rr):
        a0 = 0.529
        z = rr[:, 2]
        r = LA.norm(rr, axis=1)
        gpz = (32 * np.pi) ** -0.5 * (a0 ** -1.5) / a0 * z * np.exp(-r / 2 / a0)
        return gpz

    def get_gauss_pz(self, rr, a=0.7):
        '''
        ----------------
        borden=0.7~1.0 rr=(10,10,24)
        ----------------
        '''  #
        a0 = 0.529
        borden = a0 * a
        z = rr[:, 2]
        r = LA.norm(rr, axis=1)
        gpz = (z / borden) * np.exp(-0.5 * (r / borden) ** 2)
        return gpz

    def get_gpz(self, rr):
        return self.get_gauss_pz(rr)

    def get_2d_orbital_for_Hatomic(self, rr, n, l, m, scale_a0=1):
        '''
        * reture 2d H atom orbitals in scale of a0
        * to get a Morris scale orbital, rr should be 
        * replaced by rr - ro, a0 should be replaced by
        * a0 * scale_a0

        R10 = 2 * a0 ** (-3/2) * np.exp(-rho)
        R20 = 1/np.sqrt(2) * a0 ** (-3/2) * np.exp(-rho) * (1-0.5*r/a0)
        R21 = 1/np.sqrt(24) * a0 ** (-3/2) * np.exp(-rho) * (r/a0)

        R30 = 2 / np.sqrt(27) * a0 ** (-3/2) * np.exp(-rho) * (1-2/3*r/a0+2/27*(r/a0)**2)
        R31 = 8/27/np.sqrt(6) * a0 ** (-3/2) * np.exp(-rho) * (1-1/6*r/a0)*(r/a0)
        R32 = 4/81/np.sqrt(30) * a0 ** (-3/2) * np.exp(-rho) * (r/a0)**2

        plt.plot(r, Rn, color='red', linewidth='3', alpha=0.5)
        plt.plot(r, R20, color='k', linewidth='1')

        print('int R={}'.format((R30 * r **2).sum()*dr))
        print('int R={}'.format((R31 * r **2).sum()*dr))
        print('int R={}'.format((R32 * r **2).sum()*dr))
        print('int R={}'.format((R10 * r **2).sum()*dr))
        print('int R={}'.format((R20 * r **2).sum()*dr))
        print('int R={}'.format((R21 * r **2).sum()*dr))
        print('int R={}'.format(R31.sum()*dr))
        print('int R={}'.format(R32.sum()*dr))
        '''  #
        x, y, z = rr.T[0], rr.T[1], rr.T[2]
        r = LA.norm(rr, axis=1)

        a0 = 0.529 * scale_a0
        rho = r / a0 / n
        vrho = rho ** 0
        cj = 1
        for j in range(n - l - 1):
            cj *= 2 * (j + l - n + 1) / (j + 2 * l + 2) / (j + 1)
            vrho += cj * rho ** (j + 1)
            print(cj)

        Rnl = (2 * rho) ** l * np.exp(-rho) * vrho
        Rnl *= 2 * a0 ** (-3 / 2) / (n ** 2 * np.math.factorial(2 * l + 1)) * np.sqrt(
            np.math.factorial(n + l) / np.math.factorial(n - l - 1))

        if l == 0 and m == 0:
            Ylm = np.sqrt(1 / 2 / np.pi)
        elif l == 1 and m == 0:
            Ylm = np.sqrt(1 / 2 / np.pi)
        elif l == 1 and m == -1:
            # Ylm = np.sqrt(1 / 2 / np.pi) * (x - 1j * y) / r
            Ylm = np.sqrt(1 / 2 / np.pi) * np.real(x / (r + 0.1j * a0))
        elif l == 1 and m == 1:
            # Ylm = -np.sqrt(1 / 2 / np.pi) * (x + 1j * y) / r
            Ylm = -np.sqrt(1 / 2 / np.pi) * np.real(y / (r + 0.1j * a0))
        else:
            Ylm = 0
            print('only s and p orbital are suppoted')

        gn = Rnl * Ylm

        return gn

    '''
      * Atomic Orbitals v2
    '''
    def get_moire_gs(self, rr, sigmax=1.0, _sigmay=None):
        sigmay = sigmax if _sigmay is None else _sigmay
        xx = rr[:, 0]
        yy = rr[:, 1]

        Rn = np.exp(-0.5 * ((xx/sigmax)**2 + (yy/sigmay)**2))
        Ylm = np.sqrt(1/4/np.pi)
        g = Rn * Ylm
        return g

    def get_moire_gpz(self, rr, sigmax=1.0, _sigmay=None):
        sigmay = sigmax if _sigmay is None else _sigmay
        xx = rr[:, 0]
        yy = rr[:, 1]
        zz = rr[:, 2]
        r = LA.norm(rr, axis=1)
        invr = np.real(1 / (r - 0.001j))

        Rn = np.exp(-0.5 * ((xx/sigmax)**2 + (yy/sigmay)**2))
        Ylm = np.sqrt(3/4/np.pi) * zz * invr
        g = Rn * Ylm
        return g

    def get_moire_gpx(self, rr, sigmax=1.0, _sigmay=None):
        sigmay = sigmax if _sigmay is None else _sigmay
        xx = rr[:, 0]
        yy = rr[:, 1]
        r = LA.norm(rr, axis=1)
        invr = np.real(1 / (r - 0.001j))

        Rn = np.exp(-0.5 * ((xx/sigmax)**2 + (yy/sigmay)**2))
        Ylm = np.sqrt(3/4/np.pi) * xx * invr
        g = Rn * Ylm
        return g

    def get_moire_gpy(self, rr, sigmax=1.0, _sigmay=None):
        sigmay = sigmax if _sigmay is None else _sigmay
        xx = rr[:, 0]
        yy = rr[:, 1]
        r = LA.norm(rr, axis=1)
        invr = np.real(1 / (r - 0.001j))

        Rn = np.exp(-0.5 * ((xx/sigmax)**2 + (yy/sigmay)**2))
        Ylm = np.sqrt(3/4/np.pi) * yy * invr
        g = Rn * Ylm
        return g

    def get_moire_orbi(self, rr, n=1, l='s', sigma=1.0):
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


class W90_interpolation(object):

    def __init__(self, nw, ngridR, latt, NNKP, EIG, wcc=None, load_u=False):
        self.fermi = 0

        self.nw = nw
        self.nk, self.nb = EIG.nk, EIG.nb

        # self.CELL = CELL
        self.latt = latt
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

        self.NNKP = NNKP
        self.meshk = NNKP.kk
        self.meshkc = LA.multi_dot([self.lattG, self.meshk.T]).T

        self.nR, self.ndegenR, self.gridR = self._init_ws_gridR(ngridR, self.latt)
        self.gridRc = LA.multi_dot([self.latt, self.gridR.T]).T

        self.eig = EIG.eig

        self.h_Rmn = None
        self.r_Ramn = None

        self.wcc = wcc
        if wcc is not None:
            self.wccf = LA.multi_dot([LA.inv(self.latt), wcc.T]).T
            self.eiktau = self._init_eiktau(wcc)
        self.mmn = None

        self.umat = None
        self.umat_dis = None
        self.vmat = None
        self._init_vmat(load_u)

        self.nD = 0
        self.D_namelist = []
        self.D_iRmn = None

    # def load_from_w90(self, ngridR, load_u=True):
    #     '''
    #     latt: FROM POSCAR
    #     meshk: FROM NNKP
    #     umat, umat_dis, vmat: FROM UMAT,UMAT_DIS
    #     CELL: FROM POSCAR
    #     wcc, wccf: FROM CELL
    #     eig: FROM wannier.eig
    #     ''' #
    #     NNKP = W90_nnkp()
    #     NNKP.load_from_w90()
    #     UMAT = W90_umat()
    #     UMAT_DIS = W90_umat_dis()
    #     CELL = Cell()
    #     CELL.load_poscar()
    #
    #     self.latt = CELL.lattice
    #     self.lattG = CELL.latticeG
    #     self.wcc = CELL.ions_car
    #     self.wccf = CELL.ions
    #     self.CELL = CELL
    #
    #     self.nR, self.ndegenR, self.gridR = self._init_ws_gridR(ngridR, self.latt)
    #     self.gridRc = LA.multi_dot([self.latt, self.gridR.T]).T
    #     self.meshk = NNKP.kk
    #     self.meshkc = LA.multi_dot([self.lattG, self.meshk.T]).T
    #
    #     EIG = W90_eig(self.nb, self.nk)
    #     EIG.load_from_w90()
    #
    #     self.eig = EIG.eig
    #
    #     if load_u:
    #         UMAT.load_from_w90()
    #         UMAT_DIS.load_from_w90()
    #     else:
    #         UMAT.nk, UMAT.nb, UMAT.nw = self.nk, self.nb, self.nw
    #         UMAT_DIS.nk, UMAT_DIS.nb, UMAT_DIS.nw = self.nk, self.nb, self.nw
    #         UMAT.umat = np.zeros([self.nk, self.nw, self.nw], dtype='complex128')
    #         UMAT_DIS.umat_dis = np.zeros([self.nk, self.nb, self.nw], dtype='complex128')
    #
    #     self.umat = UMAT.umat
    #     self.umat_dis = UMAT_DIS.umat_dis
    #     self.vmat = np.einsum('kmi,kin->kmn', self.umat_dis, self.umat)

    def _init_vmat(self, load_u):
        UMAT = W90_umat()
        UMAT_DIS = W90_umat_dis()
        if load_u:
            UMAT.load_from_w90()
            UMAT_DIS.load_from_w90()
        else:
            UMAT.nk, UMAT.nb, UMAT.nw = self.nk, self.nb, self.nw
            UMAT_DIS.nk, UMAT_DIS.nb, UMAT_DIS.nw = self.nk, self.nb, self.nw
            UMAT.umat = np.zeros([self.nk, self.nw, self.nw], dtype='complex128')
            UMAT_DIS.umat_dis = np.zeros([self.nk, self.nb, self.nw], dtype='complex128')

        self.umat = UMAT.umat
        self.umat_dis = UMAT_DIS.umat_dis
        self.vmat = np.einsum('kmi,kin->kmn', self.umat_dis, self.umat)

    def _init_ws_gridR(self, ngridR, latt):
        '''
        :param Rgrid:
        :return:
                nrpts: int
                ndegen: list
                irvec: [array([i1,i2,i3])]

        '''
        # ***********
        # init
        # ***********
        a1 = latt.T[0]
        a2 = latt.T[1]
        a3 = latt.T[2]

        # ***********
        # main
        # ***********
        nR = 0  # -1
        ndegen = []
        gridR = []

        g_matrix = np.dot(np.array([a1, a2, a3]),
                          np.array([a1, a2, a3]).T)

        for n1 in range(-ngridR[0], ngridR[0] + 1):
            for n2 in range(-ngridR[1], ngridR[1] + 1):
                for n3 in range(-ngridR[2], ngridR[2] + 1):
                    # Loop 125 R
                    icnt = -1
                    dist = np.zeros((125))
                    for i1 in [-2, -1, 0, 1, 2]:
                        for i2 in [-2, -1, 0, 1, 2]:
                            for i3 in [-2, -1, 0, 1, 2]:
                                icnt += 1
                                ndiff = np.array([
                                    n1 - i1 * ngridR[0],
                                    n2 - i2 * ngridR[1],
                                    n3 - i3 * ngridR[2]
                                ])
                                dist[icnt] = ndiff.dot(g_matrix).dot(ndiff)
                    # print(dist)

                    # dist_min = min(dist.tolist())
                    dist_min = np.min(dist)
                    if np.abs((dist[62] - dist_min)) < 10 ** -7:
                        # nrpts += 1
                        ndegen.append(0)
                        for i in range(0, 125):
                            if np.abs(dist[i] - dist_min) < 10 ** -7:
                                ndegen[nR] += 1
                        nR += 1

                        # irvec.append(n1 * a1 + n2 * a2 + n3 * a3)
                        gridR.append(np.array([n1, n2, n3]))

        ndegen = np.array(ndegen, dtype='int64')
        gridR = np.array(gridR, dtype='int64')
        # print('nrpts={}'.format(nrpts_s))
        # print('ndegen=\n', ndegen_s)
        # print('irvec=\n')
        # pp.pprint(irvec_s)
        print('*=============================================================================*')
        print('|                                   R Grid                                     |')
        print('|    number of R Grid = {:4>}                                                  |'.format(nR))
        print('*=============================================================================*')
        for i in range(nR):
            print('|{: 4}). {: 3} {: 3} {: 3}   *{:2>}  '.format(i + 1, gridR[i, 0], gridR[i, 1], gridR[i, 2], ndegen[i]),
                  end='')
            if (i + 1) % 3 == 0:
                print('|')
        print('')
        print('*--------------------------------------------------------------------------------*')
        return nR, ndegen, gridR

    def _init_eiktau(self, wcc):
        return np.exp(1j * np.einsum('ka,na->kn', self.meshkc, wcc))

    def init_symmOP(self, nD):
        self.nD = nD
        self.D_namelist = [None for i in range(nD)]
        self.D_iRmn = np.zeros([nD, self.nR, self.nw, self.nw], dtype='complex128')

    def interpolate_hR(self, tbgauge=False):
        ek = np.array([
            np.diag(self.eig[ik])
            for ik in range(self.nk)
        ], dtype='float64')
        hwk = np.einsum('kim,kij,kjn->kmn', self.vmat.conj(), ek, self.vmat, optimize=True)
        # hwk = np.einsum('kim,kij,kjn->kmn', self.umat_dis.conj(), ek, self.umat_dis, optimize=True)
        hwk = 0.5 * (hwk + np.einsum('kmn->knm', hwk).conj())

        eikR = np.exp(-2j * np.pi * np.einsum('ka,Ra->kR', self.meshk, self.gridR))
        if tbgauge:
            self.h_Rmn = np.einsum('kR,km,kmn,kn->Rmn', eikR, self.eiktau, hwk, self.eiktau.conj(), optimize=True) / self.nk
        else:
            self.h_Rmn = np.einsum('kR,kmn->Rmn', eikR, hwk) / self.nk

    def interpolate_hR_selec(self, selec_orbi):
        self.nw = selec_orbi.shape[0]
        self.umat_dis1 = np.zeros([self.nk, self.nb, self.nw], dtype='complex128')
        for i in range(self.nw):
            self.umat_dis1[:, :, i] = self.umat_dis[:, :, selec_orbi[i]]
        self.umat_dis = self.umat_dis1

    def interpolate_rR_onsite(self, tbgauge=False):
        self.r_Ramn = np.zeros([self.nR, 3, self.nw, self.nw], dtype='complex128')
        if not tbgauge:
            self.r_Ramn[self.nR//2, 0] = np.diag(self.wcc.T[0])
            self.r_Ramn[self.nR//2, 1] = np.diag(self.wcc.T[1])
            self.r_Ramn[self.nR//2, 2] = np.diag(self.wcc.T[2])

    def interpolate_DR(self, eigD, i, name, tbgauge=False):
        '''
          * interpolate symmetric operators D in R space
            eigD: eignvalues in eignstates representation
        ''' #
        Dk_diag = np.array([
            np.diag(eigD[ik])
            for ik in range(self.nk)
        ], dtype='float64')
        Dwk = np.einsum('kim,kij,kjn->kmn', self.vmat.conj(), Dk_diag, self.vmat, optimize=True)
        Dwk = 0.5 * (Dwk + np.einsum('kmn->knm', Dwk).conj())

        eikR = np.exp(-2j * np.pi * np.einsum('ka,Ra->kR', self.meshk, self.gridR))
        self.D_namelist[i] = name
        if tbgauge:
            self.D_iRmn[i] = np.einsum('kR,km,kmn,kn->Rmn', eikR, self.eiktau, Dwk, self.eiktau.conj()) / self.nk
        else:
            self.D_iRmn[i] = np.einsum('kR,kmn->Rmn', eikR, Dwk) / self.nk

    def get_hk(self, k, tbgauge=False):
        eikR = np.exp(2j * np.pi * np.einsum('a,Ra->R', k, self.gridR)) / self.ndegenR
        if tbgauge:
            eiktau = np.exp(2j * np.pi * np.einsum('a,na', k, self.wccf))
            hk = np.einsum('R,m,Rmn,n->mn', eikR, eiktau.conj(), self.h_Rmn, eiktau, optimize=True)
        else:
            hk = np.einsum('R,Rmn->mn', eikR, self.h_Rmn, optimize=True)
        return hk

    def get_Dk(self, k, i, tbgauge=False):
        eikR = np.exp(2j * np.pi * np.einsum('a,Ra->R', k, self.gridR)) / self.ndegenR
        if tbgauge:
            eiktau = np.exp(2j * np.pi * np.einsum('a,na', k, self.wccf))
            Dk = np.einsum('R,m,Rmn,n->mn', eikR, eiktau.conj(), self.D_iRmn[i], eiktau, optimize=True)
        else:
            Dk = np.einsum('R,Rmn->mn', eikR, self.D_iRmn[i], optimize=True)
        return Dk

    def svd_init_guess(self, amn):
        '''
          * extract subspace of target bands from inital guess
            by make using of Lowdin's symmetric orthogonalization procedure.
            For entangled energy bands, using eq.23 [PRB 65 035109, 2001].
        ''' #
        nk, nb, nw = amn.shape
        vmat = np.zeros([nk, nb, nw], dtype='complex128')
        for i in range(self.nk):
            u, s, vh = LA.svd(amn[i])
            vmat[i] = LA.multi_dot([u, np.eye(nb, nw), vh])
        return vmat

    # def get_wannier_ob_dev(self):
    #     ngridR = np.array([6, 6, 1])
    #     w90intp = W90_interpolation()
    #     w90intp.load_from_w90(ngridR)
    #     nk = w90intp.nk
    #     meshk = w90intp.meshk
    #     R = make_mesh([3, 3, 1], type='discrete', centersym=True)
    #     eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', meshk, R))
    #
    #     initwanWF = np.load(r'initwanWF.npz')
    #     gn0 = initwanWF.get('gn')
    #     gnR0 = initwanWF.get('gnR')
    #     rr = initwanWF.get('rr')
    #     nmeshR = initwanWF.get('nmeshR')
    #
    #     vmat = w90intp.vmat
    #
    #     # amn = AMN.amn
    #
    #     # SR = np.einsum('kml,kmn,kR->Rln', amn.conj(), vmat, eikR) / nk
    #     # gn = np.einsum('Rln,nRLAr->lLAr', SR, gnR0)
    #     # gn = np.einsum('Rln,nRLAr->lr', SR, gnR0)
    #     # plot_orbital(rr, gn[2], nmeshR)

    def build_htb(self, CELL, tbgauge=False):
        htb = Htb()
        htb.name = 'W90_interpolation'
        htb.fermi = self.fermi
        htb.nw = self.nw
        htb.nR = self.nR
        htb.R = self.gridR
        htb.Rc = self.gridRc
        htb.ndegen = self.ndegenR
        htb.N_ucell = 1

        htb.latt = self.latt
        htb.lattG = self.lattG

        htb.cell = CELL
        htb.wcc = self.wcc
        htb.wccf = self.wccf

        htb.hr_Rmn = self.h_Rmn
        htb.r_Ramn = self.r_Ramn

        htb.nD = self.nD
        htb.D_namelist = self.D_namelist
        htb.D_iRmn = self.D_iRmn
        return htb


if __name__ == '__main__':
    os.chdir('')
    # nnkp = W90_nnkp()
    # nnkp.load_from_w90()

    # amn = W90_amn()
    # amn.load_from_w90(fname=r'silicon.amn')
    # amn.save_w90(fname=r'silicon.wanpy.amn')
    #
    # mmn = W90_mmn()
    # mmn.load_from_w90(fname=r'silicon.mmn')
    # mmn.save_w90(fname=r'silicon.wanpy.mmn')
    #
    # eig = W90_eig(nw=12, nk=64)
    # eig.load_from_w90(fname=r'silicon.eig')
    # eig.save_w90(fname=r'silicon.wanpy.eig')


    # umat = W90_umat()
    # umat.load_from_w90(fname=r'wannier90_u.mat')
    # umat.save_w90(fname=r'wannier90_u.wanpy.mat')

    # umat_dis = W90_umat_dis()
    # umat_dis.load_from_w90(fname=r'wannier90_u_dis.mat')
    # umat_dis.save_w90(fname=r'wannier90_u_dis.wanpy.mat')

    # band = W90_band()
