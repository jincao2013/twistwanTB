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

__date__ = "May. 23, 2020"


import os
import errno
import sys
import math
import re
from collections import defaultdict
import numpy as np
from numpy import linalg as LA
# from wanpy.core.units import *
from twistwanTB.wanpy.mesh import make_mesh

import h5py


'''
  Basic object
'''
class Cell(object):

    def __init__(self):
        self._container = ['name', 'lattice', 'latticeG', 'N', 'spec', 'ions', 'ions_car']
        self.name = None
        self.lattice = None
        self.latticeG = None
        self.N = None
        self._spec = None
        self.ions = None
        self.ions_car = None

    @property
    def spec(self):
        return [i.decode('utf-8') for i in self._spec]

    @spec.setter
    def spec(self, value):
        self._spec = [i if type(i) is bytes else i.encode('utf-8') for i in value]

    def get_latticeG(self):
        return 2 * np.pi * LA.inv(self.lattice.T)

    def get_ions(self):
        if self.ions_car is None:
            print('self.ions_car is None')
            return None
        return LA.multi_dot([LA.inv(self.lattice), self.ions_car.T]).T

    def get_ions_car(self):
        if self.ions is None:
            print('self.ions is None')
            return None
        return LA.multi_dot([self.lattice, self.ions.T]).T

    def get_vcell(self):
        return LA.det(self.lattice)

    def load_poscar(self, fname=r'POSCAR'):
        with open(fname, 'r') as poscar:
            name = poscar.readline().strip()
            t = float(poscar.readline())
            a1 = t * np.array(poscar.readline().split(), dtype='float64')
            a2 = t * np.array(poscar.readline().split(), dtype='float64')
            a3 = t * np.array(poscar.readline().split(), dtype='float64')
            lattice = np.array([a1, a2, a3]).T
            latticeG = 2 * np.pi * LA.inv(lattice.T)

            spec_name = poscar.readline().split()
            spec_num = np.array(poscar.readline().split(), dtype='int64')
            num_ion = spec_num.sum()
            ion_spec = []
            for i in range(len(spec_name)):
                for num in range(spec_num[i]):
                    ion_spec.append(spec_name[i])

            ion_unit = poscar.readline().strip().upper()[0]

            ions = np.zeros((num_ion, 3))
            for i in range(num_ion):
                inline = poscar.readline().split()
                ions[i] = np.array(inline[:3], dtype='float64')
        poscar.close()

        self.name = name
        self.lattice = lattice
        self.latticeG = latticeG
        self.N = num_ion
        self.spec = ion_spec

        if ion_unit == 'D':
            self.ions = ions
            self.ions_car = LA.multi_dot([lattice, ions.T]).T
        elif ion_unit == 'C':
            self.ions = LA.multi_dot([LA.inv(lattice), ions.T]).T
            self.ions_car = ions
        else:
            print('ION_UNIT error in reading poscar')
            sys.exit(1)

    def save_poscar(self, fname='POSCAR.vasp', cartesian=False):
        if os.path.exists(fname):
            os.remove(fname)

        ions_unit = 'D'
        ions = self.ions
        spec_name, spec_index, spec_num = np.unique(self.spec, return_index=True, return_counts=True)
        spec_name = spec_name[np.argsort(spec_index)]
        spec_num = spec_num[np.argsort(spec_index)]

        if cartesian:
            ions_unit = 'C'
            ions = self.ions_car

        with open(fname, 'a') as poscar:
            poscar.write('writen by wanpy\n')
            poscar.write('   1.0\n')
            for i in self.lattice.T: # crystal_data['lattice'].T:
                poscar.write('   {: 2.16f}    {: 2.16f}    {: 2.16f}\n'.format(i[0], i[1], i[2]))

            poscar.write('    '.join(spec_name))
            poscar.write('\n')
            poscar.write('    '.join(np.array(spec_num, dtype='U')))
            poscar.write('\n')

            poscar.write('{}\n'.format(ions_unit))

            for i in ions:
                poscar.write('  {: 2.16f}  {: 2.16f}  {: 2.16f}\n'.format(i[0], i[1], i[2]))
            poscar.write('\n')
        poscar.close()

    def save_h5(self, fname='cell.h5'):
        with h5py.File(fname, "a") as f:
            f.create_group('cell')
            cell = f['cell']
            h5st = h5py.string_dtype(encoding='utf-8')
            hdf5_create_dataset(cell, 'name', data=self.name, dtype=h5st)
            hdf5_create_dataset(cell, 'lattice', data=self.lattice, dtype='float64')
            hdf5_create_dataset(cell, 'latticeG', data=self.latticeG, dtype='float64')
            hdf5_create_dataset(cell, 'N', data=self.N, dtype='int64')
            hdf5_create_dataset(cell, 'spec', data=self._spec, dtype=h5st)
            hdf5_create_dataset(cell, 'ions', data=self.ions, dtype='float64')
            hdf5_create_dataset(cell, 'ions_car', data=self.ions_car, dtype='float64')
            f.close()

    def load_h5(self, fname='cell.h5'):
        f = h5py.File(fname, "r")
        cell = f.get('cell')
        for i in self._container:
            item = cell.get(i)
            if item is not None:
                self.__dict__[i] = item[()]
        self.spec = hdf5_read_dataset(cell, 'spec', default=[])
        f.close()

    def printer(self, site=True):
        lattice = self.lattice
        latticeG = self.latticeG
        spec = self.spec
        print('                                    ------')
        print('                                    SYSTEM')
        print('                                    ------')
        print('')
        print('                              Lattice Vectors (Ang)')
        print('                    a_1  {: 11.6f}{: 11.6f}{: 11.6f}'.format(lattice[0, 0], lattice[1, 0], lattice[2, 0]))
        print('                    a_2  {: 11.6f}{: 11.6f}{: 11.6f}'.format(lattice[0, 1], lattice[1, 1], lattice[2, 1]))
        print('                    a_3  {: 11.6f}{: 11.6f}{: 11.6f}'.format(lattice[0, 2], lattice[1, 2], lattice[2, 2]))
        print('')
        print('                   Unit Cell Volume: {: 13.5f}  (Ang^3)'.format(LA.det(lattice)))
        print('')
        print('                        Reciprocal-Space Vectors (Ang^-1)')
        print('                    b_1  {: 11.6f}{: 11.6f}{: 11.6f}'.format(latticeG[0, 0], latticeG[1, 0], latticeG[2, 0]))
        print('                    b_2  {: 11.6f}{: 11.6f}{: 11.6f}'.format(latticeG[0, 1], latticeG[1, 1], latticeG[2, 1]))
        print('                    b_3  {: 11.6f}{: 11.6f}{: 11.6f}'.format(latticeG[0, 2], latticeG[1, 2], latticeG[2, 2]))
        print('')
        if site:
            print(' *-----------------------------------------------------------------------*')
            print(' | Site      Fractional Coordinate        Cartesian Coordinate (Ang)     |')
            print(' *-----------------------------------------------------------------------*')
            for i in range(self.ions.shape[0]):
                print(' | {} {: 10.5f}{: 10.5f}{: 10.5f}   | {: 10.5f}{: 10.5f}{: 10.5f}   |'.format(
                    spec[i],
                    self.ions[i, 0], self.ions[i, 1], self.ions[i, 2],
                    self.ions_car[i, 0], self.ions_car[i, 1], self.ions_car[i, 2],
                ))
            print(' *-----------------------------------------------------------------------*')


class Htb(object):

    def __init__(self, fermi=0.):
        self._head_container = [
            'name', 'fermi', 'nw', 'nR', 'R', 'Rc', 'ndegen', 'N_ucell',
            'latt', 'lattG',
            'wcc', 'wccf',
            'nD', 'D_namelist',
        ]
        self._head_container_large = ['hr_Rmn', 'r_Ramn', 'D_iRmn']

        self.name = None
        self.fermi = fermi
        self.nw = None
        self.nR = None
        self.R = None
        self.Rc = None
        self.ndegen = None
        self.N_ucell = 1

        self.latt = None
        self.lattG = None

        self.cell = Cell()
        # self.wout = Wout()
        self.wcc = None
        self.wccf = None

        self.hr_Rmn = None
        self.r_Ramn = None

        # symmetric operators {D}
        self.nD = None
        self._D_namelist = None
        self.D_iRmn = None

        # etc.
        self.dk = None
        self._R_hr = None
        self._R_r = None
        self._Rc_hr = None
        self._Rc_r = None

    @property
    def D_namelist(self):
        return [i.decode('utf-8') for i in self._D_namelist]

    @D_namelist.setter
    def D_namelist(self, value):
        self._D_namelist = [i if type(i) is bytes else i.encode('utf-8') for i in value]

    @property
    def R_hr(self):
        if self._R_hr is not None:
            return self._R_hr
        else:
            return self.R

    @R_hr.setter
    def R_hr(self, value):
        self._R_hr = value

    @property
    def R_r(self):
        if self._R_r is not None:
            return self._R_r
        else:
            return self.R

    @R_r.setter
    def R_r(self, value):
        self._R_r = value

    @property
    def Rc_hr(self):
        if self._Rc_hr is not None:
            return self._Rc_hr
        else:
            return self.Rc

    @Rc_hr.setter
    def Rc_hr(self, value):
        self._Rc_hr = value

    @property
    def Rc_r(self):
        if self._Rc_r is not None:
            return self._Rc_r
        else:
            return self.Rc

    @Rc_r.setter
    def Rc_r(self, value):
        self._Rc_r = value


    '''
      * I/O level 1
    '''
    def load_htb(self, htb_fname=r'htb.h5'):
        # self.load_npz(htb_fname)
        self.load_h5(htb_fname)

    def save_htb(self, htb_fname=r'htb.h5'):
        # self.save_npz(htb_fname)
        self.save_h5(htb_fname)

    def load_npz(self, htb_fname=r'htb.npz'):
        data = np.load(htb_fname, allow_pickle=True)
        head = data['head'].item()
        self.hr_Rmn = data.get('hr_Rmn')
        self.r_Ramn = data.get('r_Ramn')
        self.D_iRmn = data.get('D_iRmn')

        for i in self._head_container:
            self.__dict__[i] = head[i]

        self.cell = head['cell']
        self.latt = self.cell.lattice
        self.lattG = self.cell.latticeG
        self.Rc = LA.multi_dot([self.latt, self.R.T]).T

    def save_npz(self, save_fname=r'htb.npz'):
        head = defaultdict(f_none)

        for i in self._head_container:
            head[i] = self.__dict__[i]

        np.savez_compressed(save_fname,
                            head=head,
                            hr_Rmn=self.hr_Rmn,
                            r_Ramn=self.r_Ramn,
                            D_iRmn=self.D_iRmn,
                            )

    def save_h5(self, fname='htb.h5'):
        self.Rc = LA.multi_dot([self.latt, self.R.T]).T

        if os.path.exists(fname):
            os.remove(fname)

        with h5py.File(fname, "w") as f:
            f.create_group('htb')
            htb = f['htb']
            h5st = h5py.string_dtype(encoding='utf-8')
            hdf5_create_dataset(htb, 'name', data=self.name, dtype=h5st)
            hdf5_create_dataset(htb, 'fermi', data=self.fermi, dtype='float64')
            hdf5_create_dataset(htb, 'nw', data=self.nw, dtype='int64')
            hdf5_create_dataset(htb, 'nR', data=self.nR, dtype='int64')
            hdf5_create_dataset(htb, 'R', data=self.R, dtype='int64')
            hdf5_create_dataset(htb, 'Rc', data=self.Rc, dtype='float64')
            hdf5_create_dataset(htb, 'ndegen', data=self.ndegen, dtype='int64')
            hdf5_create_dataset(htb, 'N_ucell', data=self.N_ucell, dtype='int64')

            hdf5_create_dataset(htb, 'latt', data=self.latt, dtype='float64')
            hdf5_create_dataset(htb, 'lattG', data=self.lattG, dtype='float64')
            hdf5_create_dataset(htb, 'wcc', data=self.wcc, dtype='float64')
            hdf5_create_dataset(htb, 'wccf', data=self.wccf, dtype='float64')

            hdf5_create_dataset(htb, 'nD', data=self.nD, dtype='int64')
            hdf5_create_dataset(htb, 'D_namelist', data=self._D_namelist, dtype=h5st)

            hdf5_create_dataset(htb, 'hr_Rmn', data=self.hr_Rmn, dtype='complex128', compression="gzip")
            hdf5_create_dataset(htb, 'r_Ramn', data=self.r_Ramn, dtype='complex128', compression="gzip")
            hdf5_create_dataset(htb, 'D_iRmn', data=self.D_iRmn, dtype='complex128', compression="gzip")
            f.close()

        self.cell.save_h5(fname)

    def load_h5(self, fname='htb.h5'):
        f = h5py.File(fname, 'r')
        htb = f.get('htb')
        for i in self._head_container + self._head_container_large:
            item = htb.get(i)
            if item is not None:
                self.__dict__[i] = item[()]
        if htb.get('D_namelist') is not None:                                       # WANNING!!!!!! Dec.2 2020
            self.D_namelist = hdf5_read_dataset(htb, 'D_namelist', default=[])
        f.close()
        self.cell.load_h5(fname)

    def load_wannier90_dat(self,
                           poscar_fname=r'POSCAR',
                           seedname='wannier90',
                           ):
        wout_fname = seedname + '.wout'
        hr_fname = seedname + '_hr.dat'
        r_fname = seedname + '_r.dat'
        if os.path.exists(poscar_fname):
            self.cell.load_poscar(fname=poscar_fname)
            self.name = self.cell.name
            self.latt = self.cell.lattice
            self.lattG = self.cell.latticeG
            print('loaded from {}'.format(poscar_fname))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), poscar_fname)

        if os.path.exists(wout_fname):
            # wout = W90_wout()
            # wout.load_wc(fname=wout_fname)
            #
            # self.wcc = wout.wcc
            # self.wccf = wout.wccf
            self.wcc, self.wccf = self._w90_load_wcc(fname=wout_fname)
            print('loaded from {}'.format(wout_fname))
        else:
            print('\033[0;31mfile not found: {} \033[0m'.format(wout_fname))

        if os.path.exists(hr_fname):
            self.nw, self.nR, self.ndegen, self.R, self.hr_Rmn = self._read_hr(fname=hr_fname)
            # self.r_Ramn = self._read_rr_v2x(fname=r_fname, nR=self.nR)
            print('loaded from {}'.format(hr_fname))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), hr_fname)
            # print('\033[1;31m[WANNING] \033[0m' + '{} file not found'.format(hr_fname))

        if os.path.exists(r_fname):
            self.r_Ramn = self._read_rr(fname=r_fname)
            print('loaded from {}'.format(r_fname))
        else:
            print('\033[0;31mfile not found: {} \033[0m'.format(r_fname))


    def save_wannier90_hr_dat(self, seedname='wannier90'):
        hr_fname = seedname + '_hr.dat'

        hr_Rmn = self.hr_Rmn

        if os.path.exists(hr_fname):
            os.remove(hr_fname)

        with open(hr_fname, 'a') as f:
            f.write('written by wanpy\n')
            f.write('          {}\n'.format(self.nw))
            f.write('          {}\n'.format(self.nR))

            countor = 0
            for i in range(self.nR):
                if countor == 15:
                    countor = 0
                    f.write(' \n')
                f.write('    ')
                f.write(str(self.ndegen[i]))
                countor += 1
            f.write(' \n')

            for i, _R in zip(range(self.nR), self.R):
                for wi in range(self.nw):
                    for wj in range(self.nw):
                        f.write(
                            '{: >5d}{: >5d}{: >5d}{: >5d}{: >5d}{: >12.6f}{: >12.6f}\n'.format(_R[0], _R[1], _R[2],
                                                                                               wj + 1, wi + 1,
                                                                                               hr_Rmn[i, wj, wi].real,
                                                                                               hr_Rmn[i, wj, wi].imag,
                                                                                               )
                        )
        f.close()

    def save_wannier90_r_dat(self, seedname='wannier90'):
        r_fname = seedname + '_r.dat'
        if os.path.exists(r_fname):
            os.remove(r_fname)

        r_Ramn = self.r_Ramn
        with open(r_fname, 'a') as f:
            f.write('written by wanpy\n')
            f.write('          {}\n'.format(self.nw))

            for i, _R in zip(range(self.nR), self.R):
                for wi in range(self.nw):
                    for wj in range(self.nw):
                        f.write(
                            '{: >5d}{: >5d}{: >5d}{: >5d}{: >5d}{: >12.6f}{: >12.6f}{: >12.6f}{: >12.6f}{: >12.6f}{: >12.6f}\n'.format(
                                _R[0], _R[1], _R[2],
                                wj + 1, wi + 1,
                                r_Ramn[i, 0, wj, wi].real, r_Ramn[i, 0, wj, wi].imag,
                                r_Ramn[i, 1, wj, wi].real, r_Ramn[i, 1, wj, wi].imag,
                                r_Ramn[i, 2, wj, wi].real, r_Ramn[i, 2, wj, wi].imag,
                                )
                        )
        f.close()

    def save_D_dat(self, seedname=r'wanpy_symmOP'):
        for _isymmOP in range(self.nD):
            fname = seedname + '_' + str(_isymmOP+1) + '.dat'
            symm_name = self.D_namelist[_isymmOP]
            D_Rmn = self.D_iRmn[_isymmOP]

            if os.path.exists(fname):
                os.remove(fname)

            with open(fname, 'a') as f:
                f.write('{} \n'.format(symm_name))
                f.write('          {}\n'.format(self.nw))
                f.write('          {}\n'.format(self.nR))

                countor = 0
                for i in range(self.nR):
                    if countor == 15:
                        countor = 0
                        f.write(' \n')
                    f.write('    ')
                    f.write(str(self.ndegen[i]))
                    countor += 1
                f.write(' \n')

                for i, _R in zip(range(self.nR), self.R):
                    for wi in range(self.nw):
                        for wj in range(self.nw):
                            f.write(
                                '{: >5d}{: >5d}{: >5d}{: >5d}{: >5d}{: >12.6f}{: >12.6f}\n'.format(_R[0], _R[1], _R[2],
                                                                                                   wj + 1, wi + 1,
                                                                                                   D_Rmn[i, wj, wi].real,
                                                                                                   D_Rmn[i, wj, wi].imag,
                                                                                                   )
                            )
            f.close()

    def save_wcc(self, fname=r'wcc.vasp', cartesian=False):
        if os.path.exists(fname):
            os.remove(fname)

        wcc_unit = 'D'
        wcc = self.wccf

        if cartesian:
            wcc_unit = 'C'
            wcc = self.wcc

        with open(fname, 'a') as poscar:
            poscar.write('writen by wanpy\n')
            poscar.write('   1.0\n')
            for i in self.cell.lattice.T:
                poscar.write('   {: 2.16f}    {: 2.16f}    {: 2.16f}\n'.format(i[0], i[1], i[2]))

            poscar.write('H   \n')
            poscar.write('{}    \n'.format(self.nw))

            poscar.write('{}\n'.format(wcc_unit))

            for i in wcc:
                poscar.write('  {: 2.16f}  {: 2.16f}  {: 2.16f}\n'.format(i[0], i[1], i[2]))
            poscar.write('\n')
        poscar.close()

    '''
      * I/O level 2
    '''
    def _read_hr(self, fname):
        with open(fname, 'r') as hr_file:
            hr_file.readline()

            nw = int(hr_file.readline().strip())
            nR = int(hr_file.readline().strip())

            ndegen = np.zeros((nR), dtype='int64')
            index = 0
            for i in range(math.ceil(nR / 15)):
                for j in hr_file.readline().split():
                    ndegen[index] = int(j)
                    index += 1

            R = np.zeros((nR, 3), dtype='int64')
            hr_Rmn = np.zeros((nR, nw, nw), dtype='complex128')
            for nrpts_i in range(nR):

                for m in range(nw):
                    for n in range(nw):
                        inline = hr_file.readline().split()
                        inline_m = int(inline[3]) - 1
                        inline_n = int(inline[4]) - 1
                        hr_Rmn[nrpts_i, inline_m, inline_n] = complex(float(inline[5]), float(inline[6]))
                R[nrpts_i] = np.array(inline[:3], dtype='int64')

        return nw, nR, ndegen, R, hr_Rmn

    def _read_rr(self, fname):
        with open(fname, 'r') as r_file:
            r_file.readline()
            nw = int(r_file.readline().strip())
            nR = int(r_file.readline().strip())

            R = np.zeros((nR, 3), dtype='int64')
            r_Ramn = np.zeros((nR, 3, nw, nw), dtype='complex128')
            for nrpts_i in range(nR):

                for m in range(nw):
                    for n in range(nw):
                        inline = r_file.readline().split()
                        inline_m = int(inline[3]) - 1
                        inline_n = int(inline[4]) - 1
                        r_Ramn[nrpts_i, 0, inline_m, inline_n] = np.complex(float(inline[5]), float(inline[6]))
                        r_Ramn[nrpts_i, 1, inline_m, inline_n] = np.complex(float(inline[7]), float(inline[8]))
                        r_Ramn[nrpts_i, 2, inline_m, inline_n] = np.complex(float(inline[9]), float(inline[10]))

                R[nrpts_i] = np.array(inline[:3], dtype='int64')

        return r_Ramn

    def _read_rr_v2x(self, fname, nR):
        '''
         interface with wannier90 v2.x
        '''
        with open(fname, 'r') as r_file:
            r_file.readline()
            nw = int(r_file.readline().strip())

            R = np.zeros((nR, 3), dtype='int64')
            r_Ramn = np.zeros((nR, 3, nw, nw), dtype='complex128')
            for nrpts_i in range(nR):

                for m in range(nw):
                    for n in range(nw):
                        inline = r_file.readline().split()
                        inline_m = int(inline[3]) - 1
                        inline_n = int(inline[4]) - 1
                        r_Ramn[nrpts_i, 0, inline_m, inline_n] = np.complex(float(inline[5]), float(inline[6]))
                        r_Ramn[nrpts_i, 1, inline_m, inline_n] = np.complex(float(inline[7]), float(inline[8]))
                        r_Ramn[nrpts_i, 2, inline_m, inline_n] = np.complex(float(inline[9]), float(inline[10]))

                R[nrpts_i] = np.array(inline[:3], dtype='int64')

        return r_Ramn

    def _w90_load_wcc(self, fname, shiftincell=True):
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

        return wcc, wccf

    '''
      * operation
    '''
    def set_ndegen_ones(self):
        self.hr_Rmn = np.einsum('R,Rmn->Rmn', 1/self.ndegen, self.hr_Rmn)
        self.r_Ramn = np.einsum('R,Ramn->Ramn', 1/self.ndegen, self.r_Ramn)
        if self.D_iRmn is not None:
            self.D_iRmn = np.einsum('R,iRmn->iRmn', 1/self.ndegen, self.D_iRmn)
        self.ndegen = np.ones(self.nR, dtype='int64')
        print('[FROM Htb] htb.ndegen have seted to ones')

    def reduce_htb(self, tmin, tmin_r=-1.0, tb=False, open_boundary=-1):
        nR = self.nR
        if not np.isclose(self.ndegen, 1.0).all():
            print('*********************************************************************************')
            print('*****  [WARNING IN REDUCE HTB] np.isclose(self.ndegen, 1.0).all() = False.  *****')
            print('*********************************************************************************')

        if open_boundary in [0, 1, 2]:
            slab_remain = np.zeros(nR, dtype='int')
            for i in range(nR):
                if self.R[i, open_boundary] == 0:
                    slab_remain[i] = 1
                else:
                    slab_remain[i] = 0
        else:
            slab_remain = np.ones(nR, dtype='int')

        hr_remain = np.array([1 if np.abs(self.hr_Rmn[i]).max() > tmin else 0 for i in range(nR)])
        r_remain = np.array([1 if np.abs(self.r_Ramn[i]).max() > tmin_r else 0 for i in range(nR)])
        # if tmin_r > 0:
        #     r_remain = np.array([1 if np.abs(self.r_Ramn[i]).max() > tmin_r else 0 for i in range(nR)])
        # elif tmin_r < 0:
        #     r_remain = np.array([1 if (self.R[i] == 0).all() else 0 for i in range(nR)])

        if tb:
            r_remain = np.array([1 if (self.R[i] == 0).all() else 0 for i in range(nR)])

        hr_remain *= slab_remain
        r_remain *= slab_remain

        if (hr_remain == 1).all():
            hr_Rmn = np.array(self.hr_Rmn, dtype='complex128')
            R_hr = np.array(self.R, dtype='int64')
        else:
            hr_Rmn = np.array([self.hr_Rmn[i] for i in range(nR) if hr_remain[i]], dtype='complex128')
            R_hr = np.array([self.R[i] for i in range(nR) if hr_remain[i]], dtype='int64')

        if tb:
            for i in range(self.nR):
                if (self.R[i] == 0).all():
                    r_Ramn = np.array([[
                        np.diag(np.diag(self.r_Ramn[i, 0])),
                        np.diag(np.diag(self.r_Ramn[i, 1])),
                        np.diag(np.diag(self.r_Ramn[i, 2])),
                    ]], dtype='complex128')
            R_r = np.array([[0., 0., 0.]], dtype='int64')
        else:
            if (r_remain == 1).all():
                r_Ramn = np.array(self.r_Ramn, dtype='complex128')
                R_r = np.array(self.R, dtype='int64')
            else:
                r_Ramn = np.array([self.r_Ramn[i] for i in range(nR) if r_remain[i]], dtype='complex128')
                R_r = np.array([self.R[i] for i in range(nR) if r_remain[i]], dtype='int64')

        nR_hr = R_hr.shape[0]
        nR_r = R_r.shape[0]
        print('[reduce hr procedure] nR have reduced from {} to {}'.format(nR, nR_hr))
        print('[reduce r procedure] nR have reduced from {} to {}'.format(nR, nR_r))

        return nR_hr, nR_r, R_hr, R_r, hr_Rmn, r_Ramn

    def printer(self):
        print('*=============================================================================*')
        print('|                                     R Grid                                  |')
        print('|    number of R Grid = {:4>}                                                  |'.format(self.nR))
        print('*=============================================================================*')
        for i in range(self.nR):
            print('|{: 4}). {: 3} {: 3} {: 3}   *{:2>}  '.format(i + 1, self.R[i, 0], self.R[i, 1], self.R[i, 2], self.ndegen[i]), end='')
            if (i + 1) % 3 == 0:
                print('|')
        print('')
        print('*--------------------------------------------------------------------------------*')

    def get_dk(self, nkmesh):
        self.dk = LA.norm(self.cell.latticeG.T, axis=1) / nkmesh
        return self.dk

    def setup(self):
        if self.r_Ramn is None:
            print('[WANNING] htb.r_Ramn is None, setting proper values according to htb.wcc.')
            r_Ramn = np.zeros([self.nR, 3, self.nw, self.nw], dtype='complex128')
            r_Ramn[self.nR//2, 0] += np.diag(self.wcc.T[0])
            r_Ramn[self.nR//2, 1] += np.diag(self.wcc.T[1])
            r_Ramn[self.nR//2, 2] += np.diag(self.wcc.T[2])
            self.r_Ramn = r_Ramn

        self.set_ndegen_ones()
        self.Rc_hr = LA.multi_dot([self.latt, self.R_hr.T]).T
        self.Rc_r = LA.multi_dot([self.latt, self.R_r.T]).T

    def get_hk(self, k):
        eikr = np.exp(2j * np.pi * np.einsum('a,Ra', k, self.R)) / self.ndegen
        hk = np.einsum('R,Rmn->mn', eikr, self.hr_Rmn)
        return hk


'''
  calculated
'''
class Bandstructure(object):

    def __init__(self, nb=None, nk=None, latt=None, nmesh=None):
        self.nb = nb
        self.nk = nk

        self.latt = latt
        self.lattG = 2 * np.pi * LA.inv(latt.T)

        self.nmesh = nmesh
        self.meshk = make_mesh(nmesh, type='continuous')
        self.meshkc = LA.multi_dot([self.lattG, self.meshk.T]).T

        self.eig = None # (nk, nb)
        self.BC = 3 # (3, nk, nb)


    '''
      * Plot
    '''
    def plot_distribution(self, dist, vmax=None):
        import matplotlib.pyplot as plt

        # dist = np.sum(meshkc, axis=1)

        meshkc = self.meshkc
        nk1, nk2, nk3 = self.nmesh
        cmap = 'seismic'

        # nk, nw = bandE.shape
        # dos = np.log(dos)

        fig = plt.figure('dist', figsize=(8, 4))
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

        if vmax == None:
            vmax = np.max(np.abs(dist))
        levels = np.linspace(-vmax, vmax, 500)

        cs = ax.contourf(meshkc_2D[:, :, 0], meshkc_2D[:, :, 1], dist_2D, levels, vmax=vmax, vmin=-vmax, cmap=cmap)
        # plt.xlabel('$k_x$')
        # plt.ylabel('$k_y$')
        # plt.title('Fermi={:.4f} eV')

        cbar = plt.colorbar(cs)
        # cbar.set_label('Density of States')
        cbar.set_ticks(np.linspace(-vmax, vmax, 5))

        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()

        fig.show()


class BandstructureHSP(object):

    def __init__(self):
        self.nb = None
        self.nk = None

        self.eig = None # (nk, nb)
        self.eig_ref = None # (nk, nb)
        self.U = None
        self.HSP_list = None
        self.HSP_name = None
        self.HSP_path_frac = None
        self.HSP_path_car = None

        self.BC = 0 #  (3, nk, nb)
        self.bandprojection = None # (nw, nk, nb)
        self.surfDOS = None

    '''
      * Plot
    '''
    def plot_band(self, eemin=-3.0, eemax=3.0, unit='C', savefname='band.pdf'):
        import matplotlib.pyplot as plt

        nline = self.HSP_list.shape[0] - 1
        xlabel = self.HSP_name
        if unit.upper() == 'C':
            kpath = kmold(self.HSP_path_car)
        elif unit.upper() == 'D':
            kpath = kmold(self.HSP_path_frac)
        else:
            kpath = None

        '''
          * plot band
        '''
        fig = plt.figure('band', figsize=[4, 3], dpi=150)
        # fig = plt.figure('band')
        fig.clf()
        ax = fig.add_subplot(111)
        ax.axis([kpath.min(), kpath.max(), eemin, eemax])
        ax.axhline(0, color='k', linestyle="--", linewidth=1, zorder=101)

        ax.plot(kpath, self.eig, linewidth=1, linestyle="-", color='k', alpha=1, zorder=12)

        for i in range(1, nline):
            ax.axvline(x=kpath[i * self.nk // nline], linestyle='-', color='k', linewidth=1, alpha=1, zorder=101)

        if xlabel is not None:
            # xlabel = ['K', 'G', '-K', '-M', 'G', 'M', 'K']
            num_xlabel = len(xlabel)
            plt.xticks(kpath[np.arange(num_xlabel) * (self.nk // nline)], xlabel)

        ax.set_ylabel('Energy (eV)')

        fig.tight_layout()
        fig.show()
        # fig.savefig(savefname)

        # with open('band.dat', 'a') as f:
        #     for i in range(self.nk):
        #         f.write('{: <15.6f}'.format(kpath[i]))
        #         for j in range(self.nb):
        #             f.write('{: <15.6f}'.format(self.eig[i, j]))
        #         f.write('\n')
        # f.close()

    def plot_2band_compare(self, eemin=-3.0, eemax=3.0, unit='C', save=False, savefname='compareband.png'):
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab

        # matplotlib.rcdefaults()
        # params = {
        #     'axes.labelsize': '12',
        #     'xtick.labelsize': '12',
        #     'ytick.labelsize': '12',
        #     'legend.fontsize': '12',
        #     'xtick.direction': 'in',
        #     'ytick.direction': 'in',
        #     'ytick.minor.visible': True,
        #     'xtick.minor.visible': True,
        #     'xtick.top': True,
        #     'ytick.right': True,
        #     # 'figure.figsize': '5, 4',  # set figure size
        #     # 'figure.dpi': '100',  # set figure size
        #     'pdf.fonttype': '42',  # set figure size
        #     'font.family': 'sans-serif',  # set figure size
        #     # 'font.serif': 'Times',  # set figure size
        #     'font.sans-serif': 'Arial',  # set figure size
        # }
        # pylab.rcParams.update(params)

        nline = self.HSP_list.shape[0] - 1
        xlabel = self.HSP_name
        if unit.upper() == 'C':
            kpath = kmold(self.HSP_path_car)
        elif unit.upper() == 'D':
            kpath = kmold(self.HSP_path_frac)
        else:
            kpath = None

        '''
          * plot band
        '''
        # fig = plt.figure('compare', figsize=[4, 3], dpi=150)
        fig = plt.figure('compare')
        fig.clf()
        ax = fig.add_subplot(111)

        ax.axis([kpath.min(), kpath.max(), eemin, eemax])
        ax.axhline(0, color='k', linewidth=0.5, zorder=101)

        ax.plot(kpath, self.eig, linewidth=6, linestyle="-", color='#ff1744', alpha=0.3, zorder=12)
        ax.plot(kpath, self.eig_ref, linewidth=2, linestyle="-", color='k', alpha=1, zorder=11)

        for i in range(1, nline):
            ax.axvline(x=kpath[i * self.nk // nline], linestyle='-', color='k', linewidth=0.5, alpha=1, zorder=101)

        if xlabel is not None:
            # xlabel = ['K', 'G', '-K', '-M', 'G', 'M', 'K']
            num_xlabel = len(xlabel)
            plt.xticks(kpath[np.arange(num_xlabel) * (self.nk // nline)], xlabel)
        ax.set_ylabel('$Energy / (meV)$')

        fig.tight_layout()
        fig.show()
        fig.savefig(savefname)

    def plot_distribution_in_band(self, distri, eemin=-3.0, eemax=3.0, unit='C', S=30, vmax=None):
        import matplotlib.pyplot as plt

        # distri (nk, nb)

        if vmax == None:
            vmax = np.abs(distri).max()
        vmin = -vmax

        cmap = 'seismic'
        nline = self.HSP_list.shape[0] - 1
        xlabel = self.HSP_name
        if unit.upper() == 'C':
            kpath = kmold(self.HSP_path_car)
        elif unit.upper() == 'D':
            kpath = kmold(self.HSP_path_frac)
        else:
            kpath = None

        '''
          * plot band
        '''
        fig = plt.figure('BC dist')
        fig.clf()
        ax = fig.add_subplot(111)

        ax.axis([kpath.min(), kpath.max(), eemin, eemax])
        ax.axhline(0, color='k', linewidth=0.5, zorder=101)

        for i in range(1, nline):
            ax.axvline(x=kpath[i * self.nk // nline], linestyle='-', color='k', linewidth=0.5, alpha=1, zorder=101)

        if xlabel is not None:
            # xlabel = ['K', 'G', '-K', '-M', 'G', 'M', 'K']
            num_xlabel = len(xlabel)
            plt.xticks(kpath[np.arange(num_xlabel) * (self.nk // nline)], xlabel)
        # ax.set_ylabel('$Energy / (eV)$')

        ax.plot(kpath, self.eig, linewidth=0.5, linestyle="-", color='k', alpha=1, zorder=21)
        im = ax.scatter(np.kron(kpath, np.ones([self.nb])).T, self.eig, cmap=cmap, c=distri, s=S, alpha=1, vmin=vmin, vmax=vmax, zorder=11)
        plt.colorbar(im)

        fig.tight_layout()
        fig.show()
        # fig.savefig(savefname)

    def plot_bandprojection(self, S, eemin=-3.0, eemax=3.0, unit='D', savefname=None):
        import matplotlib.pyplot as plt

        # matplotlib.rcdefaults()
        # params = {
        #     'axes.labelsize': '12',
        #     'xtick.labelsize': '12',
        #     'ytick.labelsize': '12',
        #     'legend.fontsize': '12',
        #     'xtick.direction': 'in',
        #     'ytick.direction': 'in',
        #     'ytick.minor.visible': True,
        #     'xtick.minor.visible': True,
        #     'xtick.top': True,
        #     'ytick.right': True,
        #     # 'figure.figsize': '5, 4',  # set figure size
        #     # 'figure.dpi': '100',  # set figure size
        #     'pdf.fonttype': '42',  # set figure size
        #     'font.family': 'sans-serif',  # set figure size
        #     # 'font.serif': 'Times',  # set figure size
        #     'font.sans-serif': 'Arial',  # set figure size
        # }
        # pylab.rcParams.update(params)

        proj = np.abs(S) # (nw, nk, nb)

        vmax = np.max(proj)
        vmin = np.min(proj)

        cmap = 'Reds'
        cmap = 'seismic'
        nline = self.HSP_list.shape[0] - 1
        xlabel = self.HSP_name
        if unit.upper() == 'C':
            kpath = kmold(self.HSP_path_car)
        elif unit.upper() == 'D':
            kpath = kmold(self.HSP_path_frac)
        else:
            kpath = None

        '''
          * plot band
        '''
        # fig = plt.figure('proj band', figsize=[3, 3], dpi=150) # for FIG.S1&2
        # fig = plt.figure('proj band', figsize=[3.5, 4.5], dpi=150) # for FIG.2
        fig = plt.figure('proj band', figsize=[3.5, 3.5], dpi=150) # for FIG.3
        # fig = plt.figure('proj band')
        fig.clf()
        ax = fig.add_subplot(111)
        ax.axis([kpath.min(), kpath.max(), eemin, eemax])
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=1)

        for i in range(1, nline):
            ax.axvline(x=kpath[i * self.nk // nline], linestyle='--', color='black', linewidth=0.5, alpha=1, zorder=101)

        if xlabel is not None:
            # xlabel = ['K', 'G', '-K', '-M', 'G', 'M', 'K']
            num_xlabel = len(xlabel)
            plt.xticks(kpath[np.arange(num_xlabel) * (self.nk // nline)], xlabel)

        # S = 100 * np.sum(self.bandprjection, axis=0)
        # S = 100 * self.bandprjection[3]
        # S = 100 * np.sum(self.bandprjection[2:6], axis=0)
        ax.plot(kpath, self.eig, linewidth=1, linestyle="-", color='k', alpha=0)
        # ax.scatter(np.kron(kpath, np.ones([self.nb])).T, self.eig, color='red', s=proj[i], alpha=0.3)
        cs = ax.scatter(np.kron(kpath, np.ones([self.nb])).T, self.eig, cmap=cmap, c=proj, s=3, alpha=1, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(cs, ticks=np.linspace(vmin, vmax, 2))
        cbar.ax.set_yticklabels(['Low', 'High'])
        print('vmin={}, vmax={}'.format(vmin, vmax))

        ax.set_ylabel('$Energy / (eV)$')

        fig.tight_layout()
        fig.show()
        if savefname != None:
            fig.savefig(savefname)

    def plot_surfDOS(self, ee, surfDOS, eemin=-3.0, eemax=3.0, unit='D', cmap='seismic'):
        import matplotlib.pyplot as plt

        nk, ne = surfDOS.shape

        # cmap = 'seismic'
        nline = self.HSP_list.shape[0] - 1
        xlabel = self.HSP_name
        if unit.upper() == 'C':
            kpath = kmold(self.HSP_path_car)
        elif unit.upper() == 'D':
            kpath = kmold(self.HSP_path_frac)
        else:
            kpath = None

        '''
          * plot band
        '''
        fig = plt.figure('dos', figsize=[4, 3], dpi=150)
        fig.clf()
        ax = fig.add_subplot(111)

        ax.axis([kpath.min(), kpath.max(), eemin, eemax])
        ax.axhline(0, color='k', linewidth=0.6, linestyle='-', alpha=1, zorder=10)

        for i in range(1, nline):
            ax.axvline(x=kpath[i * nk // nline], linestyle='-', color='k', linewidth=0.6, alpha=1, zorder=10)

        if xlabel is not None:
            # xlabel = ['K', 'G', '-K', '-M', 'G', 'M', 'K']
            num_xlabel = len(xlabel)
            plt.xticks(kpath[np.arange(num_xlabel) * (nk // nline)], xlabel)

        kk = np.kron(np.ones([ne, 1]), kpath).T
        ee = np.kron(np.ones([nk, 1]), ee)
        # im = plt.contour(k, ek, a_k_ek, contourf_N, cmap=cmap, linewidth=0)
        im = ax.contourf(kk, ee, surfDOS, 100, alpha=1, cmap=cmap, zorder=0)
        fig.colorbar(im, ax=ax)

        ax.set_ylabel('Energy (eV)')

        fig.tight_layout()
        fig.show()

'''
  etc.
'''
def f_none():
    '''
      !!! Do not change the function name !!!
    ''' #
    return None

def list2str(alist):
    astr = ';'.join(alist)
    return astr

def str2list(astr):
    alist = astr.split(';')
    return alist

def hdf5_create_dataset(group, name, data, dtype, **kwds):
    if data is not None:
        group.create_dataset(name, data=data, dtype=dtype, **kwds)

def hdf5_read_dataset(group, name, default=None):
    data = group.get(name)
    if data is not None:
        return data[()]
    else:
        return default

def trans_npz_2_hdf5(seedname=r'htb'):
    htb = Htb()
    htb.load_npz(seedname+'.npz')

    cell = Cell()
    for i in ['name', 'lattice', 'latticeG', 'N', 'ions', 'ions_car']:
        cell.__dict__[i] = htb.cell.__dict__[i]
    cell.spec = htb.cell.__dict__['spec']
    htb.cell = cell

    htb.save_h5(seedname+'.h5')

def kmold(kkc):
    nk = kkc.shape[0]
    kkc = kkc[1:, :] - kkc[:-1, :]
    kkc = np.vstack([np.array([0, 0, 0]), kkc])
    k_mold = np.sqrt(np.einsum('ka,ka->k', kkc, kkc))
    k_mold = LA.multi_dot([np.tri(nk), k_mold])
    return k_mold