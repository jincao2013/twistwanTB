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
__date__ = "May. 22, 2019"

import os
import sys
import numpy as np
from numpy import linalg as LA
from twistwanTB.wanpy.structure import Cell, Htb
# from wanpy.core.greenfunc import self_energy
from twistwanTB.wanpy.units import *


'''
  * cell
'''
class Cell_tbg(Cell):

    def __init__(self, m, n, latt_graphene=None):
        #
        # m, n : moire index, theta=1.05 when (32, 33)
        # latt_graphene: lattice of graphene [a1, a2], the generated tbg lattice [A1, A2]
        #   will keep the same direction.
        #
        Cell.__init__(self)

        self.m = m
        self.n = n

        self.lattA_graphene = 2.46
        self.lattC_graphene = 3.015

        self.nsuper = m ** 2 + n ** 2 + m * n
        # self.theta = np.arccos((3 * m ** 2 + 3 * m + 1 / 2) / (3 * m ** 2 + 3 * m + 1)) / np.pi * 180
        self.theta = np.arccos(0.5 * (m**2 + n**2 + 4*m*n) / (m**2 + n**2 + m*n)) / np.pi * 180
        self.natom = self.nsuper * 4
        self.N = self.natom
        self.lattA_tbg = self.lattA_graphene * self.nsuper ** 0.5

        self.name = 'Real TBG theta:{: 6.4f}'.format(self.theta)

        if latt_graphene is None:
            latt_graphene = np.array([
                [3 ** 0.5 / 2, -0.5],
                [3 ** 0.5 / 2,  0.5],
            ]).T
        self.lattice = np.zeros([3, 3], dtype='float64')
        self.lattice[:2, :2] = self.lattA_tbg * latt_graphene
        self.lattice[2, 2] = 3 * self.lattC_graphene
        self.latticeG = self.get_latticeG()

        self.init_ions()


    def init_ions(self):
        m = self.m
        n = self.n
        nsuper = self.nsuper
        sup_u = np.array([
            [-m, n + m, 0],
            [n, m, 0],
            [0, 0, 1],
        ])
        sup_d = np.array([
            [-n, m + n, 0],
            [m, n, 0],
            [0, 0, 1],
        ])
        ions_graphene = np.array([
            [0, 0, 0],
            [1 / 3, 1 / 3, 0],
        ])
        '''
          * build Cell object for TBG 
        '''
        self.spec = ['C' for i in range(self.natom)]

        ions_u = np.kron(np.ones([nsuper, 1]), ions_graphene)
        ions_u.T[0] += np.kron(np.arange(nsuper), np.ones(2))
        ions_u = LA.multi_dot([ions_u, LA.inv(sup_u)])
        ions_u = np.remainder(ions_u, np.array([1., 1., 1.]))
        # ions_u += np.kron(np.ones([2*nsuper, 1]), np.array([0.5, 0.5, 0]))

        ions_d = np.kron(np.ones([nsuper, 1]), ions_graphene)
        ions_d.T[0] += np.kron(np.arange(nsuper), np.ones(2))
        ions_d = LA.multi_dot([ions_d, LA.inv(sup_d)])
        ions_d = np.remainder(ions_d, np.array([1., 1., 1.]))

        ions_u = LA.multi_dot([self.lattice, ions_u.T]).T
        ions_d = LA.multi_dot([self.lattice, ions_d.T]).T
        ions_u.T[2] = 2 * self.lattC_graphene
        ions_d.T[2] = 1 * self.lattC_graphene
        ions_car = np.vstack([ions_u, ions_d])

        self.ions_car = ions_car
        self.ions = self.get_ions()

        return ions_car

class Cell_twisted_hex(Cell):

    def __init__(self, layer_u, layer_d, m, n, d=3):
        #
        # m, n : moire index, theta=1.05 when (32, 33)
        # latt_graphene: lattice of graphene [a1, a2], the generated tbg lattice [A1, A2]
        #                will keep the same direction.
        #
        Cell.__init__(self)
        # layer_u = Cell()
        # layer_d = Cell()

        self.m = m
        self.n = n
        self.d = d
        self.d_layer_u = np.max(layer_u.ions_car.T[2]) - np.min(layer_u.ions_car.T[2])
        self.d_layer_d = np.max(layer_d.ions_car.T[2]) - np.min(layer_d.ions_car.T[2])

        self.natom_u = layer_u.ions.shape[0]
        self.natom_d = layer_d.ions.shape[0]
        self.lattA_hex = LA.norm(layer_d.lattice.T[0])

        self.nsuper = m ** 2 + n ** 2 + m * n
        self.theta = np.arccos((3 * m ** 2 + 3 * m + 1 / 2) / (3 * m ** 2 + 3 * m + 1)) / np.pi * 180
        self.natom = self.nsuper * (self.natom_u + self.natom_d)
        self.N = self.natom
        self.lattA_twisted_hex = self.lattA_hex * self.nsuper ** 0.5

        self.name = 'Realistic twisted hex system theta:{: 6.4f} deg'.format(self.theta)

        self.lattice = layer_u.lattice / self.lattA_hex * self.lattA_twisted_hex
        self.lattice[2, 2] = self.d_layer_u + self.d_layer_d + 3 * self.d
        self.latticeG = self.get_latticeG()

        self.init_ions(layer_u.ions, layer_d.ions)

    def init_ions(self, Rs_u, Rs_d):
        m = self.m
        n = self.n
        nsuper = self.nsuper

        sup_u = np.array([
            [-m, n + m, 0],
            [n, m, 0],
            [0, 0, 1],
        ])
        sup_d = np.array([
            [-n, m + n, 0],
            [m, n, 0],
            [0, 0, 1],
        ])
        '''
          * build Cell object  
        '''
        self.spec = ['C' for i in range(self.natom)]

        ions_u = np.kron(np.ones([nsuper, 1]), Rs_u)
        ions_u.T[0] += np.kron(np.arange(nsuper), np.ones(self.natom_u))
        ions_u = LA.multi_dot([ions_u, LA.inv(sup_u)])
        ions_u = np.remainder(ions_u, np.array([1., 1., 1.]))
        # ions_u += np.kron(np.ones([2*nsuper, 1]), np.array([0.5, 0.5, 0]))

        ions_d = np.kron(np.ones([nsuper, 1]), Rs_d)
        ions_d.T[0] += np.kron(np.arange(nsuper), np.ones(self.natom_d))
        ions_d = LA.multi_dot([ions_d, LA.inv(sup_d)])
        ions_d = np.remainder(ions_d, np.array([1., 1., 1.]))

        ions_car_u = LA.multi_dot([self.lattice, ions_u.T]).T
        ions_car_d = LA.multi_dot([self.lattice, ions_d.T]).T

        ions_car_u.T[2] += -np.min(ions_car_u.T[2]) + self.d
        ions_car_d.T[2] += -np.min(ions_car_d.T[2]) + self.d + self.d_layer_u + self.d

        ions_car = np.vstack([ions_car_u, ions_car_d])

        self.ions_car = ions_car
        self.ions = self.get_ions()

        return ions_car

'''
  * kp model for few layer graphene
'''
class FLG_KP(object):

    def __init__(self):
        self.nw = None
        self.latt = np.eye(3)
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)
        self.wcc = None
        self.wccf = None

class FLG_KP_A(FLG_KP):

    def __init__(self):
        FLG_KP.__init__(self)
        self.nw = 2
        self.grapheneA = 2.42
        self.grapheneC = 3
        self.latt = self.grapheneA * np.array([
            [1, 0, 0],
            [1/2, np.sqrt(3)/2, 0],
            [0, 0, 1],
        ]).T
        self.latt[2, 2] = self.grapheneC
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)
        self.wccf = np.array([
            [0,0,0],
            [1/3, 1/3, 0]
        ])
        self.wcc = (self.latt @ self.wccf.T).T

    def get_hk(self, kkc):
        gamma0 = 2.1354 * self.grapheneA
        gamma0 = 2.464 * self.grapheneA
        sigma_vp = np.array([sigmax, sigmay, sigmaz], dtype='complex128')
        sigma_vn = np.array([-sigmax, sigmay, sigmaz], dtype='complex128')
        # Dirac hamiltonian k*sigma
        nk = kkc.shape[0]
        h0 = -gamma0 * np.einsum('ka,amn->kmn', kkc, sigma_vp, optimize=True)
        # h0 = np.einsum('ij,jmn->imjn', np.eye(nk, dtype='complex128'), h0).reshape(self.nw * nk, self.nw * nk)
        return h0

    def get_vk(self):
        pass

class FLG_KP_AB(FLG_KP):

    def __init__(self):
        FLG_KP.__init__(self)
        self.nw = 4
        self.grapheneA = 2.42
        self.grapheneC = 3
        self.latt = self.grapheneA * np.array([
            [1, 0, 0],
            [1/2, np.sqrt(3)/2, 0],
            [0, 0, 1],
        ]).T
        self.latt[2, 2] = self.grapheneC
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)
        self.wccf = np.array([
            [0,0,0],
            [1/3, 1/3, 0],
            [0,0,0],
            [1/3, 1/3, 0]
        ])
        self.wcc = (self.latt @ self.wccf.T).T

    def get_hk(self, kkc):
        gamma0 = 2.464 * self.grapheneA
        gamma1 = 0.400
        gamma3 = 0.200
        gamma4 = 0
        delta = 0

        v3 = gamma3 * self.grapheneA * np.sqrt(3)/2
        v4 = gamma4 * self.grapheneA * np.sqrt(3)/2

        sigma_vp = np.array([sigmax, sigmay, sigmaz], dtype='complex128')
        sigma_vn = np.array([-sigmax, sigmay, sigmaz], dtype='complex128')

        def get_hk_k(kc):
            kp = kc[0] + 1j * kc[1]
            kn = kc[0] - 1j * kc[1]
            h0 = np.kron(sigma0, -gamma0 * np.einsum('a,amn->mn', kc, sigma_vp))
            gk = np.array([
                [v4 * kp, gamma1],
                [v3 * kn, v4 * kp],
            ])
            h0 += np.block([
                [np.zeros([2,2]), gk.T.conj()],
                [gk, np.zeros([2,2])]
            ])
            h0 += np.diag([0, delta, delta, 0])
            return h0

        hk = np.array([
            get_hk_k(kc)
            for kc in kkc
        ])
        return hk

    def get_vk(self):
        pass


'''
  * MacDonald model
'''
class Tbg_MacDonald_kp(object):

    # MacDonald continous model for TBG

    def __init__(self, m=31, n=32, N=3, w1=0.0797, w2=0.0975, nvalley=1, valley=1, latt_graphene=None, rotk=False):
        """
          ** Rot1, Rot2 : rotation matrix with angle (theta / 2) and (-theta / 2)
          ** m, n : moire index, theta=1.05 when (32, 33)
          ** N : decide the num of nearist gridG
          ** nG : num of hopping in SBZ
          ** nb : num of band = nG * 2(sites per lattice) * 2(layers)
          ** Gmesh : G mesh in SBZ which is taken into account for layer coupling
          ** kku, kkd : k points in coupling matrix Tkp=<k|T|p>=<kku|T|kkp>
                        belong to upper-layer and under-layer.
          ** hT : layer coupling
          ** h0 : onsite(kku and kkd) hopping
          ** gKu, gKd : K vallye in UBZ for upper-layer and under-layer.
          ** latt_graphene: lattice of graphene [a1, a2]
                            the generated tbg lattice [A1, A2] will be in the same direction.
        """ #

        self.m = m
        self.n = n
        self.N = N
        self.nvalley = nvalley
        self.valley = valley
        self.w1 = w1    # layer coupling constant
        self.w2 = w2    # layer coupling constant

        self.grapheneA = 2.46
        self.grapheneC = 3.015
        if latt_graphene is None:
            latt_slg = np.array([
                [1, 0],
                [1 / 2, np.sqrt(3) / 2],
            ]).T
        else:
            latt_slg = LA.multi_dot([np.array([[0, -1], [1, 0]]), latt_graphene])
        latt_slg = self.grapheneA * latt_slg
        latt_slg3 = np.zeros([3, 3], dtype='float64')
        latt_slg3[:2, :2] = latt_slg
        latt_slg3[2, 2] = self.grapheneC * 3

        self.nsuper = m ** 2 + n ** 2 + m * n
        self.theta_rad = np.arccos((3 * m ** 2 + 3 * m + 1 / 2) / (3 * m ** 2 + 3 * m + 1))
        self.theta_deg = self.theta_rad / np.pi * 180
        self.theta = self.theta_deg
        self.natom = self.nsuper * 4

        self.fermi = 0.

        self.Rot = np.identity(3, dtype='float64') # Rot +theta_rad/2 i.e. counter clock direction
        self.Rot[:2, :2] = np.array([
            [np.cos(self.theta_rad / 2), -np.sin(self.theta_rad / 2)],
            [np.sin(self.theta_rad / 2), np.cos(self.theta_rad / 2)],
        ])
        self.Rot_inv = self.Rot.T # Rot -theta_rad/2
        self.rotk = rotk

        self.latt, self.lattG = self._init_lattice(latt_slg3)
        self.gridG, self.gridGc = self._init_gridG_symm(N, self.lattG)
        self.nG = self.gridG.shape[0]
        # self.nb = 4 * self.nG
        self.nlayers = 2
        self.nw_slg = 2
        self.nb = self.nG * self.nlayers * self.nw_slg * self.nvalley
        self.lattA_tbg = self.nsuper**0.5 * self.grapheneA

        self.hT = self._init_layer_coupling(self.gridG)

        # self.vw = self.get_vw()

    def _init_lattice(self, latt_slg):
        """
          * Lattice
          ** gKu, gKd : K valley in UBZ of graphene for upper layer(u) and underlayer layer(d)
        """
        latt_slg_u = LA.multi_dot([self.Rot_inv, latt_slg])
        latt_slg_d = LA.multi_dot([self.Rot, latt_slg])
        lattG_slg = 2 * np.pi * LA.inv(latt_slg.T)
        lattG_slg_u = 2 * np.pi * LA.inv(latt_slg_u.T)
        lattG_slg_d = 2 * np.pi * LA.inv(latt_slg_d.T)

        lattG = lattG_slg_u - lattG_slg_d
        lattG[2, 2] = 2 * np.pi * (self.grapheneC*3)
        latt = LA.inv(lattG.T / 2 / np.pi)

        self.K1uc = -(2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        # self.K2uc = (1 * lattG_slg.T[0] + 2 * lattG_slg.T[1]) / 3
        self.K2uc = -self.K1uc

        self.Ku1 = -(1 * lattG.T[0] - 1 * lattG.T[1]) / 3
        self.Kd1 = (1 * lattG.T[0] + 2 * lattG.T[1]) / 3
        # self.Ku2 = (2 * lattG.T[0] + 1 * lattG.T[1]) / 3
        # self.Kd2 = (1 * lattG.T[0] - 1 * lattG.T[1]) / 3
        self.Ku2 = -self.Ku1
        self.Kd2 = -self.Kd1

        return latt, lattG

    def _init_gridG(self, N, lattG):
        """
          * gridG
        """
        gridG = np.array([
            [n1, n2, 0]
            for n1 in range(-N, N + 1)
            for n2 in range(-N, N + 1)
        ], dtype='float64')
        gridGc = LA.multi_dot([lattG, gridG.T]).T

        return gridG, gridGc

    def _init_gridG_symm(self, N, lattG):
        """
          * Symmetrized gridG
        """  #
        Gcut = (N + 0.001) * LA.norm(lattG.T[0])

        _gridG = np.array([
            [n1, n2, 0]
            for n1 in range(-2*N, 2*N + 1)
            for n2 in range(-2*N, 2*N + 1)
        ], dtype='float64')
        gridG_norm = LA.norm(LA.multi_dot([lattG, _gridG.T]).T, axis=1)

        is_in_WS = gridG_norm < Gcut
        nG = is_in_WS[is_in_WS].shape[0]

        gridG = np.zeros([nG, 3], dtype='float64')
        j = 0
        for g, remain in zip(_gridG, is_in_WS):
            if remain:
                gridG[j] = g
                j += 1

        gridGc = LA.multi_dot([lattG, gridG.T]).T

        return gridG, gridGc

    def _init_layer_coupling(self, gridG):
        """
          * Layer coupling
          ** T1, T2, T3 : layer coupling constant
        """  #
        nG = self.nG
        w1 = self.w1
        w2 = self.w2

        z = np.exp(2j * np.pi / 3)
        T1 = np.array([
            [w1, w2],
            [w2, w1],
        ], dtype='complex128')
        T2 = np.array([
            [w1,     w2 * z ** -1],
            [w2 * z ** 1, w1           ],
        ], dtype='complex128')
        T3 = np.array([
            [w1,            w2 * z ** 1],
            [w2 * z ** -1, w1    ],
        ], dtype='complex128')

        '''
          * valley +
        '''
        tkp1 = np.zeros([nG * 2, nG * 2])
        tkp2 = np.zeros([nG * 2, nG * 2])
        tkp3 = np.zeros([nG * 2, nG * 2])
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([1, 0, 0]), atol=1e-2).all():
                    tkp2[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([1, 1, 0]), atol=1e-2).all():
                    tkp3[i + nG, j] = 1

        hT1 = np.kron(tkp1, T1)
        hT2 = np.kron(tkp2, T2)
        hT3 = np.kron(tkp3, T3)

        hT = hT1 + hT2 + hT3
        hT_valley_p = hT + hT.T.conj()

        '''
          * valley -
        '''
        tkp1 = np.zeros([nG * 2, nG * 2])
        tkp2 = np.zeros([nG * 2, nG * 2])
        tkp3 = np.zeros([nG * 2, nG * 2])
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], -np.array([1, 0, 0]), atol=1e-2).all():
                    tkp2[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], -np.array([1, 1, 0]), atol=1e-2).all():
                    tkp3[i + nG, j] = 1

        hT1 = np.kron(tkp1, T1.conj())
        hT2 = np.kron(tkp2, T2.conj())
        hT3 = np.kron(tkp3, T3.conj())

        hT = hT1 + hT2 + hT3
        hT_valley_n = hT + hT.T.conj()


        '''
          * plot hopping
        '''
        # kkc = self.get_kkc(np.array([0, 0, 0]))
        # plt.scatter(kkc.T[0, :self.nG], kkc.T[1, :self.nG], color='red')
        # plt.scatter(kkc.T[0, self.nG:], kkc.T[1, self.nG:], color='black')
        # for i in range(self.nG*2):
        #     kci = kkc[i]
        #     for j in range(self.nG*2):
        #         kcj = kkc[j]
        #         hopping = np.vstack([kci, kcj])
        #         if tkp1[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='red')
        #         elif tkp2[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='blue')
        #         elif tkp3[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='black')
        #
        # plt.axis('equal')

        if self.nvalley == 1:
            if self.valley == 1:
                hT = hT_valley_p
            elif self.valley == -1:
                hT = hT_valley_n
        elif self.nvalley == 2:
            hT = np.zeros([nG * 8, nG * 8], dtype='complex128')
            hT[:nG*4,:nG*4] = hT_valley_p
            hT[nG*4:,nG*4:] = hT_valley_n

        return hT

    def get_kkc(self, kc):
        """
          * Calculate {ku} and {kd} which down-fold into same k in moire BZ
            (ku - Ku) - (kd - Kd) = q1 + GM
        """  #
        # kk_G = kc + self.gridGc  # orginal from G in SBZ
        # kku = kk_G - (-self.lattG.T[0] + self.lattG.T[1]) / 3     # orginal from Ku in moire BZ
        # kkd = kk_G - (self.lattG.T[0] + 2 * self.lattG.T[1]) / 3  # orginal from Kd in moire BZ

        kcs_u_vp = kc + self.gridGc - self.Ku1
        kcs_d_vp = kc + self.gridGc - self.Kd1

        kcs_u_vn = kc + self.gridGc - self.Ku2
        kcs_d_vn = kc + self.gridGc - self.Kd2

        # if self.nvalley == 2:
        #     kkc = np.vstack([kcs_u_valley1, kcs_d_valley1, kcs_u_valley2, kcs_d_valley2])
        # elif self.nvalley == 1:
        #     if self.valley == 1:
        #         kkc = np.vstack([kcs_u_valley1, kcs_d_valley1])
        #     elif self.valley == -1:
        #         kkc = np.vstack([kcs_u_valley2, kcs_d_valley2])
        #     else:
        #         kkc = None
        # else:
        #     kkc = None

        # kkc_valley_p = np.vstack([kcs_u_valley1, kcs_d_valley1])
        # kkc_valley_n = np.vstack([kcs_u_valley2, kcs_d_valley2])

        # kku = LA.multi_dot([self.Rot1, kku.T]).T
        # kkd = LA.multi_dot([self.Rot2, kkd.T]).T
        if self.rotk:
            kcs_u_vp = LA.multi_dot([self.Rot, kcs_u_vp.T]).T
            kcs_d_vp = LA.multi_dot([self.Rot_inv, kcs_d_vp.T]).T
            kcs_u_vn = LA.multi_dot([self.Rot, kcs_u_vn.T]).T
            kcs_d_vn = LA.multi_dot([self.Rot_inv, kcs_d_vn.T]).T

        return kcs_u_vp, kcs_d_vp, kcs_u_vn, kcs_d_vn

    def get_h0(self, k):
        vF = 2.1354 * self.grapheneA
        sigma_vp = np.array([sigmax, sigmay, sigmaz], dtype='complex128')
        sigma_vn = np.array([-sigmax, sigmay, sigmaz], dtype='complex128')
        kcs_u_vp, kcs_d_vp, kcs_u_vn, kcs_d_vn = self.get_kkc(k)

        # Dirac hamiltonian k*sigma
        h0_u_vp = -vF * np.einsum('ka,amn->kmn', kcs_u_vp, sigma_vp, optimize=True)
        h0_d_vp = -vF * np.einsum('ka,amn->kmn', kcs_d_vp, sigma_vp, optimize=True)
        h0_u_vn = -vF * np.einsum('ka,amn->kmn', kcs_u_vn, sigma_vn, optimize=True)
        h0_d_vn = -vF * np.einsum('ka,amn->kmn', kcs_d_vn, sigma_vn, optimize=True)

        if self.nvalley == 2:
            h0 = np.vstack([h0_u_vp, h0_d_vp, h0_u_vn, h0_d_vn])
        elif self.nvalley == 1:
            if self.valley == 1:
                h0 = np.vstack([h0_u_vp, h0_d_vp])
            elif self.valley == -1:
                h0 = np.vstack([h0_u_vn, h0_d_vn])
            elif self.valley == 2:
                h0 = np.vstack([h0_u_vp, h0_u_vn])
            elif self.valley == -2:
                h0 = np.vstack([h0_d_vp, h0_d_vn])
            else:
                h0 = None
        else:
            h0 = None

        h0 = np.einsum('ij,jmn->imjn', np.eye(self.nb//self.nw_slg, dtype='complex128'), h0).reshape(self.nb, self.nb)

        return h0

    def get_hk(self, k):
        h0 = self.get_h0(k)
        hk = h0 + self.hT
        return hk

    # def get_vw(self, k):
    #     vF = 2.1354 * self.grapheneA
    #     sigma = np.array([self.valley * sigmax, sigmay], dtype='complex128')
    #     kku, kkd = self.get_kkc(k)
    #
    #     eit1 = np.exp(0.5j * self.theta_rad)
    #     eit2 = np.exp(-0.5j * self.theta_rad)
    #     vxu = -vF * LA.multi_dot([np.diagflat([eit1, eit2]), sigma[0]])
    #     vxd = -vF * LA.multi_dot([np.diagflat([eit2, eit1]), sigma[0]])
    #     vyu = -vF * LA.multi_dot([np.diagflat([eit1, eit2]), sigma[1]])
    #     vyd = -vF * LA.multi_dot([np.diagflat([eit2, eit1]), sigma[1]])
    #
    #     # vx = np.kron(np.diagflat([np.ones(self.nG), np.zeros(self.nG)]), vxu) + \
    #     #      np.kron(np.diagflat([np.zeros(self.nG), np.ones(self.nG)]), vxd)
    #     # vy = np.kron(np.diagflat([np.ones(self.nG), np.zeros(self.nG)]), vyu) + \
    #     #      np.kron(np.diagflat([np.zeros(self.nG), np.ones(self.nG)]), vyd)
    #
    #     vx = np.kron(np.diagflat([kku.T[0], np.zeros(self.nG)]), vxu) + \
    #          np.kron(np.diagflat([np.zeros(self.nG), -kkd.T[0]]), vxd) + 1j * self.hT
    #     vy = np.kron(np.diagflat([kku.T[1], np.zeros(self.nG)]), vyu) + \
    #          np.kron(np.diagflat([np.zeros(self.nG), -kkd.T[1]]), vyd) + 1j * self.hT
    #
    #     vw = np.array([vx, vy], dtype='complex128')
    #
    #     return vw

    # def get_vk(self, k):
    #     vF = 2.1354 * self.grapheneA
    #     sigma = np.array([self.valley * sigma_x, sigma_y], dtype='complex128')
    #     kku, kkd = self.get_kk(k)
    #
    #     # Dirac hamiltonian k*sigma
    #     vu_x = -vF * np.einsum('k,mn->kmn', kku[:, 0], sigma[0])
    #     vu_y = -vF * np.einsum('k,mn->kmn', kku[:, 1], sigma[1])
    #     vd_x = -vF * np.einsum('k,mn->kmn', kkd[:, 0], sigma[0])
    #     vd_y = -vF * np.einsum('k,mn->kmn', kkd[:, 1], sigma[1])
    #
    #     vx = np.vstack([vu_x, vd_x])
    #     vy = np.vstack([vu_y, vd_y])
    #     vx = np.einsum('ij,jmn->imjn', np.eye(self.nG * 2, dtype='complex128'), vx).reshape(self.nb, self.nb)
    #     vy = np.einsum('ij,jmn->imjn', np.eye(self.nG * 2, dtype='complex128'), vy).reshape(self.nb, self.nb)
    #
    #     vk = np.array([vx, vy], dtype='complex128')
    #     return vk

    def w90gethk(self, kc):
        return self.get_hk(kc)

    '''
      View in real space
    '''
    def get_bloch_sum_v1_in_Rspace(self, kc, rr):
        if self.nvalley != 1:
            sys.exit(1)
        nrr = rr.shape[0]

        K1uc = 0  # self.K1uc
        K2uc = 0  # self.K2uc

        gridGc_u = self.gridGc - self.Ku1 + K1uc
        gridGc_d = self.gridGc - self.Kd1 + K1uc

        eikR = np.exp(1j * np.einsum('a,Ra->R', kc, rr))
        eiGR_u = np.exp(1j * np.einsum('Ga,Ra->GR', gridGc_u, rr))
        eiGR_d = np.exp(1j * np.einsum('Ga,Ra->GR', gridGc_d, rr))

        bloch_sum_k_rr = np.zeros([self.nb, 4, nrr], dtype='complex128')

        bloch_sum_k_rr[:self.nb//2, :2] = np.einsum('ms,R,GR->GmsR', np.eye(2), eikR, eiGR_u, optimize=True).reshape([self.nb // 2, 2, nrr])
        bloch_sum_k_rr[self.nb//2:, 2:] = np.einsum('ms,R,GR->GmsR', np.eye(2), eikR, eiGR_d, optimize=True).reshape([self.nb // 2, 2, nrr])

        return bloch_sum_k_rr

    def get_bloch_sum_in_Rspace(self, kc, rr):
        if self.nvalley != 2:
            sys.exit(1)
        nrr = rr.shape[0]

        K1uc = 0 # self.K1uc
        K2uc = 0 # self.K2uc

        gridGc_u = np.array([
            self.gridGc - self.Ku1 + K1uc,
            -self.gridGc - self.Ku2 + K2uc,
        ])
        gridGc_d = np.array([
            self.gridGc - self.Kd1 + K1uc,
            -self.gridGc - self.Kd2 + K2uc,
        ])
        eikR = np.exp(1j * np.einsum('a,Ra->R', kc, rr))
        eiGR_u = np.exp(1j * np.einsum('VGa,Ra->VGR', gridGc_u, rr))
        eiGR_d = np.exp(1j * np.einsum('VGa,Ra->VGR', gridGc_d, rr))

        bloch_sum_v1_k_rr = np.zeros([self.nb//2, 4, nrr], dtype='complex128')
        bloch_sum_v2_k_rr = np.zeros([self.nb//2, 4, nrr], dtype='complex128')
        bloch_sum_k_rr = np.zeros([self.nb, 4, nrr], dtype='complex128')

        bloch_sum_v1_k_rr[:self.nb//4, :2] = np.einsum('mn,R,GR->GmnR', np.eye(2), eikR, eiGR_u[0], optimize=True).reshape([self.nb//4, 2, nrr])
        bloch_sum_v1_k_rr[self.nb//4:, 2:] = np.einsum('mn,R,GR->GmnR', np.eye(2), eikR, eiGR_d[0], optimize=True).reshape([self.nb//4, 2, nrr])
        bloch_sum_v2_k_rr[:self.nb//4, :2] = np.einsum('mn,R,GR->GmnR', np.eye(2), eikR, eiGR_u[1], optimize=True).reshape([self.nb//4, 2, nrr])
        bloch_sum_v2_k_rr[self.nb//4:, 2:] = np.einsum('mn,R,GR->GmnR', np.eye(2), eikR, eiGR_d[1], optimize=True).reshape([self.nb//4, 2, nrr])

        bloch_sum_k_rr[:self.nb//2] = bloch_sum_v1_k_rr
        bloch_sum_k_rr[self.nb//2:] = bloch_sum_v2_k_rr

        return bloch_sum_k_rr

    def get_phi_in_Rspace(self, kc, rr):
        E, U = LA.eigh(self.get_hk(kc))
        bloch_sum_k_rr = self.get_bloch_sum_in_Rspace(kc, rr)
        psi_k_rr = np.einsum('nb,bsR->nsR', U.T, bloch_sum_k_rr)
        return psi_k_rr

    '''
      calculators
    '''
    def cal_wilson_loop(self, i1, i2, nk1=100, nk2=100):
        """
          * cal. Wilson loop W(kk2) = Int[A(k).dkk1] on closed loop
        """  #
        nb = self.nb
        nw = 4
        kk1 = np.linspace(0, 1, nk1+1)[:-1]
        # kk2 = np.linspace(0, 0.5, nk2+1)[:-1]
        kk2 = np.linspace(0, 1, nk2+1)[:-1]

        theta = np.zeros([nk2, nw], dtype='float64')
        for ik2 in range(nk2):
            print('cal wcc at ({}/{}) ky={:.3f}'.format(ik2 + 1, nk2, kk2[ik2]))
            Dky = np.identity(nb)
            for ik1 in range(nk1):
                kc = LA.multi_dot([self.lattG, np.array([kk1[ik1], kk2[ik2], 0])])
                E, U = LA.eigh(self.get_hk(kc))
                V = U[:, i1:i2]
                if ik1+1 != nk1:
                    Dky = LA.multi_dot([Dky, V, V.conj().T])
                else:
                    Dky = LA.multi_dot([V.conj().T, Dky, V])
            theta[ik2] = np.sort(np.imag(np.log(LA.eigvals(Dky))))
        theta /= np.pi * 2
        return kk2, theta


class Tbg_MacDonald_wan_debug(object):
    """
      * MacDonald continous model for TBG
        used for debug
        valley = 1, -1, 2, -2 
    """  #

    def __init__(self, m=31, n=32, N=3, w1=0.0797, w2=0.0975, nvalley=1, valley=1):
        """
          ** Rot1, Rot2 : rotation matrix with angle (theta / 2) and (-theta / 2)
          ** m, n : moire index, theta=1.05 when (32, 33)
          ** N : decide the num of nearist gridG
          ** nG : num of hopping in SBZ
          ** nb : num of band = nG * 2(sites per lattice) * 2(layers)
          ** Gmesh : G mesh in SBZ which is taken into account for layer coupling
          ** kku, kkd : k points in coupling matrix Tkp=<k|T|p>=<kku|T|kkp>
                        belong to upper-layer and under-layer.
          ** hT : layer coupling
          ** h0 : onsite(kku and kkd) hopping
          ** gKu, gKd : K vallye in UBZ for upper-layer and under-layer.
          ** latt_graphene: lattice of graphene [a1, a2]
                            the generated tbg lattice [A1, A2] will be in the same direction.
        """  #
        htb_SLG = Htb()
        htb_SLG.load_htb(r'htb.npz')
        htb_SLG.set_ndegen_ones()

        self.m = m
        self.n = n
        self.N = N
        self.w1 = w1    # layer coupling constant
        self.w2 = w2    # layer coupling constant
        self.nvalley = nvalley
        self.valley = valley
        self.nlayers = 2
        self.nw_slg = htb_SLG.nw

        self.fermi = htb_SLG.fermi

        self.grapheneA = 2.46
        self.grapheneC = 3.015

        self.wcc_slg = htb_SLG.cell.ions_car
        self.wcc_slg.T[2] = 0
        self.R_slg = htb_SLG.R
        self.Rc_slg = LA.multi_dot([htb_SLG.cell.lattice, htb_SLG.R.T]).T
        self.hr_Rmn_slg = htb_SLG.hr_Rmn
        self.r_Ramn_slg = htb_SLG.r_Ramn
        self.latt_slg = htb_SLG.cell.lattice
        self.lattG_slg = htb_SLG.cell.latticeG
        self.latt_slg[2, 2] = self.grapheneC * 3

        self.nsuper = m ** 2 + n ** 2 + m * n
        self.theta_rad = np.arccos((3 * m ** 2 + 3 * m + 1 / 2) / (3 * m ** 2 + 3 * m + 1))
        self.theta_deg = self.theta_rad / np.pi * 180
        self.theta = self.theta_deg
        self.natom = self.nsuper * 4

        self.Rot = np.identity(3, dtype='float64') # Rot +theta_rad/2 i.e. counter clock direction
        self.Rot[:2, :2] = np.array([
            [np.cos(self.theta_rad / 2), -np.sin(self.theta_rad / 2)],
            [np.sin(self.theta_rad / 2), np.cos(self.theta_rad / 2)],
        ])
        self.Rot_inv = self.Rot.T # Rot -theta_rad/2

        self.latt, self.lattG = self._init_lattice(self.latt_slg)
        self.gridG, self.gridGc = self._init_gridG_symm(N, self.lattG)
        # self.gridG, self.gridGc = self._init_gridG(N, self.lattG)
        self.nG = self.gridG.shape[0]
        self.nb = self.nG * self.nlayers * self.nw_slg * self.nvalley
        self.lattA_tbg = self.nsuper**0.5 * self.grapheneA
        self._init_sym_operators()

        self.hT = self._init_layer_coupling(self.gridG)

    def _init_lattice(self, latt_slg):
        #
        # Lattice
        # gKu, gKd : K valley in UBZ of graphene for upper layer(u) and underlayer layer(d)
        #
        latt_slg_u = LA.multi_dot([self.Rot_inv, latt_slg])
        latt_slg_d = LA.multi_dot([self.Rot, latt_slg])
        lattG_slg = 2 * np.pi * LA.inv(latt_slg.T)
        lattG_slg_u = 2 * np.pi * LA.inv(latt_slg_u.T)
        lattG_slg_d = 2 * np.pi * LA.inv(latt_slg_d.T)

        lattG = lattG_slg_u - lattG_slg_d
        lattG[2, 2] = 2 * np.pi / (self.grapheneC*3)
        latt = LA.inv(lattG.T / 2 / np.pi)

        q1 = (2 * lattG.T[0] + 1 * lattG.T[1]) / 3
        q2 = (-1 * lattG.T[0] + 1 * lattG.T[1]) / 3
        q3 = -(1 * lattG.T[0] + 2 * lattG.T[1]) / 3
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

        # self.K1uc = (2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        # self.K2uc = (1 * lattG_slg.T[0] + 2 * lattG_slg.T[1]) / 3
        # self.Ku1 = -q2
        # self.Ku2 = q1
        # self.Kd1 = q3
        # self.Kd2 = -q2

        self.K1uc = (2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        self.K2uc = -self.K1uc
        self.Ku1 = -q2
        self.Ku2 = q2
        self.Kd1 = q3
        self.Kd2 = -q3

        # self.K1uc = -(2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        # self.K2uc = -self.K1uc
        # self.Ku1 = q2
        # self.Ku2 = -q2
        # self.Kd1 = -q3
        # self.Kd2 = q3

        return latt, lattG

    def _init_gridG(self, N, lattG):
        #
        # gridG
        #
        gridG = np.array([
            [n1, n2, 0]
            for n1 in range(-N, N + 1)
            for n2 in range(-N, N + 1)
        ], dtype='float64')
        gridGc = LA.multi_dot([lattG, gridG.T]).T

        return gridG, gridGc

    def _init_gridG_symm(self, N, lattG):
        #
        # Symmetrized gridG
        #
        Gcut = (N + 0.001) * LA.norm(lattG.T[0])

        _gridG = np.array([
            [n1, n2, 0]
            for n1 in range(-2*N, 2*N + 1)
            for n2 in range(-2*N, 2*N + 1)
        ], dtype='float64')
        gridG_norm = LA.norm(LA.multi_dot([lattG, _gridG.T]).T, axis=1)

        is_in_WS = gridG_norm < Gcut
        nG = is_in_WS[is_in_WS].shape[0]

        gridG = np.zeros([nG, 3], dtype='float64')
        j = 0
        for g, remain in zip(_gridG, is_in_WS):
            if remain:
                gridG[j] = g
                j += 1

        gridGc = LA.multi_dot([lattG, gridG.T]).T

        return gridG, gridGc

    def _init_layer_coupling(self, gridG):
        """
          * Layer coupling
          ** T1, T2, T3 : layer coupling constant
        """  #
        nG = self.nG
        w1 = self.w1
        w2 = self.w2

        z = np.exp(2j * np.pi / 3)
        T1 = np.array([
            [w1, w2],
            [w2, w1],
        ], dtype='complex128')
        T2 = np.array([
            [w1,      w2 * z ** -1],
            [w2 * z , w1           ],
        ], dtype='complex128')
        T3 = np.array([
            [w1,           w2 * z],
            [w2 * z ** -1, w1    ],
        ], dtype='complex128')

        tkp1 = np.zeros([nG * 2, nG * 2])
        tkp2 = np.zeros([nG * 2, nG * 2])
        tkp3 = np.zeros([nG * 2, nG * 2])
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
                    tkp2[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
                    tkp3[i + nG, j] = 1

        hT1 = np.kron(tkp1, T1.conj())
        hT2 = np.kron(tkp2, T2.conj())
        hT3 = np.kron(tkp3, T3.conj())

        hT = hT1 + hT2 + hT3
        hT_valley1 = hT + hT.T.conj()

        tkp1 = np.zeros([nG * 2, nG * 2])
        tkp2 = np.zeros([nG * 2, nG * 2])
        tkp3 = np.zeros([nG * 2, nG * 2])

        # used for K -K
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
                    tkp2[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
                    tkp3[i + nG, j] = 1

        hT1 = np.kron(tkp1, T1)
        hT2 = np.kron(tkp2, T2)
        hT3 = np.kron(tkp3, T3)

        # # used for K1uc K2uc in unit BZ cell
        # for i in range(nG):
        #     for j in range(nG):
        #         if np.isclose(gridG[i] - gridG[j], np.array([1, 1, 0]), atol=1e-2).all():
        #             tkp1[i + nG, j] = 1
        #         elif np.isclose(gridG[i] - gridG[j], np.array([0, 1, 0]), atol=1e-2).all():
        #             tkp2[i + nG, j] = 1
        #         elif np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
        #             tkp3[i + nG, j] = 1
        # hT1 = np.kron(tkp1, T2)
        # hT2 = np.kron(tkp2, T3)
        # hT3 = np.kron(tkp3, T1)

        hT = hT1 + hT2 + hT3
        hT_valley2 = hT + hT.T.conj()

        '''
          * plot hopping
        '''
        # kkc = self.get_kkc(np.array([0, 0, 0]))
        # plt.scatter(kkc.T[0, :self.nG], kkc.T[1, :self.nG], color='red')
        # plt.scatter(kkc.T[0, self.nG:], kkc.T[1, self.nG:], color='black')
        # for i in range(self.nG*2):
        #     kci = kkc[i]
        #     for j in range(self.nG*2):
        #         kcj = kkc[j]
        #         hopping = np.vstack([kci, kcj])
        #         if tkp1[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='red')
        #         elif tkp2[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='blue')
        #         elif tkp3[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='black')
        #
        # plt.axis('equal')

        if self.nvalley == 1:
            if self.valley == 1:
                hT = hT_valley1
            elif self.valley == -1:
                hT = hT_valley2
        elif self.nvalley == 2:
            hT = np.zeros([nG * 8, nG * 8], dtype='complex128')
            hT[:nG*4,:nG*4] = hT_valley1
            hT[nG*4:,nG*4:] = hT_valley2

        return hT

    def _init_sym_operators(self):
        self.sym_T = np.kron(sigmax, np.kron(np.eye(2*self.nG), sigmax))

    def get_kkc(self, kc):
        kcs_u_valley1 = kc + self.gridGc - self.Ku1 + self.K1uc
        kcs_d_valley1 = kc + self.gridGc - self.Kd1 + self.K1uc
        kcs_u_valley2 = kc - self.gridGc - self.Ku2 + self.K2uc
        kcs_d_valley2 = kc - self.gridGc - self.Kd2 + self.K2uc

        # kcs_u_valley1 = self.K1uc + LA.multi_dot([self.Rot_inv, kcs_u_valley1.T]).T
        # kcs_d_valley1 = self.K1uc + LA.multi_dot([self.Rot, kcs_d_valley1.T]).T
        # kcs_u_valley2 = self.K2uc + LA.multi_dot([self.Rot_inv, kcs_u_valley2.T]).T
        # kcs_d_valley2 = self.K2uc + LA.multi_dot([self.Rot, kcs_d_valley2.T]).T

        if self.nvalley == 2:
            kkc = np.array([
                [kcs_u_valley1, kcs_d_valley1],
                [kcs_u_valley2, kcs_d_valley2],
            ])
        elif self.nvalley == 1:
            if self.valley == 1:
                kkc = np.vstack([kcs_u_valley1, kcs_d_valley1])
            elif self.valley == -1:
                kkc = np.vstack([kcs_u_valley2, kcs_d_valley2])
            elif self.valley == 2: # Top layer
                kkc = np.vstack([kcs_u_valley1, kcs_u_valley2])
            elif self.valley == -2: # Buttom layer
                kkc = np.vstack([kcs_d_valley1, kcs_d_valley2])
            else:
                kkc = None
        else:
            kkc = None

        return kkc

    def get_h0(self, kc):
        kkc = self.get_kkc(kc)
        # Dirac hamiltonian k*sigma
        Sk = np.exp(-1j * np.einsum('Ga,na->Gn', kkc, self.wcc_slg))
        # Sk[:, :] = 1
        eikr = np.exp(1j * np.einsum('Ga,Ra->GR', kkc, self.Rc_slg))
        h0 = np.einsum('GR,Gm,Rmn,Gn->Gmn', eikr, Sk, self.hr_Rmn_slg, Sk.conj(), optimize=True)
        h0 = np.einsum('ij,jmn->imjn', np.eye(self.nb // self.nw_slg, dtype='complex128'), h0).reshape(self.nb, self.nb)
        h0 = 0.5 * (h0 + h0.conj().T)
        return h0

    def get_hk(self, kc):
        h0 = self.get_h0(kc)
        hk = h0 + self.hT
        # if self.nvalley == 2:
        #     h02 = self.get_h0(-kc)
        #     hk2 = h02 + self.hT
        #     hk = 0.5 * (hk + LA.multi_dot([self.sym_T, hk2.conj(), self.sym_T.T]))
        return hk

    def w90gethk(self, kc):
        return self.get_hk(kc)

    def get_br(self, kc, bc):
        kkc = self.get_kkc(kc)
        Sk = np.exp(-1j * np.einsum('Ga,na->Gn', kkc, self.wcc_slg))
        # Sk[:, :] = 1
        eikr = np.exp(1j * np.einsum('Ga,na->Gn', kkc, self.Rc_slg))
        br = np.einsum('a,GR,Gm,Ramn,Gn->Gmn', bc, eikr, Sk, self.r_Ramn_slg, Sk.conj(), optimize=True)
        br = np.einsum('ij,jmn->imjn', np.eye(self.nb // self.nw_slg, dtype='complex128'), br).reshape(self.nb, self.nb)
        br = 0.5 * (br + br.conj().T)
        return br


class Tbg_MacDonald_wan(object):
    """
      * MacDonald continous model for TBG
        wannier based
        two valley included
    """  #

    def __init__(self, m=31, n=32, N=3, w1=0.0797, w2=0.0975, vac=500):
        """
          ** Rot1, Rot2 : rotation matrix with angle (theta / 2) and (-theta / 2)
          ** m, n : moire index, theta=1.05 when (32, 33)
          ** N : decide the num of nearist gridG
          ** nG : num of hopping in SBZ
          ** nb : num of band = nG * 2(sites per lattice) * 2(layers)
          ** Gmesh : G mesh in SBZ which is taken into account for layer coupling
          ** kku, kkd : k points in coupling matrix Tkp=<k|T|p>=<kku|T|kkp>
                        belong to upper-layer and under-layer.
          ** hT : layer coupling
          ** h0 : onsite(kku and kkd) hopping
          ** gKu, gKd : K vallye in UBZ for upper-layer and under-layer.
          ** latt_graphene: lattice of graphene [a1, a2]
                            the generated tbg lattice [A1, A2] will be in the same direction.
        """  #
        hL1 = Htb()
        hL1.load_htb(r'htb_SK_layer1.npz')
        hL1.set_ndegen_ones()
        hL2 = Htb()
        hL2.load_htb(r'htb_SK_layer2.npz')
        hL2.set_ndegen_ones()

        self.m = m
        self.n = n
        self.N = N
        self.w1 = w1    # layer coupling constant
        self.w2 = w2    # layer coupling constant
        self.nvalley = 2
        self.nlayers = 2
        self.nw_lg = hL1.nw

        self.fermi = hL1.fermi

        self.grapheneA = 2.46
        self.grapheneC = 3.015
        self.vac = vac

        self.wcc = np.array([hL1.wcc, hL2.wcc])

        # self.R = np.array([hL1.R, hL2.R])
        self.Rc = np.array([LA.multi_dot([hL1.cell.lattice, hL1.R.T]).T, LA.multi_dot([hL2.cell.lattice, hL2.R.T]).T])
        self.hr_Rmn = np.array([hL1.hr_Rmn, hL2.hr_Rmn])
        self.r_Ramn = np.array([hL1.r_Ramn, hL2.r_Ramn])
        self.latt_lg = hL1.cell.lattice
        self.lattG_lg = hL1.cell.latticeG

        self.nsuper = m ** 2 + n ** 2 + m * n
        self.theta_rad = np.arccos((3 * m ** 2 + 3 * m + 1 / 2) / (3 * m ** 2 + 3 * m + 1))
        self.theta_deg = self.theta_rad / np.pi * 180
        self.theta = self.theta_deg
        self.natom = self.nsuper * 4

        self.Rot = np.identity(3, dtype='float64') # Rot +theta_rad/2 i.e. counter clock direction
        self.Rot[:2, :2] = np.array([
            [np.cos(self.theta_rad / 2), -np.sin(self.theta_rad / 2)],
            [np.sin(self.theta_rad / 2), np.cos(self.theta_rad / 2)],
        ])
        self.Rot_inv = self.Rot.T # Rot -theta_rad/2

        self.latt, self.lattG = self._init_lattice(self.latt_lg)
        self.gridG, self.gridGc = self._init_gridG_symm(N, self.lattG)
        # self.gridG, self.gridGc = self._init_gridG(N, self.lattG)
        self.nG = self.gridG.shape[0]
        self.nb = self.nG * self.nlayers * self.nw_lg * self.nvalley
        self.lattA_tbg = self.nsuper**0.5 * self.grapheneA
        self._init_sym_operators()

        self.hT = self._init_layer_coupling(self.gridG)

    # def _init_wcc(self, htb):
    #     wcc_layer = np.array([htb.wcc, htb.wcc])
    #     dAB = np.max(htb.wcc.T[2]) - np.min(htb.wcc.T[2])
    #     wcc_layer[1].T[2] = wcc_layer[0].T[2] + self.grapheneC + dAB
    #     wcc = wcc_layer.reshape(8, 3)
    #     return wcc_layer, wcc

    def _init_lattice(self, latt_lg):
        """
          * Lattice
          ** gKu, gKd : K valley in UBZ of graphene for upper layer(u) and underlayer layer(d)
        """  #
        latt_lg_u = LA.multi_dot([self.Rot_inv, latt_lg])
        latt_lg_d = LA.multi_dot([self.Rot, latt_lg])
        lattG_lg = 2 * np.pi * LA.inv(latt_lg.T)
        lattG_lg_u = 2 * np.pi * LA.inv(latt_lg_u.T)
        lattG_lg_d = 2 * np.pi * LA.inv(latt_lg_d.T)

        lattG = lattG_lg_u - lattG_lg_d
        lattG[2, 2] = 2 * np.pi / self.vac
        latt = LA.inv(lattG.T / 2 / np.pi)

        q1 = (2 * lattG.T[0] + 1 * lattG.T[1]) / 3
        q2 = (-1 * lattG.T[0] + 1 * lattG.T[1]) / 3
        q3 = -(1 * lattG.T[0] + 2 * lattG.T[1]) / 3
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

        # self.K1uc = (2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        # self.K2uc = (1 * lattG_slg.T[0] + 2 * lattG_slg.T[1]) / 3
        # self.Ku1 = -q2
        # self.Ku2 = q1
        # self.Kd1 = q3
        # self.Kd2 = -q2

        self.K1uc = (2 * lattG_lg.T[0] + 1 * lattG_lg.T[1]) / 3
        self.K2uc = -self.K1uc
        self.Ku1 = -q2
        self.Ku2 = q2
        self.Kd1 = q3
        self.Kd2 = -q3

        # self.K1uc = -(2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        # self.K2uc = -self.K1uc
        # self.Ku1 = q2
        # self.Ku2 = -q2
        # self.Kd1 = -q3
        # self.Kd2 = q3

        return latt, lattG

    def _init_gridG(self, N, lattG):
        """
          * gridG
        """  #
        gridG = np.array([
            [n1, n2, 0]
            for n1 in range(-N, N + 1)
            for n2 in range(-N, N + 1)
        ], dtype='float64')
        gridGc = LA.multi_dot([lattG, gridG.T]).T

        return gridG, gridGc

    def _init_gridG_symm(self, N, lattG):
        """
          * Symmetrized gridG
        """  #
        Gcut = (N + 0.001) * LA.norm(lattG.T[0])

        _gridG = np.array([
            [n1, n2, 0]
            for n1 in range(-2*N, 2*N + 1)
            for n2 in range(-2*N, 2*N + 1)
        ], dtype='float64')
        gridG_norm = LA.norm(LA.multi_dot([lattG, _gridG.T]).T, axis=1)

        is_in_WS = gridG_norm < Gcut
        nG = is_in_WS[is_in_WS].shape[0]

        gridG = np.zeros([nG, 3], dtype='float64')
        j = 0
        for g, remain in zip(_gridG, is_in_WS):
            if remain:
                gridG[j] = g
                j += 1

        gridGc = LA.multi_dot([lattG, gridG.T]).T

        return gridG, gridGc

    def _init_layer_coupling(self, gridG):
        """
          * Layer coupling
          ** T1, T2, T3 : layer coupling constant
        """  #
        nG = self.nG
        w1 = self.w1
        w2 = self.w2

        z = np.exp(2j * np.pi / 3)
        T1 = np.array([
            [w1, w2],
            [w2, w1],
        ], dtype='complex128')
        T2 = np.array([
            [w1,      w2 * z ** -1],
            [w2 * z , w1           ],
        ], dtype='complex128')
        T3 = np.array([
            [w1,           w2 * z],
            [w2 * z ** -1, w1    ],
        ], dtype='complex128')

        tkp1 = np.zeros([nG * 2, nG * 2])
        tkp2 = np.zeros([nG * 2, nG * 2])
        tkp3 = np.zeros([nG * 2, nG * 2])
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
                    tkp2[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
                    tkp3[i + nG, j] = 1

        hT1 = np.kron(tkp1, T1.conj())
        hT2 = np.kron(tkp2, T2.conj())
        hT3 = np.kron(tkp3, T3.conj())

        hT = hT1 + hT2 + hT3
        hT_valley1 = hT + hT.T.conj()

        tkp1 = np.zeros([nG * 2, nG * 2])
        tkp2 = np.zeros([nG * 2, nG * 2])
        tkp3 = np.zeros([nG * 2, nG * 2])

        # used for K -K
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
                    tkp2[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
                    tkp3[i + nG, j] = 1

        hT1 = np.kron(tkp1, T1)
        hT2 = np.kron(tkp2, T2)
        hT3 = np.kron(tkp3, T3)

        # # used for K1uc K2uc in unit BZ cell
        # for i in range(nG):
        #     for j in range(nG):
        #         if np.isclose(gridG[i] - gridG[j], np.array([1, 1, 0]), atol=1e-2).all():
        #             tkp1[i + nG, j] = 1
        #         elif np.isclose(gridG[i] - gridG[j], np.array([0, 1, 0]), atol=1e-2).all():
        #             tkp2[i + nG, j] = 1
        #         elif np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
        #             tkp3[i + nG, j] = 1
        # hT1 = np.kron(tkp1, T2)
        # hT2 = np.kron(tkp2, T3)
        # hT3 = np.kron(tkp3, T1)

        hT = hT1 + hT2 + hT3
        hT_valley2 = hT + hT.T.conj()

        '''
          * plot hopping
        '''
        # kkc = self.get_kkc(np.array([0, 0, 0]))
        # plt.scatter(kkc.T[0, :self.nG], kkc.T[1, :self.nG], color='red')
        # plt.scatter(kkc.T[0, self.nG:], kkc.T[1, self.nG:], color='black')
        # for i in range(self.nG*2):
        #     kci = kkc[i]
        #     for j in range(self.nG*2):
        #         kcj = kkc[j]
        #         hopping = np.vstack([kci, kcj])
        #         if tkp1[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='red')
        #         elif tkp2[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='blue')
        #         elif tkp3[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='black')
        #
        # plt.axis('equal')

        hT = np.zeros([nG * 8, nG * 8], dtype='complex128')
        hT[:nG * 4, :nG * 4] = hT_valley1
        hT[nG * 4:, nG * 4:] = hT_valley2

        return hT

    def _init_sym_operators(self):
        self.sym_T = np.kron(sigmax, np.kron(np.eye(2*self.nG), sigmax))

    def get_kkc(self, kc):
        kcs_u_valley1 = kc + self.gridGc - self.Ku1 + self.K1uc
        kcs_d_valley1 = kc + self.gridGc - self.Kd1 + self.K1uc
        kcs_u_valley2 = kc - self.gridGc - self.Ku2 + self.K2uc
        kcs_d_valley2 = kc - self.gridGc - self.Kd2 + self.K2uc

        # kcs_u_valley1 = self.K1uc + LA.multi_dot([self.Rot_inv, kcs_u_valley1.T]).T
        # kcs_d_valley1 = self.K1uc + LA.multi_dot([self.Rot, kcs_d_valley1.T]).T
        # kcs_u_valley2 = self.K2uc + LA.multi_dot([self.Rot_inv, kcs_u_valley2.T]).T
        # kcs_d_valley2 = self.K2uc + LA.multi_dot([self.Rot, kcs_d_valley2.T]).T

        kkc = np.array([
            [kcs_u_valley1, kcs_d_valley1],
            [kcs_u_valley2, kcs_d_valley2],
        ])
        return kkc

    def get_h0(self, kc):
        """
          * Notice that we have used hybird gauge here
            i.e. tb gauge in xy plane whereas wannier gauge in z direction
            this gauge choise guarantees the wannier center in all direction 
            according to our test before in single layer graphene.  
        """  #
        kkc = self.get_kkc(kc)
        wcc = self.wcc
        wcc.T[2] = 0
        Sk = np.exp(-1j * np.einsum('VLGa,Lna->VLGn', kkc, wcc))
        eikr = np.exp(1j * np.einsum('VLGa,LRa->VLGR', kkc, self.Rc))
        h0 = np.einsum('VLGR,VLGm,LRmn,VLGn->VLGmn', eikr, Sk, self.hr_Rmn, Sk.conj(), optimize=True).reshape(
            [4 * self.nG, self.nw_lg, self.nw_lg])
        h0 = np.einsum('ij,jmn->imjn', np.eye(self.nb // self.nw_lg, dtype='complex128'), h0).reshape(self.nb, self.nb)
        h0 = 0.5 * (h0 + h0.conj().T)
        return h0

    def get_hk(self, kc):
        h0 = self.get_h0(kc)
        hk = h0 + self.hT
        return hk

    def w90gethk(self, kc):
        return self.get_hk(kc)

    # def get_br(self, kc, bc):
    #     kkc = self.get_kkc(kc)
    #     Sk = np.exp(-1j * np.einsum('VLGa,Lna->VLGn', kkc, self.wcc))
    #     eikr = np.exp(1j * np.einsum('VLGa,LRa->VLGR', kkc, self.Rc))
    #     br = np.einsum('a,VLGR,VLGm,LRamn,VLGn->VLGmn', bc, eikr, Sk, self.r_Ramn, Sk.conj(), optimize=True).reshape(
    #         [4 * self.nG, self.nw_lg, self.nw_lg])
    #     br = np.einsum('ij,jmn->imjn', np.eye(self.nb // self.nw_lg, dtype='complex128'), br).reshape(self.nb, self.nb)
    #     br = 0.5 * (br + br.conj().T)
    #     return br

    def get_basis_overlap(self, kc, bc):
        """
          * Basis overlap ~Mmn(k,k+b)=<~um,k|~un,k+b>
            from which we can calculate M(k,k+b)=U(k).H . ~M . U(k+b)
          * hybird gauge used here
        """  #
        bc[:2] = 0
        kkc = self.get_kkc(kc)
        Sk = np.exp(-1j * np.einsum('VLGa,Lna->VLGn', kkc, self.wcc))
        Sk[:, :, :, :] = 1
        eikr = np.exp(1j * np.einsum('VLGa,LRa->VLGR', kkc, self.Rc))
        br = np.einsum('a,VLGR,VLGm,LRamn,VLGn->VLGmn', bc, eikr, Sk, self.r_Ramn, Sk.conj(), optimize=True).reshape(
            [4 * self.nG, self.nw_lg, self.nw_lg])
        br = np.einsum('ij,jmn->imjn', np.eye(self.nb // self.nw_lg, dtype='complex128'), br).reshape(self.nb, self.nb)
        br = 0.5 * (br + br.conj().T)

        W, V = LA.eigh(br)
        eibr_mn = LA.multi_dot([V, np.diag(np.exp(1j * W)), V.conj().T])
        gxx = np.einsum('mn->nm', eibr_mn).conj()

        # return br, gxx
        return gxx

    def cal_wilson_loop_k(self, ks, i1, i2):
        """
          * calculae Wilson loop 
            W(C) = Int{A(k1, k2, k3)*dk[C]} 
            on closed loop C
        """  #
        Dky = np.identity(self.nb)
        nk = ks.shape[0]
        kcs = LA.multi_dot([self.lattG, ks.T]).T
        for i in range(nk):
            E, U = LA.eigh(self.get_hk(kcs[i]))
            V = U[:, i1:i2]
            if i + 1 != nk:
                Dky = LA.multi_dot([Dky, V, V.conj().T])
            else:
                Dky = LA.multi_dot([V.conj().T, Dky, V])

        return Dky

    def cal_wilson_loop(self, i1, i2, nk1=100, nk2=100):
        """
          * cal. Wilson loop W(k2) = Int{A(k1, k2)*d[k1]} on closed loop
        """  #
        nw = 4
        kk1 = np.linspace(0, 1, nk1 + 1)[:-1]
        kk2 = np.linspace(0, 1, nk2 + 1)[:-1]

        theta = np.zeros([nk2, nw], dtype='float64')
        for ik2 in range(nk2):
            print('cal wcc at ({}/{})'.format(ik2 + 1, nk2))

            ks = np.zeros([nk1, 3], dtype='float64')
            ks.T[0] = kk1
            ks.T[1] = kk2[ik2]

            Dky = self.cal_wilson_loop_k(ks, i1, i2)
            theta[ik2] = np.sort(np.imag(np.log(LA.eigvals(Dky))))

        theta /= np.pi * 2
        return kk2, theta

    # def cal_wilson_loop(self, i1, i2, nk1=100, nk2=100):
    #     '''
    #       * cal. Wilson loop W(kk2) = Int[A(k).dkk1] on closed loop
    #     ''' #
    #     nb = self.nb
    #     nw = 4
    #     # nk1 = 100
    #     # nk2 = 100
    #     kk1 = np.linspace(0, 1, nk1+1)[:-1]
    #     # kk2 = np.linspace(0, 0.5, nk2+1)[:-1]
    #     kk2 = np.linspace(0, 1, nk2+1)[:-1]
    #
    #     theta = np.zeros([nk2, nw], dtype='float64')
    #     for ik2 in range(nk2):
    #         print('cal wcc at ({}/{}) ky={:.3f}'.format(ik2 + 1, nk2, kk2[ik2]))
    #         Dky = np.identity(nb)
    #         for ik1 in range(nk1):
    #             kc = LA.multi_dot([self.lattG, np.array([kk1[ik1], kk2[ik2], 0])])
    #             E, U = LA.eigh(self.get_hk(kc))
    #             V = U[:, i1:i2]
    #             if ik1+1 != nk1:
    #                 Dky = LA.multi_dot([Dky, V, V.conj().T])
    #             else:
    #                 Dky = LA.multi_dot([V.conj().T, Dky, V])
    #         theta[ik2] = np.sort(np.imag(np.log(LA.eigvals(Dky))))
    #     theta /= np.pi * 2
    #     return kk2, theta


class TDBG_MacDonald_wan(object):
    """
      * MacDonald continous model for TDBG
        wannier based
        two valley included
    """  #

    def __init__(self, m=16, n=17, N=3, w1=0.0797, w2=0.0975, vac=500, htbfname=r'htb.npz'):
        htb = Htb()
        # htb.load_htb(r'htb_DFT_SCAN.npz')
        htb.load_htb(htbfname)
        htb.set_ndegen_ones()

        self.m = m
        self.n = n
        self.N = N
        self.w1 = w1    # layer coupling constant
        self.w2 = w2    # layer coupling constant
        self.nvalley = 2
        self.nlayers = 2
        self.nw_lg = htb.nw

        self.fermi = htb.fermi

        self.grapheneA = 2.46
        self.grapheneC = 3.015
        self.vac = vac

        self.wcc_layer, self.wcc = self._init_wcc(htb)

        # self.R = np.array([hL1.R, hL2.R])
        self.Rc = np.array([LA.multi_dot([htb.cell.lattice, htb.R.T]).T, LA.multi_dot([htb.cell.lattice, htb.R.T]).T])
        self.hr_Rmn = np.array([htb.hr_Rmn, htb.hr_Rmn])
        self.r_Ramn = self._init_r_Ramn(htb)
        self.latt_lg = htb.cell.lattice
        self.lattG_lg = htb.cell.latticeG

        self.nsuper = m ** 2 + n ** 2 + m * n
        self.theta_rad = np.arccos((3 * m ** 2 + 3 * m + 1 / 2) / (3 * m ** 2 + 3 * m + 1))
        self.theta_deg = self.theta_rad / np.pi * 180
        self.theta = self.theta_deg
        self.natom = self.nsuper * 4

        self.Rot = np.identity(3, dtype='float64') # Rot +theta_rad/2 i.e. counter clock direction
        self.Rot[:2, :2] = np.array([
            [np.cos(self.theta_rad / 2), -np.sin(self.theta_rad / 2)],
            [np.sin(self.theta_rad / 2), np.cos(self.theta_rad / 2)],
        ])
        self.Rot_inv = self.Rot.T # Rot -theta_rad/2

        self.latt, self.lattG = self._init_lattice(self.latt_lg)
        self.gridG, self.gridGc = self._init_gridG_symm(N, self.lattG)
        # self.gridG, self.gridGc = self._init_gridG(N, self.lattG)
        self.nG = self.gridG.shape[0]
        self.nb = self.nG * self.nlayers * self.nw_lg * self.nvalley
        self.lattA_tbg = self.nsuper**0.5 * self.grapheneA

        self.hT = self._init_layer_coupling(self.gridG)

    def _init_wcc(self, htb):
        wcc_layer = np.array([htb.wcc, htb.wcc])
        dAB = np.max(htb.wcc.T[2]) - np.min(htb.wcc.T[2])
        wcc_layer[1].T[2] = wcc_layer[0].T[2] + self.grapheneC + dAB
        wcc = wcc_layer.reshape(8, 3)
        return wcc_layer, wcc

    def _init_r_Ramn(self, htb):
        nR = htb.nR
        r_Ramn = htb.r_Ramn
        r_Ramn[:, 2] = 0

        r_Ramn = np.array([r_Ramn, r_Ramn])
        r_Ramn[0, nR//2, 2] = np.diag(self.wcc_layer[0].T[2])
        r_Ramn[1, nR//2, 2] = np.diag(self.wcc_layer[1].T[2])
        return r_Ramn

    def _init_lattice(self, latt_lg):
        """
          * Lattice
          ** gKu, gKd : K valley in UBZ of graphene for upper layer(u) and underlayer layer(d)
        """  #
        latt_lg_u = LA.multi_dot([self.Rot_inv, latt_lg])
        latt_lg_d = LA.multi_dot([self.Rot, latt_lg])
        lattG_lg = 2 * np.pi * LA.inv(latt_lg.T)
        lattG_lg_u = 2 * np.pi * LA.inv(latt_lg_u.T)
        lattG_lg_d = 2 * np.pi * LA.inv(latt_lg_d.T)

        lattG = lattG_lg_u - lattG_lg_d
        lattG[2, 2] = 2 * np.pi / self.vac
        latt = LA.inv(lattG.T / 2 / np.pi)

        q1 = (2 * lattG.T[0] + 1 * lattG.T[1]) / 3
        q2 = (-1 * lattG.T[0] + 1 * lattG.T[1]) / 3
        q3 = -(1 * lattG.T[0] + 2 * lattG.T[1]) / 3
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

        # self.K1uc = (2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        # self.K2uc = (1 * lattG_slg.T[0] + 2 * lattG_slg.T[1]) / 3
        # self.Ku1 = -q2
        # self.Ku2 = q1
        # self.Kd1 = q3
        # self.Kd2 = -q2

        self.K1uc = (2 * lattG_lg.T[0] + 1 * lattG_lg.T[1]) / 3
        self.K2uc = -self.K1uc
        self.Ku1 = -q2
        self.Ku2 = q2
        self.Kd1 = q3
        self.Kd2 = -q3

        # self.K1uc = -(2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        # self.K2uc = -self.K1uc
        # self.Ku1 = q2
        # self.Ku2 = -q2
        # self.Kd1 = -q3
        # self.Kd2 = q3

        return latt, lattG

    def _init_gridG_symm(self, N, lattG):
        """
          * Symmetrized gridG
        """  #
        Gcut = (N + 0.001) * LA.norm(lattG.T[0])

        _gridG = np.array([
            [n1, n2, 0]
            for n1 in range(-2*N, 2*N + 1)
            for n2 in range(-2*N, 2*N + 1)
        ], dtype='float64')
        gridG_norm = LA.norm(LA.multi_dot([lattG, _gridG.T]).T, axis=1)

        is_in_WS = gridG_norm < Gcut
        nG = is_in_WS[is_in_WS].shape[0]

        gridG = np.zeros([nG, 3], dtype='float64')
        j = 0
        for g, remain in zip(_gridG, is_in_WS):
            if remain:
                gridG[j] = g
                j += 1

        gridGc = LA.multi_dot([lattG, gridG.T]).T

        return gridG, gridGc

    def _init_layer_coupling(self, gridG):
        """
          * Layer coupling
          ** T1, T2, T3 : layer coupling constant
        """  #
        nG = self.nG
        w1 = self.w1
        w2 = self.w2

        z = np.exp(2j * np.pi / 3)
        zero = np.zeros([2, 2], dtype='complex128')
        t1 = np.array([
            [w1, w2],
            [w2, w1],
        ], dtype='complex128')
        t2 = np.array([
            [w1,      w2 * z ** -1],
            [w2 * z , w1           ],
        ], dtype='complex128')
        t3 = np.array([
            [w1,           w2 * z],
            [w2 * z ** -1, w1    ],
        ], dtype='complex128')

        T1 = np.block([
            [zero, t1],
            [zero, zero],
        ]).T
        T2 = np.block([
            [zero, t2],
            [zero, zero],
        ]).T
        T3 = np.block([
            [zero, t3],
            [zero, zero],
        ]).T

        '''
          * valley 1
        '''
        tkp1 = np.zeros([nG * 2, nG * 2])
        tkp2 = np.zeros([nG * 2, nG * 2])
        tkp3 = np.zeros([nG * 2, nG * 2])
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
                    tkp2[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
                    tkp3[i + nG, j] = 1

        hT1 = np.kron(tkp1, T1.conj())
        hT2 = np.kron(tkp2, T2.conj())
        hT3 = np.kron(tkp3, T3.conj())

        hT = hT1 + hT2 + hT3
        hT_valley1 = hT + hT.T.conj()

        '''
          * valley -1
        '''
        tkp1 = np.zeros([nG * 2, nG * 2])
        tkp2 = np.zeros([nG * 2, nG * 2])
        tkp3 = np.zeros([nG * 2, nG * 2])

        # used for K -K
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
                    tkp2[i + nG, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
                    tkp3[i + nG, j] = 1

        hT1 = np.kron(tkp1, T1)
        hT2 = np.kron(tkp2, T2)
        hT3 = np.kron(tkp3, T3)

        # # used for K1uc K2uc in unit BZ cell
        # for i in range(nG):
        #     for j in range(nG):
        #         if np.isclose(gridG[i] - gridG[j], np.array([1, 1, 0]), atol=1e-2).all():
        #             tkp1[i + nG, j] = 1
        #         elif np.isclose(gridG[i] - gridG[j], np.array([0, 1, 0]), atol=1e-2).all():
        #             tkp2[i + nG, j] = 1
        #         elif np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
        #             tkp3[i + nG, j] = 1
        # hT1 = np.kron(tkp1, T2)
        # hT2 = np.kron(tkp2, T3)
        # hT3 = np.kron(tkp3, T1)

        hT = hT1 + hT2 + hT3
        hT_valley2 = hT + hT.T.conj()

        '''
          * plot hopping
        '''
        # kkc = self.get_kkc(np.array([0, 0, 0]))
        # plt.scatter(kkc.T[0, :self.nG], kkc.T[1, :self.nG], color='red')
        # plt.scatter(kkc.T[0, self.nG:], kkc.T[1, self.nG:], color='black')
        # for i in range(self.nG*2):
        #     kci = kkc[i]
        #     for j in range(self.nG*2):
        #         kcj = kkc[j]
        #         hopping = np.vstack([kci, kcj])
        #         if tkp1[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='red')
        #         elif tkp2[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='blue')
        #         elif tkp3[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='black')
        #
        # plt.axis('equal')

        hT = np.zeros([self.nb, self.nb], dtype='complex128')
        hT[:self.nb//2, :self.nb//2] = hT_valley1
        hT[self.nb//2:, self.nb//2:] = hT_valley2

        return hT

    def get_kkc(self, kc):
        kcs_u_valley1 = kc + self.gridGc - self.Ku1 + self.K1uc
        kcs_d_valley1 = kc + self.gridGc - self.Kd1 + self.K1uc
        kcs_u_valley2 = kc - self.gridGc - self.Ku2 + self.K2uc
        kcs_d_valley2 = kc - self.gridGc - self.Kd2 + self.K2uc

        # kcs_u_valley1 = self.K1uc + LA.multi_dot([self.Rot_inv, kcs_u_valley1.T]).T
        # kcs_d_valley1 = self.K1uc + LA.multi_dot([self.Rot, kcs_d_valley1.T]).T
        # kcs_u_valley2 = self.K2uc + LA.multi_dot([self.Rot_inv, kcs_u_valley2.T]).T
        # kcs_d_valley2 = self.K2uc + LA.multi_dot([self.Rot, kcs_d_valley2.T]).T

        kkc = np.array([
            [kcs_u_valley1, kcs_d_valley1],
            [kcs_u_valley2, kcs_d_valley2],
        ])
        return kkc

    def get_graphene_AB(self, kc, valley=1):
        """
          * parameters adapted from [Koshino, PRB 99, 235406 (2019)]
        """  #
        delta_prime = 0.050 # represents the on-site potential of dimer sites with respect to nondimer sites
        gamma1 = 0.400 #  coupling between dimer sites
        gamma3 = 0.320 #  trigonal warping
        gamma4 = 0.044 #  electron-hole asymmetry

        v3 = 0.5 * 3 ** 0.5 * self.grapheneA * gamma3
        v4 = 0.5 * 3 ** 0.5 * self.grapheneA * gamma4
        vF = 2.1354 * self.grapheneA # graphene dirac fermi verlocity

        nvy = np.array([valley, 1, 1], dtype='complex128')
        k_plus = kc.dot(np.array([valley, 1j, 1]))
        k_minus = kc.dot(np.array([valley, -1j, 1]))

        delta_prime_11 = np.array([[delta_prime, 0], [0, 0]], dtype='complex128')
        delta_prime_22 = np.array([[0, 0], [0, delta_prime]], dtype='complex128')

        sigma = np.array([sigmax, sigmay, sigmaz])
        h_dirac = -vF * np.einsum('a,a,amn->mn', nvy, kc, sigma, optimize=True)

        couple_AB = np.array([
            [v4 * k_plus,  gamma1],
            [v3 * k_minus, v4 * k_plus]
        ])

        # couple_AB[:,:] = 0

        h0 = np.block([
            [h_dirac+delta_prime_22,  couple_AB.T.conj()],
            [couple_AB, h_dirac+delta_prime_11],
        ])

        return h0

    def get_h0_kp(self, kc):
        self.fermi = 0

        kcs_u_vp = kc + self.gridGc - self.Ku1
        kcs_d_vp = kc + self.gridGc - self.Kd1
        kcs_u_vn = kc - self.gridGc - self.Ku2
        kcs_d_vn = kc - self.gridGc - self.Kd2

        # Dirac hamiltonian k*sigma
        h0_u_vp = np.array([self.get_graphene_AB(_kc, valley=1) for _kc in kcs_u_vp])
        h0_d_vp = np.array([self.get_graphene_AB(_kc, valley=1) for _kc in kcs_d_vp])
        h0_u_vn = np.array([self.get_graphene_AB(_kc, valley=-1) for _kc in kcs_u_vn])
        h0_d_vn = np.array([self.get_graphene_AB(_kc, valley=-1) for _kc in kcs_d_vn])

        h0 = np.vstack([h0_u_vp, h0_d_vp, h0_u_vn, h0_d_vn])
        h0 = np.einsum('ij,jmn->imjn', np.eye(self.nb//self.nw_lg, dtype='complex128'), h0).reshape(self.nb, self.nb)

        return h0

    def get_h0(self, kc):
        """
          * Notice that we have used hybird gauge here
            i.e. tb gauge in xy plane whereas wannier gauge in z direction
            this gauge choise guarantees the wannier center in all direction
            according to our test before in single layer graphene.
        """  #
        kkc = self.get_kkc(kc)
        # wcc = self.wcc
        # wcc.T[2] = 0
        Sk = np.exp(-1j * np.einsum('VLGa,Lna->VLGn', kkc, self.wcc_layer))
        eikr = np.exp(1j * np.einsum('VLGa,LRa->VLGR', kkc, self.Rc))
        h0 = np.einsum('VLGR,VLGm,LRmn,VLGn->VLGmn', eikr, Sk, self.hr_Rmn, Sk.conj(), optimize=True).reshape(
            [4 * self.nG, self.nw_lg, self.nw_lg])
        h0 = np.einsum('ij,jmn->imjn', np.eye(self.nb // self.nw_lg, dtype='complex128'), h0).reshape(self.nb, self.nb)
        h0 = 0.5 * (h0 + h0.conj().T)
        return h0

    def get_hk(self, kc):
        # h0 = self.get_h0_kp(kc)
        h0 = self.get_h0(kc)
        hk = h0 + self.hT
        return hk

    def w90gethk(self, kc):
        return self.get_hk(kc)

    def get_basis_overlap(self, kc, bc):
        """
          * Basis overlap ~Mmn(k,k+b)=<~um,k|~un,k+b>
            from which we can calculate M(k,k+b)=U(k).H . ~M . U(k+b)
          * hybird gauge used here
        """  #
        bc[:2] = 0
        kkc = self.get_kkc(kc)
        Sk = np.exp(-1j * np.einsum('VLGa,Lna->VLGn', kkc, self.wcc_layer))
        Sk[:, :, :, :] = 1
        eikr = np.exp(1j * np.einsum('VLGa,LRa->VLGR', kkc, self.Rc))
        br = np.einsum('a,VLGR,VLGm,LRamn,VLGn->VLGmn', bc, eikr, Sk, self.r_Ramn, Sk.conj(), optimize=True).reshape(
            [4 * self.nG, self.nw_lg, self.nw_lg])
        br = np.einsum('ij,jmn->imjn', np.eye(self.nb // self.nw_lg, dtype='complex128'), br).reshape(self.nb, self.nb)
        br = 0.5 * (br + br.conj().T)

        W, V = LA.eigh(br)
        eibr_mn = LA.multi_dot([V, np.diag(np.exp(1j * W)), V.conj().T])
        gxx = np.einsum('mn->nm', eibr_mn).conj()

        # return br, gxx
        return gxx

    def cal_wilson_loop_k(self, ks, i1, i2):
        """
          * calculae Wilson loop 
            W(C) = Int{A(k1, k2, k3)*dk[C]} 
            on closed loop C
        """  #
        Dky = np.identity(self.nb)
        nk = ks.shape[0]
        kcs = LA.multi_dot([self.lattG, ks.T]).T
        for i in range(nk):
            E, U = LA.eigh(self.get_hk(kcs[i]))
            V = U[:, i1:i2]
            if i + 1 != nk:
                Dky = LA.multi_dot([Dky, V, V.conj().T])
            else:
                Dky = LA.multi_dot([V.conj().T, Dky, V])

        return Dky

    def cal_wilson_loop(self, i1, i2, nk1=100, nk2=100):
        """
          * cal. Wilson loop W(k2) = Int{A(k1, k2)*d[k1]} on closed loop
        """  #
        nw = 4
        kk1 = np.linspace(0, 1, nk1 + 1)[:-1]
        kk2 = np.linspace(0, 1, nk2 + 1)[:-1]

        theta = np.zeros([nk2, nw], dtype='float64')
        for ik2 in range(nk2):
            print('cal wcc at ({}/{})'.format(ik2 + 1, nk2))

            ks = np.zeros([nk1, 3], dtype='float64')
            ks.T[0] = kk1
            ks.T[1] = kk2[ik2]

            Dky = self.cal_wilson_loop_k(ks, i1, i2)
            theta[ik2] = np.sort(np.imag(np.log(LA.eigvals(Dky))))

        theta /= np.pi * 2
        return kk2, theta


# class MacDonald_TMG_kp(object):
#     '''
#       * MacDonald continous model for TMG
#         * Stacking htb_u and htb_d along z direction with a twist angle
#         * wannier kernal
#         * two valley included
#       * Notice that
#         * htb_u and htb_d should have same lattice, also the fermi level should set at 0eV.
#         * wannier orbitals should arange along stacking direction, otherwise tLu tLd should be specified.
#       * example
#         twisted double bilayer graphene(TDBG) with ABBA stacking, (x) indicate sequence of wannier orbital
#         [htb_u]:
#            ---pz(3)---pz(4)---- A
#            ---pz(1)---pz(2)---- B # tLu = 0
#         [htb_d]:
#            ---pz(3)---pz(4)---- B # tLd = -1
#            ---pz(1)---pz(2)---- A
#
#     ''' #
#
#     def __init__(self, kpu, kpd, m=16, n=17, N=3, w1=0.0797, w2=0.0975, tLu=0, tLd=-1, vac=500, Umax=0, rotk=False):
#         self.m = m
#         self.n = n
#         self.N = N
#         self.w1 = w1    # layer coupling constant
#         self.w2 = w2    # layer coupling constant
#         self.tLu = tLu  # twist coupling added at tLu th graphene layer in upper(Top part) htb
#         self.tLd = tLd  # twist coupling added at tLd th graphene layer in down(Bottom part) htb
#         self.nlayers = 2
#         self.rotk = rotk
#         self.Umax = Umax
#
#         self.fermi = 0
#
#         self.grapheneA = 2.46
#         self.grapheneC = 3.015
#         self.vac = vac
#
#         self.kpu = kpu
#         self.kpd = kpd
#         self.nw_u = kpu.nw
#         self.nw_d = kpd.nw
#         self.nw = self.nw_u + self.nw_d
#         # self.htb_u, self.htb_d = htb_u, htb_d
#         # self.htb_u.set_ndegen_ones()
#         # self.htb_d.set_ndegen_ones()
#         # self.nw_u = self.htb_u.nw
#         # self.nw_d = self.htb_d.nw
#         # self.nw = self.nw_u + self.nw_d
#
#         # self.latt_lg = self.htb_u.cell.lattice
#         # self.lattG_lg = self.htb_u.cell.latticeG
#
#         self.nsuper = m ** 2 + n ** 2 + m * n
#         self.theta_rad = np.arccos((3 * m ** 2 + 3 * m + 1 / 2) / (3 * m ** 2 + 3 * m + 1))
#         self.theta_deg = self.theta_rad / np.pi * 180
#         self.theta = self.theta_deg
#         self.natom = self.nsuper * 4
#
#         self.Rot = np.identity(3, dtype='float64') # Rot +theta_rad/2 i.e. counter clock direction
#         self.Rot[:2, :2] = np.array([
#             [np.cos(self.theta_rad / 2), -np.sin(self.theta_rad / 2)],
#             [np.sin(self.theta_rad / 2), np.cos(self.theta_rad / 2)],
#         ])
#         self.Rot_inv = self.Rot.T # Rot -theta_rad/2
#
#         self.latt_lg = self.grapheneA * np.array([
#             [1, 0, 0],
#             [1/2, np.sqrt(3)/2, 0],
#             [0, 0, 1],
#         ]).T
#         self.latt_lg[2,2] = self.grapheneC
#         self.latt, self.lattG = self._init_lattice(self.latt_lg)
#         self.gridG, self.gridGc = self._init_gridG_symm(N, self.lattG)
#         # self.gridG, self.gridGc = self._init_gridG(N, self.lattG)
#         self.nG = self.gridG.shape[0]
#
#         self.nb_u = self.nG * self.nw_u
#         self.nb_d = self.nG * self.nw_d
#         self.nb = self.nb_u + self.nb_d
#         self.lattA_tbg = self.nsuper**0.5 * self.grapheneA
#
#         self._init_symmetric_operators()
#         self.wcc = self._init_wcc()
#
#         self.hT = self._init_layer_coupling(self.gridG)
#         self.hD = self.get_displacement_field(self.Umax)
#         self.hBN = 0 # self.get_hBN()
#
#     def _init_wcc(self):
#         wcc = np.vstack([
#             np.kron(np.ones([self.nG]), self.kpu.wcc.T).T, # for valley +, top parts
#             np.kron(np.ones([self.nG]), self.kpd.wcc.T).T, # for valley +, bottom parts
#         ])
#         return wcc
#
#     def _init_lattice(self, latt_lg):
#         print('[FROM MacDonald_TMG_wan] inintal lattice for twisted system')
#         '''
#           * Lattice
#           ** gKu, gKd : K valley in UBZ of graphene for upper layer(u) and underlayer layer(d)
#         ''' #
#         latt_lg_u = LA.multi_dot([self.Rot_inv, latt_lg])
#         latt_lg_d = LA.multi_dot([self.Rot, latt_lg])
#         lattG_lg = 2 * np.pi * LA.inv(latt_lg.T)
#         lattG_lg_u = 2 * np.pi * LA.inv(latt_lg_u.T)
#         lattG_lg_d = 2 * np.pi * LA.inv(latt_lg_d.T)
#
#         lattG = lattG_lg_u - lattG_lg_d
#         lattG[2, 2] = 2 * np.pi / self.vac
#         latt = LA.inv(lattG.T / 2 / np.pi)
#
#         q1 = (2 * lattG.T[0] + 1 * lattG.T[1]) / 3
#         q2 = (-1 * lattG.T[0] + 1 * lattG.T[1]) / 3
#         q3 = -(1 * lattG.T[0] + 2 * lattG.T[1]) / 3
#         self.q1 = q1
#         self.q2 = q2
#         self.q3 = q3
#
#         # self.K1uc = (2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
#         # self.K2uc = (1 * lattG_slg.T[0] + 2 * lattG_slg.T[1]) / 3
#         # self.Ku1 = -q2
#         # self.Ku2 = q1
#         # self.Kd1 = q3
#         # self.Kd2 = -q2
#
#         self.K1uc = (2 * lattG_lg.T[0] + 1 * lattG_lg.T[1]) / 3
#         self.K2uc = -self.K1uc
#         self.Ku1 = -q2
#         self.Ku2 = q2
#         self.Kd1 = q3
#         self.Kd2 = -q3
#
#         # self.K1uc = -(2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
#         # self.K2uc = -self.K1uc
#         # self.Ku1 = q2
#         # self.Ku2 = -q2
#         # self.Kd1 = -q3
#         # self.Kd2 = q3
#
#         return latt, lattG
#
#     def _init_gridG_symm(self, N, lattG):
#         print('[FROM MacDonald_TMG_wan] inintal gridG for twisted system')
#         '''
#           * Symmetrized gridG
#         ''' #
#         Gcut = (N + 0.001) * LA.norm(lattG.T[0])
#
#         _gridG = np.array([
#             [n1, n2, 0]
#             for n1 in range(-2*N, 2*N + 1)
#             for n2 in range(-2*N, 2*N + 1)
#         ], dtype='float64')
#         gridG_norm = LA.norm(LA.multi_dot([lattG, _gridG.T]).T, axis=1)
#
#         is_in_WS = gridG_norm < Gcut
#         nG = is_in_WS[is_in_WS].shape[0]
#
#         gridG = np.zeros([nG, 3], dtype='float64')
#         j = 0
#         for g, remain in zip(_gridG, is_in_WS):
#             if remain:
#                 gridG[j] = g
#                 j += 1
#
#         gridGc = LA.multi_dot([lattG, gridG.T]).T
#
#         return gridG, gridGc
#
#     def _init_layer_coupling(self, gridG):
#         print('[FROM MacDonald_TMG_wan] inintal layer coupling for twisted system')
#         '''
#           * Layer coupling
#           ** T1, T2, T3 : layer coupling constant
#         ''' #
#         nG = self.nG
#         w1 = self.w1
#         w2 = self.w2
#
#         z = np.exp(2j * np.pi / 3)
#         t1 = np.array([
#             [w1, w2],
#             [w2, w1],
#         ], dtype='complex128')
#         t2 = np.array([
#             [w1,      w2 * z ** -1],
#             [w2 * z , w1           ],
#         ], dtype='complex128')
#         t3 = np.array([
#             [w1,           w2 * z],
#             [w2 * z ** -1, w1    ],
#         ], dtype='complex128')
#         TT1 = np.zeros([self.nw_d//2, self.nw_u//2], dtype='complex128')
#         TT2 = np.zeros([self.nw_d//2, self.nw_u//2], dtype='complex128')
#         TT3 = np.zeros([self.nw_d//2, self.nw_u//2], dtype='complex128')
#         TT1[self.tLd, self.tLu] = 1.0
#         TT2[self.tLd, self.tLu] = 1.0
#         TT3[self.tLd, self.tLu] = 1.0
#         T1 = np.kron(TT1, t1)
#         T2 = np.kron(TT2, t2)
#         T3 = np.kron(TT3, t3)
#
#         '''
#           * valley 1
#         '''
#         tkp1 = np.zeros([nG, nG])
#         tkp2 = np.zeros([nG, nG])
#         tkp3 = np.zeros([nG, nG])
#         for i in range(nG):
#             for j in range(nG):
#                 if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
#                     tkp1[i, j] = 1
#                 elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
#                     tkp2[i, j] = 1
#                 elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
#                     tkp3[i, j] = 1
#
#         hT1 = np.kron(tkp1, T1.conj())
#         hT2 = np.kron(tkp2, T2.conj())
#         hT3 = np.kron(tkp3, T3.conj())
#
#         hT = hT1 + hT2 + hT3
#         hT_valley1 = np.block([
#             [np.zeros([self.nb_u, self.nb_u]), hT.T.conj()],
#             [hT, np.zeros([self.nb_d, self.nb_d])],
#         ])
#
#         '''
#           * valley -1
#         '''
#         tkp1 = np.zeros([nG, nG])
#         tkp2 = np.zeros([nG, nG])
#         tkp3 = np.zeros([nG, nG])
#
#         # used for K -K
#         for i in range(nG):
#             for j in range(nG):
#                 if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
#                     tkp1[i, j] = 1
#                 elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
#                     tkp2[i, j] = 1
#                 elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
#                     tkp3[i, j] = 1
#
#         hT1 = np.kron(tkp1, T1)
#         hT2 = np.kron(tkp2, T2)
#         hT3 = np.kron(tkp3, T3)
#
#         hT = hT1 + hT2 + hT3
#         hT_valley2 = np.block([
#             [np.zeros([self.nb_u, self.nb_u]), hT.T.conj()],
#             [hT, np.zeros([self.nb_d, self.nb_d])],
#         ])
#
#         # '''
#         #   * plot hopping
#         # '''
#         # kkc = self.get_kkc(np.array([0, 0, 0]))[2:]
#         # plt.scatter(kkc[0].T[0], kkc[0].T[1], color='red')
#         # plt.scatter(kkc[1].T[0], kkc[1].T[1], color='black')
#         # for i in range(self.nG):
#         #     kci = kkc[1, i]
#         #     for j in range(self.nG):
#         #         kcj = kkc[0, j]
#         #         hopping = np.vstack([kci, kcj])
#         #         if tkp1[i, j] == 1:
#         #             plt.plot(hopping.T[0], hopping.T[1], color='red')
#         #         elif tkp2[i, j] == 1:
#         #             plt.plot(hopping.T[0], hopping.T[1], color='blue')
#         #         elif tkp3[i, j] == 1:
#         #             plt.plot(hopping.T[0], hopping.T[1], color='black')
#         #
#         # plt.axis('equal')
#
#         hT = hT_valley2
#
#         return hT
#
#     def _init_symmetric_operators(self):
#         nb = self.nb
#         self.op_v1_proj = np.zeros([nb, nb])
#         self.op_v2_proj = np.zeros([nb, nb])
#
#         self.op_v1_proj[:nb // 2, :nb // 2] = np.identity(nb // 2)
#         self.op_v2_proj[nb // 2:, nb // 2:] = -np.identity(nb // 2)
#
#         self.op_valley = self.op_v1_proj + self.op_v2_proj
#
#     def get_kkc(self, kc):
#         kcs_v1_u = kc + self.gridGc - self.Ku1
#         kcs_v1_d = kc + self.gridGc - self.Kd1
#         kcs_v2_u = kc - self.gridGc - self.Ku2
#         kcs_v2_d = kc - self.gridGc - self.Kd2
#
#         if self.rotk:
#             kcs_v1_u = LA.multi_dot([self.Rot, kcs_v1_u.T]).T
#             kcs_v1_d = LA.multi_dot([self.Rot_inv, kcs_v1_d.T]).T
#             kcs_v2_u = LA.multi_dot([self.Rot, kcs_v2_u.T]).T
#             kcs_v2_d = LA.multi_dot([self.Rot_inv, kcs_v2_d.T]).T
#
#         # kcs_v1_u = kcs_v1_u + self.K1uc
#         # kcs_v1_d = kcs_v1_d + self.K1uc
#         # kcs_v2_u = kcs_v2_u + self.K2uc
#         # kcs_v2_d = kcs_v2_d + self.K2uc
#
#         kcs = np.array([kcs_v1_u, kcs_v1_d, kcs_v2_u, kcs_v2_d], dtype='float64')
#         return kcs
#
#     def get_h0(self, kc):
#         '''
#           * Notice that we have used hybird gauge here
#             i.e. tb gauge in xy plane whereas wannier gauge in z direction
#             this gauge choise guarantees the wannier center in all direction
#             according to our test before in single layer graphene.
#         ''' #
#         kcs_v1_u, kcs_v1_d, kcs_v2_u, kcs_v2_d = self.get_kkc(kc)
#
#         h0_u = self.kpu.get_hk(kcs_v1_u)
#         h0_d = self.kpd.get_hk(kcs_v1_d)
#         h0_u = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), h0_u).reshape(self.nw_u * self.nG, self.nw_u * self.nG)
#         h0_d = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), h0_d).reshape(self.nw_d * self.nG, self.nw_d * self.nG)
#         h0 = np.block([
#             [h0_u, np.zeros([self.nb_u, self.nb_d])],
#             [np.zeros([self.nb_u, self.nb_d]).T, h0_d],
#         ])
#
#         h0 = 0.5 * (h0 + h0.conj().T)
#         return h0
#
#     def get_displacement_field(self, Umax):
#         nlayer = self.nw // 2
#         # U = Umax * np.kron(np.ones([2]), np.linspace(0, 1, nlayer))
#         # U = Umax * np.kron(np.linspace(0, 1, nlayer), np.ones([2]))
#         U = Umax * np.kron(np.linspace(-0.5, 0.5, nlayer), np.ones([2]))
#
#         hD_u_flg = U[self.nw_d:]
#         hD_d_flg = U[:self.nw_d]
#
#         hD = np.zeros([self.nb])
#         hD[:self.nb_u] = np.kron(np.ones([self.nG]), hD_u_flg)
#         hD[self.nb_u:] = np.kron(np.ones([self.nG]), hD_d_flg)
#
#         hD = np.diag(hD)
#         return hD
#
#     def get_hBN(self, U_BN_u, U_BN_d, enable=False):
#         '''
#           * To active the h-BN substrate effect
#             manually change `U_BN_u` and `U_BN_d`
#         '''
#         if not enable:
#             hBN = 0
#             return hBN
#         else:
#             # # fot mTBG
#             # U_BN_u = np.array([0])
#             # U_BN_d = np.array([0.03])
#             # fot mTTG
#             # U_BN_u = np.array([-0.0, -0.0])
#             # U_BN_d = np.array([0.05])
#
#             U_BN_u_flg = np.kron(U_BN_u, np.array([1, -1]))
#             U_BN_d_flg = np.kron(U_BN_d, np.array([1, -1]))
#
#             hBN = np.zeros([self.nb])
#             hBN[:self.nb_u] = np.kron(np.ones([self.nG]), U_BN_u_flg)
#             hBN[self.nb_u:] = np.kron(np.ones([self.nG]), U_BN_d_flg)
#
#             hBN = np.diag(hBN)
#             return hBN
#
#     def get_hk(self, kc):
#         '''
#           * The basis for contunuum hamiltonian in order of:
#             valley, twist u/d part, nG, nw_bottom_few_layer_graphene/nw_top_few_layer_graphene
#             nb = nvalley * (nb_u + nb_d)
#             nb_u = nG * nw_u
#             nb_d = nG * nw_d
#
#             h0 + hT =
#             [h0_{T,valley+} T.H            0                0             ]
#             [T              h0_{B,valley+} 0                0             ]
#             [0              0              h0_{T,valley-}   T.H           ]
#             [0              0              T                H0_{B,valley-}]
#
#             h0_{T,valley+} =
#             [h0(kc + K1uc + Ku1 + G1) 0                        . ]
#             [0                        h0(kc + K1uc + Ku1 + G2) . ]
#             [.                        .                        . ]
#
#             h0 = k.sigma for TBG
#
#             For instance of TDBG, in order of
#             |1,A> |1,B> |2,A> |2,B> for Top parts
#             |-2,A> |-2,B> |-1,A> |-1,B> for Bottom parts
#
#             hD is displacement field
#             hBN is the effect from the h-BN substrate
#
#         '''
#         h0 = self.get_h0(kc)
#         hk = h0 + self.hT + self.hD + self.hBN
#         return hk
#
#     # def w90gethk(self, kc):
#     #     return self.get_hk(kc)
#     #
#     # def get_eigh(self, H):
#     #     nb_per_valley = self.nb//2
#     #     h1 = H[:nb_per_valley, :nb_per_valley]
#     #     h2 = H[nb_per_valley:, nb_per_valley:]
#     #     e1, v1 = LA.eigh(h1)
#     #     e2, v2 = LA.eigh(h2)
#     #     E = np.hstack([e1, e2])
#     #     U = np.block([
#     #         [v1, np.zeros([nb_per_valley, nb_per_valley])],
#     #         [np.zeros([nb_per_valley, nb_per_valley]), v2],
#     #     ])
#     #     # U *= np.exp(-1j * np.angle(U)[0])
#     #     return E, U
#
#     def get_br_V_L(self, kkc, bc, htb):
#         '''
#           * br for each valley and layer
#         ''' #
#         Sk = np.exp(-1j * np.einsum('Ga,na->Gn', kkc, htb.wcc))
#         # Sk[:, :] = 1
#         eikr = np.exp(1j * np.einsum('Ga,Ra->GR', kkc, htb.Rc))
#         br = np.einsum('a,GR,Gm,Ramn,Gn->Gmn', bc, eikr, Sk, htb.r_Ramn, Sk.conj(), optimize=True).reshape([self.nG, htb.nw, htb.nw])
#         br = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), br).reshape(htb.nw * self.nG, htb.nw * self.nG)
#         return br
#
#     '''
#       * calculators
#     '''
#     def get_vtbk_V_L(self, kkc, htb):
#         '''
#           * get v^tb(k), from which we can get v(k) = U* v^tb(k) U
#             v^tb(k) = dH^tb(k) + i[H^tb(k), A^tb(k)],
#             where we take A^tb(k) = 0
#             dH^tb(k) = sum_{R} [e^(ik(R-tau_m+tau_n))<m0|H|nR>]
#         ''' #
#         Rmn = np.zeros([htb.nR, 3, htb.nw, htb.nw])
#         for i in range(htb.nR):
#             for a in range(3):
#                 M, N = np.meshgrid(htb.wcc.T[a], htb.wcc.T[a])
#                 Rmn[i, a] = htb.Rc[i, a] - (M - N)
#
#         eikRmn = np.exp(1j * np.einsum('Ga,Ramn->GRmn', kkc, Rmn))
#         hkk = 1j * np.einsum('Ramn,GRmn,Rmn->aGmn', Rmn, eikRmn, htb.hr_Rmn, optimize=True)
#         hkkx = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), hkk[0]).reshape(htb.nw * self.nG, htb.nw * self.nG)
#         hkky = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), hkk[1]).reshape(htb.nw * self.nG, htb.nw * self.nG)
#         hkkz = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), hkk[2]).reshape(htb.nw * self.nG, htb.nw * self.nG)
#         return hkkx, hkky, hkkz
#
#     def get_vtbk(self, kc):
#         kcs_v1_u, kcs_v1_d, kcs_v2_u, kcs_v2_d = self.get_kkc(kc)
#         def block(A, B, nA, nB):
#             C = np.zeros([nA, nB])
#             D = np.block([[A,C],[C.T,B]])
#             return D
#
#         hkkx_v1_u, hkky_v1_u, hkkz_v1_u = self.get_vtbk_V_L(kcs_v1_u, self.htb_u)
#         hkkx_v1_d, hkky_v1_d, hkkz_v1_d = self.get_vtbk_V_L(kcs_v1_d, self.htb_d)
#         hkkx_v2_u, hkky_v2_u, hkkz_v2_u = self.get_vtbk_V_L(kcs_v2_u, self.htb_u)
#         hkkx_v2_d, hkky_v2_d, hkkz_v2_d = self.get_vtbk_V_L(kcs_v2_d, self.htb_d)
#
#         hkkx_v1 = block(hkkx_v1_u, hkkx_v1_d, self.nb_u, self.nb_d)
#         hkky_v1 = block(hkky_v1_u, hkky_v1_d, self.nb_u, self.nb_d)
#         hkkz_v1 = block(hkkz_v1_u, hkkz_v1_d, self.nb_u, self.nb_d)
#
#         hkkx_v2 = block(hkkx_v2_u, hkkx_v2_d, self.nb_u, self.nb_d)
#         hkky_v2 = block(hkky_v2_u, hkky_v2_d, self.nb_u, self.nb_d)
#         hkkz_v2 = block(hkkz_v2_u, hkkz_v2_d, self.nb_u, self.nb_d)
#
#         hkkx = block(hkkx_v1, hkkx_v2, self.nb//2, self.nb//2)
#         hkky = block(hkky_v1, hkky_v2, self.nb//2, self.nb//2)
#         hkkz = block(hkkz_v1, hkkz_v2, self.nb//2, self.nb//2)
#         hkk = np.array([hkkx, hkky, hkkz])
#         return hkk
#
#     def cal_berry_curvature(self, kc, ewide=0.001):
#         hk = self.get_hk(kc)
#         # E, U = LA.eigh(hk)
#         E, U = self.get_eigh(hk)
#         vtbk = self.get_vtbk(kc)
#         vk = np.einsum('mi,aij,jn->amn', U.conj().T, vtbk, U, optimize=True)
#
#         e1, e2 = np.meshgrid(E, E)
#         invE = np.real(1 / (e2 - e1 - 1j * ewide))
#         invE = invE - np.diag(np.diag(invE))
#
#         bc = np.zeros([3, self.nb], dtype='float64')
#         bc[0] = -2. * np.imag(np.einsum('nm,mn,mn->n', vk[1], vk[2], invE ** 2, optimize=True))
#         bc[1] = -2. * np.imag(np.einsum('nm,mn,mn->n', vk[2], vk[0], invE ** 2, optimize=True))
#         bc[2] = -2. * np.imag(np.einsum('nm,mn,mn->n', vk[0], vk[1], invE ** 2, optimize=True))
#
#         return E, bc
#
#     def _cal_wilson_loop_on_a_closed_loop(self, kcs, bandindex):
#         '''
#           * calculae Wilson loop
#             W(C) = Int{A(k)*dk[C]}
#             on closed loop C
#         '''  #
#         nk = kcs.shape[0]
#
#         Dky = np.identity(self.nb)
#         for i in range(nk):
#             E, U = LA.eigh(self.get_hk(kcs[i]))
#             V = U[:, bandindex].reshape(self.nb, bandindex.shape[0])
#             if i + 1 != nk:
#                 Dky = LA.multi_dot([Dky, V, V.conj().T])
#             else:
#                 V_iGtau = np.diag(np.exp(-1j * self.lattG.T[1] @ self.wcc.T))
#                 Dky = LA.multi_dot([V.conj().T, Dky, V_iGtau, V])
#                 # Dky = LA.multi_dot([V.conj().T, Dky, V])
#
#         theta = np.sort(np.imag(np.log(LA.eigvals(Dky)))) / 2 / np.pi
#         return theta
#
#     def cal_wilson_loop(self, bandindex, e1=0, e2=1, e3=2, k3=0, nk1=30, nk2=30):
#         '''
#
#         :param i1, i2: track wannier center for bands index i1-i2
#         :param e1: direction to show wannier center
#         :param e2: direction to integration
#         :param e3: principle direction of plane
#         :param k3: position of the 2D plane
#         '''
#         theta = np.zeros([nk1, bandindex.shape[0]], dtype='float64')
#         kk1 = np.linspace(0, 1, nk1 + 1)[:-1]
#         # kk1 = 0.5 * (2*kk1)**7 # enable adapted sampling
#         for i in range(nk1):
#             print('cal wcc at ({}/{})'.format(i + 1, nk1))
#             _kk = np.zeros([nk2, 3], dtype='float64')
#             _kk.T[e1] = kk1[i]
#             _kk.T[e2] = np.linspace(0, 1, nk2 + 1)[:-1]
#             _kk.T[e3] = k3
#             _kcs = LA.multi_dot([self.lattG, _kk.T]).T
#             theta[i] = self._cal_wilson_loop_on_a_closed_loop(_kcs, bandindex)
#         return kk1, theta


class MacDonald_TMG_wan(object):
    """
      * MacDonald continous model for TMG
        * Stacking htb_u and htb_d along z direction with a twist angle
        * wannier kernal
        * two valley included
      * Notice that
        * htb_u and htb_d should have same lattice, also the fermi level should set at 0eV.
        * wannier orbitals should arange along stacking direction, otherwise tLu tLd should be specified.
      * example
        twisted double bilayer graphene(TDBG) with ABBA stacking, (x) indicate sequence of wannier orbital
        [htb_u]:
           ---pz(3)---pz(4)---- A
           ---pz(1)---pz(2)---- B # tLu = 0
        [htb_d]:
           ---pz(3)---pz(4)---- B # tLd = -1
           ---pz(1)---pz(2)---- A
    """  #

    def __init__(self, htb_u, htb_d, m=16, n=17, N=3, w1=0.0797, w2=0.0975, tLu=0, tLd=-1, vac=500, Umax=0, rotk=False):
        self.m = m
        self.n = n
        self.N = N
        self.w1 = w1    # layer coupling constant
        self.w2 = w2    # layer coupling constant
        self.tLu = tLu  # twist coupling added at tLu th graphene layer in upper(Top part) htb
        self.tLd = tLd  # twist coupling added at tLd th graphene layer in down(Bottom part) htb
        self.nvalley = 2
        self.nlayers = 2
        self.rotk = rotk
        self.Umax = Umax

        self.fermi = 0

        self.grapheneA = 2.46
        self.grapheneC = 3.015
        self.vac = vac

        # self.htb_u, self.htb_d = self._load_htb(htbfname_u, htbfname_d)
        self.htb_u, self.htb_d = htb_u, htb_d
        self.htb_u.set_ndegen_ones()
        self.htb_d.set_ndegen_ones()
        self.nw_u = self.htb_u.nw
        self.nw_d = self.htb_d.nw
        self.nw = self.nw_u + self.nw_d

        self.latt_lg = self.htb_u.cell.lattice
        self.lattG_lg = self.htb_u.cell.latticeG

        self.nsuper = m ** 2 + n ** 2 + m * n
        self.theta_rad = np.arccos((3 * m ** 2 + 3 * m + 1 / 2) / (3 * m ** 2 + 3 * m + 1))
        self.theta_deg = self.theta_rad / np.pi * 180
        self.theta = self.theta_deg
        self.natom = self.nsuper * 4

        self.Rot = np.identity(3, dtype='float64') # Rot +theta_rad/2 i.e. counter clock direction
        self.Rot[:2, :2] = np.array([
            [np.cos(self.theta_rad / 2), -np.sin(self.theta_rad / 2)],
            [np.sin(self.theta_rad / 2), np.cos(self.theta_rad / 2)],
        ])
        self.Rot_inv = self.Rot.T # Rot -theta_rad/2

        self.latt, self.lattG = self._init_lattice(self.latt_lg)
        self.gridG, self.gridGc = self._init_gridG_symm(N, self.lattG)
        # self.gridG, self.gridGc = self._init_gridG(N, self.lattG)
        self.nG = self.gridG.shape[0]
        self.nb_u = self.nG * self.nw_u
        self.nb_d = self.nG * self.nw_d
        self.nb = self.nvalley * (self.nb_u + self.nb_d)
        self.lattA_tbg = self.nsuper**0.5 * self.grapheneA

        self._init_symmetric_operators()
        self.wcc, self.wcc_lg_d, self.wcc_lg_u = self._init_wcc()

        self.hT = self._init_layer_coupling(self.gridG)
        self.hD = self.get_displacement_field(self.Umax)
        self.hBN = 0 # self.get_hBN()

    # def _load_htb(self, htbfname_u, htbfname_d):
    #     print('[FROM MacDonald_TMG_wan] loading htb')
    #     htb_u = Htb()
    #     htb_d = Htb()
    #     htb_u.load_htb(htbfname_u)
    #     htb_d.load_htb(htbfname_d)
    #     htb_u.set_ndegen_ones()
    #     htb_d.set_ndegen_ones()
    #
    #     htb_u.Rc = LA.multi_dot([htb_u.cell.lattice, htb_u.R.T]).T
    #     htb_d.Rc = LA.multi_dot([htb_d.cell.lattice, htb_d.R.T]).T
    #
    #     return htb_u, htb_d

    def _init_wcc(self):
        print('[FROM MacDonald_TMG_wan] inintal wcc for twisted system')
        '''
          * wcc: ?
          * wcc_lg_d: wcc for buttom layer htb
          * wcc_lg_u: wcc for top layer htb
        ''' #
        dU = np.max(self.htb_u.wcc.T[2]) - np.min(self.htb_u.wcc.T[2])
        dD = np.max(self.htb_d.wcc.T[2]) - np.min(self.htb_d.wcc.T[2])

        self.htb_d.wcc.T[2] = self.htb_d.wcc.T[2] - np.min(self.htb_d.wcc.T[2]) + self.grapheneC
        self.htb_u.wcc.T[2] = self.htb_u.wcc.T[2] - np.min(self.htb_u.wcc.T[2]) + self.grapheneC + dD + self.grapheneC

        # wcc = np.vstack([self.htb_d.wcc, self.htb_u.wcc])
        wcc_lg_d = self.htb_d.wcc
        wcc_lg_u = self.htb_u.wcc
        self.dU = dU
        self.dD = dD

        wcc = np.vstack([
            np.kron(np.ones([self.nG]), wcc_lg_u.T).T, # for valley +, top parts
            np.kron(np.ones([self.nG]), wcc_lg_d.T).T, # for valley +, bottom parts
            np.kron(np.ones([self.nG]), wcc_lg_u.T).T, # for valley -, top parts
            np.kron(np.ones([self.nG]), wcc_lg_d.T).T, # for valley -, bottom parts
        ])

        return wcc, wcc_lg_d, wcc_lg_u

    def _init_lattice(self, latt_lg):
        print('[FROM MacDonald_TMG_wan] inintal lattice for twisted system')
        '''
          * Lattice
          ** gKu, gKd : K valley in UBZ of graphene for upper layer(u) and underlayer layer(d)
        ''' #
        latt_lg_u = LA.multi_dot([self.Rot_inv, latt_lg])
        latt_lg_d = LA.multi_dot([self.Rot, latt_lg])
        lattG_lg = 2 * np.pi * LA.inv(latt_lg.T)
        lattG_lg_u = 2 * np.pi * LA.inv(latt_lg_u.T)
        lattG_lg_d = 2 * np.pi * LA.inv(latt_lg_d.T)

        lattG = lattG_lg_u - lattG_lg_d
        lattG[2, 2] = 2 * np.pi / self.vac
        latt = LA.inv(lattG.T / 2 / np.pi)

        q1 = (2 * lattG.T[0] + 1 * lattG.T[1]) / 3
        q2 = (-1 * lattG.T[0] + 1 * lattG.T[1]) / 3
        q3 = -(1 * lattG.T[0] + 2 * lattG.T[1]) / 3
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

        # self.K1uc = (2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        # self.K2uc = (1 * lattG_slg.T[0] + 2 * lattG_slg.T[1]) / 3
        # self.Ku1 = -q2
        # self.Ku2 = q1
        # self.Kd1 = q3
        # self.Kd2 = -q2

        self.K1uc = (2 * lattG_lg.T[0] + 1 * lattG_lg.T[1]) / 3
        self.K2uc = -self.K1uc
        self.Ku1 = -q2
        self.Ku2 = q2
        self.Kd1 = q3
        self.Kd2 = -q3

        # self.K1uc = -(2 * lattG_slg.T[0] + 1 * lattG_slg.T[1]) / 3
        # self.K2uc = -self.K1uc
        # self.Ku1 = q2
        # self.Ku2 = -q2
        # self.Kd1 = -q3
        # self.Kd2 = q3

        return latt, lattG

    # def _init_cell(self):
    #     cell = Cell()
    #     cell.name = r'Twisted muilti-layer graphene (TMG)'
    #     cell.lattice = self.latt
    #     cell.latticeG = self.lattG
    #     cell.N = None
    #     cell.spec = None
    #     cell.ions = None
    #     cell.ions_car = None

    def _init_gridG_symm(self, N, lattG):
        print('[FROM MacDonald_TMG_wan] inintal gridG for twisted system')
        '''
          * Symmetrized gridG
        ''' #
        Gcut = (N + 0.001) * LA.norm(lattG.T[0])

        _gridG = np.array([
            [n1, n2, 0]
            for n1 in range(-2*N, 2*N + 1)
            for n2 in range(-2*N, 2*N + 1)
        ], dtype='float64')
        gridG_norm = LA.norm(LA.multi_dot([lattG, _gridG.T]).T, axis=1)

        is_in_WS = gridG_norm < Gcut
        nG = is_in_WS[is_in_WS].shape[0]

        gridG = np.zeros([nG, 3], dtype='float64')
        j = 0
        for g, remain in zip(_gridG, is_in_WS):
            if remain:
                gridG[j] = g
                j += 1

        gridGc = LA.multi_dot([lattG, gridG.T]).T

        return gridG, gridGc

    def _init_layer_coupling(self, gridG):
        print('[FROM MacDonald_TMG_wan] inintal layer coupling for twisted system')
        '''
          * Layer coupling
          ** T1, T2, T3 : layer coupling constant
        ''' #
        nG = self.nG
        w1 = self.w1
        w2 = self.w2

        z = np.exp(2j * np.pi / 3)
        t1 = np.array([
            [w1, w2],
            [w2, w1],
        ], dtype='complex128')
        t2 = np.array([
            [w1,      w2 * z ** -1],
            [w2 * z , w1           ],
        ], dtype='complex128')
        t3 = np.array([
            [w1,           w2 * z],
            [w2 * z ** -1, w1    ],
        ], dtype='complex128')
        TT1 = np.zeros([self.nw_d//2, self.nw_u//2], dtype='complex128')
        TT2 = np.zeros([self.nw_d//2, self.nw_u//2], dtype='complex128')
        TT3 = np.zeros([self.nw_d//2, self.nw_u//2], dtype='complex128')
        TT1[self.tLd, self.tLu] = 1.0
        TT2[self.tLd, self.tLu] = 1.0
        TT3[self.tLd, self.tLu] = 1.0
        T1 = np.kron(TT1, t1)
        T2 = np.kron(TT2, t2)
        T3 = np.kron(TT3, t3)

        '''
          * valley 1
        '''
        tkp1 = np.zeros([nG, nG])
        tkp2 = np.zeros([nG, nG])
        tkp3 = np.zeros([nG, nG])
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
                    tkp2[i, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
                    tkp3[i, j] = 1

        hT1 = np.kron(tkp1, T1.conj())
        hT2 = np.kron(tkp2, T2.conj())
        hT3 = np.kron(tkp3, T3.conj())

        hT = hT1 + hT2 + hT3
        hT_valley1 = np.block([
            [np.zeros([self.nb_u, self.nb_u]), hT.T.conj()],
            [hT, np.zeros([self.nb_d, self.nb_d])],
        ])

        '''
          * valley -1
        '''
        tkp1 = np.zeros([nG, nG])
        tkp2 = np.zeros([nG, nG])
        tkp3 = np.zeros([nG, nG])

        # used for K -K
        for i in range(nG):
            for j in range(nG):
                if np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
                    tkp1[i, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, 0, 0]), atol=1e-2).all():
                    tkp2[i, j] = 1
                elif np.isclose(gridG[i] - gridG[j], np.array([-1, -1, 0]), atol=1e-2).all():
                    tkp3[i, j] = 1

        hT1 = np.kron(tkp1, T1)
        hT2 = np.kron(tkp2, T2)
        hT3 = np.kron(tkp3, T3)

        # # used for K1uc K2uc in unit BZ cell
        # for i in range(nG):
        #     for j in range(nG):
        #         if np.isclose(gridG[i] - gridG[j], np.array([1, 1, 0]), atol=1e-2).all():
        #             tkp1[i + nG, j] = 1
        #         elif np.isclose(gridG[i] - gridG[j], np.array([0, 1, 0]), atol=1e-2).all():
        #             tkp2[i + nG, j] = 1
        #         elif np.isclose(gridG[i] - gridG[j], np.array([0, 0, 0]), atol=1e-2).all():
        #             tkp3[i + nG, j] = 1
        # hT1 = np.kron(tkp1, T2)
        # hT2 = np.kron(tkp2, T3)
        # hT3 = np.kron(tkp3, T1)

        hT = hT1 + hT2 + hT3
        hT_valley2 = np.block([
            [np.zeros([self.nb_u, self.nb_u]), hT.T.conj()],
            [hT, np.zeros([self.nb_d, self.nb_d])],
        ])

        # '''
        #   * plot hopping
        # '''
        # kkc = self.get_kkc(np.array([0, 0, 0]))[2:]
        # plt.scatter(kkc[0].T[0], kkc[0].T[1], color='red')
        # plt.scatter(kkc[1].T[0], kkc[1].T[1], color='black')
        # for i in range(self.nG):
        #     kci = kkc[1, i]
        #     for j in range(self.nG):
        #         kcj = kkc[0, j]
        #         hopping = np.vstack([kci, kcj])
        #         if tkp1[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='red')
        #         elif tkp2[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='blue')
        #         elif tkp3[i, j] == 1:
        #             plt.plot(hopping.T[0], hopping.T[1], color='black')
        #
        # plt.axis('equal')

        hT = np.zeros([self.nb, self.nb], dtype='complex128')
        hT[:self.nb//2, :self.nb//2] = hT_valley1
        hT[self.nb//2:, self.nb//2:] = hT_valley2

        return hT

    def _init_symmetric_operators(self):
        nb = self.nb
        self.op_v1_proj = np.zeros([nb, nb])
        self.op_v2_proj = np.zeros([nb, nb])

        self.op_v1_proj[:nb // 2, :nb // 2] = np.identity(nb // 2)
        self.op_v2_proj[nb // 2:, nb // 2:] = -np.identity(nb // 2)

        self.op_valley = self.op_v1_proj + self.op_v2_proj

    def get_kkc(self, kc):
        kcs_v1_u = kc + self.gridGc - self.Ku1
        kcs_v1_d = kc + self.gridGc - self.Kd1
        kcs_v2_u = kc - self.gridGc - self.Ku2
        kcs_v2_d = kc - self.gridGc - self.Kd2

        if self.rotk:
            kcs_v1_u = LA.multi_dot([self.Rot, kcs_v1_u.T]).T
            kcs_v1_d = LA.multi_dot([self.Rot_inv, kcs_v1_d.T]).T
            kcs_v2_u = LA.multi_dot([self.Rot, kcs_v2_u.T]).T
            kcs_v2_d = LA.multi_dot([self.Rot_inv, kcs_v2_d.T]).T

        kcs_v1_u = kcs_v1_u + self.K1uc
        kcs_v1_d = kcs_v1_d + self.K1uc
        kcs_v2_u = kcs_v2_u + self.K2uc
        kcs_v2_d = kcs_v2_d + self.K2uc

        kcs = np.array([kcs_v1_u, kcs_v1_d, kcs_v2_u, kcs_v2_d], dtype='float64')
        return kcs

    def get_h0_V_L(self, kkc, htb):
        """
          * h for each valley and layer
        """  #
        Sk = np.exp(-1j * np.einsum('Ga,na->Gn', kkc, htb.wcc))
        eikr = np.exp(1j * np.einsum('Ga,Ra->GR', kkc, htb.Rc))
        h0 = np.einsum('GR,Gm,Rmn,Gn->Gmn', eikr, Sk, htb.hr_Rmn, Sk.conj(), optimize=True).reshape([self.nG, htb.nw, htb.nw])
        h0 = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), h0).reshape(htb.nw * self.nG, htb.nw * self.nG)
        return h0

    def get_h0(self, kc):
        """
          * Notice that we have used hybird gauge here
            i.e. tb gauge in xy plane whereas wannier gauge in z direction
            this gauge choise guarantees the wannier center in all direction
            according to our test before in single layer graphene.
        """  #
        kcs_v1_u, kcs_v1_d, kcs_v2_u, kcs_v2_d = self.get_kkc(kc)

        h0_v1 = np.block([
            [self.get_h0_V_L(kcs_v1_u, self.htb_u), np.zeros([self.nb_u, self.nb_d])],
            [np.zeros([self.nb_u, self.nb_d]).T, self.get_h0_V_L(kcs_v1_d, self.htb_d)],
        ])
        h0_v2 = np.block([
            [self.get_h0_V_L(kcs_v2_u, self.htb_u), np.zeros([self.nb_u, self.nb_d])],
            [np.zeros([self.nb_u, self.nb_d]).T, self.get_h0_V_L(kcs_v2_d, self.htb_d)],
        ])
        h0 = np.block([
            [h0_v1, np.zeros([self.nb//2, self.nb//2])],
            [np.zeros([self.nb//2, self.nb//2]), h0_v2],
        ])
        h0 = 0.5 * (h0 + h0.conj().T)
        return h0

    def get_displacement_field(self, Umax):
        nlayer = self.nw // 2
        # U = Umax * np.kron(np.ones([2]), np.linspace(0, 1, nlayer))
        # U = Umax * np.kron(np.linspace(0, 1, nlayer), np.ones([2]))
        U = Umax * np.kron(np.linspace(-0.5, 0.5, nlayer), np.ones([2]))

        hD_u_flg = U[self.nw_d:]
        hD_d_flg = U[:self.nw_d]

        hD_v1 = np.zeros([self.nb//2])
        hD_v1[:self.nb_u] = np.kron(np.ones([self.nG]), hD_u_flg)
        hD_v1[self.nb_u:] = np.kron(np.ones([self.nG]), hD_d_flg)

        hD = np.hstack([hD_v1, hD_v1])
        hD = np.diag(hD)
        return hD

    def get_hBN(self, U_BN_u, U_BN_d, enable=False):
        """
          * To active the h-BN substrate effect
            manually change `U_BN_u` and `U_BN_d`
        """
        if not enable:
            hBN = 0
            return hBN
        else:
            # # fot mTBG
            # U_BN_u = np.array([0])
            # U_BN_d = np.array([0.03])
            # fot mTTG
            # U_BN_u = np.array([-0.0, -0.0])
            # U_BN_d = np.array([0.05])

            U_BN_u_flg = np.kron(U_BN_u, np.array([1, -1]))
            U_BN_d_flg = np.kron(U_BN_d, np.array([1, -1]))

            hBN_v1 = np.zeros([self.nb//2])
            hBN_v1[:self.nb_u] = np.kron(np.ones([self.nG]), U_BN_u_flg)
            hBN_v1[self.nb_u:] = np.kron(np.ones([self.nG]), U_BN_d_flg)

            hBN = np.hstack([hBN_v1, hBN_v1])
            hBN = np.diag(hBN)
            return hBN

    def get_hk(self, kc):
        """
          * The basis for contunuum hamiltonian in order of:
            valley, twist u/d part, nG, nw_bottom_few_layer_graphene/nw_top_few_layer_graphene
            nb = nvalley * (nb_u + nb_d)
            nb_u = nG * nw_u
            nb_d = nG * nw_d

            h0 + hT =
            [h0_{T,valley+} T.H            0                0             ]
            [T              h0_{B,valley+} 0                0             ]
            [0              0              h0_{T,valley-}   T.H           ]
            [0              0              T                H0_{B,valley-}]

            h0_{T,valley+} =
            [h0(kc + K1uc + Ku1 + G1) 0                        . ]
            [0                        h0(kc + K1uc + Ku1 + G2) . ]
            [.                        .                        . ]

            h0 = k.sigma for TBG

            For instance of TDBG, in order of
            |1,A> |1,B> |2,A> |2,B> for Top parts
            |-2,A> |-2,B> |-1,A> |-1,B> for Bottom parts

            hD is displacement field
            hBN is the effect from the h-BN substrate

        """
        h0 = self.get_h0(kc)
        hk = h0 + self.hT + self.hD + self.hBN
        return hk

    def w90gethk(self, kc):
        return self.get_hk(kc)

    def get_eigh(self, H):
        nb_one_valley = self.nb//2
        h1 = H[:nb_one_valley, :nb_one_valley]
        h2 = H[nb_one_valley:, nb_one_valley:]
        e1, v1 = LA.eigh(h1)
        e2, v2 = LA.eigh(h2)
        E = np.hstack([e1, e2])
        zero = np.zeros([nb_one_valley, nb_one_valley])
        U = np.block([
            [v1, zero],
            [zero, v2],
        ])
        U *= np.exp(-1j * np.angle(U)[0])
        return E, U

    def get_fb_index(self):
        i1 = self.nb // 4 - 1
        i2 = self.nb // 4
        fb_index = np.array([i1 + self.nb // 2, i2 + self.nb // 2, i1, i2])  # (valley-,VB); (-,CB); (+,VB); (+,CB);
        print('[FROM MacDonald_TMG_wan.get_fb_index]')
        print('[{}, (valley-,VB)]\n [{}, (-,CB)]\n [{}, (+,VB)]\n [{}, (+,CB)]'.format(fb_index[0], fb_index[1], fb_index[2], fb_index[3]), )
        return fb_index

    def get_br_V_L(self, kkc, bc, htb):
        """
          * br for each valley and layer
        """  #
        Sk = np.exp(-1j * np.einsum('Ga,na->Gn', kkc, htb.wcc))
        # Sk[:, :] = 1
        eikr = np.exp(1j * np.einsum('Ga,Ra->GR', kkc, htb.Rc))
        br = np.einsum('a,GR,Gm,Ramn,Gn->Gmn', bc, eikr, Sk, htb.r_Ramn, Sk.conj(), optimize=True).reshape([self.nG, htb.nw, htb.nw])
        br = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), br).reshape(htb.nw * self.nG, htb.nw * self.nG)
        return br

    def get_basis_overlap(self, kc, bc):
        """
          * Basis overlap ~Mmn(k,k+b)=<~um,k|~un,k+b>
            from which we can calculate M(k,k+b)=U(k).H . ~M . U(k+b)
          * hybird gauge used here
        """  #
        bc[:2] = 0
        # Sk = np.exp(-1j * np.einsum('VLGa,Lna->VLGn', kkc, self.wcc_layer))
        # Sk[:, :, :, :] = 1
        # eikr = np.exp(1j * np.einsum('VLGa,LRa->VLGR', kkc, self.Rc))
        # br = np.einsum('a,VLGR,VLGm,LRamn,VLGn->VLGmn', bc, eikr, Sk, self.r_Ramn, Sk.conj(), optimize=True).reshape(
        #     [4 * self.nG, self.nw_lg, self.nw_lg])
        # br = np.einsum('ij,jmn->imjn', np.eye(self.nb // self.nw_lg, dtype='complex128'), br).reshape(self.nb, self.nb)
        # br = 0.5 * (br + br.conj().T)

        kcs_v1_u, kcs_v1_d, kcs_v2_u, kcs_v2_d = self.get_kkc(kc)

        br_v1 = np.block([
            [self.get_br_V_L(kcs_v1_u, bc, self.htb_u), np.zeros([self.nb_u, self.nb_d])],
            [np.zeros([self.nb_u, self.nb_d]).T, self.get_br_V_L(kcs_v1_d, bc, self.htb_d)],
        ])
        br_v2 = np.block([
            [self.get_br_V_L(kcs_v2_u, bc, self.htb_u), np.zeros([self.nb_u, self.nb_d])],
            [np.zeros([self.nb_u, self.nb_d]).T, self.get_br_V_L(kcs_v2_d, bc, self.htb_d)],
        ])
        br = np.block([
            [br_v1, np.zeros([self.nb//2, self.nb//2])],
            [np.zeros([self.nb//2, self.nb//2]), br_v2],
        ])
        br = 0.5 * (br + br.conj().T)

        W, V = LA.eigh(br)
        eibr_mn = LA.multi_dot([V, np.diag(np.exp(1j * W)), V.conj().T])
        gxx = np.einsum('mn->nm', eibr_mn).conj()

        # return br, gxx
        return gxx

    '''
      view in real space
    '''
    def get_bloch_sum_in_Rspace(self, kc, rr):
        nrr = rr.shape[0]

        K1uc = 0 # self.K1uc
        K2uc = 0 # self.K2uc

        gridGc = LA.multi_dot([self.lattG, self.gridG.T]).T

        gridGc_u = np.array([
            gridGc - self.Ku1 + K1uc,
            -gridGc - self.Ku2 + K2uc,
        ])
        gridGc_d = np.array([
            gridGc - self.Kd1 + K1uc,
            -gridGc - self.Kd2 + K2uc,
        ])
        eikR = np.exp(1j * np.einsum('a,Ra->R', kc, rr))
        eiGR_u = np.exp(1j * np.einsum('VGa,Ra->VGR', gridGc_u, rr))
        eiGR_d = np.exp(1j * np.einsum('VGa,Ra->VGR', gridGc_d, rr))

        bloch_sum_v1_k_rr = np.zeros([self.nb//2, self.nw, nrr], dtype='complex128')
        bloch_sum_v2_k_rr = np.zeros([self.nb//2, self.nw, nrr], dtype='complex128')
        bloch_sum_k_rr = np.zeros([self.nb, self.nw, nrr], dtype='complex128')

        # for instance of TDBG, nw here in order of
        # |1,A> |1,B> |2,A> |2,B> ; |-2,A> |-2,B> |-1,A> |-1,B>
        bloch_sum_v1_k_rr[:self.nb_u, :self.nw_u] = np.einsum('mn,R,GR->GmnR', np.eye(self.nw_u), eikR, eiGR_u[0], optimize=True).reshape([self.nb_u, self.nw_u, nrr])
        bloch_sum_v1_k_rr[self.nb_u:, self.nw_u:] = np.einsum('mn,R,GR->GmnR', np.eye(self.nw_d), eikR, eiGR_d[0], optimize=True).reshape([self.nb_d, self.nw_d, nrr])
        bloch_sum_v2_k_rr[:self.nb_u, :self.nw_u] = np.einsum('mn,R,GR->GmnR', np.eye(self.nw_u), eikR, eiGR_u[1], optimize=True).reshape([self.nb_u, self.nw_u, nrr])
        bloch_sum_v2_k_rr[self.nb_u:, self.nw_u:] = np.einsum('mn,R,GR->GmnR', np.eye(self.nw_d), eikR, eiGR_d[1], optimize=True).reshape([self.nb_d, self.nw_d, nrr])

        bloch_sum_k_rr[:self.nb//2] = bloch_sum_v1_k_rr
        bloch_sum_k_rr[self.nb//2:] = bloch_sum_v2_k_rr

        return bloch_sum_k_rr

    def get_phi_in_Rspace(self, kc, rr):
        # eign states in real space
        E, U = LA.eigh(self.get_hk(kc))
        bloch_sum_k_rr = self.get_bloch_sum_in_Rspace(kc, rr)
        psi_k_rr = np.einsum('nb,bsR->nsR', U.T, bloch_sum_k_rr)
        return psi_k_rr

    '''
      * calculators
    '''
    def get_vtbk_V_L(self, kkc, htb):
        """
          * get v^tb(k), from which we can get v(k) = U* v^tb(k) U
            v^tb(k) = dH^tb(k) + i[H^tb(k), A^tb(k)],
            where we take A^tb(k) = 0
            dH^tb(k) = sum_{R} [e^(ik(R-tau_m+tau_n))<m0|H|nR>]
        """  #
        Rmn = np.zeros([htb.nR, 3, htb.nw, htb.nw])
        for i in range(htb.nR):
            for a in range(3):
                M, N = np.meshgrid(htb.wcc.T[a], htb.wcc.T[a])
                Rmn[i, a] = htb.Rc[i, a] - (M - N)

        eikRmn = np.exp(1j * np.einsum('Ga,Ramn->GRmn', kkc, Rmn))
        hkk = 1j * np.einsum('Ramn,GRmn,Rmn->aGmn', Rmn, eikRmn, htb.hr_Rmn, optimize=True)
        hkkx = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), hkk[0]).reshape(htb.nw * self.nG, htb.nw * self.nG)
        hkky = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), hkk[1]).reshape(htb.nw * self.nG, htb.nw * self.nG)
        hkkz = np.einsum('ij,jmn->imjn', np.eye(self.nG, dtype='complex128'), hkk[2]).reshape(htb.nw * self.nG, htb.nw * self.nG)
        return hkkx, hkky, hkkz

    def get_vtbk(self, kc):
        kcs_v1_u, kcs_v1_d, kcs_v2_u, kcs_v2_d = self.get_kkc(kc)
        def block(A, B, nA, nB):
            C = np.zeros([nA, nB])
            D = np.block([[A,C],[C.T,B]])
            return D

        hkkx_v1_u, hkky_v1_u, hkkz_v1_u = self.get_vtbk_V_L(kcs_v1_u, self.htb_u)
        hkkx_v1_d, hkky_v1_d, hkkz_v1_d = self.get_vtbk_V_L(kcs_v1_d, self.htb_d)
        hkkx_v2_u, hkky_v2_u, hkkz_v2_u = self.get_vtbk_V_L(kcs_v2_u, self.htb_u)
        hkkx_v2_d, hkky_v2_d, hkkz_v2_d = self.get_vtbk_V_L(kcs_v2_d, self.htb_d)

        hkkx_v1 = block(hkkx_v1_u, hkkx_v1_d, self.nb_u, self.nb_d)
        hkky_v1 = block(hkky_v1_u, hkky_v1_d, self.nb_u, self.nb_d)
        hkkz_v1 = block(hkkz_v1_u, hkkz_v1_d, self.nb_u, self.nb_d)

        hkkx_v2 = block(hkkx_v2_u, hkkx_v2_d, self.nb_u, self.nb_d)
        hkky_v2 = block(hkky_v2_u, hkky_v2_d, self.nb_u, self.nb_d)
        hkkz_v2 = block(hkkz_v2_u, hkkz_v2_d, self.nb_u, self.nb_d)

        hkkx = block(hkkx_v1, hkkx_v2, self.nb//2, self.nb//2)
        hkky = block(hkky_v1, hkky_v2, self.nb//2, self.nb//2)
        hkkz = block(hkkz_v1, hkkz_v2, self.nb//2, self.nb//2)
        hkk = np.array([hkkx, hkky, hkkz])
        return hkk

    def cal_berry_curvature(self, kc, ewide=0.001):
        hk = self.get_hk(kc)
        # E, U = LA.eigh(hk)
        E, U = self.get_eigh(hk)
        vtbk = self.get_vtbk(kc)
        vk = np.einsum('mi,aij,jn->amn', U.conj().T, vtbk, U, optimize=True)

        e1, e2 = np.meshgrid(E, E)
        invE = np.real(1 / (e2 - e1 - 1j * ewide))
        invE = invE - np.diag(np.diag(invE))

        bc = np.zeros([3, self.nb], dtype='float64')
        bc[0] = -2. * np.imag(np.einsum('nm,mn,mn->n', vk[1], vk[2], invE ** 2, optimize=True))
        bc[1] = -2. * np.imag(np.einsum('nm,mn,mn->n', vk[2], vk[0], invE ** 2, optimize=True))
        bc[2] = -2. * np.imag(np.einsum('nm,mn,mn->n', vk[0], vk[1], invE ** 2, optimize=True))

        return E, bc

    def _cal_wilson_loop_on_a_closed_loop(self, kcs, bandindex, e2=1):
        """ 
          * calculae Wilson loop
            W(C) = Int{A(k)*dk[C]}
            on closed loop C
        """ #
        nk = kcs.shape[0]

        Dky = np.identity(self.nb, dtype='complex128')
        for i in range(nk):
            hk = self.get_hk(kcs[i])
            E, U = self.get_eigh(hk)
            # E, U = LA.eigh(self.get_hk(kcs[i]))
            V = U[:, bandindex]
            if i + 1 != nk:
                Dky = LA.multi_dot([Dky, V, V.conj().T])
            else:
                # u, s, vh = LA.svd(Dky)
                # Dky = u @ vh
                V_iGtau = np.diag(np.exp(-1j * self.lattG.T[e2] @ self.wcc.T))
                Dky = LA.multi_dot([V.conj().T, Dky, V_iGtau, V])
                # Dky = LA.multi_dot([V.conj().T, Dky, V])

        theta = np.sort(np.imag(np.log(LA.eigvals(Dky)))) / 2 / np.pi
        return theta

    def cal_wilson_loop(self, bandindex, e1=0, e2=1, e3=2, k3=0, nk1=30, nk2=30):
        """

        :param bandindex: track wannier center for bands index in bandindex
        :param e1: direction to show wannier center
        :param e2: direction to integration
        :param e3: principle direction of plane
        :param k3: position of the 2D plane
        :param nk1: num of k
        :param nk2: num of k
        """
        theta = np.zeros([nk1, bandindex.shape[0]], dtype='float64')
        kk1 = np.linspace(0, 1, nk1 + 1)[:-1]
        # kk1 = 0.5 * (2*kk1)**7
        for i in range(nk1):
            print('cal wcc at ({}/{})'.format(i + 1, nk1))
            _kk = np.zeros([nk2, 3], dtype='float64')
            _kk.T[e1] = kk1[i]
            _kk.T[e2] = np.linspace(0, 1, nk2 + 1)[:-1]
            _kk.T[e3] = k3
            _kcs = LA.multi_dot([self.lattG, _kk.T]).T
            theta[i] = self._cal_wilson_loop_on_a_closed_loop(_kcs, bandindex, e2)
        return kk1, theta

    # def get_closed_loop(self, ik2=0, nk1=100, nk2=100):
    #     kk1 = np.linspace(0, 1, nk1 + 1)[:-1]
    #     kk2 = np.linspace(0, 1, nk2 + 1)[:-1]
    #     ks = np.zeros([nk1, 3], dtype='float64')
    #     ks.T[0] = kk1
    #     ks.T[1] = kk2[ik2]
    #     return kk2, ks
    #
    # def cal_wilson_loop_k(self, ik2, i1, i2, nk1=100, nk2=100):
    #     '''
    #       * calculae Wilson loop
    #         W(C) = Int{A(k1, k2, k3)*dk[C]}
    #         on closed loop C
    #     '''  #
    #     kk2, ks = self.get_closed_loop(ik2, nk1, nk2)
    #
    #     Dky = np.identity(self.nb)
    #     nk1 = ks.shape[0]
    #     kcs = LA.multi_dot([self.lattG, ks.T]).T
    #     for i in range(nk1):
    #         E, U = LA.eigh(self.get_hk(kcs[i]))
    #         V = U[:, i1:i2]
    #         if i + 1 != nk1:
    #             Dky = LA.multi_dot([Dky, V, V.conj().T])
    #         else:
    #             Dky = LA.multi_dot([V.conj().T, Dky, V])
    #
    #     theta = np.sort(np.imag(np.log(LA.eigvals(Dky)))) / 2 / np.pi
    #     return theta
    #
    # def cal_wilson_loop(self, i1, i2, nk1=100, nk2=100):
    #     '''
    #       * cal. Wilson loop W(k2) = Int{A(k1, k2)*d[k1]} on closed loop
    #     '''  #
    #     nw = i2 - i1
    #     kk2 = np.linspace(0, 1, nk2 + 1)[:-1]
    #     theta = np.zeros([nk2, nw], dtype='float64')
    #     for ik2 in range(nk2):
    #         print('cal wcc at ({}/{})'.format(ik2 + 1, nk2))
    #         theta[ik2] = self.cal_wilson_loop_k(ik2, i1, i2, nk1, nk2)
    #
    #     return kk2, theta

    def cal_chern_latt_method(self, bandindex, nk1=20, nk2=20):
        def get_UU(k):
            kc = self.lattG @ k
            hk = self.get_hk(kc)
            E, U = self.get_eigh(hk)
            return U

        kk1 = np.linspace(0, 1, nk1 + 1)  # [:-1]
        kk2 = np.linspace(0, 1, nk2 + 1)  # [:-1]

        C = 0
        UU = np.zeros([2, nk2 + 1, self.nb, bandindex.shape[0]], dtype='complex128')

        for j in range(nk2 + 1):
            UU[0, j] = get_UU(np.array([kk1[0], kk2[j], 0]))[:, bandindex]
        # UU[0, -1] = UU[0, 0]

        for i in range(nk1):
            print('cal at {}/{}'.format(i + 1, nk1))
            for j in range(nk2 + 1):
                UU[1, j] = get_UU(np.array([kk1[i + 1], kk2[j], 0]))[:, bandindex]
            # UU[1, -1] = UU[1, 0]

            for j in range(nk2):
                # F = UU[0, j].T.conj() @ UU[1, j] @ \
                #     UU[1, j].T.conj() @ UU[1, j+1] @ \
                #     UU[1, j + 1].T.conj() @ UU[0, j+1] @ \
                #     UU[0, j+1].T.conj() @ UU[0, j]
                # F = np.imag(np.log(LA.det(F)))
                F = LA.det(UU[0, j].T.conj() @ UU[1, j]) * \
                    LA.det(UU[1, j].T.conj() @ UU[1, j + 1]) * \
                    LA.det(UU[1, j + 1].T.conj() @ UU[0, j + 1]) * \
                    LA.det(UU[0, j + 1].T.conj() @ UU[0, j])
                F = -np.imag(np.log(F))
                C += F

            UU[0] = UU[1]
            UU[1] = 0

        C = C / np.pi / 2
        return C

    def get_hk_slab(self, k, N2=10, ribbon='zigzag'):
        """
          ribbon 
        """  #

        if ribbon == 'armchair':
            surlattG = np.array([
                self.lattG.T[0] + self.lattG.T[1],
                self.lattG.T[1] - self.lattG.T[0],
                self.lattG.T[2]
            ]).T
            surlatt = LA.inv(surlattG.T / 2 / np.pi)
        elif ribbon == 'zigzag':
            surlattG = np.array([
                self.lattG.T[1] - self.lattG.T[0],
                self.lattG.T[0] + self.lattG.T[1],
                self.lattG.T[2]
            ]).T
            surlatt = LA.inv(surlattG.T / 2 / np.pi)
        else:
            surlattG = None
            surlatt = None

        kk = np.zeros([N2, 3], dtype='float64')
        kk.T[0] = k[0]
        kk.T[1] = np.linspace(0, 1, N2+1)[:-1]
        kkc = LA.multi_dot([surlattG, kk.T]).T

        nR = N2+1
        R = np.zeros([nR, 3], dtype='float64')
        degen = np.zeros([nR], dtype='int64')
        R.T[1] = np.arange(nR) - N2 // 2
        degen[np.array([0, -1])] = 2

        hk = np.array([
            self.get_hk(kkc[ik])
            for ik in range(N2)
        ])
        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', kk, R))
        hR = np.einsum('kR,kmn->Rmn', eikR, hk) / N2

        h0 = hR[nR//2]
        h1R = hR[nR//2+1]
        h1L = hR[nR//2-1]
        hk_slab = np.kron(np.eye(N2), h0) + \
                  np.kron(np.eye(N2, N2, 1), h1R) + \
                  np.kron(np.eye(N2, N2, -1), h1L)
        return hk_slab

    # def get_slab_green(self, ee, k, N2=20, eps=0.01, ribbon='zigzag'):
    #     ne = ee.shape[0]
    #
    #     if ribbon == 'armchair':
    #         surlattG = np.array([
    #             self.lattG.T[0] + self.lattG.T[1],
    #             self.lattG.T[1] - self.lattG.T[0],
    #             self.lattG.T[2]
    #         ]).T
    #         surlatt = LA.inv(surlattG.T / 2 / np.pi)
    #     elif ribbon == 'zigzag':
    #         surlattG = np.array([
    #             self.lattG.T[1] - self.lattG.T[0],
    #             self.lattG.T[0] + self.lattG.T[1],
    #             self.lattG.T[2]
    #         ]).T
    #         surlatt = LA.inv(surlattG.T / 2 / np.pi)
    #     else:
    #         surlattG = None
    #         surlatt = None
    #
    #     '''
    #       * F.T. to real space
    #     '''
    #     kk = np.zeros([N2, 3], dtype='float64')
    #     kk.T[0] = k[0]
    #     kk.T[1] = np.linspace(0, 1, N2+1)[:-1]
    #     kkc = LA.multi_dot([surlattG, kk.T]).T
    #
    #     nR = N2+1
    #     R = np.zeros([nR, 3], dtype='float64')
    #     degen = np.zeros([nR], dtype='int64')
    #     R.T[1] = np.arange(nR) - N2 // 2
    #     degen[np.array([0, -1])] = 2
    #
    #     hk = np.array([
    #         self.get_hk(kkc[ik])
    #         for ik in range(N2)
    #     ])
    #     eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', kk, R))
    #     hR = np.einsum('kR,kmn->Rmn', eikR, hk) / N2
    #
    #     h0 = hR[nR//2]
    #     h1R = hR[nR//2+1]
    #     h1L = hR[nR//2-1]
    #     # h0 = np.block([
    #     #     [hR[nR//2], hR[nR//2+1]],
    #     #     [hR[nR//2-1], hR[nR//2]],
    #     # ])
    #     # h1R = np.block([
    #     #     [hR[nR//2+2], hR[nR//2+3]],
    #     #     [hR[nR//2+1], hR[nR//2+2]],
    #     # ])
    #     # h1L = h1R.T.conj()
    #
    #     '''
    #       * green func
    #     '''
    #     GLr = np.zeros([ne, self.nb, self.nb], dtype='complex128')
    #     GRr = np.zeros([ne, self.nb, self.nb], dtype='complex128')
    #     DOS_L = np.zeros([ne], dtype='float64')
    #     DOS_R = np.zeros([ne], dtype='float64')
    #
    #     for i, _e in zip(range(ne), ee):
    #         e = (_e + 1j * eps) * np.identity(self.nb)
    #         selfenLr = self_energy(e - h0, e - h0, h1L, h1L.T.conj())
    #         selfenRr = self_energy(e - h0, e - h0, h1R, h1R.T.conj())
    #         GLr[i] = LA.inv(e - h0 - selfenLr)
    #         GRr[i] = LA.inv(e - h0 - selfenRr)
    #         DOS_L[i] = -2 * np.imag(np.trace(GLr[i]))
    #         DOS_R[i] = -2 * np.imag(np.trace(GRr[i]))
    #
    #     return DOS_L, DOS_R


def return_twist_par(m):
    n = m + 1
    nsuper = m ** 2 + n ** 2 + m * n
    theta = np.arccos((3 * m ** 2 + 3 * m + 1 / 2) / (3 * m ** 2 + 3 * m + 1)) / np.pi * 180
    natom = nsuper * 4
    return nsuper, natom, theta


def return_twist_par2(m, n):
    nsuper = m ** 2 + n ** 2 + m * n
    theta = np.arccos(0.5 * (m**2 + n**2 + 4*m*n) / (m**2 + n**2 + m*n)) / np.pi * 180
    natom = nsuper * 4
    return nsuper, natom, theta


if __name__ == '__main__':
    wdir = r''
    os.chdir(wdir)

    layer_u = Cell()
    layer_d = Cell()
    layer_u.load_poscar(r'graphene_AB_2.vasp')
    layer_d.load_poscar(r'graphene_AB_2.vasp')

    cell = Cell_twisted_hex(layer_u, layer_d, m=5, n=6)
    cell.save_poscar()

    # htb_u = Htb()
    # htb_d = Htb()
    # htb_u.load_htb(r'htb_SLG_SK.h5')
    # htb_d.load_htb(r'htb_SLG_SK.h5')
    # ham = MacDonald_TMG_wan(htb_u, htb_d, m=30, n=31, N=2)
    #
    # kkc = ham.get_kkc(np.array([0, 0, 0]))[0]
    #
    # lattG = ham.lattG
    # lattG_lg = ham.lattG_lg
    #
    # theta = 2 * np.pi / 3
    # rot = np.array([
    #     [np.cos(theta), -np.sin(theta), 0],
    #     [np.sin(theta), np.cos(theta), 0],
    #     [0, 0, 0]
    # ])
    #
    # e_kk = np.remainder(LA.multi_dot([LA.inv(lattG_lg), kkc.T]).T, 1)
    # e_kkc = LA.multi_dot([lattG_lg, e_kk.T]).T
    #
    # c3_kkc = LA.multi_dot([rot, kkc.T]).T
    # c3c3_kkc = LA.multi_dot([rot, rot, kkc.T]).T
    #
    # c3_kk = LA.multi_dot([LA.inv(lattG_lg), c3_kkc.T]).T
    # c3_kk = np.remainder(c3_kk + 1e-10, 1.0)
    #
    # c3c3_kk = LA.multi_dot([LA.inv(lattG_lg), c3c3_kkc.T]).T
    # c3c3_kk = np.remainder(c3c3_kk + 1e-10, 1.0)
    #
    # c3_kkc = LA.multi_dot([lattG_lg, c3_kk.T]).T
    # c3c3_kkc = LA.multi_dot([lattG_lg, c3c3_kk.T]).T
    #
    # # index in kkc after applied c3 and c3c3
    # c3_index = np.argmin(distance.cdist(kkc, c3_kkc, 'euclidean'), 0)
    # c3c3_index = np.argmin(distance.cdist(kkc, c3c3_kkc, 'euclidean'), 0)
    #
    #
    # plot_grid(kkc)
    # plot_grid(c3_kkc)
    # plot_grid(c3c3_kkc)
    #
    # plot_grid(np.vstack([kkc, c3_kkc, c3c3_kkc]))


if __name__ == '__main__':
    ROOT_WDIR = r''
    wdir = os.path.join(ROOT_WDIR, r'TMG_w90/tmg')
    input_dir = os.path.join(ROOT_WDIR, r'TMG_w90/hamlib')

    os.chdir(input_dir)

    m = 12
    Umax = 0.02
    enable_hBN = False
    # # for others
    # U_BN_u = None
    # U_BN_d = None
    # # fot mTBG
    # U_BN_u = np.array([-0.0])
    # U_BN_d = np.array([-0.03])
    # fot mTTG
    # U_BN_u = np.array([-0.0, -0.0])
    # U_BN_d = np.array([0.05])
    # U_BN_u = np.array([-0.02, -0.02])
    # U_BN_d = np.array([0.08])
    # U_BN_u = np.array([-0.1, -0.])
    # U_BN_d = np.array([0.1])
    # for 3+3
    U_BN_u = np.array([0, 0, 0, 0])
    U_BN_d = np.array([0.01, 0.01, 0.0, 0.0])

    # htbfname_u = r'htb_SLG_SK.h5'
    # htbfname_d = r'htb_SLG_SK.h5'
    # htbfname_u = r'htb_SL_DFT.h5'
    # htbfname_d = r'htb_SL_DFT.h5'
    # htbfname_u = r'htb_AB_SCAN.h5'
    # htbfname_d = r'htb_AB_SCAN.h5'
    # htbfname_u = r'htb_AB_SCAN.h5'
    # htbfname_d = r'htb_SL_DFT.h5'
    # htbfname_u = r'htb_ABCA_SCAN.h5'
    # htbfname_d = r'htb_AB_SCAN.h5'
    htbfname_u = r'htb_ABC_SCAN.h5'
    htbfname_d = r'htb_ABC_SCAN.h5'

    htb_u = Htb()
    htb_d = Htb()
    htb_u.load_htb(htbfname_u)
    htb_d.load_htb(htbfname_d)
    ham = MacDonald_TMG_wan(htb_u, htb_d, m=m, n=m+1, N=3, w1=1*0.0797, w2=1*0.0975, tLu=0, tLd=-1, vac=300,
                            rotk=True, Umax=Umax)
    ham.hBN = ham.get_hBN(U_BN_u, U_BN_d, enable=enable_hBN)

    os.chdir(wdir)

    fb_index = ham.get_fb_index()
    k = np.array([0, 0, 0])
    kc = ham.lattG @ k
    hk = ham.get_hk(kc)
    E, U = ham.get_eigh(hk)

