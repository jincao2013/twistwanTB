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

__date__ = "Aug. 11, 2017"

__all__ = [
    'make_mesh',
    'make_kpath',
]

import time
import numpy as np

'''
  * k mesh
'''
def make_mesh(nmesh, type='continuous', centersym=False, basis=np.identity(3), mesh_shift=0., info=False):
    T0 = time.time()
    '''
    * type = continuous, discrete
    * centersym = False, True
    * basis = np.array([f1, f2, f3]) 
        
    `basis` is used to get custom shape of BZ, 
    the original BZ is defined by `lattG`
        lattG = np.array([
            [b11, b21, b31],
            [b12, b22, b32],
            [b13, b23, b33],
        ])
    the new BZ is defined as  
        lattG'[0] = f11 b1 + f12 b2 + f13 b3
        lattG'[1] = f21 b1 + f22 b2 + f23 b3
        lattG'[2] = f31 b1 + f32 b2 + f33 b3
    it is obtained by 
        lattG' = lattG @ basis.T                (1)
    where basis is defined as 
        basis.T = np.array([
            [f11, f21, f31],
            [f12, f22, f32],
            [f13, f23, f33],
        ])
    or 
        basis = np.array([f1, f2, f3])          (2)

    '''  #
    N1, N2, N3 = nmesh
    N = N1 * N2 * N3
    # n2, n1, n3 = np.meshgrid(np.arange(N2), np.arange(N1), np.arange(N3))
    n1, n2, n3 = np.mgrid[0:N1:1, 0:N2:1, 0:N3:1]
    mesh = np.array([n1.flatten(), n2.flatten(), n3.flatten()], dtype='float64').T

    if centersym:
        if not (np.mod(nmesh, 2) == 1).all():
            print('centersym mesh need odd number of [nmesh]')
            # sys.exit(1)
        else:
            mesh -= mesh[N // 2]
    if type[0].lower() == 'c':
        mesh /= nmesh

    # mesh = LA.multi_dot([basis.T, mesh.T]).T + mesh_shift
    mesh = (basis.T @ mesh.T).T + mesh_shift
    if info:
        print('Make mesh complited. Time consuming {} s'.format(time.time()-T0))

    return mesh

def make_kpath(kpath_list, nk1):

    # print('\n')
    # print("[from make_kpath] Attention : ")
    # print("[from make_kpath] kpath_list should in unit of b1 b2 b3")

    kpaths = [np.array(i) for i in kpath_list]
    kpaths_delta = [kpaths[i] - kpaths[i-1] for i in range(1,len(kpaths))]

    stage = len(kpaths_delta)

    kmesh_1d = np.zeros((nk1 * stage + 1, 3))
    i = 0
    for g_index in range(stage):   # g is high sym kpoint witch given by kpath_list
        for n1 in range(nk1):
            k = kpaths_delta[g_index] * n1/nk1 + kpaths[g_index]
            kmesh_1d[i] = k
            i = i + 1

    kmesh_1d[-1] = kpaths[-1]

    return kmesh_1d
    