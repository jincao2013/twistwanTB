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
    'trans001', 'trans002', 'kmold'
]


import numpy as np
import numpy.linalg as LA

'''
   Tools
'''
def trans001(astr, asarray=False):
    '''
      astr = '1-3;5-7'
      return [0,1,2, 4,5,6]
      astr = '1-3;5-7;9'
      return [0,1,2, 4,5,6, 8]
    '''
    alist = []

    astr1 = astr.split(';')   # = ['1-3', '5-7']
    for _astr1 in astr1:
        astr2 = _astr1.split('-') # = ['1', '3']
        alist.extend([i for i in range(int(astr2[0])-1, int(astr2[-1]))])
    if asarray:
        alist = np.array(alist, dtype='int64')
    return alist

def trans002(spec, astr):
    '''
      spec = "Fe"
      astr = ["Li", "Fe", "Fe", "P", "P", "P"]
      return [1,2]
    '''
    alist = [i for i, _spec in enumerate(astr) if _spec == spec]
    return alist

def kmold(kkc):
    nk = kkc.shape[0]
    kkc = kkc[1:, :] - kkc[:-1, :]
    kkc = np.vstack([np.array([0, 0, 0]), kkc])
    k_mold = np.sqrt(np.einsum('ka,ka->k', kkc, kkc))
    k_mold = LA.multi_dot([np.tri(nk), k_mold])
    return k_mold