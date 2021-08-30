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

import sys
import os

__author__ = 'Jin Cao'
__email__ = "caojin.phy@gmail.com"
__version__ = "0.2.0"


ROOT_WDIR = os.path.split(__file__)[0]
PYGUI = True

# if os.getenv('WANPY_ROOT_DIR') is not None:
#     ROOT_WDIR = os.getenv('WANPY_ROOT_DIR')

# if os.getenv('PYGUI') in ['True', '1']:
#     PYGUI = True


# ROOT_WDIR = r'/Users/jincao/scidata'
# ROOT_WDIR = r'/Volumes/jindedata/scidata'