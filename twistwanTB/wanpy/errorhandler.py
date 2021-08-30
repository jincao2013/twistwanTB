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

__date__ = "Otc. 21, 2020"

'''
  * Wanpy build-in Exception
'''
class WanpyInputError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

class CommandNotFoundError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo=" No command '{}'.".format(ErrorInfo)
    def __str__(self):
        return self.errorinfo


