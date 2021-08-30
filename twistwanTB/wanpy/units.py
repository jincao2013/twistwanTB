# Copyright (C) 2011 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy projects nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np

__all__ = ['sigma', 'sigma0', 'sigmax', 'sigmay', 'sigmaz', 'levi_civita',
           'kb_J', 'PlanckConstant', 'Hbar', 'Avogadro', 'SpeedOfLight', 'AMU', 'Newton',
           'Joule', 'EV', 'Angstrom', 'THz', 'Mu0', 'Epsilon0', 'Me', 'Bohr', 'Hartree',
           'Rydberg', 'NiuB', 'THzToEv', 'Kb', 'THzToCm', 'CmToEv', 'VaspToEv', 'VaspToTHz',
           'VaspToCm', 'EvTokJmol', 'Wien2kToTHz', 'AbinitToTHz', 'PwscfToTHz', 'ElkToTHz',
           'SiestaToTHz', 'CP2KToTHz', 'CrystalToTHz', 'EVAngstromToGPa',
           'Hbar_IS', 'NiuB_SI', 'PlanckConstant_IS']

pi = np.pi

'''
  * Physical Constants
  **  copy from phonopy code 
'''
kb_J = 1.3806504e-23 # [J/K]
PlanckConstant = 4.13566733e-15 # [eV s]        # h/e
Hbar = PlanckConstant/(2*pi) # [eV s]
Avogadro = 6.02214179e23
SpeedOfLight = 299792458 # [m/s]
AMU = 1.6605402e-27 # [kg]
Newton = 1.0        # [kg m / s^2]
Joule = 1.0         # [kg m^2 / s^2]
EV = 1.60217733e-19 # [J]
Angstrom = 1.0e-10  # [m]
THz = 1.0e12        # [/s]
Mu0 = 4.0e-7 * pi   # [Henry/m] or [N/A^2]
Epsilon0 = 1.0 / Mu0 / SpeedOfLight**2 # [C^2 / N m^2]
Me = 9.10938215e-31

Bohr = 4e10 * pi * Epsilon0 * Hbar**2 / Me  # Bohr radius [A] 0.5291772
Hartree = Me * EV / 16 / pi**2 / Epsilon0**2 / Hbar**2 # Hartree [eV] 27.211398
Rydberg = Hartree / 2 # Rydberg [eV] 13.6056991
NiuB = EV * Hbar / Me / 2  # Bohr magneton [eV/T] 5.78838e-05

THzToEv = PlanckConstant * 1e12 # [eV]
Kb = kb_J / EV  # [eV/K] 8.6173383e-05
THzToCm = 1.0e12 / (SpeedOfLight * 100) # [cm^-1] 33.356410
CmToEv = THzToEv / THzToCm # [eV] 1.2398419e-4
VaspToEv = np.sqrt(EV/AMU)/Angstrom/(2*pi)*PlanckConstant # [eV] 6.46541380e-2
VaspToTHz = np.sqrt(EV/AMU)/Angstrom/(2*pi)/1e12 # [THz] 15.633302
VaspToCm =  VaspToTHz * THzToCm # [cm^-1] 521.47083
EvTokJmol = EV / 1000 * Avogadro # [kJ/mol] 96.4853910
Wien2kToTHz = np.sqrt(Rydberg/1000*EV/AMU)/(Bohr*1e-10)/(2*pi)/1e12 # [THz] 3.44595837
AbinitToTHz = np.sqrt(EV/(AMU*Bohr))/Angstrom/(2*pi)/1e12 # [THz] 21.49068
PwscfToTHz = np.sqrt(Rydberg*EV/AMU)/(Bohr*1e-10)/(2*pi)/1e12 # [THz] 108.97077
ElkToTHz = np.sqrt(Hartree*EV/AMU)/(Bohr*1e-10)/(2*pi)/1e12 # [THz] 154.10794
SiestaToTHz = np.sqrt(EV/(AMU*Bohr))/Angstrom/(2*pi)/1e12 # [THz] 21.49068
CP2KToTHz = ElkToTHz  # CP2K uses a.u. units (Hartree/Bohr)
CrystalToTHz = VaspToTHz
EVAngstromToGPa = EV * 1e21


'''
  * Physical Constants
  **  wanpy added
'''
Hbar_IS = Hbar * EV
PlanckConstant_IS = PlanckConstant * EV
NiuB_SI = NiuB * EV

'''
  * Numerical Convergence Constants 
'''
# Inf = np.inf
# Eps2 = np.float64(1.0e-2)
# Eps3 = np.float64(1.0e-3)
# Eps4 = np.float64(1.0e-4)
# Eps5 = np.float64(1.0e-5)
# Eps6 = np.float64(1.0e-6)
# Eps7 = np.float64(1.0e-7)
# Eps8 = np.float64(1.0e-8)
# Eps9 = np.float64(1.0e-9)
# Eps10 = np.float64(1.0e-10)
# Eps16 = np.float64(1.0e-16)
# Eps32 = np.float64(1.0e-32)
# Eps64 = np.float64(1.0e-64)

'''
  * Dirac Algebra 
'''
sigma_0 = np.array([
    [1, 0],
    [0, 1]
], dtype='complex128')
sigma_x = np.array([
    [0, 1],
    [1, 0]
], dtype='complex128')
sigma_y = np.array([
    [0, -1j],
    [1j,  0]
], dtype='complex128')
sigma_z = np.array([
    [1,  0],
    [0, -1]
], dtype='complex128')
sigma0 = np.array([
    [1, 0],
    [0, 1]
], dtype='complex128')
sigmax = np.array([
    [0, 1],
    [1, 0]
], dtype='complex128')
sigmay = np.array([
    [0, -1j],
    [1j,  0]
], dtype='complex128')
sigmaz = np.array([
    [1,  0],
    [0, -1]
], dtype='complex128')

sigma = np.array([
    sigma_0, sigma_x, sigma_y, sigma_z
])

levi_civita = np.array([
    [
        [ 0,  0,  0],
        [ 0,  0,  1],
        [ 0, -1,  0]
    ],
    [
        [ 0,  0, -1],
        [ 0,  0,  0],
        [ 1,  0,  0]
    ],
    [
        [ 0,  1,  0],
        [-1,  0,  0],
        [ 0,  0,  0]
    ]
])

