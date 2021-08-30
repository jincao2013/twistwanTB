# twistwanTB
`twistwanTB` is an open-source code for building the Ab initio four-band Wannier tight-binding model for generic twisted multilayer graphene (TMG). 

### Build Wannier TB model for generic TMG

1. set parameters in `build_wanTB.py`. 

| Parameters      | Description                                               |
| --------------- | --------------------------------------------------------- |
| `m` and `n`     | They control the twist angle.                             |
| `NG`            | NG controls the num of plane waves, default is 3.         |
| `w1` and `w2`   | Layer coupling constant                                   |
| `Umax`          | Displacement field                                        |
| `enable_hBN`    | If consider BN substrate                                  |
| `U_BN_u(d)`     | hBN potential for the upper (lower) parts                 |
| `htbfname_u(d)` | htb filename for the upper (lower) part                   |
| `mp_grid`       | Wigner-Seitz grid used for interpolating Wannier TB model |
| `if_cal_wfs`    | If give real-space Wannier functions                      |
| `fname_wfs`     | Filename for the Wannier functions                        |

2. `python build_wanTB.py`, all of the Wannier tight-binding information will be saved in a single file `htb.twistTMG.h5`.
3. The `.dat` formated tight-binding model can be obtained by

```python
from twistwanTB.wanpy.structure import Htb
# create Htb instance
htb = Htb()
# load data from .h5 file
htb.load('htb.twistTMG.h5')
# save the wannier charge center in POSCAR.vasp file
htb.save_wcc()
# save the Hamiltonian H(R)=<m0|H|nR> in wannier90_hr.dat formate
htb.save_wannier90_hr_dat()
# save the valley operators V(R)=<m0|V|nR> in wanpy_symmOP_xx.dat formate
htb.save_D_dat()
```

### Estimate Coulomb parameters

1. Set `if_cal_wfs = True` in `build_wanTB.py`  to obtain the real-space Wannier functions. It will be saved in `fname_wfs=wfs.npz` by default. 
2. Set the filename `fname_wfs`  of the calculated Wannier functions in `cal_coulomb.py`, and run the python code `python cal_coulomb.py`. The results will be saved in `xxx.dat` file. 

### Examples

- Magic angle twist bilayer graphene (mTBG) (theta=1.08 degree)
- Magic angle twist bilayer graphene (mTBG) with h-BN substrate (theta=1.08 degree)
- Twist double bilayer graphene (TDBG) (theta=2.0 degree) 
- Magic angle twist double bilayer graphene (mTDBG) with displacement field turned on (theta=1.248 degree) 
- Magic angle twist trilayer graphene (mTTG) with h-BN substrate and displacement field turned on  (theta=1.248 degree) 
- Twist multilayer graphene  (theta=2.646 degree) 
- The real-space Wannier functions for magic angle TBG, TDBG and TTG. 

### Reference

Jin Cao, Maoyuan Wang, Shi-Feng Qian, Cheng-Cheng Liu, and Yugui Yao, [Phys. Rev. B **104**, L081403 (2021)](http://link.aps.org/doi/10.1103/PhysRevB.104.L081403).

