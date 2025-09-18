import numpy as np
from scipy.integrate import simpson

from nrgljubljana_interface import Solver, Flat, SemiCircular
from h5 import *
import time
from scipy import interpolate
from triqs.utility import mpi
import scipy.fftpack as ft
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpi4py import MPI
import os
import setproctitle

U = os.getenv('U_VALUE', '0.0')
setproctitle.setproctitle(f'foreward_T1e-4_U{U}')

#setproctitle.setproctitle('u3.5')

D = 1.0               
U = 2.4
#eta = 1e-6            
n_loops = 1000          
T = 1e-4
Delta_min = 1e-5
S = Solver(model = "SIAM", symtype = "QS", mesh_max = 10.0, mesh_min = 1e-10,
        mesh_ratio = 1.01)

w=S.G_w.mesh
freq = np.array(list(w.values()))
omega = freq

#S.Delta_w['imp'] << SemiCircular(D)
#hyb = S.Delta_w['imp'].data
#Delta = hyb[:,0,0].real + 1j*hyb[:,0,0].imag

data = np.loadtxt("delta.dat")
Delta = data[:,1]+1j*data[:,2]
omega = data[:,0]

sp = { "T": T, "Lambda": 3, "Nz": 8, "Tmin": 1e-10, "keep": 10000,
        "keepenergy": 10.0}


ht1 = lambda z: 2*(z-1j*np.sign(z.imag)*np.sqrt(1-z**2))
ht0 = lambda z: ht1(z/D)
EPS = 1e-20
ht = lambda z: ht0(z.real+1j*(z.imag if z.imag>0.0 else EPS))

def compute_typical_rho(omega, Delta):
        e_f = -U/2.0 
        # Model Parameters
        mp = { "U1": U, "eps1": e_f }
        sp["model_parameters"] = mp

        S.solve(**sp)

        Sw = S.Sigma_w['imp']
        Sigma = Sw[0,0].data.real + 1j*Sw[0,0].data.imag

        Gii = 1.0 / (omega + 1j*1e-8 - e_f - Delta - Sigma)
        Gloc = np.zeros_like(Sigma)
        for m in range(len(omega)):
            Gloc[m] = ht(omega[m] - e_f - Sigma[m])
        rho_i = -np.imag(Gloc) / np.pi
        return rho_i, Gloc, Sigma, Gii

# DMFT loop
for i in range(n_loops):
    if mpi.is_master_node(): print(f"DMFT loop {i} -------------------")

    S.Delta_w['imp'].data[:, 0, 0] = Delta

    rho, Gloc, Sigma, Gii = compute_typical_rho(omega, Delta)

    np.savetxt(f"{i}_rho.txt", np.column_stack([omega, rho]))
    np.savetxt(f"{i}_sigma.txt", np.column_stack([omega, Sigma.real, Sigma.imag]))
    np.savetxt(f"{i}_gloc.txt", np.column_stack([omega, Gloc.real, Gloc.imag]))
    np.savetxt(f"{i}_gimp.txt", np.column_stack([omega, Gii.real, Gii.imag]))

    Gloc_inv = 1.0 / Gloc
    Delta_mod = (D**2)*Gloc/4.0
    np.savetxt(f"{i}_delta.txt", np.column_stack([omega, Delta_mod.real, Delta_mod.imag]))

    #diff = np.sum(np.abs(Delta_mod - Delta))
    #norm = np.sum(np.abs(Delta)) + 1e-12
    diff = np.sum(np.abs(Gloc - Gii))
    norm = np.sum(np.abs(Gii)) + 1e-12
    err = diff / norm
    if mpi.is_master_node(): print(f"Convergence: {err}")
    if err < 1e-5:
        break

    #Delta = 0.5*Delta + 0.5*Delta_mod
    Delta_new = 0.5*Delta + 0.5*Delta_mod

    for w in range(len(omega)):
        r = Delta_new[w].real
        i = Delta_new[w].imag
        Delta[w] = r + 1j*(i if i<-Delta_min else -Delta_min)

np.savetxt("rho_final.txt", np.column_stack([omega, rho]))
np.savetxt("delta_final.txt", np.column_stack([omega, Delta_mod.real, Delta_mod.imag]))
np.savetxt("sigma_final.txt", np.column_stack([omega, Sigma.real, Sigma.imag]))
np.savetxt("gloc.txt", np.column_stack([omega, Gloc.real, Gloc.imag]))
np.savetxt("gimp.txt", np.column_stack([omega, Gii.real, Gii.imag]))

