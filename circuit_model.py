# --------------------------------------------------------------------
# Description: This module is used to define the testing circuit model for the resonators.
# --------------------------------------------------------------------

import numpy as np

L_expected = 3.3257e-6
Z0 = 50

def Z_Res(f, L, C, Rs, Rp):
    Zc = 1/(1j*2*np.pi*f*C)
    Rc = Rs + 1j*2*np.pi*f*L + Zc*Rp/(Zc+Rp)
    return Rc

def S21(f, Z0, L, C, Rs):
    Res = Rs + 1j*2*np.pi*f*L + 1/(1j*2*np.pi*f*C)
    ZL = Res*Z0/(Res+Z0)
    return 1+(ZL-Z0)/(ZL+Z0)

def S21_res(f, Z0, C, Rs, Rp, Ra, Rb):
    L = L_expected
    Zc = 1/(1j*2*np.pi*f*C)
    Rc = Rs + 1j*2*np.pi*f*L + Zc*Rp/(Zc+Rp)
    return 2*Rc*Z0/(Z0**2+Ra*Z0+Rb*Z0+2*Rc*Z0+Ra*Rb+Ra*Rc+Rb*Rc)

def S21_ind(f, Z0, L, Rs):
    Rc = 1j*2*np.pi*f*L + Rs
    return 2*Rc*Z0/(Z0**2+2*Rc*Z0)

def simple_lorentzian(f, f0, Q, L):
    # a = Rs/4piL
    # b = Z0/8piL
    # Q = f0/2a
    # L = L_expected
    a = f0/(2*Q)
    b = Z0/(8*np.pi*L)
    A = 2 * a * b + b**2
    B = (f-f0)**2 + a**2
    return 10*np.log10(1/(1+A/B))

def simple_ind(f, L):
    Z_T = 1j*2*np.pi*f*L
    Z_L = Z_T*Z0/(Z_T+Z0)
    S = 2*Z_L/(Z0+Z_L)
    return 20*np.log10(S)   