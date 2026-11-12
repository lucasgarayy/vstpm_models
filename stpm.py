import numpy as np
import scipy.constants as c
from scipy.optimize import brentq

class StellaratorTwoPointModel():
    def __init__(self, q_par, gamma, chi, theta, alpha, L_c):
        self.md = 3.3435837768e-27
        self.kappa_0e = 2000
        self.q_par = q_par
        self.gamma = gamma
        self.chi = chi
        self.Theta = theta
        self.alpha = alpha
        self.L_c =  L_c

    def cs_d(self, Td):
        return np.sqrt(2*(c.e * Td)/self.md)        

    def nd_of_Td(self, Td):
        nd = self.q_par / (self.gamma * self.cs_d(Td) * c.e * Td)
        return nd 

    def f_m(self, Td):
        return self.alpha / np.sqrt(Td)

    def Tu_mom(self, Td, nu):
        nd = self.nd_of_Td(Td=Td)
        Tu = (2*nd*Td*(1+self.f_m(Td=Td)))/nu
        return Tu

    def energy_res(self, Td, nu):
        nd = self.nd_of_Td(Td=Td)
        Tu = self.Tu_mom(Td=Td, nu=nu)
        perp_coeff = (7 * self.chi * (nu + nd) * c.e) / (4 * self.kappa_0e * self.Theta ** 2)
        para_coeff = (7 * self.q_par * self.L_c) / (2 * self.kappa_0e)
        res = (Td**3.5 - Tu**3.5) + para_coeff - perp_coeff * (Tu - Td)
        return res

    def solve(self, nu, lo=1e-3, hi=1e2, lo_min=1e-8, hi_max=1e5):
        f_lo = self.energy_res(lo, nu)
        while f_lo > 0.0:                      
            lo /= 4.0
            if lo < lo_min:
                raise RuntimeError(f"no lower bracket above Td={lo_min:g} eV")
            f_lo = self.energy_res(lo, nu)
 
        while self.energy_res(hi, nu) < 0.0:   
            hi *= 2.0
            if hi > hi_max:
                raise RuntimeError(f"no upper bracket below Td={hi_max:g} eV")
 
        Td = brentq(self.energy_res, lo, hi, args=(nu,),
                    xtol=1e-12, rtol=8.9e-16, maxiter=2000)
        nd = self.nd_of_Td(Td)
        Tu = self.Tu_mom(Td, nu)
        return dict(Td=Td, Tu=Tu, nd=nd, nu=nu,
                    pd=2 * nd * Td, pu=2 * nu * Tu,
                    fm=self.f_m(Td), csd=self.cs_d(Td))
    
    def scan(self, nu_array):
        keys = ("Td", "Tu", "nd", "fm", "pd", "pu")
        out = {k: np.full(nu_array.size, np.nan) for k in keys}
        for i, nu in enumerate(nu_array):
            try:
                s = self.solve(nu)
            except (RuntimeError, ValueError) as err:
                print(f"  [warn] nu={nu:.3e}: {err}")
                continue
            for k in keys:
                out[k][i] = s[k]
        return out


def stpm_scan(theta, L_c, n_u, gamma, chi, alpha, q_x):
    """The STPM over the upstream densities n_u for a line of constant pitch theta and
    connection length L_c, with q_par = q_x / theta."""
    return StellaratorTwoPointModel(q_par=q_x / theta, gamma=gamma, chi=chi, theta=theta,
                                    alpha=alpha, L_c=L_c).scan(nu_array=n_u)
