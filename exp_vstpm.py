import numpy as np
import scipy.constants as c
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

class ExpVariationalStellaratorTwoPointModel():
    def __init__(self, q_par, gamma, chi, alpha, theta_arr, ell_arr, q_par_arr = None, q_x=None, rtol=1e-9, atol=1e-11, method='DOP853'):
        self.md = 3.3435837768e-27
        self.kappa_0e = 2000
        self.q_par = q_par
        self.gamma = gamma
        self.chi = chi
        self.alpha = alpha
        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.ell_arr = np.asarray(ell_arr, dtype=float)      # ensure its in meters, target-first
        self.theta_arr = np.asarray(theta_arr, dtype=float) 
        # Safeguarding   
        if self.theta_arr.shape != self.ell_arr.shape:
            raise ValueError(f"theta_arr {self.theta_arr.shape} and ell_arr "
                             f"{self.ell_arr.shape} must have the same shape")
        if np.any(np.diff(self.ell_arr) <= 0.0):
            raise ValueError("ell_arr must be strictly increasing (target first)")
        if np.min(self.theta_arr) <= 0.0:
            raise ValueError("Theta must be strictly positive")

        self.ell_0, self.ell_1 = self.ell_arr[0], self.ell_arr[-1]
        self.L_c = self.ell_1 - self.ell_0
        self.theta_d = float(self.theta_arr[0])

        self.q_x = self.q_par * self.theta_d if q_x is None else float(q_x)
        self.q_par_target = self.q_x / self.theta_d
        self.q_par_arr = q_par_arr

    def theta_of(self, ell):
        return np.interp(ell, self.ell_arr, self.theta_arr)

    def q_par_of(self, ell):
        if self.q_par_arr:
            return np.interp(ell, self.ell_arr, self.q_par_arr)
        else:
            return self.q_par_target
        
    def cs_d(self, Td):
        return np.sqrt(2*(c.e * Td)/self.md)

    def nd_of_Td(self, Td):
        return self.q_par_target / (self.gamma * self.cs_d(Td) * c.e * Td)

    def f_m(self, Td):
        return self.alpha / np.sqrt(Td)

    def Tu_mom(self, Td, nu):
        nd = self.nd_of_Td(Td=Td)
        Tu = (2*nd*Td*(1+self.f_m(Td=Td)))/nu
        return Tu

    def dT_dl(self, ell, T, perp_coeff):
        theta = self.theta_of(ell=ell)
        q_par = self.q_par_of(ell=ell)
        T = np.maximum(T, 1e-12)
        return (q_par * theta**2) / (self.kappa_0e * T**2.5 * theta**2 + perp_coeff)
      
    def T_of_ell(self, ell, Td=None, nu=None):
        """Temperature [eV] along the field line, integrated from the target (ell_0, Td).

        ell may be a scalar or an array (any order, in [ell_0, ell_1]); the result has
        the same shape.  Td and nu default to the last state returned by solve(), so
        T_of_ell(ell) gives the profile of the solved solution.  The ODE and the
        perpendicular coefficient are identical to Tu_ode, hence
        T_of_ell(ell_1) == Tu_ode(Td, nu).
        """
        if Td is None: Td = getattr(self, "Td_sol", None)
        if nu is None: nu = getattr(self, "nu_sol", None)
        if Td is None or nu is None:
            raise ValueError("Td and nu must be given, or solve() must be run first")
        Td, nu = float(Td), float(nu)

        ell = np.asarray(ell, dtype=float)
        tol = 1e-12 * self.L_c
        if np.any(ell < self.ell_0 - tol) or np.any(ell > self.ell_1 + tol):
            raise ValueError(f"ell must be within [{self.ell_0}, {self.ell_1}]")
        ell_c = np.clip(ell, self.ell_0, self.ell_1)

        ell_eval = np.unique(ell_c.ravel())          # sorted, as t_eval requires
        T_eval = np.full(ell_eval.shape, Td)
        if ell_eval[-1] > self.ell_0:
            perp_coeff = self.chi * (nu + self.nd_of_Td(Td)) * c.e / 2
            sol = solve_ivp(self.dT_dl, (self.ell_0, ell_eval[-1]), [Td],
                            args=(perp_coeff,), t_eval=ell_eval,
                            method=self.method, rtol=self.rtol, atol=self.atol)
            if not sol.success:
                raise RuntimeError(f"ODE integration failed: {sol.message}")
            T_eval = sol.y[0]

        T = T_eval[np.searchsorted(ell_eval, ell_c.ravel())].reshape(ell.shape)
        return float(T) if T.ndim == 0 else T

    def Tu_ode(self, Td, nu):
        Td = np.atleast_1d(np.asarray(Td, dtype=float))
        nu = np.atleast_1d(np.asarray(nu, dtype=float))

        Td, nu = np.broadcast_arrays(Td, nu)

        n_avg = (nu + self.nd_of_Td(Td))/2
        perp_coeff = self.chi * n_avg * c.e

        sol = solve_ivp(self.dT_dl, (self.ell_0, self.ell_1), Td.ravel(),
                        args=(perp_coeff.ravel(),), t_eval=[self.ell_1],
                        method=self.method, rtol=self.rtol, atol=self.atol)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol.y[:, -1].reshape(Td.shape)

    def T_profile(self, Td, nu):
        """Temperature on ell_arr for every (Td, nu); shape broadcast(Td, nu).shape + (len(ell_arr),)."""
        Td, nu = np.broadcast_arrays(np.asarray(Td, dtype=float), np.asarray(nu, dtype=float))
        shape = Td.shape
        Td, nu = Td.ravel(), nu.ravel()
        perp_coeff = self.chi * (nu + self.nd_of_Td(Td)) * c.e / 2
        sol = solve_ivp(self.dT_dl, (self.ell_0, self.ell_1), Td,
                        args=(perp_coeff,), t_eval=self.ell_arr,
                        method=self.method, rtol=self.rtol, atol=self.atol)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol.y.reshape(shape + (self.ell_arr.size,))

    def t_star_ell(self, nu, nd, T):
        """t* = sqrt(perp/para) = sqrt(chi * n_avg * e / (kappa_0e * T**(5/2))), n_avg = (nu + nd)/2.

        Same perpendicular coefficient as the ODE (dT_dl), so that the ODE
        denominator reads kappa_0e * T**2.5 * (theta**2 + t*(T)**2).
        """
        perp = self.chi * (nu + nd) * c.e / 2
        para = self.kappa_0e * np.maximum(T, 1e-12)**2.5
        return np.sqrt(perp / para)

    def t_star_profile(self, T_ell, nu, nd):
        """t* on ell_arr from a temperature profile T_ell of shape (..., len(ell_arr)).

        nu and nd broadcast against the leading dimensions of T_ell.
        """
        T_ell = np.asarray(T_ell, dtype=float)
        nu = np.asarray(nu, dtype=float)[..., None]
        nd = np.asarray(nd, dtype=float)[..., None]
        return self.t_star_ell(nu, nd, T_ell)

    def residual(self, Td, nu):
        return self.Tu_ode(Td, nu) - self.Tu_mom(Td, nu)

    def solve(self, nu, lo=1e-3, hi=1e2, lo_min=1e-8, hi_max=1e5):
        res = lambda Td: float(self.residual(Td, nu).item())
        f_lo = res(lo)
        f_hi = res(hi)

        while f_lo > 0.0 and lo > lo_min:
            hi, f_hi = lo, f_lo
            lo = max(lo * 0.1, lo_min)
            f_lo = res(lo)

        while f_hi < 0.0 and hi < hi_max:
            lo, f_lo = hi, f_hi
            hi = min(hi * 10.0, hi_max)
            f_hi = res(hi)

        if f_lo * f_hi > 0.0:
            raise RuntimeError(
                f"no bracket for nu={nu:.3e} within [{lo_min:.1e}, {hi_max:.1e}] eV: "
                f"res({lo:.3g})={f_lo:.3g}, res({hi:.3g})={f_hi:.3g}")

        Td = brentq(res, lo, hi, xtol=1e-12, rtol=1e-12)
        self.Td_sol, self.nu_sol = float(Td), float(nu)   # used by T_of_ell defaults
        return self._state(Td, nu)

    def _state(self, Td, nu):
        nd = self.nd_of_Td(Td)
        T_ell = self.T_profile(Td, nu)
        Tu = T_ell[..., -1]
        Td, nd, Tu = np.asarray(Td), np.asarray(nd), np.asarray(Tu)
        out = {"Td": Td, "Tu": Tu, "nd": nd, "nu": np.asarray(nu),
            "pd": 2 * nd * Td, "pu": 2 * np.asarray(nu) * Tu,
            "fm": np.asarray(self.f_m(Td)), "csd": np.asarray(self.cs_d(Td))}
        if np.ndim(nu) == 0:
            out = {k: float(v) for k, v in out.items()}
        # temperature along the field line, [..., len(ell)] on ell_arr
        out["ell"] = self.ell_arr.copy()
        out["T_ell"] = T_ell
        # t* = sqrt(perp/para) along the field line, [..., len(ell)] on ell_arr
        out["t_star_ell"] = self.t_star_profile(T_ell, nu, nd)
        return out

    def scan(self, nu_array, lo=1e-3, hi=1e2, lo_min=1e-8, hi_max=1e5,
             rtol=1e-11, max_iter=60):
        """Solve the whole nu scan at once.

        Every nu shares one ODE integration per iteration -- the ODE is linear in
        the number of unknowns but its step count is not, so bisecting all of
        them together costs about as much as bisecting one.  Bisection is done in
        log(Td), which is why a 5-decade start bracket converges in ~40 steps.
        """
        nu = np.atleast_1d(np.asarray(nu_array, dtype=float))
        lo = np.full(nu.shape, float(lo))
        hi = np.full(nu.shape, float(hi))
        f_lo = self.residual(lo, nu)
        f_hi = self.residual(hi, nu)

        while np.any(bad := (f_lo > 0.0) & (lo > lo_min)):
            hi, f_hi = np.where(bad, lo, hi), np.where(bad, f_lo, f_hi)
            lo = np.where(bad, np.maximum(lo * 0.1, lo_min), lo)
            f_lo = np.where(bad, self.residual(lo, nu), f_lo)

        while np.any(bad := (f_hi < 0.0) & (hi < hi_max)):
            lo, f_lo = np.where(bad, hi, lo), np.where(bad, f_hi, f_lo)
            hi = np.where(bad, np.minimum(hi * 10.0, hi_max), hi)
            f_hi = np.where(bad, self.residual(hi, nu), f_hi)

        unbracketed = f_lo * f_hi > 0.0
        if unbracketed.any():
            for i in np.flatnonzero(unbracketed):
                print(f"  [warn] nu={nu[i]:.3e}: no bracket within "
                      f"[{lo_min:.1e}, {hi_max:.1e}] eV")
            # park them on a finite dummy so they cannot poison the shared ODE
            lo = np.where(unbracketed, 1.0, lo)
            hi = np.where(unbracketed, 1.0, hi)

        for _ in range(max_iter):
            if np.all(hi - lo <= rtol * hi):
                break
            mid = np.sqrt(lo * hi)
            neg = self.residual(mid, nu) < 0.0
            lo = np.where(neg, mid, lo)
            hi = np.where(neg, hi, mid)

        Td = np.sqrt(lo * hi)
        out = self._state(Td, nu)
        return {k: v if k == "ell" else
                np.where(unbracketed.reshape(unbracketed.shape + (1,) * (np.ndim(v) - 1)), np.nan, v)
                for k, v in out.items()}
    

def vstpm_scan(theta, ell, n_u, gamma, chi, alpha, q_x):
    """The VSTPM over the upstream densities n_u for the pitch profile theta(ell), ell in
    metres from the target, with q_par = q_x / theta[0]."""
    return ExpVariationalStellaratorTwoPointModel(q_par=q_x / theta[0], gamma=gamma, chi=chi,
                                               alpha=alpha, theta_arr=theta, ell_arr=ell,
                                               q_x=q_x).scan(nu_array=n_u)
