#!/usr/bin/env python3
"""Environment: an agent searches for an odor source it cannot see.

A source at an unknown position emits particles that are carried by the
wind and spread by turbulence. The agent does not measure a smooth
concentration: at each step it receives a number of DETECTIONS h, drawn
from a Poisson distribution, and most of the time h = 0.

Physics (Vergassola, Villermaux & Shraiman, Nature 2007): stationary
advection-diffusion with finite particle lifetime,

    D lap c - V . grad c - c/tau + R delta(r - r0) = 0

whose 2D solution is

    c(r) = (R / 2 pi D) exp( V.(r-r0) / 2D ) K_0( |r-r0| / lam )
    1/lam^2 = V^2/(4 D^2) + 1/(D tau)

The detection rate of a sensor of radius a is proportional to c,
normalised with ln(lam/a). Detections in one step are Poisson with mean
rate * dt.

Units: lattice units (dx = 1, dt = 1). With D=1, V=0.2, tau=2000 the
plume is long and narrow: lam ~ 9.8, decay length ~400 cells downwind and
~5 cells upwind.

Note: in the thesis this file is called the 'plume' environment.
"""
import numpy as np
from scipy.special import k0

AZIONI = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
N_AZIONI = 5
NOMI_AZIONI = ["fermo", "+x (sottovento)", "-x (sopravvento)", "+y", "-y"]


class Pennacchio:
    def __init__(self, n=41, D=1.0, V=0.2, tau=2000.0, R_em=0.5, a=1.0,
                 dt=1.0, passi=600, d_minima_iniziale=8.0, h_max=3):
        self.n = n
        self.D, self.V, self.tau = D, V, tau
        self.R_em, self.a, self.dt = R_em, a, dt
        self.passi = passi
        self.d0 = d_minima_iniziale
        self.h_max = h_max          # counts above this value are merged
        self.lam = 1.0 / np.sqrt(V ** 2 / (4 * D ** 2) + 1.0 / (D * tau))
        self._norm = R_em / np.log(self.lam / a)
        # grid of candidate source positions
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self.celle = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)

    # ------------------------------------------------------------ physics
    def tasso(self, r, r0):
        """Detection rate at r for a source at r0.

        r0 can be an (m, 2) array: returns (m,). This lets the Bayesian update
        evaluate all candidate source cells at once.
        """
        r = np.atleast_2d(np.asarray(r, dtype=float))
        r0 = np.atleast_2d(np.asarray(r0, dtype=float))
        d = r - r0                                   # from source to sensor
        dist = np.maximum(np.linalg.norm(d, axis=-1), self.a)
        # exp( V.(r-r0)/2D ) with V along +x
        esp = self.V * d[..., 0] / (2.0 * self.D)
        # combine the exponents before exponentiating: K_0(z) ~ exp(-z)/sqrt(z),
        # and keeping them separate overflows at large distances
        return self._norm * np.exp(esp - dist / self.lam) * \
            (k0(dist / self.lam) * np.exp(dist / self.lam))

    def media_conteggi(self, r, r0):
        return self.tasso(r, r0) * self.dt

    # ------------------------------------------------------------- state
    def reset(self, rng):
        self.rng = rng
        self.t = 0
        self.r0 = rng.integers(0, self.n, size=2).astype(float)
        for _ in range(10000):
            r = rng.integers(0, self.n, size=2).astype(float)
            if np.linalg.norm(r - self.r0) >= self.d0:
                self.r = r
                break
        else:
            self.r = (self.r0 + np.array([self.d0, 0.0])) % self.n
        return self._osserva()

    def _osserva(self):
        mu = float(self.media_conteggi(self.r, self.r0)[0])
        return int(self.rng.poisson(mu))

    def passo(self, azione):
        self.r = np.clip(self.r + AZIONI[azione], 0, self.n - 1).astype(float)
        self.t += 1
        trovato = bool(np.linalg.norm(self.r - self.r0) < self.a)
        h = 0 if trovato else self._osserva()
        finito = trovato or self.t >= self.passi
        return h, trovato, finito, {"t": self.t, "trovato": trovato}

    # --------------------------------------------------------- utilities
    def passi_oracolo(self):
        """Steps needed if the source position were known (Manhattan distance).

        With 4 unit moves this is the absolute lower bound (the 'oracle').
        """
        return float(np.abs(self.r - self.r0).sum())

    def campo(self):
        """Detection rate over the whole grid (used only for plotting)."""
        return self.tasso(self.celle, self.r0).reshape(self.n, self.n)

    # ------------------------------------------------------- speed-up
    _tab = None

    def tabella(self):
        """Precomputed mean count mu as a function of the sensor-source offset.

        mu only depends on r - r0 and everything lives on a lattice, so the
        (2n-1)^2 possible values are computed once. Without this table infotaxis
        spends ~95% of its time evaluating K_0.
        """
        if self._tab is None:
            d = np.arange(-(self.n - 1), self.n)
            DX, DY = np.meshgrid(d, d, indexing="ij")
            pts = np.stack([DX.ravel(), DY.ravel()], axis=1).astype(float)
            self._tab = (self.tasso(pts, np.zeros((1, 2)))
                         .reshape(len(d), len(d)) * self.dt)
        return self._tab

    def mu_griglia(self, r):
        """mu for EVERY candidate source cell, with the sensor at r."""
        tab = self.tabella()
        ix = (int(r[0]) - np.arange(self.n)) + (self.n - 1)
        iy = (int(r[1]) - np.arange(self.n)) + (self.n - 1)
        return tab[np.ix_(ix, iy)]

    def vicini(self, r):
        """Positions reachable with the 5 actions, clipped to the grid."""
        return [tuple(np.clip(np.asarray(r) + a, 0, self.n - 1).astype(int))
                for a in AZIONI]
