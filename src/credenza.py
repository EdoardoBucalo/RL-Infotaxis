#!/usr/bin/env python3
"""Belief: the posterior probability of the source position, given everything seen so far.

The belief is a 41x41 map updated with Bayes' rule. It is EXACT (no
approximation, no network) and it is the same for every strategy: the
strategies only differ in how they DECIDE the next move given the belief.

Two updates are applied at each step:
  1. the observed count h: the likelihood of h for a source in r0 is
     Poisson with mean mu(r, r0);
  2. 'I am here and I did not find it': the current cell is set to zero.
"""
import numpy as np
from scipy.special import gammaln


def _poisson(h, mu):
    """P(h | mu), numerically stable also for very small mu."""
    mu = np.maximum(mu, 1e-300)
    return np.exp(-mu + h * np.log(mu) - gammaln(h + 1.0))


class Credenza:
    def __init__(self, pennacchio, prior=None):
        self.p = pennacchio
        n = pennacchio.n
        self.b = np.full((n, n), 1.0 / (n * n)) if prior is None \
            else prior.copy()

    # ---------------------------------------------------------- update
    def aggiorna(self, r, h):
        mu = self.p.mu_griglia(r)
        self.b = self.b * _poisson(h, mu)
        i, j = int(r[0]), int(r[1])
        self.b[i, j] = 0.0            # I am here and I did not find it
        s = self.b.sum()
        if s <= 0.0:
            # all hypotheses are zero: impossible with the correct model, so it is
            # a bug and must not be hidden by silently going back to the prior
            raise RuntimeError("credenza azzerata: modello incoerente "
                               "con le osservazioni")
        self.b /= s

    # ----------------------------------------------------------- readouts
    def entropia(self):
        q = self.b[self.b > 0.0]
        return float(-(q * np.log(q)).sum())

    def media(self):
        n = self.p.n
        i = np.arange(n)
        return np.array([float(self.b.sum(1) @ i), float(self.b.sum(0) @ i)])

    def massimo(self):
        k = int(np.argmax(self.b))
        return np.array([k // self.p.n, k % self.p.n], dtype=float)

    def copia(self):
        c = Credenza.__new__(Credenza)
        c.p, c.b = self.p, self.b.copy()
        return c

    # ------------------------------------------------- used by infotaxis
    def entropia_attesa(self, r_nuovo):
        """Expected entropy of the belief AFTER moving to r_nuovo (used by infotaxis).

        Averages over the counts that could be observed there, weighted by their
        probability under the current belief. It is a prediction: it never uses
        the true source position.
        """
        i, j = int(r_nuovo[0]), int(r_nuovo[1])
        p_trovo = float(self.b[i, j])          # source found there: entropy goes to 0
        mu = self.p.mu_griglia(r_nuovo)
        b = self.b.copy()
        b[i, j] = 0.0
        resto = b.sum()
        if resto <= 0.0:
            return 0.0
        b /= resto
        att = 0.0
        tot_rho = 0.0
        for h in range(self.p.h_max + 1):
            ver = _poisson(h, mu)
            post = b * ver
            rho = float(post.sum())
            if rho <= 0.0:
                continue
            post /= rho
            q = post[post > 0.0]
            att += rho * float(-(q * np.log(q)).sum())
            tot_rho += rho
        if tot_rho > 0.0:
            att /= tot_rho                      # renormalise over the truncated counts
        return (1.0 - p_trovo) * att            # the 'found' case contributes 0
