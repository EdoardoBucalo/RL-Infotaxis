#!/usr/bin/env python3
"""Reference strategies used in the thesis.

  Casuale       (random)  random move: the floor.
  VersoMassimo  (MLS, most likely state)  exact Bayesian inference, then
                move towards the most probable cell. Never explores.
  Infotassi     (infotaxis, Vergassola et al., Nature 2007)  move where the
                EXPECTED entropy of the belief after the move is lowest.
  Oracolo       (oracle)  knows the source: the absolute lower bound.

The question of the project is whether a learned policy can beat infotaxis.
"""
import numpy as np
from pennacchio import AZIONI, N_AZIONI
from credenza import Credenza


def _verso(da, a):
    """Action that reduces the Manhattan distance from 'da' to 'a' the most."""
    d = np.asarray(a, dtype=float) - np.asarray(da, dtype=float)
    if np.abs(d).max() < 0.5:
        return 0
    if abs(d[0]) >= abs(d[1]):
        return 1 if d[0] > 0 else 2
    return 3 if d[1] > 0 else 4


class Casuale:
    nome = "casuale"
    usa_credenza = False

    def __call__(self, p, cred, rng):
        return int(rng.integers(N_AZIONI))


class VersoMassimo:
    """MLS: exact inference, then go to the most probable cell. No exploration."""
    nome = "verso-massimo"
    usa_credenza = True

    def __call__(self, p, cred, rng):
        return _verso(p.r, cred.massimo())


class Infotassi:
    """Infotaxis: minimise the EXPECTED entropy of the belief after the move.

    It is greedy (one step ahead): this is exactly where a learned policy
    could in principle do better, by looking further ahead.
    """
    nome = "infotassi"
    usa_credenza = True

    def __call__(self, p, cred, rng):
        att = [cred.entropia_attesa(v) for v in p.vicini(p.r)]
        m = min(att)
        pari = [k for k, v in enumerate(att) if v <= m + 1e-12]
        return int(rng.choice(pari))      # ties are broken at random


class Oracolo:
    """Oracle: knows the source position (lower bound on the number of steps)."""
    nome = "oracolo"
    usa_credenza = False

    def __call__(self, p, cred, rng):
        return _verso(p.r, p.r0)


TUTTE = [Casuale, VersoMassimo, Infotassi, Oracolo]


def gioca(pennacchio, politica, rng, registra=False):
    """Play one episode. Returns (found, steps, oracle_steps, trace).

    The belief is ALWAYS updated, also for strategies that do not use it.
    """
    h = pennacchio.reset(rng)
    cred = Credenza(pennacchio)
    cred.aggiorna(pennacchio.r, h)
    n_orac = pennacchio.passi_oracolo()
    tr = [] if registra else None
    while True:
        if registra:
            tr.append((pennacchio.r.copy(), cred.b.copy(), h))
        a = politica(pennacchio, cred, rng)
        h, trovato, finito, info = pennacchio.passo(a)
        if not trovato:
            cred.aggiorna(pennacchio.r, h)
        if finito:
            return trovato, info["t"], n_orac, tr
