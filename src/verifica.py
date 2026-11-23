#!/usr/bin/env python3
"""Sanity checks of the ENVIRONMENT and of the INFERENCE, before training anything.

A broken environment still gives plausible learning curves: the agent
learns the wrong game very well. Three things are checked separately:
the physics, the Bayesian inference and the ranking of the reference
strategies (Table 4.1 of the thesis: 60 paired episodes).

Run:  python src/verifica.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pennacchio as pn
from credenza import Credenza, _poisson
import politiche as P

esiti = []


def prova(nome, cond, det=""):
    esiti.append((nome, bool(cond)))
    print("  [%s] %-44s %s" % ("ok" if cond else "NO", nome, det))


print("VERIFICA\n--- la fisica ---")
p = pn.Pennacchio()
r0 = np.array([20.0, 20.0])

# 1 the precomputed table must match the direct computation
rng = np.random.default_rng(0)
err = 0.0
for _ in range(50):
    r = rng.integers(0, p.n, size=2).astype(float)
    diretto = p.media_conteggi(r, p.celle).reshape(p.n, p.n)
    err = max(err, float(np.abs(diretto - p.mu_griglia(r)).max()))
prova("tabella = conto diretto", err < 1e-12, "scarto massimo %.2e" % err)

# 2 the plume points downwind, not upwind
giu = p.media_conteggi(r0 + [20, 0], r0)[0]
su = p.media_conteggi(r0 - [20, 0], r0)[0]
prova("il pennacchio e' allungato SOTTOVENTO", giu > 10 * su,
      "a 20 celle: %.4f contro %.6f (x%.0f)" % (giu, su, giu / su))

# 3 the rate decreases with distance along the axis
assi = [p.media_conteggi(r0 + [d, 0], r0)[0] for d in range(1, 40)]
prova("il tasso cala con la distanza", all(np.diff(assi) < 0),
      "da %.3f a %.3f" % (assi[0], assi[-1]))

# 4 the observed count really is Poisson with the declared mean
p.reset(np.random.default_rng(1))
p.r = p.r0 + np.array([6.0, 0.0])
mu = float(p.media_conteggi(p.r, p.r0)[0])
camp = np.array([p._osserva() for _ in range(40000)])
z = abs(camp.mean() - mu) / (camp.std() / np.sqrt(len(camp)))
prova("i conteggi sono Poisson(mu)", z < 4,
      "media %.4f contro mu %.4f, %.1f sigma" % (camp.mean(), mu, z))
prova("Poisson: varianza = media", abs(camp.var() / camp.mean() - 1) < 0.05,
      "var/media = %.4f" % (camp.var() / camp.mean()))

# 5 moves: one cell per step, never outside the grid
p.reset(np.random.default_rng(2))
rng = np.random.default_rng(2)
peggio, fuori = 0.0, 0
for _ in range(2000):
    prima = p.r.copy()
    p.passo(int(rng.integers(pn.N_AZIONI)))
    peggio = max(peggio, float(np.abs(p.r - prima).sum()))
    if p.r.min() < 0 or p.r.max() > p.n - 1:
        fuori += 1
prova("una cella per passo", peggio <= 1.0, "spostamento massimo %.1f" % peggio)
prova("mai fuori dalla griglia", fuori == 0, "violazioni %d" % fuori)

print("\n--- l'inferenza ---")
# 6 the belief stays normalised and the TRUE source cell never goes to zero.
#   If it did, the model used for the update would not be the one that
#   generates the data: the worst bug, because the agent keeps working
#   and simply searches in the wrong place.
peggio_norm, minimo_vero = 0.0, 1.0
for s in range(30):
    p.reset(np.random.default_rng(100 + s))
    c = Credenza(p)
    i0, j0 = int(p.r0[0]), int(p.r0[1])
    h = p._osserva()
    c.aggiorna(p.r, h)
    for _ in range(150):
        h, trov, fin, _ = p.passo(int(np.random.default_rng(s).integers(5)))
        if trov:
            break
        c.aggiorna(p.r, h)
        peggio_norm = max(peggio_norm, abs(float(c.b.sum()) - 1.0))
        minimo_vero = min(minimo_vero, float(c.b[i0, j0]))
prova("la credenza resta normalizzata", peggio_norm < 1e-10,
      "scarto massimo da 1: %.2e" % peggio_norm)
prova("la sorgente vera non si azzera mai", minimo_vero > 0.0,
      "probabilita' minima della cella giusta: %.2e" % minimo_vero)

# 7 information: with infotaxis the belief must CONCENTRATE
ent_i, ent_f, pv_i, pv_f = [], [], [], []
for s in range(20):
    p.reset(np.random.default_rng(200 + s))
    c = Credenza(p)
    i0, j0 = int(p.r0[0]), int(p.r0[1])
    pol = P.Infotassi()
    rng = np.random.default_rng(s)
    h = p._osserva()
    c.aggiorna(p.r, h)
    ent_i.append(c.entropia()); pv_i.append(float(c.b[i0, j0]))
    for _ in range(80):
        h, trov, fin, _ = p.passo(pol(p, c, rng))
        if trov:
            break
        c.aggiorna(p.r, h)
    ent_f.append(c.entropia()); pv_f.append(float(c.b[i0, j0]))
prova("l'entropia della credenza scende",
      np.mean(ent_f) < np.mean(ent_i) - 0.1,
      "%.3f -> %.3f nat" % (np.mean(ent_i), np.mean(ent_f)))
prova("la credenza si sposta sulla cella giusta",
      np.mean(pv_f) > 20 * np.mean(pv_i),
      "P(vera) %.2e -> %.2e (x%.0f)"
      % (np.mean(pv_i), np.mean(pv_f), np.mean(pv_f) / np.mean(pv_i)))

# 8 reproducibility
a = P.gioca(pn.Pennacchio(), P.Infotassi(), np.random.default_rng(7))[:3]
b = P.gioca(pn.Pennacchio(), P.Infotassi(), np.random.default_rng(7))[:3]
prova("stesso seme, stesso episodio", a == b, "%s e %s" % (a[:2], b[:2]))

print("\n--- l'ordinamento delle strategie (60 episodi, semi appaiati) ---")
# HOW TO MEASURE. The steps/oracle ratio can only be computed on the
# SUCCESSFUL episodes, so for a strategy that rarely succeeds it is a
# SELECTED number: it only closes the easy episodes.
# 
# The honest metric is the CENSORED COST: failed episodes count as the
# step cap. It is equally pessimistic for everyone and it is defined on
# all episodes. The conditional ratio is reported next to it, always
# together with the success rate.
SEMI = list(range(60))
TETTO = pn.Pennacchio().passi
ris = {}
print("    %-14s %-9s %-16s %-18s" % ("", "trovata", "costo con censura",
                                      "rapporto se trovata"))
for cls in P.TUTTE:
    pol = cls()
    t0 = time.time()
    r = [P.gioca(pn.Pennacchio(), pol, np.random.default_rng(s)) for s in SEMI]
    trov = np.array([x[0] for x in r])
    # censored cost: actual steps if found, the cap otherwise
    cens = np.array([x[1] if x[0] else TETTO for x in r], dtype=float)
    rap = np.array([x[1] / x[2] for x in r if x[0]])
    ris[pol.nome] = {"trovata": trov.mean(),
                     "cens": cens.mean(), "cens_se": cens.std() / np.sqrt(len(cens)),
                     "rap": rap.mean() if len(rap) else np.nan,
                     "n_rap": len(rap)}
    d = ris[pol.nome]
    print("    %-14s %6.0f%%    %6.1f +- %5.1f    %s   (%.1fs)"
          % (pol.nome, 100 * d["trovata"], d["cens"], d["cens_se"],
             ("%5.2f  su %2d ep." % (d["rap"], d["n_rap"])) if d["n_rap"]
             else "    -        ", time.time() - t0))

prova("l'oracolo e' il pavimento assoluto",
      abs(ris["oracolo"]["rap"] - 1.0) < 1e-9,
      "rapporto %.6f" % ris["oracolo"]["rap"])
prova("l'infotassi batte verso-massimo (costo censurato)",
      ris["infotassi"]["cens"] < ris["verso-massimo"]["cens"]
      - 2 * ris["verso-massimo"]["cens_se"],
      "%.1f contro %.1f passi" % (ris["infotassi"]["cens"],
                                  ris["verso-massimo"]["cens"]))
prova("l'infotassi batte il caso di molto",
      ris["infotassi"]["cens"] < 0.5 * ris["casuale"]["cens"],
      "%.1f contro %.1f passi" % (ris["infotassi"]["cens"],
                                  ris["casuale"]["cens"]))

n_no = sum(1 for _, ok in esiti if not ok)
print("\n  ESITO: %d controlli, %d falliti  ->  %s"
      % (len(esiti), n_no, "tutto sano" if n_no == 0 else "DA SISTEMARE"))
sys.exit(1 if n_no else 0)
