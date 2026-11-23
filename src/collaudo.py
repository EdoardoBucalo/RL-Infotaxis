#!/usr/bin/env python3
"""Final test (collaudo). Run ONCE, at the end.

All policies on the same 400 test seeds (20000..20399), never used before,
same step cap, at two signal levels (R_em = 0.5 nominal, 0.05 rare):
  reference strategies: random, MLS, infotaxis, oracle
  step 0:  6 MLP networks (runs/rete_nominale_*.pt)
  clones:  3 CNN clones of infotaxis (runs/clone_mappa_*.pt)
  step 1:  3 CNN networks after REINFORCE (runs/g1_mappa_*.pt)
Paired comparisons per seed: final - clone, final - infotaxis.

Output: runs/collaudo.json (Chapter 4 of the thesis).
"""
import os, sys, json, time
import numpy as np
import torch
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pennacchio as pn
import politiche as P
from reinforce import Rete, PoliticaAppresa, SEMI_COLLAUDO
from mappa import ReteMappa, PoliticaClone

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
TETTO = 600
LIVELLI = [0.5, 0.05]
RIFERIMENTI = ["casuale", "verso-massimo", "infotassi", "oracolo"]
GRUPPI = {"gradino 0 (8 car.)": ["rete_nominale_%d" % s for s in range(6)],
          "clone a mappa": ["clone_mappa_%d" % s for s in range(3)],
          "gradino 1 (finale)": ["g1_mappa_%d" % s for s in range(3)]}


def politica(nome):
    rif = {"casuale": P.Casuale, "verso-massimo": P.VersoMassimo,
           "infotassi": P.Infotassi, "oracolo": P.Oracolo}
    if nome in rif:
        return rif[nome]()
    pth = os.path.join(RUNS, nome + ".pt")
    if nome.startswith("rete_nominale"):
        r = Rete(); r.load_state_dict(torch.load(pth)); r.eval()
        return PoliticaAppresa(r, nome)
    r = ReteMappa(); r.load_state_dict(torch.load(pth)); r.eval()
    return PoliticaClone(r, "mappa", nome)


def compito(args):
    nome, R_em = args
    torch.set_num_threads(1)
    pol = politica(nome)
    passi, trov = [], []
    for s in SEMI_COLLAUDO:
        p = pn.Pennacchio(passi=TETTO, R_em=R_em)
        t, n, _, _ = P.gioca(p, pol, np.random.default_rng(s))
        passi.append(n if t else TETTO); trov.append(t)
    return nome, R_em, passi, trov


def stat(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else 0.0)


if __name__ == "__main__":
    t0 = time.time()
    nomi = RIFERIMENTI + [n for g in GRUPPI.values() for n in g]
    with get_context("spawn").Pool(10) as pool:
        ris = pool.map(compito, [(n, R) for R in LIVELLI for n in nomi])
    tab = {(n, R): (np.array(p, float), np.array(t, float)) for n, R, p, t in ris}
    out = {"semi": [SEMI_COLLAUDO[0], SEMI_COLLAUDO[-1]], "tetto": TETTO, "politiche": {}, "test": {}}

    def media_gruppo(g, R, k):
        return np.mean([tab[(n, R)][k] for n in GRUPPI[g]], 0)

    print("COLLAUDO: %d semi mai visti (%d-%d), tetto %d"
          % (len(SEMI_COLLAUDO), SEMI_COLLAUDO[0], SEMI_COLLAUDO[-1], TETTO))
    for R in LIVELLI:
        print("\n=== R_em %s ===" % R)
        print("  %-22s %-18s %s" % ("politica", "costo censurato", "trovata"))
        for n in RIFERIMENTI:
            ps, tv = tab[(n, R)]
            print("  %-22s %6.1f +- %4.1f    %5.1f%%" % (n, ps.mean(), stat(ps)[1], 100 * tv.mean()))
            out["politiche"]["%s|%s" % (R, n)] = {"costo": ps.mean(), "trovata": tv.mean()}
        for g, membri in GRUPPI.items():
            cc = np.array([tab[(n, R)][0].mean() for n in membri])
            tt = np.array([tab[(n, R)][1].mean() for n in membri])
            print("  %-22s %6.1f +- %4.1f    %5.1f%%   (%d reti: %s)"
                  % (g, cc.mean(), cc.std(ddof=1) / np.sqrt(len(cc)), 100 * tt.mean(),
                     len(membri), " ".join("%.0f" % c for c in cc)))
            out["politiche"]["%s|%s" % (R, g)] = {"costo": cc.mean(), "trovata": tt.mean(),
                                                  "per_rete": cc.tolist(), "trov_per_rete": tt.tolist()}

        info_p, info_t = tab[("infotassi", R)]
        fin_p, fin_t = media_gruppo("gradino 1 (finale)", R, 0), media_gruppo("gradino 1 (finale)", R, 1)
        clo_p = media_gruppo("clone a mappa", R, 0)
        print("  confronti appaiati per seme (media delle reti):")
        for chiave, d in [("finale - clone, passi", fin_p - clo_p),
                          ("finale - infotassi, passi", fin_p - info_p),
                          ("finale - infotassi, riuscita", fin_t - info_t)]:
            m, se, t = stat(d)
            unita = "punti" if "riuscita" in chiave else "passi"
            if unita == "punti":
                m, se = 100 * m, 100 * se
            print("    %-30s %+6.1f +- %4.1f %s   t %+5.2f" % (chiave, m, se, unita, t))
            out["test"]["%s|%s" % (R, chiave)] = {"diff": m, "se": se, "t": t}
        # per network: success rate vs infotaxis (third claim tested)
        for n in GRUPPI["gradino 1 (finale)"]:
            m, se, t = stat(tab[(n, R)][1] - info_t)
            print("      %-26s riuscita - infotassi %+5.1f +- %3.1f punti  t %+5.2f" % (n, 100 * m, 100 * se, t))

    with open(os.path.join(RUNS, "collaudo.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n  %.0f s" % (time.time() - t0))
