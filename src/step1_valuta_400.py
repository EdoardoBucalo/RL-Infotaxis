#!/usr/bin/env python3
"""Step 1, part 3: re-evaluation of clones and REINFORCE networks on 400 extra development seeds.

On only 40 development seeds the evaluation noise (~10 steps) was as large
as the effect being measured. This script re-evaluates the 3 CNN clones
and the 3 networks after REINFORCE on 400 ADDITIONAL development seeds
(10040..10439), disjoint from the test seeds, with paired comparisons.

Output: runs/gradino1_esteso.json (Figure 3.3 and Table 3.5 of the thesis).
"""
import os, sys, json, time
import numpy as np
import torch
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pennacchio as pn
import politiche as P
from mappa import ReteMappa, PoliticaClone

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
SEMI = list(range(10_040, 10_440))
TETTO = 600
LIVELLI = [0.5, 0.05]


def compito(args):
    nome, R_em = args
    torch.set_num_threads(1)
    if nome == "infotassi":
        pol = P.Infotassi()
    else:
        rete = ReteMappa()
        rete.load_state_dict(torch.load(os.path.join(RUNS, nome + ".pt")))
        rete.eval()
        pol = PoliticaClone(rete, "mappa", nome)
    passi, trov = [], []
    for s in SEMI:
        p = pn.Pennacchio(passi=TETTO, R_em=R_em)
        t, n, _, _ = P.gioca(p, pol, np.random.default_rng(s))
        passi.append(n if t else TETTO); trov.append(t)
    return nome, R_em, passi, trov


def t_app(d):
    d = np.asarray(d, float)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


if __name__ == "__main__":
    t0 = time.time()
    nomi = ["infotassi"] + ["clone_mappa_%d" % s for s in range(3)] + \
           ["g1_mappa_%d" % s for s in range(3)]
    with get_context("spawn").Pool(10) as pool:
        ris = pool.map(compito, [(n, R) for R in LIVELLI for n in nomi])
    tab = {(n, R): (np.array(p, float), np.array(t, float)) for n, R, p, t in ris}

    out = {}
    for R in LIVELLI:
        info = tab[("infotassi", R)][0]
        print("\n=== R_em %s, %d semi di sviluppo aggiuntivi ===" % (R, len(SEMI)))
        print("  infotassi        %6.1f (%3.0f%%)" % (info.mean(), 100 * tab[("infotassi", R)][1].mean()))
        print("  rete  clone            +REINFORCE       dopo-prima            dopo-infotassi")
        for s in range(3):
            pc, tc = tab[("clone_mappa_%d" % s, R)]
            pg, tg = tab[("g1_mappa_%d" % s, R)]
            m1, e1, t1 = t_app(pg - pc)
            m2, e2, t2 = t_app(pg - info)
            print("  %d     %6.1f (%3.0f%%)   %6.1f (%3.0f%%)   %+6.1f +-%4.1f t %+5.2f   %+6.1f +-%4.1f t %+5.2f"
                  % (s, pc.mean(), 100 * tc.mean(), pg.mean(), 100 * tg.mean(),
                     m1, e1, t1, m2, e2, t2))
        pc = np.mean([tab[("clone_mappa_%d" % s, R)][0] for s in range(3)], 0)
        pg = np.mean([tab[("g1_mappa_%d" % s, R)][0] for s in range(3)], 0)
        for nome, d in [("clone - infotassi", pc - info), ("REINFORCE - clone", pg - pc),
                        ("REINFORCE - infotassi", pg - info)]:
            m, e, t = t_app(d)
            print("  media 3 reti, %-22s %+6.1f +- %4.1f   t %+5.2f" % (nome, m, e, t))
            out["%s|%s" % (R, nome)] = {"diff": m, "se": e, "t": t}
        for n in nomi:
            out["%s|%s" % (R, n)] = {"costo": tab[(n, R)][0].mean(), "trovata": tab[(n, R)][1].mean()}
    with open(os.path.join(RUNS, "gradino1_esteso.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n  %.0f s" % (time.time() - t0))
