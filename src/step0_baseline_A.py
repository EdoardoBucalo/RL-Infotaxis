#!/usr/bin/env python3
"""Step 0: REINFORCE started from imitation, with the LEARNED baseline, on 6 seeds.

Paired comparison: for each seed the same network is evaluated BEFORE
(imitation only) and AFTER (imitation + REINFORCE) on the development
seeds. Result in runs/semi.json (Table 3.2, 'baseline appresa').

Note: this script uses the learned baseline, which was the default at the
time; the current default in reinforce.py is 'nessuna'.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import politiche as P
from reinforce import (imita, addestra, valuta, PoliticaAppresa,
                       SEMI_SVILUPPO)

N_SEMI, LOTTI, PER_LOTTO, LR, TETTO = 6, 150, 16, 3e-4, 600
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")

print("%d semi, %d lotti da %d episodi, lr %.0e, tetto %d, semi di sviluppo"
      % (N_SEMI, LOTTI, PER_LOTTO, LR, TETTO))
rif = {}
for cls in (P.VersoMassimo, P.Infotassi):
    rif[cls.nome] = valuta(cls(), SEMI_SVILUPPO, TETTO)
    print("  %-14s costo %6.1f   trovata %3.0f%%"
          % (cls.nome, rif[cls.nome][0], 100 * rif[cls.nome][1]))

print("\n  seme   prima   dopo    differenza")
righe = []
t0 = time.time()
for s in range(N_SEMI):
    rete, acc = imita(seme_torch=s, verboso=False)
    c_pre, t_pre = valuta(PoliticaAppresa(rete), SEMI_SVILUPPO, TETTO)
    rete, st, _ = addestra(lotti=LOTTI, per_lotto=PER_LOTTO, lr=LR,
                           tetto=TETTO, seme_torch=s, curriculum=False,
                           rete_iniziale=rete, ogni=LOTTI, verboso=False,
                           baseline="appresa")
    c_post, t_post = valuta(PoliticaAppresa(rete), SEMI_SVILUPPO, TETTO)
    righe.append({"seme": s, "imitazione_acc": acc,
                  "pre": c_pre, "pre_trov": t_pre,
                  "post": c_post, "post_trov": t_post})
    print("  %4d   %6.1f  %6.1f   %+7.1f" % (s, c_pre, c_post, c_post - c_pre))

pre = np.array([r["pre"] for r in righe])
post = np.array([r["post"] for r in righe])
d = post - pre
se = d.std(ddof=1) / np.sqrt(len(d))
print("\n  prima  (imitazione sola) %6.1f +- %4.1f" % (pre.mean(), pre.std(ddof=1)/np.sqrt(len(pre))))
print("  dopo   (+ REINFORCE)     %6.1f +- %4.1f" % (post.mean(), post.std(ddof=1)/np.sqrt(len(post))))
print("  differenza appaiata      %+6.1f +- %4.1f passi   (%d su %d migliorati)"
      % (d.mean(), se, int((d < 0).sum()), len(d)))
print("  t appaiato = %.2f con %d gradi di liberta'" % (d.mean()/se if se else np.nan, len(d)-1))
print("\n  riferimenti: verso-massimo %.1f, infotassi %.1f"
      % (rif["verso-massimo"][0], rif["infotassi"][0]))
print("  divario residuo con l'infotassi: %+.1f passi" % (post.mean() - rif["infotassi"][0]))
print("\n  %.0f s" % (time.time() - t0))

os.makedirs(RUNS, exist_ok=True)
with open(os.path.join(RUNS, "semi.json"), "w") as f:
    json.dump({"impostazioni": {"n_semi": N_SEMI, "lotti": LOTTI,
                                "per_lotto": PER_LOTTO, "lr": LR, "tetto": TETTO},
               "riferimenti": rif, "righe": righe}, f, indent=1)
