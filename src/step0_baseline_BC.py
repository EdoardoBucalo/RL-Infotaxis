#!/usr/bin/env python3
"""Step 0: the other two baseline variants, on the same 6 seeds as step0_baseline_A.py.

  A  learned baseline                 (step0_baseline_A.py, runs/semi.json)
  B  no learned baseline              only the batch normalisation
  C  learned baseline + warm-up of V  V is trained alone for 30 batches,
                                      with the policy frozen

Hypothesis: the instability of A comes from the untrained value head,
whose noisy estimates destroy the policy learned by imitation.
Same paired comparison (before/after) on the development seeds.
Result in runs/diagnosi.json (Table 3.2).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import politiche as P
from reinforce import imita, addestra, valuta, PoliticaAppresa, SEMI_SVILUPPO

N_SEMI, LOTTI, PER_LOTTO, LR, TETTO, SCALDA = 6, 150, 16, 3e-4, 600, 30
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
CONDIZIONI = [("B  baseline nessuna", dict(baseline="nessuna")),
              ("C  appresa + scaldo V", dict(baseline="appresa",
                                             scalda_valore=SCALDA))]

print("%d semi, %d lotti da %d, lr %.0e, tetto %d  (scaldamento: %d lotti)\n"
      % (N_SEMI, LOTTI, PER_LOTTO, LR, TETTO, SCALDA), flush=True)
rif = valuta(P.Infotassi(), SEMI_SVILUPPO, TETTO)
print("  infotassi (termometro, non avversario): %.1f passi\n" % rif[0], flush=True)

tutto = {}
for nome, extra in CONDIZIONI:
    print("=== %s ===" % nome, flush=True)
    print("  seme   prima   dopo    differenza", flush=True)
    righe = []
    for s in range(N_SEMI):
        rete, _ = imita(seme_torch=s, verboso=False)
        c_pre, t_pre = valuta(PoliticaAppresa(rete), SEMI_SVILUPPO, TETTO)
        rete, st, _ = addestra(lotti=LOTTI, per_lotto=PER_LOTTO, lr=LR,
                               tetto=TETTO, seme_torch=s, curriculum=False,
                               rete_iniziale=rete, ogni=LOTTI, verboso=False,
                               **extra)
        c_post, t_post = valuta(PoliticaAppresa(rete), SEMI_SVILUPPO, TETTO)
        righe.append({"seme": s, "pre": c_pre, "post": c_post,
                      "pre_trov": t_pre, "post_trov": t_post})
        print("  %4d   %6.1f  %6.1f   %+7.1f"
              % (s, c_pre, c_post, c_post - c_pre), flush=True)
    pre = np.array([r["pre"] for r in righe]); post = np.array([r["post"] for r in righe])
    d = post - pre; se = d.std(ddof=1) / np.sqrt(len(d))
    peggio = float(d.max())
    print("  prima %6.1f   dopo %6.1f +- %4.1f   differenza %+6.1f +- %4.1f"
          % (pre.mean(), post.mean(), post.std(ddof=1)/np.sqrt(len(post)), d.mean(), se),
          flush=True)
    print("  migliorati %d/%d   PEGGIOR CROLLO %+.1f passi\n"
          % (int((d < 0).sum()), len(d), peggio), flush=True)
    tutto[nome] = {"righe": righe, "media_pre": float(pre.mean()),
                   "media_post": float(post.mean()), "diff": float(d.mean()),
                   "se": float(se), "peggior_crollo": peggio,
                   "migliorati": int((d < 0).sum())}

print("=== CONFRONTO COL CASO GIA' MISURATO ===", flush=True)
print("  A  appresa (step0_baseline_A) differenza  +62.8 +- 51.8   crollo +308.9", flush=True)
for nome in tutto:
    t = tutto[nome]
    print("  %-28s differenza %+6.1f +- %4.1f   crollo %+.1f"
          % (nome, t["diff"], t["se"], t["peggior_crollo"]), flush=True)
with open(os.path.join(RUNS, "diagnosi.json"), "w") as f:
    json.dump(tutto, f, indent=1)
