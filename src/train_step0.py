#!/usr/bin/env python3
"""Step 0, final networks: 6 MLPs (imitation of MLS + REINFORCE, no learned baseline).

Each network starts from its own imitation (seed s) and is trained for
150 batches of 16 episodes. The 6 networks are saved in
runs/rete_nominale_*.pt and evaluated on the test seeds by collaudo.py.

Run:  python src/train_step0.py
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reinforce import imita, addestra

N_RETI, LOTTI, PER_LOTTO, LR, TETTO = 6, 150, 16, 3e-4, 600
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


print("addestro %d reti al NOMINALE (R_em=0.5, d0=8), baseline nessuna"
      % N_RETI, flush=True)
reti = []
t0 = time.time()
for s in range(N_RETI):
    rete, _ = imita(seme_torch=s, verboso=False)
    rete, _, _ = addestra(lotti=LOTTI, per_lotto=PER_LOTTO, lr=LR, tetto=TETTO,
                          seme_torch=s, curriculum=False, rete_iniziale=rete,
                          baseline="nessuna", ogni=LOTTI, verboso=False)
    reti.append(rete)
    print("  rete %d/%d  (%.0f s)" % (s + 1, N_RETI, time.time() - t0), flush=True)

os.makedirs(RUNS, exist_ok=True)
for i, r in enumerate(reti):
    torch.save(r.state_dict(), os.path.join(RUNS, "rete_nominale_%d.pt" % i))
print("\n  %.0f s   reti salvate in runs/rete_nominale_*.pt" % (time.time() - t0),
      flush=True)
