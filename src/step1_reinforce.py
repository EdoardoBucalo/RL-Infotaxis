#!/usr/bin/env python3
"""Step 1, part 2: REINFORCE starting from the CNN clones of infotaxis.

Same policy gradient as reinforce.py, with the configuration that worked
in step 0 (no learned baseline, normalised advantage). The observation is
the 81x81 egocentric belief map instead of the 8 features.

Settings (Table 3.4 of the thesis): 16 episodes per batch, Adam with
learning rate 1e-4, entropy bonus 0.004, gradient norm clipped at 1,
150 batches, 3 networks trained in parallel.

To save memory each episode stores the belief (float16) and the position,
and the maps are rebuilt in chunks when the gradient is computed.

Outputs: runs/g1_mappa_{0,1,2}.pt and runs/gradino1.json.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pennacchio as pn
from credenza import Credenza
import politiche as P
from reinforce import da_adesso_in_poi, SEMI_SVILUPPO
from mappa import ReteMappa, PoliticaClone, egocentrica

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
TETTO = 600
LIVELLI_EVAL = [0.5, 0.05]
SEME_TR = 40_000                      # training seeds, disjoint from the others


def episodio(rete, seme, tetto):
    rng = np.random.default_rng(seme)
    p = pn.Pennacchio(passi=tetto)
    h = p.reset(rng)
    cred = Credenza(p)
    cred.aggiorna(p.r, h)
    B, Pos, A, R = [], [], [], []
    while True:
        B.append(cred.b.astype(np.float16)); Pos.append(p.r.astype(np.int16))
        x = torch.from_numpy(egocentrica(cred.b[None], p.r[None]))
        with torch.no_grad():
            logit, _ = rete(x)
        pr = torch.softmax(logit, -1).numpy().ravel().astype(np.float64)
        a = int(rng.choice(pn.N_AZIONI, p=pr / pr.sum()))
        h, trov, fin, _ = p.passo(a)
        A.append(a)
        R.append(0.0 if trov else -1.0)
        if not trov:
            cred.aggiorna(p.r, h)
        if fin:
            return np.array(B), np.array(Pos), np.array(A), np.array(R, np.float32), trov


def valuta(rete, R_em, semi=SEMI_SVILUPPO):
    """Steps per seed (censored at the cap) and success, on the given seeds."""
    pol = PoliticaClone(rete, "mappa", "mappa")
    passi, trov = [], []
    for s in semi:
        p = pn.Pennacchio(passi=TETTO, R_em=R_em)
        t, ps, _, _ = P.gioca(p, pol, np.random.default_rng(s))
        passi.append(ps if t else TETTO); trov.append(t)
    return np.array(passi, float), np.array(trov, float)


def addestra(rete, lotti, per_lotto=16, lr=1e-4, peso_entropia=0.004,
             taglio=1.0, pezzo=512, seme_base=SEME_TR, ogni=50, log=print):
    ott = torch.optim.Adam(rete.parameters(), lr=lr)
    seme = seme_base
    curva = []
    for it in range(1, lotti + 1):
        t0 = time.time()
        Bs, Ps, As, Gs, riusc = [], [], [], [], []
        rete.eval()
        for _ in range(per_lotto):
            B, Pos, A, R, trov = episodio(rete, seme, TETTO)
            seme += 1
            # tail value: 0 if found (terminated); -1 if truncated, i.e. 'at least
            # another full budget'. The value head is NOT trained here (no learned
            # baseline) and must not be used.
            Gs.append(da_adesso_in_poi(R / TETTO, 1.0, 0.0 if trov else -1.0))
            Bs.append(B); Ps.append(Pos); As.append(A); riusc.append(trov)
        B = np.concatenate(Bs); Pos = np.concatenate(Ps)
        A = torch.from_numpy(np.concatenate(As)).long()
        G = torch.from_numpy(np.concatenate(Gs))
        van = (G - G.mean()) / (G.std() + 1e-8)
        n = len(A)
        rete.train()
        ott.zero_grad()
        ent_tot = 0.0
        # gradient accumulated in chunks of 512 steps: the mean is over the whole batch
        for k in range(0, n, pezzo):
            sl = slice(k, k + pezzo)
            x = torch.from_numpy(egocentrica(B[sl], Pos[sl]))
            logit, _ = rete(x)
            logp_all = torch.log_softmax(logit, -1)
            logp = logp_all.gather(1, A[sl].unsqueeze(1)).squeeze(1)
            ent = -(logp_all.exp() * logp_all).sum(1)
            perdita = (-(logp * van[sl]).sum() - peso_entropia * ent.sum()) / n
            perdita.backward()
            ent_tot += float(ent.detach().sum())
        gn = float(torch.nn.utils.clip_grad_norm_(rete.parameters(), taglio))
        ott.step()
        if it == 1 or it % ogni == 0:
            rete.eval()
            c, t = valuta(rete, 0.5)
            curva.append({"lotto": it, "costo": c.mean(), "trovata": t.mean(),
                          "riusc_tr": float(np.mean(riusc)), "grad": gn,
                          "entropia": ent_tot / n, "passi_lotto": n})
            log("  lotto %4d  costo_ev %6.1f (%3.0f%%)  riusc_tr %3.0f%%  |grad| %.3f  ent %.3f  %d passi  %.1f s"
                % (it, c.mean(), 100 * t.mean(), 100 * np.mean(riusc), gn,
                   ent_tot / n, n, time.time() - t0))
    return rete, curva


def lavoro(args):
    """One network: evaluation before, training, evaluation after (pool worker)."""
    s, lotti, fili = args
    torch.set_num_threads(fili)
    torch.manual_seed(s)
    rete = ReteMappa()
    rete.load_state_dict(torch.load(os.path.join(RUNS, "clone_mappa_%d.pt" % s)))
    rete.eval()
    righe = []
    def log(m):
        righe.append(m); print("[rete %d] %s" % (s, m), flush=True)
    prima = {R: valuta(rete, R) for R in LIVELLI_EVAL}
    rete, curva = addestra(rete, lotti, seme_base=SEME_TR + 100_000 * s, log=log)
    rete.eval()
    dopo = {R: valuta(rete, R) for R in LIVELLI_EVAL}
    torch.save(rete.state_dict(), os.path.join(RUNS, "g1_mappa_%d.pt" % s))
    return s, prima, dopo, curva


def t_appaiato(d):
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std() > 0 else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotti", type=int, default=150)
    ap.add_argument("--prova", action="store_true", help="2 lotti, una rete: misura i tempi")
    a = ap.parse_args()
    t0 = time.time()
    ctx = get_context("spawn")
    if a.prova:
        torch.set_num_threads(3)
        rete = ReteMappa()
        rete.load_state_dict(torch.load(os.path.join(RUNS, "clone_mappa_0.pt")))
        addestra(rete, 2, ogni=1)
        print("%.0f s" % (time.time() - t0))
        sys.exit()

    # infotaxis on the same seeds, for the paired comparison
    info = {}
    for R in LIVELLI_EVAL:
        ps, tv = [], []
        for sd in SEMI_SVILUPPO:
            p = pn.Pennacchio(passi=TETTO, R_em=R)
            t, n, _, _ = P.gioca(p, P.Infotassi(), np.random.default_rng(sd))
            ps.append(n if t else TETTO); tv.append(t)
        info[R] = (np.array(ps, float), np.array(tv, float))

    with ctx.Pool(3) as pool:
        ris = pool.map(lavoro, [(s, a.lotti, 3) for s in range(3)])

    out = {"lotti": a.lotti, "reti": {}}
    print("\n=== PRIMA (clone) e DOPO (+REINFORCE), 40 semi di sviluppo ===")
    for R in LIVELLI_EVAL:
        pi, ti = info[R]
        print("\n  R_em %s   infotassi %.1f (%.0f%%)" % (R, pi.mean(), 100 * ti.mean()))
        print("  rete   prima          dopo           dopo-prima        dopo-infotassi (t)")
        dd, dopo_m = [], []
        for s, prima, dopo, curva in ris:
            (p0, t0_), (p1, t1) = prima[R], dopo[R]
            d = p1 - p0; di = p1 - pi
            dd.append(d.mean()); dopo_m.append(p1.mean())
            print("  %d      %6.1f (%3.0f%%)  %6.1f (%3.0f%%)  %+6.1f (t %+.2f)   %+6.1f (t %+.2f)"
                  % (s, p0.mean(), 100 * t0_.mean(), p1.mean(), 100 * t1.mean(),
                     d.mean(), t_appaiato(d), di.mean(), t_appaiato(di)))
            out["reti"].setdefault(str(s), {})[str(R)] = {
                "prima": p0.tolist(), "dopo": p1.tolist(),
                "trov_prima": t0_.mean(), "trov_dopo": t1.mean()}
        dd = np.array(dd); dm = np.array(dopo_m)
        print("  media  dopo %.1f +- %.1f    dopo-prima %+.1f +- %.1f (%d/3 migliorate)"
              % (dm.mean(), dm.std(ddof=1) / np.sqrt(3), dd.mean(),
                 dd.std(ddof=1) / np.sqrt(3), int((dd < 0).sum())))
        out["infotassi_" + str(R)] = pi.tolist()
    for s, _, _, curva in ris:
        out["reti"][str(s)]["curva"] = curva
    with open(os.path.join(RUNS, "gradino1.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n  %.0f s   reti in runs/g1_mappa_*.pt" % (time.time() - t0))
