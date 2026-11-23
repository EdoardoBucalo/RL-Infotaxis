#!/usr/bin/env python3
"""Step 1, part 1: CNN on the belief map vs MLP on 8 features. Who can CLONE infotaxis?

Diagnosis without RL. Infotaxis is a deterministic function of (belief,
position), so a network that sees the whole belief can represent it. If
the 8 features fail where the map succeeds, the bottleneck of step 0 was
the representation.

Data: infotaxis episodes at 5 signal levels (R_em = 0.5 ... 0.05), 60 per
level for training (seeds 30000+) and 15 held out (seeds 31000+). Each
step is a (situation, infotaxis action) example.

EGOCENTRIC MAP. The 41x41 belief is placed on an 81x81 canvas with the
agent always at the centre. Three channels:
    0  b / max(b)          where the probability mass is
    1  sqrt(b / max(b))    the tails (secondary hypotheses)
    2  inside the room     1 inside, 0 outside (zero belief vs wall)
The wind is always along +x, so no wind channel is needed.

Outputs: runs/clone_{car,mappa}_{0,1,2}.pt and runs/mappa.json
(Figure 3.2 and Table 3.3 of the thesis).
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pennacchio as pn
from credenza import Credenza
import politiche as P
from reinforce import Rete, caratteristiche, SEMI_SVILUPPO

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
N = 41
LATO = 2 * N - 1                       # 81: agent always at the centre
LIVELLI = [0.5, 0.3, 0.2, 0.1, 0.05]
TETTO = 600
EP_ADDESTRA = 60                       # episodes per level, seeds 30000+
EP_TENUTI = 15                         # episodes per level, seeds 31000+
N_INIT = 3


# ------------------------------------------------------------ the map
def egocentrica(b, r):
    """(B,41,41) beliefs, (B,2) positions -> (B,3,81,81) egocentric maps."""
    B = len(b)
    out = np.zeros((B, 3, LATO, LATO), dtype=np.float32)
    for k in range(B):
        oi, oj = N - 1 - int(r[k][0]), N - 1 - int(r[k][1])
        q = b[k].astype(np.float32)
        q = q / max(float(q.max()), 1e-30)
        out[k, 0, oi:oi + N, oj:oj + N] = q
        out[k, 1, oi:oi + N, oj:oj + N] = np.sqrt(q)
        out[k, 2, oi:oi + N, oj:oj + N] = 1.0
    return out


class ReteMappa(nn.Module):
    """Small CNN: 4 conv layers (stride 2), 81 -> 41 -> 21 -> 11 -> 6, then a
    dense layer of 128 units and two heads (logits, value). Same interface
    as reinforce.Rete.
    """

    def __init__(self, n_az=pn.N_AZIONI):
        super().__init__()
        self.tronco = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),    # 41
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 21
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),   # 11
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),   # 6
            nn.Flatten(), nn.Linear(32 * 6 * 6, 128), nn.ReLU())
        self.testa_pi = nn.Linear(128, n_az)
        self.testa_v = nn.Linear(128, 1)

    def forward(self, x):
        t = self.tronco(x)
        return self.testa_pi(t), self.testa_v(t).squeeze(-1)


class PoliticaClone:
    """Standard policy interface, for both representations ('car' = 8 features, 'mappa' = map)."""
    usa_credenza = True

    def __init__(self, rete, tipo, nome, avida=False):
        self.rete, self.tipo, self.nome, self.avida = rete, tipo, nome, avida

    def __call__(self, p, cred, rng):
        if self.tipo == "mappa":
            x = torch.from_numpy(egocentrica(cred.b[None], p.r[None]))
        else:
            x = torch.from_numpy(caratteristiche(p, cred)).unsqueeze(0)
        with torch.no_grad():
            logit, _ = self.rete(x)
        if self.avida:
            return int(logit.argmax())
        pr = torch.softmax(logit, -1).numpy().ravel().astype(np.float64)
        return int(rng.choice(pn.N_AZIONI, p=pr / pr.sum()))


# ------------------------------------------------------------- data
def episodio_infotassi(args):
    """One infotaxis episode, recording every decision."""
    seme, R_em = args
    rng = np.random.default_rng(seme)
    p = pn.Pennacchio(passi=TETTO, R_em=R_em)
    h = p.reset(rng)
    cred = Credenza(p)
    cred.aggiorna(p.r, h)
    maestro = P.Infotassi()
    car, cre, pos, az = [], [], [], []
    while True:
        car.append(caratteristiche(p, cred))
        cre.append(cred.b.astype(np.float16))
        pos.append(p.r.astype(np.int16))
        a = maestro(p, cred, rng)
        az.append(a)
        h, trov, fin, _ = p.passo(a)
        if not trov:
            cred.aggiorna(p.r, h)
        if fin:
            break
    return (np.array(car), np.array(cre), np.array(pos), np.array(az))


def raccogli(base, n_ep, pool):
    compiti = [(base + 1000 * i + s, R) for i, R in enumerate(LIVELLI)
               for s in range(n_ep)]
    ris = pool.map(episodio_infotassi, compiti)
    liv = np.concatenate([np.full(len(r[3]), LIVELLI.index(R))
                          for (_, R), r in zip(compiti, ris)])
    return {"car": np.concatenate([r[0] for r in ris]),
            "cre": np.concatenate([r[1] for r in ris]),
            "pos": np.concatenate([r[2] for r in ris]),
            "az": np.concatenate([r[3] for r in ris]),
            "liv": liv}


def ingressi(d, idx, tipo):
    if tipo == "mappa":
        return torch.from_numpy(egocentrica(d["cre"][idx], d["pos"][idx]))
    return torch.from_numpy(d["car"][idx])


# --------------------------------------------------------- cloning
def clona(d, tipo, seme, epoche, lotto, lr=1e-3):
    torch.manual_seed(seme)
    rng = np.random.default_rng(seme)
    rete = ReteMappa() if tipo == "mappa" else Rete()
    ott = torch.optim.Adam(rete.parameters(), lr=lr)
    y = torch.from_numpy(d["az"]).long()
    n = len(y)
    for _ in range(epoche):
        perm = rng.permutation(n)
        for k in range(0, n, lotto):
            idx = perm[k:k + lotto]
            logit, _ = rete(ingressi(d, idx, tipo))
            perdita = F.cross_entropy(logit, y[idx])
            ott.zero_grad(); perdita.backward(); ott.step()
    return rete


def accuratezza(rete, d, tipo):
    """Per signal level: fraction of infotaxis decisions reproduced by the network."""
    pred = []
    with torch.no_grad():
        for k in range(0, len(d["az"]), 1024):
            idx = np.arange(k, min(k + 1024, len(d["az"])))
            pred.append(rete(ingressi(d, idx, tipo))[0].argmax(1).numpy())
    giusto = np.concatenate(pred) == d["az"]
    return [float(giusto[d["liv"] == i].mean()) for i in range(len(LIVELLI))]


# ------------------------------------------------------ evaluation
def _init_lavoratore():
    torch.set_num_threads(1)


def valuta_compito(args):
    """One (policy, signal level) on the 40 development seeds (pool worker)."""
    tipo, percorso, avida, R_em = args
    if tipo == "infotassi":
        pol = P.Infotassi()
    else:
        rete = ReteMappa() if tipo == "mappa" else Rete()
        rete.load_state_dict(torch.load(percorso))
        rete.eval()
        pol = PoliticaClone(rete, tipo, tipo, avida)
    passi, trov = [], []
    for s in SEMI_SVILUPPO:
        p = pn.Pennacchio(passi=TETTO, R_em=R_em)
        t, ps, _, _ = P.gioca(p, pol, np.random.default_rng(s))
        passi.append(ps if t else TETTO)
        trov.append(t)
    return np.array(passi, dtype=float), np.array(trov, dtype=float)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(10, initializer=_init_lavoratore) as pool:
        print("raccolgo gli episodi dell'infotassi ...", flush=True)
        tr = raccogli(30_000, EP_ADDESTRA, pool)
        te = raccogli(31_000, EP_TENUTI, pool)
    print("  addestramento %d decisioni, tenute da parte %d   (%.0f s)"
          % (len(tr["az"]), len(te["az"]), time.time() - t0), flush=True)
    for i, R in enumerate(LIVELLI):
        print("    R_em %-5s %6d decisioni" % (R, int((tr["liv"] == i).sum())))

    CONF = {"car": dict(epoche=40, lotto=512), "mappa": dict(epoche=8, lotto=256)}
    acc, percorsi = {}, {}
    print("\naccuratezza sugli episodi TENUTI DA PARTE, per livello R_em")
    print("  %-6s %-4s " % ("rappr.", "init") + " ".join("%6s" % R for R in LIVELLI))
    for tipo in ("car", "mappa"):
        for s in range(N_INIT):
            t1 = time.time()
            percorsi[(tipo, s)] = os.path.join(RUNS, "clone_%s_%d.pt" % (tipo, s))
            if os.path.exists(percorsi[(tipo, s)]):
                # resume after an interruption: the data are deterministic (same seeds),
                # so the saved network is the one of this training
                rete = ReteMappa() if tipo == "mappa" else Rete()
                rete.load_state_dict(torch.load(percorsi[(tipo, s)]))
            else:
                rete = clona(tr, tipo, s, **CONF[tipo])
                torch.save(rete.state_dict(), percorsi[(tipo, s)])
            a = accuratezza(rete, te, tipo)
            acc[(tipo, s)] = a
            print("  %-6s %-4d " % (tipo, s) + " ".join("%5.1f%%" % (100 * x) for x in a)
                  + "   (%.0f s)" % (time.time() - t1), flush=True)

    print("\nvaluto sui 40 semi di SVILUPPO, tetto %d ..." % TETTO, flush=True)
    compiti = [("infotassi", None, False, R) for R in LIVELLI]
    chiavi = [("infotassi", None, False, R) for R in LIVELLI]
    for (tipo, s), pth in percorsi.items():
        for avida in (False, True):
            for R in LIVELLI:
                compiti.append((tipo, pth, avida, R))
                chiavi.append((tipo, s, avida, R))
    with ctx.Pool(10, initializer=_init_lavoratore) as pool:
        ris = dict(zip(chiavi, pool.map(valuta_compito, compiti)))

    out = {"livelli": LIVELLI, "accuratezza": {}, "costo": {}}
    for (tipo, s), a in acc.items():
        out["accuratezza"]["%s_%d" % (tipo, s)] = a
    for k, (ps, tv) in ris.items():
        out["costo"]["_".join(map(str, k))] = {"costo": ps.mean(), "trovata": tv.mean(),
                                               "passi": ps.tolist()}

    for avida in (False, True):
        print("\n=== costo censurato (trovata), %s ===" % ("AVIDA" if avida else "CAMPIONATA"))
        print("  %-6s " % "R_em" + "%-20s" % "infotassi"
              + "%-28s" % "8 caratteristiche (3 init)" + "%-28s" % "mappa (3 init)"
              + "mappa - car, per init")
        for R in LIVELLI:
            pi, ti = ris[("infotassi", None, False, R)]
            riga = "  %-6s %6.1f (%3.0f%%)       " % (R, pi.mean(), 100 * ti.mean())
            med = {}
            for tipo in ("car", "mappa"):
                c = np.array([ris[(tipo, s, avida, R)][0].mean() for s in range(N_INIT)])
                t = np.array([ris[(tipo, s, avida, R)][1].mean() for s in range(N_INIT)])
                med[tipo] = (c, t)
                riga += "%6.1f +-%5.1f (%3.0f%%)    " % (
                    c.mean(), c.std(ddof=1) / np.sqrt(N_INIT), 100 * t.mean())
            dt = 100 * (med["mappa"][1] - med["car"][1])
            dc = med["mappa"][0] - med["car"][0]
            riga += " ".join("%+.0fp/%+.0f" % (a, b) for a, b in zip(dt, dc))
            print(riga)
        print("  (ultima colonna: punti di riuscita / passi, mappa meno caratteristiche)")

    with open(os.path.join(RUNS, "mappa.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n  %.0f s   reti in runs/clone_*.pt, dati in runs/mappa.json"
          % (time.time() - t0))
