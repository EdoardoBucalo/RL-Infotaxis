#!/usr/bin/env python3
"""Step 0: REINFORCE written by hand, with an MLP on 8 features of the belief.

Policy gradient (Williams 1992):

    grad J = E[ sum_t grad log pi(a_t|s_t) * (G_t - b(s_t)) ]

G_t is the reward FROM t ONWARDS. With reward -1 per step and gamma = 1,
G_t is minus the number of steps left until the end of the episode.

Baseline: any b(s) that does not depend on the action can be subtracted
without bias. Three options are available (--baseline):
    nessuna  (none)    only the batch normalisation of the returns
    media    (mean)    running mean of the returns
    appresa  (learned) V(s) from the value head
Measured on 6 paired seeds (thesis, Section 3.3): the learned baseline
makes training unstable, 'none' gives the best result, so 'nessuna' is
the default.

Seeds for training, development and final test are disjoint (see below).

Examples:
    python src/reinforce.py --avvio zero --lotti 200          # from scratch
    python src/reinforce.py --avvio imitazione --lotti 150    # from imitation
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pennacchio as pn
from credenza import Credenza
import politiche as P

QUI = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(QUI, "..", "runs")

N_CAR = 8


# -------------------------------------------------------- observation
def caratteristiche(p, cred):
    """From the 41x41 belief to 8 features (Table 3.1 of the thesis).

        1   entropy of the belief / ln(n^2)            uncertainty, in [0, 1]
        2-3 (most probable cell - position) / n         direction of the mode
        4-5 (mean of the belief - position) / n         direction of the mean
        6   sqrt(max probability)                       height of the peak
        7   sqrt(var_x + var_y) / n                     spread of the belief
        8   t / T                                       fraction of budget used

    Everything is relative to the agent position and normalised.
    """
    n = p.n
    b = cred.b
    ent = cred.entropia() / np.log(n * n)          # in [0,1]
    dmo = (cred.massimo() - p.r) / n               # towards the mode
    dme = (cred.media() - p.r) / n                 # towards the mean
    pmax = float(b.max())
    # spread: root mean square deviation of the belief, normalised
    i = np.arange(n)
    mi, mj = cred.media()
    vi = float(b.sum(1) @ (i - mi) ** 2)
    vj = float(b.sum(0) @ (i - mj) ** 2)
    disp = np.sqrt(vi + vj) / n
    return np.array([ent, dmo[0], dmo[1], dme[0], dme[1],
                     np.sqrt(pmax), disp, p.t / p.passi], dtype=np.float32)


# ------------------------------------------------------------ network
class Rete(nn.Module):
    """MLP with a shared trunk and two heads: policy (5 logits) and value V(s)."""

    def __init__(self, n_car=N_CAR, n_nasc=64, n_az=pn.N_AZIONI):
        super().__init__()
        self.tronco = nn.Sequential(
            nn.Linear(n_car, n_nasc), nn.Tanh(),
            nn.Linear(n_nasc, n_nasc), nn.Tanh())
        self.testa_pi = nn.Linear(n_nasc, n_az)
        self.testa_v = nn.Linear(n_nasc, 1)

    def forward(self, x):
        t = self.tronco(x)
        return self.testa_pi(t), self.testa_v(t).squeeze(-1)


class PoliticaAppresa:
    """Wraps the network in the standard policy interface used by gioca().

    avida=True takes the most probable action (greedy) instead of sampling.
    """
    usa_credenza = True

    def __init__(self, rete, nome="appresa", avida=False):
        self.rete, self.nome, self.avida = rete, nome, avida

    def __call__(self, p, cred, rng):
        x = torch.from_numpy(caratteristiche(p, cred)).unsqueeze(0)
        with torch.no_grad():
            logit, _ = self.rete(x)
        if self.avida:
            return int(logit.argmax())
        pr = torch.softmax(logit, -1).numpy().ravel()
        return int(rng.choice(pn.N_AZIONI, p=pr))


# --------------------------------------------------------- one episode
def raccogli(rete, seme, tetto=600, d0=8.0, campiona=None):
    """Play one episode sampling actions from the policy and record everything.

    Returns (features, actions, rewards, final features, found, steps, oracle).
    """
    rng = np.random.default_rng(seme)
    # `campiona` chooses the WORLD of this episode (domain randomisation).
    # The belief receives the same environment, so inference stays exact:
    # only the difficulty changes, not the model.
    kw = campiona(rng) if campiona is not None else {}
    p = pn.Pennacchio(passi=tetto, d_minima_iniziale=d0, **kw)
    h = p.reset(rng)
    cred = Credenza(p)
    cred.aggiorna(p.r, h)
    n_or = p.passi_oracolo()
    X, A, R = [], [], []
    while True:
        x = caratteristiche(p, cred)
        with torch.no_grad():
            logit, _ = rete(torch.from_numpy(x).unsqueeze(0))
        pr = torch.softmax(logit, -1).numpy().ravel()
        a = int(rng.choice(pn.N_AZIONI, p=pr))
        h, trovato, finito, info = p.passo(a)
        X.append(x); A.append(a)
        # -1 per step: maximising the reward = minimising the number of steps.
        # Sparse on purpose (no reward shaping).
        R.append(0.0 if trovato else -1.0)
        if not trovato:
            cred.aggiorna(p.r, h)
        if finito:
            # state AFTER the last action: needed for the bootstrap if the
            # episode was truncated rather than terminated
            x_fin = caratteristiche(p, cred)
            return (np.array(X), np.array(A), np.array(R, dtype=np.float32),
                    x_fin, trovato, info["t"], n_or)


def da_adesso_in_poi(R, gamma, coda=0.0):
    """G_t = reward from t onwards, with `coda` as the value after the last step.

    TERMINATION and TRUNCATION are different:
      terminated  the source was found: the value after the end is really 0;
      truncated   the step budget ran out: the value is NOT 0 (steps would
                  still be needed). Using 0 makes a failure look like a success.
    For truncated episodes the tail value is a bootstrap estimate.
    """
    G = np.zeros_like(R)
    acc = float(coda)
    for t in range(len(R) - 1, -1, -1):
        acc = R[t] + gamma * acc
        G[t] = acc
    return G


# -------------------------------------------------------- imitation
def imita(maestro=None, n_episodi=120, passi=200, epoche=400, lr=3e-3,
          seme_torch=0, verboso=True):
    """Behaviour cloning: learn to copy a rule-based policy (default: MLS).

    Supervised training with cross-entropy on (features, teacher action)
    pairs from 120 episodes on their own seeds (90000+). Used as the starting
    point of REINFORCE in step 0, because REINFORCE from scratch reaches only
    8% success.
    """
    import torch.nn.functional as F
    maestro = maestro or P.VersoMassimo()
    torch.manual_seed(seme_torch)
    X, Y = [], []
    rng = np.random.default_rng(0)
    for s in range(n_episodi):
        p = pn.Pennacchio(passi=passi)
        h = p.reset(np.random.default_rng(90_000 + s))   # their own seeds, disjoint from the others
        cred = Credenza(p)
        cred.aggiorna(p.r, h)
        for _ in range(passi):
            X.append(caratteristiche(p, cred))
            a = maestro(p, cred, rng)
            Y.append(a)
            h, trov, fin, _ = p.passo(a)
            if trov or fin:
                break
            cred.aggiorna(p.r, h)
    X = torch.from_numpy(np.array(X))
    Y = torch.tensor(Y).long()
    rete = Rete()
    ott = torch.optim.Adam(rete.parameters(), lr=lr)
    for _ in range(epoche):
        logit, _ = rete(X)
        perdita = F.cross_entropy(logit, Y)
        ott.zero_grad(); perdita.backward(); ott.step()
    acc = float((rete(X)[0].argmax(1) == Y).float().mean())
    if verboso:
        print("imitazione di %s: %d campioni, accuratezza %.1f%%"
              % (maestro.nome, len(Y), 100 * acc))
    return rete, acc


# --------------------------------------------------------- training
# ========================= SEED PARTITION =========================
# Three disjoint sets, fixed in advance.
# 
#   TRAINING     0, 1, 2, ...   used in sequence, as many as needed
#   DEVELOPMENT  10000..10039   used to choose algorithms and parameters
#   TEST         20000..20399   looked at ONCE, at the end
# 
# Why three and not two: after trying many configurations on the same
# evaluation seeds, those seeds become part of the training and the
# result is no longer comparable with infotaxis.
SEMI_SVILUPPO = list(range(10_000, 10_040))
# test set enlarged from 100 to 400 seeds BEFORE any use, for statistical
# power
SEMI_COLLAUDO = list(range(20_000, 20_400))


def valuta(politica, semi, tetto):
    """Censored cost (failures count as the step cap) and success rate on the given seeds."""
    passi, trov = [], []
    for s in semi:
        p = pn.Pennacchio(passi=tetto)
        t, ps, no, _ = P.gioca(p, politica, np.random.default_rng(s))
        passi.append(ps if t else tetto)
        trov.append(t)
    return float(np.mean(passi)), float(np.mean(trov))


def addestra(lotti=400, per_lotto=16, lr=3e-3, gamma=1.0, baseline="nessuna",
             peso_v=0.5, peso_entropia=0.004, normalizza=True, tetto=600,
             seme_torch=0, ogni=25, n_eval=40, curriculum=True,
             avvio="zero", rete_iniziale=None, scalda_valore=0,
             campiona=None, verboso=True):
    """REINFORCE training loop (step 0).

    Two fixes that were needed to make it learn at all:
      (1) reward scaling: G is divided by the step cap, so it lies in [-1, 0]
          and the value loss no longer dominates the policy loss;
      (2) curriculum on the starting distance: the agent starts 3 cells from
          the source and the distance grows by 0.5 every time the training
          success rate exceeds 60%, up to 8 cells.
    Evaluation never changes: always distance 8 and cap 600 on the
    development seeds.
    """
    torch.manual_seed(seme_torch)
    if rete_iniziale is not None:
        rete = rete_iniziale          # to measure BEFORE and AFTER
    elif avvio == "imitazione":
        rete, _ = imita(seme_torch=seme_torch, verboso=verboso)
    elif avvio == "zero":
        rete = Rete()
    else:
        raise ValueError(avvio)
    ott = torch.optim.Adam(rete.parameters(), lr=lr)
    # for the WARM-UP: optimise ONLY the value head, keeping the trunk
    # frozen, so that the policy does not change during the warm-up
    ott_v = torch.optim.Adam(rete.testa_v.parameters(), lr=lr * 10)

    SEMI_EVAL = SEMI_SVILUPPO[:n_eval]   # DEVELOPMENT seeds
    prossimo_seme = 0
    media_corrente = 0.0
    storia = []
    d0 = 3.0 if curriculum else 8.0

    if verboso:
        print("riferimenti sugli stessi %d semi di SVILUPPO, distanza 8, tetto %d"
              % (n_eval, tetto))
        for cls in (P.Casuale, P.VersoMassimo, P.Infotassi):
            c, t = valuta(cls(), SEMI_EVAL, tetto)
            print("  %-14s costo %6.1f   trovata %3.0f%%" % (cls.nome, c, 100 * t))
        print("\n  lotto  costo_ev trovata_ev | d0  riusc_tr  |grad|  entropia")

    for it in range(1, lotti + 1):
        # the TRAINING step cap follows the curriculum: at distance 3 there is no
        # need for 600 steps, and a 600-step failure would weigh 12 times more
        # than a 50-step success in the gradient
        tetto_tr = int(np.clip(30 * d0, 80, 400))
        Xs, As, Gs, riusc = [], [], [], []
        for _ in range(per_lotto):
            X, A, R, x_fin, trov, ps, no = raccogli(rete, prossimo_seme,
                                                    tetto_tr, d0, campiona)
            prossimo_seme += 1
            Xs.append(X); As.append(A); riusc.append(trov)
            # reward scaled by the cap: -1/cap per step, so G lies in [-1, 0]
            Rs = R / tetto_tr
            if trov:
                coda = 0.0                     # terminated: nothing after
            else:
                with torch.no_grad():          # truncated: bootstrap with V
                    coda = float(rete(torch.from_numpy(x_fin).unsqueeze(0))[1])
            Gs.append(da_adesso_in_poi(Rs, gamma, coda))
        X = torch.from_numpy(np.concatenate(Xs))
        A = torch.from_numpy(np.concatenate(As)).long()
        G = torch.from_numpy(np.concatenate(Gs))
        tasso_tr = float(np.mean(riusc))

        logit, V = rete(X)
        logp = torch.log_softmax(logit, -1).gather(1, A.unsqueeze(1)).squeeze(1)

        if it <= scalda_valore:
            # policy frozen: only V learns to predict the returns
            pv = ((V - G) ** 2).mean()
            ott_v.zero_grad(); pv.backward(); ott_v.step()
            if verboso and (it % ogni == 0 or it == 1):
                print("  %5d  (scaldo il valore)  perdita V %.4f" % (it, float(pv)))
            continue

        if baseline == "nessuna":
            vantaggio = G
        elif baseline == "media":
            media_corrente = 0.9 * media_corrente + 0.1 * float(G.mean())
            vantaggio = G - media_corrente
        elif baseline == "appresa":
            vantaggio = G - V.detach()
        else:
            raise ValueError(baseline)

        if normalizza:
            vantaggio = (vantaggio - vantaggio.mean()) / (vantaggio.std() + 1e-8)

        perdita_pi = -(logp * vantaggio).mean()
        ent = -(torch.softmax(logit, -1) * torch.log_softmax(logit, -1)).sum(1).mean()
        perdita_v = ((V - G) ** 2).mean() if baseline == "appresa" else torch.zeros(())
        perdita = perdita_pi + peso_v * perdita_v - peso_entropia * ent

        ott.zero_grad()
        perdita.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(rete.parameters(), 10.0))
        ott.step()

        # the curriculum advances on the TRAINING success rate, not on the
        # evaluation one (at distance 8 it is zero at the beginning)
        if curriculum and tasso_tr > 0.6 and d0 < 8.0:
            d0 = min(8.0, d0 + 0.5)

        if it % ogni == 0 or it == 1:
            costo, trov = valuta(PoliticaAppresa(rete), SEMI_EVAL, tetto)
            storia.append({"lotto": it, "costo": costo, "trovata": trov,
                           "d0": d0, "riuscita_tr": tasso_tr, "grad": gnorm,
                           "entropia": float(ent.detach())})
            if verboso:
                print("  %5d  %7.1f   %4.0f%%    | %.1f   %4.0f%%   %7.2f   %6.3f"
                      % (it, costo, 100 * trov, d0, 100 * tasso_tr, gnorm,
                         float(ent.detach())))
    return rete, storia, SEMI_EVAL


def collauda(politica, tetto=600, verboso=True):
    """Final test: run ONCE at the end, on the test seeds, together with the
    reference strategies (paired comparison, same cap).
    """
    righe = []
    for p in [politica, P.Casuale(), P.VersoMassimo(), P.Infotassi(),
              P.Oracolo()]:
        c, t = valuta(p, SEMI_COLLAUDO, tetto)
        righe.append((p.nome, c, t))
    if verboso:
        print("COLLAUDO su %d semi mai visti, tetto %d"
              % (len(SEMI_COLLAUDO), tetto))
        for nome, c, t in righe:
            print("  %-16s costo %6.1f   trovata %3.0f%%" % (nome, c, 100 * t))
    return righe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotti", type=int, default=400)
    ap.add_argument("--senza-curriculum", action="store_true")
    ap.add_argument("--per-lotto", type=int, default=16)
    ap.add_argument("--baseline", default="nessuna",
                    choices=["nessuna", "media", "appresa"])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tetto", type=int, default=600)
    ap.add_argument("--seme", type=int, default=0)
    ap.add_argument("--avvio", default="zero", choices=["zero", "imitazione"])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    rete, storia, semi = addestra(lotti=a.lotti, per_lotto=a.per_lotto,
                                  baseline=a.baseline, gamma=a.gamma,
                                  tetto=a.tetto, seme_torch=a.seme,
                                  curriculum=not a.senza_curriculum,
                                  avvio=a.avvio, lr=a.lr)
    print("\n  %.0f s" % (time.time() - t0))
    if a.out:
        os.makedirs(RUNS, exist_ok=True)
        torch.save(rete.state_dict(), os.path.join(RUNS, a.out + ".pt"))
        with open(os.path.join(RUNS, a.out + ".json"), "w") as f:
            json.dump({"argomenti": vars(a), "storia": storia}, f, indent=1)
        print("  salvato in runs/%s.{pt,json}" % a.out)
