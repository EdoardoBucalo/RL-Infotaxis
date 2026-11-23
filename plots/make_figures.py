#!/usr/bin/env python3
"""Recreate the result figures of the thesis from the saved data in runs/.

No training and no new simulations: every figure is read from the json files
written by the scripts in src/ (except the signal map, which is computed
directly from the plume model). Labels are in Italian, as in the thesis.

  figures/rl_griglia_segnale.png        expected signal mu on the 41x41 grid
                                        (linear and log scale)
  figures/rl_f1_rappresentazione.png    how well each representation clones
                                        infotaxis, vs signal level  (mappa.json)
  figures/rl_f4_prima_dopo.png          the 3 CNN networks before and after
                                        REINFORCE, 400 dev seeds (gradino1_esteso.json)
  figures/rl_f2_collaudo_nominale.png   final test, nominal signal: steps
  figures/rl_f3_collaudo_raro.png       final test, rare signal: success rate
                                        (both from collaudo.json)

Run from the repository root:  python plots/make_figures.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = os.path.join(ROOT, "runs")
FIG = os.path.join(ROOT, "figures")
sys.path.insert(0, os.path.join(ROOT, "src"))
from pennacchio import Pennacchio  # noqa: E402

os.makedirs(FIG, exist_ok=True)

# colours: blue = RL network, orange = infotaxis, grey = everything else
BLU, ARANCIO, GRIGIO = "#2a78d6", "#eb6834", "#b9bdc2"
INK, INK2, GRID, SUP = "#0b0b0b", "#52514e", "#e6e6e3", "#fcfcfb"

STYLE = {
    "font.family": "sans-serif", "font.size": 10.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SUP,
    "figure.facecolor": SUP, "axes.spines.top": False, "axes.spines.right": False,
}


def load(name):
    with open(os.path.join(RUNS, name)) as f:
        return json.load(f)


def virgola(s):
    """Italian decimal comma."""
    return s.replace(".", ",")


def griglia(ax, asse="y"):
    ax.grid(axis=asse, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ------------------------------------------------------- signal map (Fig. 2.1)
def fig_griglia_segnale():
    """Expected signal on the grid, source at (10,20), agent start at (30,8)."""
    env = Pennacchio()
    env.reset(np.random.default_rng(0))
    env.r0 = np.array([10.0, 20.0])
    env.r = np.array([30.0, 8.0])
    mu = env.campo()

    with plt.rc_context({"font.size": 11}):
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        kw = dict(origin="lower", extent=[-.5, 40.5, -.5, 40.5])
        pannelli = [(None, "Segnale atteso μ, scala lineare",
                     "rilevazioni attese per passo"),
                    (LogNorm(), "Stessa mappa, scala logaritmica",
                     "rilevazioni attese per passo (log)")]
        for a, (norm, tit, lab) in zip(ax, pannelli):
            im = a.imshow(mu.T, cmap="Blues", norm=norm, **kw)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label=lab)
            # thin lines between cells
            a.set_xticks(np.arange(-.5, 41, 1), minor=True)
            a.set_yticks(np.arange(-.5, 41, 1), minor=True)
            a.grid(which="minor", color="#999", lw=0.25, alpha=0.5)
            a.tick_params(which="minor", length=0)
            a.set_xticks(range(0, 41, 10)); a.set_yticks(range(0, 41, 10))
            a.plot(*env.r0, "*", ms=18, mfc="#E8590C", mec="white", mew=1.5,
                   label="sorgente")
            a.plot(*env.r, "o", ms=10, mfc="#2B8A3E", mec="white", mew=1.5,
                   label="partenza agente")
            a.annotate("", xy=(38, 37), xytext=(30, 37),
                       arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
            a.text(34, 38.4, "vento", ha="center", color="#333")
            a.set_title(tit); a.set_xlabel("x (celle)"); a.set_ylabel("y (celle)")
        ax[0].legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG, "rl_griglia_segnale.png"), dpi=140)
        plt.close(fig)


# ------------------------------------------- cloning accuracy (Fig. 3.2, f1)
def fig_f1():
    m = load("mappa.json")
    liv = m["livelli"]
    acc = m["accuratezza"]
    car = np.array([acc["car_%d" % s] for s in range(3)]) * 100
    mpp = np.array([acc["mappa_%d" % s] for s in range(3)]) * 100
    x = np.arange(len(liv))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for dati, col, nome in [(mpp, BLU, "mappa 81×81 (CNN)"),
                            (car, ARANCIO, "8 features")]:
        # mean over the 3 networks of each representation
        ax.plot(x, dati.mean(0), color=col, linewidth=2, marker="o",
                markersize=7, markeredgecolor=SUP, markeredgewidth=2,
                label=nome, zorder=3)
        ax.text(x[-1] + 0.12, dati.mean(0)[-1], nome, color=INK,
                va="center", fontsize=10)
    ax.set_xticks(x, [virgola(str(v)) for v in liv])
    ax.set_xlim(-0.3, len(liv) - 0.3 + 1.3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("tasso di emissione R_em  (segnale più raro →)")
    ax.set_ylabel("decisioni dell'infotaxis riprodotte")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK)
    griglia(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rl_f1_rappresentazione.png"), dpi=200)
    plt.close(fig)


# ------------------------------------- before / after REINFORCE (Fig. 3.3, f4)
def fig_f4():
    e = load("gradino1_esteso.json")
    R = "0.5"                       # nominal signal
    info = e["%s|infotassi" % R]["costo"]
    prima = [e["%s|clone_mappa_%d" % (R, s)]["costo"] for s in range(3)]
    dopo = [e["%s|g1_mappa_%d" % (R, s)]["costo"] for s in range(3)]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    alfa = [0.55, 0.7, 0.85]
    for s in range(3):
        ax.plot([0, 1], [prima[s], dopo[s]], color=BLU, alpha=alfa[s],
                linewidth=2, marker="o", markersize=6,
                markeredgecolor="#cfe0f5", markeredgewidth=1.5, zorder=3)
        ax.text(-0.06, prima[s], virgola("rete %d: %.1f" % (s, prima[s])),
                ha="right", va="center", fontsize=9.5, color=INK)
    # labels on the right, moved apart where the points are too close
    yd = list(dopo)
    ordine = np.argsort(yd)
    for i in range(1, 3):
        a, b = ordine[i - 1], ordine[i]
        if yd[b] - yd[a] < 3.2:
            yd[b] = yd[a] + 3.2
    for s in range(3):
        ax.text(1.06, yd[s], virgola("rete %d: %.1f" % (s, dopo[s])),
                ha="left", va="center", fontsize=9.5, color=INK)
        if abs(yd[s] - dopo[s]) > 0.5:
            ax.plot([1.01, 1.05], [dopo[s], yd[s]], color=INK2, lw=0.7)
    ax.axhline(info, color=ARANCIO, linestyle="--", linewidth=1.6, zorder=2)
    ax.text(-0.72, info - 2.5, virgola("infotaxis: %.1f" % info),
            color=ARANCIO, fontsize=9.5, va="center")
    ax.set_xticks([0, 1], ["clone CNN\n(prima del RL)",
                           "dopo REINFORCE\n(150 batch)"])
    ax.set_xlim(-0.75, 1.6)
    ax.set_ylim(120, 165)
    ax.set_ylabel("costo censurato (passi, meno è meglio)")
    griglia(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rl_f4_prima_dopo.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------- final test (Fig. 4.1-4.2, f2 f3)
# keys used in collaudo.json -> labels used in the thesis
VOCI = [("verso-massimo", "MLS", "rif"),
        ("gradino 0 (8 car.)", "step 0: 8 features + RL", "rl"),
        ("clone a mappa", "clone a mappa dell'infotaxis (CNN)", "clone"),
        ("gradino 1 (finale)", "step 1: mappa + RL (finale)", "fin"),
        ("infotassi", "infotaxis", "info")]
COLORE = {"rif": GRIGIO, "rl": GRIGIO, "clone": GRIGIO, "fin": BLU,
          "info": ARANCIO}


def barre(P, N, R, chiave, xlabel, nome_file, fmt, xmax):
    """Horizontal bars; dots = single networks, error bars = binomial s.e."""
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    y = np.arange(len(VOCI))[::-1]
    for yi, (k, etichetta, tipo) in zip(y, VOCI):
        d = P["%s|%s" % (R, k)]
        scala = 100 if chiave == "trovata" else 1
        v = d[chiave] * scala
        ax.barh(yi, v, height=0.56, color=COLORE[tipo], zorder=2)
        per = d.get("per_rete" if chiave == "costo" else "trov_per_rete")
        if per:
            pv = np.array(per) * scala
            ax.scatter(pv, np.full(len(pv), yi), s=16, color=INK, zorder=3,
                       linewidths=0)
        elif chiave == "trovata":
            p = d["trovata"]
            se = 100 * np.sqrt(p * (1 - p) / N)
            ax.errorbar(v, yi, xerr=se, color=INK, capsize=3, linewidth=1,
                        zorder=3)
        ax.text(v + xmax * 0.012, yi + 0.27, fmt(v), va="center",
                fontsize=9.5, color=INK)
    ax.set_yticks(y, [e for _, e, _ in VOCI])
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    if chiave == "trovata":
        ax.set_xticks([0, 20, 40, 60, 80, 100],
                      ["0%", "20%", "40%", "60%", "80%", "100%"])
    griglia(ax, "x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, nome_file), dpi=200)
    plt.close(fig)


def fig_f2_f3():
    c = load("collaudo.json")
    P = c["politiche"]
    N = c["semi"][1] - c["semi"][0] + 1          # 400 test seeds
    barre(P, N, "0.5", "costo", "costo censurato (passi, meno è meglio)",
          "rl_f2_collaudo_nominale.png", lambda v: "%.0f" % v, 300)
    barre(P, N, "0.05", "trovata", "episodi in cui trova la sorgente",
          "rl_f3_collaudo_raro.png", lambda v: virgola("%.1f%%" % v), 100)


if __name__ == "__main__":
    fig_griglia_segnale()
    with plt.rc_context(STYLE):
        fig_f1()
        fig_f4()
        fig_f2_f3()
    print("figures written to", os.path.abspath(FIG))
