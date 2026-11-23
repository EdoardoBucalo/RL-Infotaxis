# RL-Infotaxis

Master's project: a reinforcement learning agent compared with infotaxis in searching for an odor source from sparse, noisy detections.

Project work for the first-level Master in Data Science and Artificial Intelligence (UniNettuno), academic year 2026/2027.
Author: Edoardo Renato Bucalo.

## The problem

An agent moves on a 41×41 grid and has to find an odor source it cannot see. Downwind of the source, odor particles are carried by the wind. At each step the agent only receives a Poisson count *h* of detections. The mean of the count, μ, comes from the stationary solution of the advection–diffusion equation (Bessel K₀, Vergassola et al., *Nature* 2007). The agent has 5 actions: stay, ±x, ±y. The agent keeps an exact Bayesian belief over the 41×41 possible source positions and updates it after every count.

The compared strategies are:

| Strategy | Code name | Idea |
|---|---|---|
| random | `Casuale` | random action |
| MLS (most likely source) | `VersoMassimo` | move towards the most probable cell of the belief |
| infotaxis | `Infotassi` | move to minimise the expected Shannon entropy of the belief |
| oracle | `Oracolo` | knows the source: shortest path (lower bound) |
| RL, step 0 | `rete_nominale_*` | MLP on 8 features of the belief, imitation of MLS + REINFORCE |
| RL, step 1 | `g1_mappa_*` | CNN on the 81×81 egocentric belief map, imitation of infotaxis + REINFORCE |

## Main results

Final test on **400 seeds never used before** (20000–20399), with a step cap of 600. The cost is censored: a failed episode counts as 600 steps. The values for the RL rows are the mean over the trained networks (6 for step 0, 3 for step 1).

| Policy | Nominal signal (R_em = 0.5): steps | success | Rare signal (R_em = 0.05): steps | success |
|---|---|---|---|---|
| random | 587.5 | 5.2% | 584.8 | 6.5% |
| MLS | 258.1 | 99.0% | 552.6 | 17.5% |
| step 0: 8 features + RL | 255.4 | 97.3% | 555.2 | 16.0% |
| CNN clone of infotaxis | 146.4 | 97.2% | 370.9 | 74.7% |
| **step 1: map + RL (final)** | **135.9** | **99.5%** | **365.6** | **77.2%** |
| infotaxis | 131.3 | 100% | 364.5 | 74.5% |
| oracle | 29.1 | 100% | 29.1 | 100% |

The paired comparisons per seed give:

- REINFORCE improves the CNN clone by 10.5 ± 3.4 steps at the nominal signal (t = −3.1).
- The final network and infotaxis are not significantly different: +4.6 ± 3.6 steps (t = +1.3) at the nominal signal, and +2.7 ± 2.4 points of success rate (t = +1.1) at the rare signal.

## Repository structure

```
src/                      environment, inference, strategies and experiments
  pennacchio.py           plume model, Poisson detections, 5 actions
  credenza.py             exact Bayesian belief over the 41x41 source positions
  politiche.py            reference strategies (random, MLS, infotaxis, oracle) and gioca()
  verifica.py             sanity checks of physics and inference + ranking of the strategies
  reinforce.py            step 0: features, MLP, imitation, REINFORCE (hand-written), seed split
  step0_baseline_A.py     step 0: REINFORCE with learned baseline, 6 seeds      -> runs/semi.json
  step0_baseline_BC.py    step 0: no learned baseline / learned + warm-up        -> runs/diagnosi.json
  train_step0.py          step 0: the 6 final MLP networks                       -> runs/rete_nominale_*.pt
  mappa.py                step 1: egocentric map, CNN, cloning of infotaxis      -> runs/clone_mappa_*.pt, mappa.json
  step1_reinforce.py      step 1: REINFORCE from the CNN clones                  -> runs/g1_mappa_*.pt, gradino1.json
  step1_valuta_400.py     step 1: before/after on 400 extra development seeds    -> runs/gradino1_esteso.json
  collaudo.py             final test on the 400 test seeds                       -> runs/collaudo.json
plots/
  make_figures.py         recreates the figures of the thesis from runs/
runs/                     saved results (json); the trained networks (.pt) are not included
figures/                  output of make_figures.py
```

## How to run

Use Python 3.10+ and install the dependencies (PyTorch on CPU is enough):

```bash
pip install -r requirements.txt
```

To recreate all the figures of the thesis from the saved results (this takes a few seconds):

```bash
python plots/make_figures.py
```

To check the environment and the inference (about 1 minute):

```bash
python src/verifica.py
```

The full pipeline runs in the same order as in the thesis. Each script writes its output to `runs/`.

```bash
# step 0 (MLP on 8 features)
python src/reinforce.py --avvio zero --lotti 200      # REINFORCE from random weights (reaches ~8% success)
python src/step0_baseline_A.py                         # learned baseline
python src/step0_baseline_BC.py                        # no learned baseline / learned + warm-up
python src/train_step0.py                              # the 6 final step-0 networks

# step 1 (CNN on the belief map)
python src/mappa.py                                    # clone infotaxis: 8 features vs map
python src/step1_reinforce.py                          # REINFORCE, 150 batches, 3 networks
python src/step1_valuta_400.py                         # re-evaluation on 400 development seeds

# final test (run once)
python src/collaudo.py
```

The trained networks (`runs/*.pt`) are not in the repository: the training scripts create them, and `step1_valuta_400.py` and `collaudo.py` need them. The figures only need the json files, which are included.

`mappa.py`, `step1_reinforce.py`, `step1_valuta_400.py` and `collaudo.py` use a process pool with up to 10 workers. Step 1 takes a few hours on a laptop CPU.

## Seeds

The seeds are split into three disjoint sets, so that the final numbers are not tuned on the data they are measured on:

| Set | Seeds | Used for |
|---|---|---|
| training | 0, 1, 2, … (step 0), 40000+ (step 1) | episodes used by REINFORCE |
| imitation data | 90000+ (MLS), 30000+ / 31000+ (infotaxis) | behaviour cloning |
| development | 10000–10039, extra 10040–10439 | choosing methods and parameters |
| test | 20000–20399 | final test only, used once |

## Glossary (Italian code → English)

The identifiers in the code are in Italian, so that the code here is exactly the code that produced the results. Comments and docstrings are in English.

| Italian | English |
|---|---|
| pennacchio | plume |
| credenza | belief |
| politica / politiche | policy / policies |
| Casuale, VersoMassimo, Infotassi, Oracolo | random, MLS, infotaxis, oracle |
| caratteristiche | features |
| rete, ReteMappa | network, CNN on the map |
| mappa (egocentrica) | (egocentric) belief map |
| imita / clona | imitation / behaviour cloning |
| addestra, lotto, lotti | train, batch, batches |
| premio, perdita | reward, loss |
| baseline nessuna / media / appresa | none / running mean / learned |
| scaldamento (scalda_valore) | warm-up of the value head |
| passo, passi, tetto | step, steps, step cap |
| costo (censurato) | (censored) cost in steps |
| trovata, riuscita | found, success rate |
| seme, semi | seed, seeds |
| gradino 0 / 1 | step 0 / 1 |
| sviluppo, collaudo | development, final test |
| prima / dopo | before / after |
| R_em | emission rate of the source |

## References

- M. Vergassola, E. Villermaux, B. I. Shraiman, *'Infotaxis' as a strategy for searching without gradients*, Nature 445, 406–409 (2007).
- R. J. Williams, *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, Machine Learning 8, 229–256 (1992).
- R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press (2018).
- S. Ross, G. Gordon, D. Bagnell, *A reduction of imitation learning and structured prediction to no-regret online learning*, AISTATS (2011).
