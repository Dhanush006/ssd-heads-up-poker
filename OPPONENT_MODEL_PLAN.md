# Plan: learned opponent-strategy model

Status: PLAN ONLY. No code written. Nothing here touches `agent.py` on `main`.

## 0. The gate (user rule, 2026-11-07)

A candidate may enter the submission only if **both** are true:
1. It beats the frozen baseline (section 1) on the acceptance tests (section 5).
2. The user reads the results and explicitly approves the methodology.

Until then the submittable bot stays v2.1. Candidate work lives on branch `opp-model` in `agent_candidate.py` / `opp_model/`. Merging to `main` needs the user's "approved".

## 1. Frozen baseline (what we must beat)

- Code: v2.1 at commit `f41e047965bfc9526d9e8ff600093bff6363abe1` (Monte Carlo equity, 700 samples, counter-based opponent model, `DEFAULT_PARAMS`).
- Tag it `baseline-v2.1` so it cannot drift.
- Re-measure it with the exact benchmark below. Logged numbers (panel mean about +7.67 bb/deal) are only a reference. The comparison uses fresh paired runs on the same seeds.

## 2. Goal

Two jobs, one small model:
1. **Learn the opponent's strategy** from observable behaviour (fold-to-raise, raise rate per street, bet-size mix, check rate, showdown strength).
2. **Adjust our strategy** from that: choose or blend tuned parameter profiles (raise/bet thresholds, bluff frequency, call margin, range width `q`).

Fallback: if confidence is low or anything throws, use `DEFAULT_PARAMS`. The candidate can then never be worse than the baseline by construction when the model is unsure.

## 3. Benchmark (built first, frozen before any model work)

**Train panel** (used for tuning and training): the 7 bots in `tools/tune.py`: `random`, `always_pot`, `calling_station`, `eq_bot`, `tight_passive`, `lag`, `station_plus`. The baseline's parameters were tuned on these, so they favour the baseline.

**Hold-out panel** (never used for training or tuning, only for acceptance):
- `always_half` and `opponents/v1` (exist, not in the tuning panel).
- New bots to write: check-raiser, 3-bettor, mixed-strategy nit, tilt bot (switches style mid-half), and an adaptive exploiter that attacks fixed profiles.
- A new equilibrium-flavoured bot, if time allows.

**Protocol**
- Duplicate deals, both halves, same pairing and CI as the judge (`tools/selfplay.py`).
- Seeds fixed in the repo (for example 101 to 108). Never reused for tuning.
- 1000 deals per opponent per seed.
- Report per-opponent bb/deal, panel mean, and the **paired difference** (candidate minus baseline on identical deals) with a 95% CI.
- Log every run in `EXPERIMENTS.md` (version, opponent, deals, bb/deal, CI, think time, errors).

## 4. Stages

| # | Stage | Output | Cost |
|---|---|---|---|
| 1 | Build benchmark and hold-out bots; record baseline numbers | `tools/bench.py`, new bots in `opponents/`, baseline table in `EXPERIMENTS.md` | CPU, low |
| 2 | Per-style tuning: fit the best parameter profile against each style cluster with `tools/tune.py` | profile table (nit, station, maniac, lag, eq-bot, ...) | CPU, low |
| 3 | Data generator: play the bot against many randomized opponents (panel plus randomly parameterized variants), log opponent stats at hands 30, 60, 100, 200, 400 with true style labels | dataset | CPU, medium |
| 4 | Train the classifier: small MLP or gradient-boosted trees, about 10 features in, style probabilities out. Check accuracy per hand count and calibration | `opp_model/model.npz` (kilobytes) | CPU, low |
| 5 | Integrate in `agent_candidate.py`: probability-weighted blend of profiles, with default-param fallback when confidence is low | candidate bot | low |
| 6 | Shadow mode first: log what the model would choose while playing baseline params, to catch misclassification before it can cost chips | log | low |
| 7 | Acceptance run (section 5) and report | results table | CPU, low |
| 8 | **Stop. User reviews and approves or rejects.** | decision | none |
| 9 | Only if approved: merge, run `./tools/check_build.sh`, record the final commit ID | submission | low |

Cheap alternatives to compare against the classifier in stage 7, so we know the model earns its place:
- A **hand-written style switch**: if fold rate over 0.5, use the "folder" profile, and so on.
- The **v2.1 counters** alone.

## 5. Acceptance criteria (all must hold)

1. **Overall:** paired-difference 95% CI on hold-out panel mean bb/deal is entirely above 0.
2. **Train panel:** candidate panel mean is at least the baseline panel mean.
3. **No big regressions:** no single opponent loses more than 0.5 bb/deal against the baseline.
4. **Hardest opponents:** `tight_passive`, `eq_bot` and `v1` each improve or stay level (CI includes 0 at worst).
5. **Safety:** 0 errors across all runs. All 39 existing tests plus new tests for the model pass.
6. **Time:** charged think time per half is no more than 10% above the baseline and under 3 s on this laptop. It must also pass the 3x slow-CPU test.
7. **Size and startup:** model file under 1 MB, startup under 10 s.
8. **Generalization check:** classifier accuracy on hold-out opponent types is reported, including behaviour on a style it never saw. It must fall back to defaults, not guess wildly.

If any item fails, we do not ship the candidate. We report which one failed and iterate or stop.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Overfitting to our own sparring bots (the baseline is already tuned on them) | Hold-out panel and unseen styles; fixed benchmark seeds |
| Misclassification early in a half costs chips | Confidence threshold, default-param fallback, shadow mode |
| Noise hides small gains (std about 20 bb per hand) | 1000 deals × 8 seeds, paired differences |
| Time budget | Tiny model, counters stay O(1) |
| Tournament opponents unlike any bot we test | Include style-switching and exploiter bots; report honestly that this is a proxy |

## 7. Deliverables for the approval review

- Baseline vs candidate table (train panel, hold-out panel, per opponent, paired CI).
- Think-time and error report.
- Classifier accuracy and confusion matrix.
- Comparison against the hand-written style switch and the v2.1 counters.
- A plain statement of what improved, what did not, and the evidence.
