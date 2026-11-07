# Experiments log

## STATUS (read first after a restart)
- Best safe version: v2.1 = f41e047965bfc9526d9e8ff600093bff6363abe1 (code). Always `git log -1 main`; main is kept submittable. v1 = 59b1f90.
- Tuning: `python tools/tune.py '{}' '{"key":val}'` (about 30 s per 3 variants, restartable; no long background runs).
- Time guard (v2.1): startup speed calibration + live slowdown EMA; budget 3.5 s soft / 6.0 s hard (estimated judge charge).
  `python tools/slowcpu.py 3` simulates a 3x slower CPU.
- Next planned step: build a stronger range-aware sparring bot (snapshot v2 as opponents/v2) and tune
  call/raise vs aggressive equity bots; preflop 3-bet ranges; verify think time on slower CPU (taskpolicy / cpulimit).

Harness: `tools/selfplay.py` (in-process duplicate matches, same pairing/CI as the judge).
Judge rows use `judge_engine.py` over HTTP (charged think time = what the tournament counts).
bb/deal is per paired deal; CI is 95%.

| Version | Opponent | Deals | bb/deal ± CI | Verdict | Think/half (charged) | Errors | Notes |
|---|---|---|---|---|---|---|---|
| v1 | baseline_random | 4x500 selfplay | +7.54 ± 0.87 | WIN all seeds | ~1.5 s raw in-proc | 0 | MC equity 700 samples, preflop table |
| v1 | baseline_always_pot | 4x500 | +8.62 ± 1.11 | WIN all | ~1.7 s raw | 0 | |
| v1 | baseline_always_half | 4x500 | +13.19 ± 0.87 | WIN all | ~2.1 s raw | 0 | |
| v1 | baseline_calling_station | 4x500 | +8.62 ± 0.54 | WIN all | ~1.5 s raw | 0 | |
| v1 | always_pot (HTTP judge, seed 11) | 1000 | +8.28 ± 1.56 | WIN | 0.53 s / 0.44 s | 0 | wall ~1.1 ms/call |
| v1 | random (HTTP judge, seed 12) | 1000 | +7.83 ± 1.25 | WIN | 0.51 s / 0.64 s | 0 | adaptive time guard |
| v1 | opponents/eq_bot (equity vs random + pot odds) | 4x500 | +3.70 ± 0.51 | WIN | ~1.5 s raw | 0 | typical competitor style |
| v1 | opponents/tight_passive | 4x500 | +1.36 ± 0.31 | WIN | | 0 | smallest edge: nit folds to bets |
| v1 | opponents/lag | 4x500 | +7.83 ± 0.74 | WIN | | 0 | |
| v1 | opponents/station_plus | 4x500 | +6.55 ± 0.54 | WIN | | 0 | |
| v2 | panel mean (7 opps, 2x400 selfplay) | 5600 | +7.67 (was +6.45 for v1) | | | 0 | thinner value bets (bet_half ~0.50-0.54), raise_eq 0.56-0.72, maniac trap (+0.10 raise_eq when opp raise freq > 0.55), bluff 0.1, own-action tracking bug fixed |
| v2 | opponents/v1 | 8x500 | +0.73 ± ~0.6 | WIN (pooled) | | 0 | head-to-head vs previous version |
| v2 | always_pot (HTTP judge, seed 21) | 1000 | +7.94 ± 1.59 | WIN | 0.71 s / 0.65 s | 0 | |
| v2 | calling_station (HTTP judge, seed 22) | 1000 | +10.87 ± 0.96 | WIN | 0.30 s / 0.40 s | 0 | |

Negative / neutral findings: bluff_freq 0.4, preflop steals, re-raise bluffs: no gain vs this panel (opponents mostly call). eps 0.03-0.15 and n_samples 400 vs 700: within noise.
Bandit meta-policy: per-hand std is about 20 bb, so within 1,000 hands UCB cannot separate profiles that differ by < 2 bb/hand. Using direct opponent-model adjustments instead.
| v2.1 | random (HTTP judge, seeds 31/32, laptop load avg 12) | 1000 x2 | +7.10 ± 1.28 / +7.17 ± 1.32 | WIN | 1.85/1.74, 1.71/1.87 s | 0 | before guard fix the same runs spiked to 4.73 s and 3.64 s per half under CPU contention |
| v2.1 | slowcpu sim (always_pot) factor 1 / 3 | 1000 | n/a | | est. 2.2 s / 3.7 s | 0 | n drops to 60 when slow |

## Phase 3 (branch phase3) - aggressive/adaptive sparring panel
New opponents: opponents/lag3bet.py (light 3-bets), opponents/river_bluffer.py, opponents/exploiter.py (adapts to our fold rate), opponents/v2 (snapshot of submitted v2.1). Panel = 11 opponents, 4 seeds x 500 paired deals per opponent.

| Candidate | Panel mean | vs v2 | Worst opponent | Verdict |
|---|---|---|---|---|
| v2 (seeds 1-4) | +6.04 | 0 | tight_passive +1.34 ± 0.34 | baseline; aggressive bots: lag3bet +4.16, river_bluffer +5.47, exploiter +4.63 (all WIN) |
| pre_reraise_add [-0.02,0,0.03,0.06] | +6.17 | +0.14 ± 0.17 | | within noise |
| pre_reraise_add [0,0.02,0.05,0.08] / [0.03,0,0,0] | +6.10 / +5.95 | ~0 | | no gain |
| pos_call 0.02 / pos_bet ±0.02 | +5.94 / +6.04 / +5.97 | ~0 | | no gain |
| call_pow 0.5 / 1.0 (narrow range after opp calls) | +5.86 / +5.78 | ~0 | | worse: too cautious |
| seeds 5-8: v2 / n=1200 / eps 0.05 / eps 0.05 + reraise | +5.53 / +5.53 / +5.49 / +5.60 | -0.04 / +0.19 / -0.14 | | within noise |

Verdict: plateau. No candidate clearly beats v2, so main stays at the submitted v2.1 (c94d9f4). New parameters are kept but off by default (behaviour identical to v2). 0 errors in every run; in-process think time <= 2.6 s per half with 8 matches running in parallel.
Next ideas if time allows: opponent-specific preflop ranges from showdown data, multi-street planning (bet sizing by SPR), river bluff-catching calibrated by observed river bluff rate.
