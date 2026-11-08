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

## Opponent-model benchmark (frozen seeds 101-104, 1000 deals)

Date 2026-11-07. Baseline commit f41e047 (agent.py == opponents/v21/agent_v21.py). Harness: tools/bench.py with the time guard pinned (n fixed at _N_BASE), duplicate deals, seeds 101-104. Baseline bb/deal vs each opponent:

| panel | opponent | baseline bb/deal | errors |
|---|---|---|---|
| train | random | +7.53 | 0 |
| train | always_pot | +8.29 | 0 |
| train | calling_station | +11.28 | 0 |
| train | eq_bot | +5.10 | 0 |
| train | tight_passive | +1.69 | 0 |
| train | lag | +8.51 | 0 |
| train | station_plus | +9.00 | 0 |
| train | **mean** | **+7.34** | 0 |
| hold-out | always_half | +12.91 | 0 |
| hold-out | agent_v1 | +0.84 | 0 |
| hold-out | check_raiser | +4.92 | 0 |
| hold-out | three_bettor | +5.31 | 0 |
| hold-out | mixed_nit | +1.92 | 0 |
| hold-out | tilt | +5.30 | 0 |
| hold-out | exploiter | +2.83 | 0 |
| hold-out | **mean** | **+4.86** | 0 |

Think time under Pool(8) is 8-18 s/half (contention, guard pinned) and differs up to ~12% between identical bots; not comparable to judge charged time.

## Strategy profiles (tuning seeds 1,2, 400 deals)

Keep rule: variant MEAN must beat default MEAN (same command) by more than 0.3 bb/deal. Timing guard not pinned, so about 0.05-0.1 noise.

value_vs_callers (panel: calling_station, station_plus, always_pot) - KEPT (+0.339)
```
MEAN +9.757 | {} | calling_station=+11.82 station_plus=+8.58 always_pot=+8.88
MEAN +10.096 | {"bluff_freq":0.0,"bet_half":[0.47,0.48,0.49,0.51],"bet_pot":[0.52,0.57,0.60,0.63]} | calling_station=+12.72 station_plus=+9.20 always_pot=+8.36
```

pressure_vs_folders (panel: tight_passive, eq_bot; mixed_nit swapped out to keep hold-out clean) - DROPPED
```
MEAN +2.558 | {} | tight_passive=+0.87 eq_bot=+4.25
MEAN +2.599 | {"bluff_freq":0.35,"steal_freq":0.5,"reraise_bluff":0.3} | tight_passive=+0.82 eq_bot=+4.38
MEAN +2.547 | {} | tight_passive=+0.84 eq_bot=+4.26
MEAN +2.529 | {"bluff_freq":0.5,"steal_freq":0.8,"reraise_bluff":0.5} | tight_passive=+0.74 eq_bot=+4.31
```

tight_vs_aggressors (panel: lag, always_pot, random; hold-out bots swapped out) - DROPPED
```
MEAN +8.933 | {} | lag=+9.36 always_pot=+8.75 random=+8.69
MEAN +8.901 | {"maniac_adj":0.2,"maniac_agg":0.45,"call_margin":[0.02,0.05,0.05,0.04],"raise_eq":[0.60,0.68,0.72,0.76]} | lag=+9.32 always_pot=+9.20 random=+8.19
MEAN +8.461 | {} | lag=+9.42 always_pot=+8.88 random=+7.08
MEAN +7.308 | {"maniac_adj":0.3,"maniac_agg":0.35,"call_margin":[0.03,0.07,0.07,0.06],"raise_eq":[0.64,0.72,0.76,0.80]} | lag=+6.89 always_pot=+8.08 random=+6.95
```

Result: only value_vs_callers survives. data/profiles.json = default + value_vs_callers.

### Profile re-check with the pinned, paired harness (bench.run, seeds 1-4, 1000 deals, 7 training opponents)
tune.py is unpinned: the same default config scored +8.93 and +8.46 on two runs, so the +0.34 tuning margin was within noise. Re-checked pairwise with the time guard pinned:

| Profile vs default | calling_station | station_plus | always_pot | eq_bot | tight_passive | lag | random | Panel paired diff |
|---|---|---|---|---|---|---|---|---|
| value_vs_callers (kept) | +0.90 ± 0.17 | +0.75 ± 0.14 | -0.07 ± 0.14 | +0.14 ± 0.15 | -0.03 ± 0.11 | +0.51 ± 0.42 | +0.12 ± 0.71 | **+0.33 ± 0.14** |
| pressure_vs_folders (dropped) | +0.00 | -0.01 | +0.00 | +0.00 | -0.07 ± 0.06 | +0.00 | -0.32 ± 0.43 | -0.06 ± 0.06 |
| tight_vs_aggressors (dropped) | -0.63 ± 0.13 | -0.49 ± 0.13 | +0.03 ± 0.37 | -0.23 ± 0.15 | -0.01 ± 0.09 | -0.68 ± 0.46 | -0.98 ± 0.72 | -0.43 ± 0.15 |

Note: value_vs_callers is not significantly worse than default against any training opponent, so a fixed switch to it is a candidate in its own right (to be compared in the acceptance run). The plan's stop rule ("fewer than two profiles beat the default") triggered; continuing anyway with K=2 at the user's request to finish all tasks.

## Opponent-model training (Task 6), 2026-11-07
Dataset: 157 opponents (7 training bots + 150 param_bot), K=2 profiles (default, value_vs_callers), seeds 11,12, 400 deals, time guard pinned. 628 rows, 12 min on 8 procs.
Per-opponent mean: default +4.517, value_vs_callers +4.638 (better for 61% of opponents), oracle best-per-opponent +4.741.

Held-out policy value (group CV over opponents, 5 folds):

| TEMP | always default | learned model | gain | oracle |
|---|---|---|---|---|
| 0.05 | +4.517 | +4.612 | +0.095 | +4.741 |
| 0.10 | +4.517 | +4.610 | +0.093 | +4.741 |
| 0.25 | +4.517 | +4.602 | +0.085 | +4.741 |
| 0.50 | +4.517 | +4.593 | +0.076 | +4.741 |
| 1.00 | +4.517 | +4.586 | +0.069 | +4.741 |
| always value_vs_callers (no model) | | +4.638 | +0.121 | |

**Stop rule triggered:** best learned gain +0.095 < required +0.2, and the learned selector is worse than simply always using value_vs_callers (+0.121). Even a perfect selector (oracle) only reaches +0.224. With these profiles the learned model cannot earn its place; Tasks 7-8 (candidate bot with the advisor) are not worth building. Next: acceptance benchmark of the fixed value_vs_callers parameters vs the frozen baseline.

## Acceptance run: fixed value_vs_callers vs frozen v2.1 (bench.py, seeds 101-104, 1000 deals), 2026-11-07
Candidate: `agent.py@{"bluff_freq":0.0,"bet_half":[0.47,0.48,0.49,0.51],"bet_pot":[0.52,0.57,0.60,0.63]}`.

| Panel | Opponent | Candidate | Baseline | Paired diff ± 95% CI |
|---|---|---|---|---|
| train | random | +7.84 | +7.53 | +0.32 ± 0.72 |
| train | always_pot | +8.02 | +8.29 | -0.27 ± 0.17 |
| train | calling_station | +12.35 | +11.28 | +1.07 ± 0.17 |
| train | eq_bot | +5.40 | +5.10 | +0.30 ± 0.15 |
| train | tight_passive | +1.63 | +1.69 | -0.05 ± 0.13 |
| train | lag | +8.77 | +8.51 | +0.25 ± 0.42 |
| train | station_plus | +9.79 | +9.00 | +0.80 ± 0.15 |
| train | **panel** | +7.69 | +7.34 | **+0.34 ± 0.14** |
| hold-out | always_half | +12.55 | +12.91 | -0.36 ± 0.13 |
| hold-out | agent_v1 | +0.73 | +0.84 | -0.12 ± 0.19 |
| hold-out | check_raiser | +4.94 | +4.92 | +0.02 ± 0.17 |
| hold-out | three_bettor | +5.60 | +5.31 | +0.29 ± 0.22 |
| hold-out | mixed_nit | +1.97 | +1.92 | +0.05 ± 0.28 |
| hold-out | tilt | +5.26 | +5.30 | -0.04 ± 0.31 |
| hold-out | exploiter | +2.60 | +2.83 | -0.23 ± 0.19 |
| hold-out | **panel** | | | **-0.06 ± 0.10** |

Gates: G1 FAIL (hold-out -0.055 ± 0.101), G2-G6 PASS. Per-seed hold-out diffs [+0.08, +0.03, -0.03, -0.30]. **NOT ACCEPTED.**
Reading: the +0.34 train gain came from opponents the profile was tuned on (calling_station, station_plus); it does not carry over to unseen opponents and loses against pot/half-pot aggressors (always_pot -0.27, always_half -0.36). v2.1 stays the submission. The learned selector (Task 6) was below this candidate offline, so it was not built into a bot.

## Profile matrix on the 14-bot training panel (paired, pinned, tuning seeds 1-4, 1000 deals), 2026-11-07
Diff vs default per opponent; panel = mean over 14 bots.
```
value_vs_callers     panel +0.120 ± 0.097
calldown_vs_aggro    panel -0.159 ± 0.090
tighter_calls        panel -0.021 ± 0.076
bluff_vs_nits        panel -0.032 ± 0.066
thin_value_only      panel +0.092 ± 0.087
raise_wider          panel +0.147 ± 0.099

opponent            base    value_vs_cal  calldown_vs_  tighter_call  bluff_vs_nit  thin_value_o   raise_wider
random              +7.95   +0.12±0.71  +0.51±0.77  -0.53±0.71  -0.30±0.70  +0.28±0.67  +0.56±0.66
always_pot          +8.87   -0.07±0.14  -0.64±0.47  +0.05±0.36  +0.00±0.03  -0.07±0.14  +0.01±0.16
calling_station    +11.46   +0.90±0.17  +0.00±0.01  -0.00±0.00  +0.00±0.00  +0.53±0.13  +0.55±0.12
eq_bot              +4.79   +0.14±0.15  -0.16±0.13  +0.05±0.11  -0.04±0.08  +0.05±0.12  +0.22±0.12
tight_passive       +1.48   -0.03±0.11  -0.18±0.08  +0.04±0.04  -0.16±0.09  -0.02±0.08  +0.06±0.09
lag                 +8.53   +0.51±0.42  -0.48±0.44  +0.20±0.42  -0.11±0.23  +0.30±0.43  -0.05±0.44
station_plus        +8.96   +0.75±0.14  -0.18±0.09  +0.05±0.06  -0.04±0.03  +0.42±0.11  +0.40±0.13
always_half        +13.00   -0.40±0.11  +0.10±0.23  -0.27±0.21  -0.01±0.01  -0.32±0.11  +0.22±0.20
agent_v1            +0.77   -0.25±0.18  -0.32±0.16  +0.10±0.11  +0.84±0.26  +0.33±0.18  -0.05±0.14
check_raiser        +4.67   +0.01±0.16  -0.10±0.10  +0.00±0.09  -0.37±0.11  -0.03±0.14  +0.02±0.12
three_bettor        +5.04   +0.48±0.22  -0.10±0.13  -0.01±0.08  -0.06±0.11  +0.34±0.20  +0.09±0.36
mixed_nit           +1.89   -0.29±0.28  -0.62±0.27  -0.10±0.22  -0.12±0.26  -0.18±0.27  -0.17±0.26
tilt                +4.89   +0.10±0.32  -0.27±0.30  +0.06±0.25  -0.04±0.26  -0.07±0.31  +0.00±0.33
exploiter           +3.17   -0.29±0.20  +0.21±0.22  +0.06±0.13  -0.03±0.10  -0.29±0.18  +0.18±0.20

best single fixed profile: raise_wider +0.147 (mean over 14 opps vs default)
oracle per-opponent pick : +0.351  -> selector ceiling over best fixed +0.204
```
Reading: raise_wider ({"raise_eq":[0.53,0.60,0.64,0.68]}) is the best single fixed profile (+0.147 ± 0.099, no significant loss). Oracle per-opponent selector ceiling over it is only +0.204 (upward-biased by noise); a learned selector would capture roughly 40% of that, below the +0.2 stop rule, so data regeneration/retraining is skipped. Next: hold-out acceptance run of fixed raise_wider on the 6 new hold-out bots.

## Acceptance run: fixed raise_wider vs frozen v2.1 (bench.py, seeds 101-104, 1000 deals, new 6-bot hold-out), 2026-11-07
Candidate: `agent.py@{"raise_eq":[0.53,0.60,0.64,0.68]}` (default raise_eq is [0.56,0.64,0.68,0.72]).
```

== TRAIN ==
opponent                       cand     base        diff ± CI  think c/b (s)  err c/b
random                        +7.26    +7.53    -0.27 ± 0.70  10.53/10.34    0/0  
always_pot                    +8.20    +8.29    -0.09 ± 0.17  12.59/12.24    0/0  
calling_station              +11.79   +11.28    +0.51 ± 0.13   9.61/ 9.52    0/0  
eq_bot                        +5.26    +5.10    +0.16 ± 0.13   9.36/ 9.30    0/0  
tight_passive                 +1.75    +1.69    +0.07 ± 0.08   8.97/ 8.68    0/0  
lag                           +8.40    +8.51    -0.11 ± 0.45  13.06/12.93    0/0  
station_plus                  +9.36    +9.00    +0.37 ± 0.12  10.66/10.39    0/0  
always_half                  +13.31   +12.91    +0.40 ± 0.20  15.74/15.77    0/0  
agent_v1                      +0.78    +0.84    -0.06 ± 0.16   8.22/ 9.13    0/0  
check_raiser                  +5.08    +4.92    +0.17 ± 0.12   9.33/10.21    0/0  
three_bettor                  +4.88    +5.31    -0.43 ± 0.36   9.34/10.68    0/0  
mixed_nit                     +1.82    +1.92    -0.10 ± 0.27   8.51/ 9.55    0/0  
tilt                          +5.15    +5.30    -0.15 ± 0.33   9.87/ 9.33    0/0  
exploiter                     +2.89    +2.83    +0.06 ± 0.19  10.53/ 9.43    0/0  
PANEL paired diff                               +0.04 ± 0.10

== HOLDOUT ==
opponent                       cand     base        diff ± CI  think c/b (s)  err c/b
sizer                         +4.78    +4.58    +0.21 ± 0.10   9.19/ 8.47    0/0  
river_bluffer                 +4.94    +4.74    +0.20 ± 0.13   9.54/ 8.84    0/0  
trapper                       +5.66    +5.45    +0.20 ± 0.12   9.22/ 9.25    0/0  
adaptive_caller               +3.46    +3.15    +0.31 ± 0.13   8.78/ 8.86    0/0  
nut_nit                       +1.09    +1.00    +0.09 ± 0.04   4.66/ 5.20    0/0  
range_eq                      +0.71    +0.72    -0.01 ± 0.09   8.06/ 8.40    0/0  
PANEL paired diff                               +0.17 ± 0.07

== ACCEPTANCE GATES (user approval still required) ==
PASS  G1 hold-out paired CI lower bound > 0  [+0.168 ± 0.070]
PASS  G2 train-panel mean >= baseline  [cand +6.139 vs base +6.102]
PASS  G3 no opponent regresses > 0.5  [worst three_bettor -0.434]
PASS  G4 hard opponents improve or tie  [significantly worse: none]
PASS  G5 zero candidate errors  [0 errors]
PASS  G6 total think within 1.15x baseline (same run; real budgets tested in pytest)  [cand 195.8s vs base 196.5s total]
train per-seed panel diffs: [0.189, -0.063, 0.122, -0.102] (+0.037 ± 0.138)
holdout per-seed panel diffs: [0.307, 0.113, 0.161, 0.092] (+0.168 ± 0.095)
ALL GATES PASS
```
Reading: ALL GATES PASS. Hold-out +0.17 ± 0.07 bb/deal, positive on all 4 seeds and on 5 of 6 bots (range_eq flat). Train panel on fresh seeds only +0.04 ± 0.10 (vs +0.15 on tuning seeds: partly regression to the mean); worst opponent three_bettor -0.43 ± 0.36. First baseline numbers on the new hold-out: sizer +4.58, river_bluffer +4.74, trapper +5.45, adaptive_caller +3.15, nut_nit +1.00, range_eq +0.72. Awaiting user approval; not merged.
