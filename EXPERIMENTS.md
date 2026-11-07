# Experiments log

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
