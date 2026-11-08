"""Check fasteval.omaha_scores == evaluator.omaha_score on random deals, and time it."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
t = time.time()
import fasteval
print("fasteval import s", round(time.time() - t, 3))
from evaluator import omaha_score

rng = np.random.default_rng(0)
N = 50000
cards = np.argsort(rng.random((N, 36)), axis=1)[:, :10]
fast = fasteval.omaha_scores(cards[:, :5], cards[:, 5:])
ref = np.array([omaha_score(list(r[:5]), list(r[5:])) for r in cards])
print("mismatches", int((fast != ref).sum()), "of", N)
for n in (200, 400, 800):
    h = cards[:n, :5].copy(); b = cards[:n, 5:].copy()
    t = time.perf_counter()
    for _ in range(200):
        fasteval.omaha_scores(h, b)
    print(n, "rows: ms per call", round((time.perf_counter() - t) / 200 * 1000, 3))
