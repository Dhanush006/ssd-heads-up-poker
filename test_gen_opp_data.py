import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "opponents"))
import gen_opp_data as G   # noqa: E402
import param_bot           # noqa: E402


def test_aggregate():
    rng = np.random.default_rng(1)
    descs = [("fixed", "a.py", 0), ("fixed", "b.py", 0)]
    res, f = [], {}
    for di, d in enumerate(descs):
        for k in range(2):
            for s in ("11", "12"):
                mean = 10.0 * di + k + (0.5 if s == "12" else 0.0)
                feats = []
                if k == 0:
                    for c in (40, 80):
                        v = rng.random(8)
                        f[(di, c, s)] = v
                        feats.append((c, v))
                res.append((d, k, s, mean, feats))
    X, Y, Gr = G.aggregate(res, 2, ckpts=(40, 80))
    assert X.shape == (4, 8) and Y.shape == (4, 2)
    for r in range(4):
        di = r // 2
        assert np.allclose(Y[r], [10.0 * di + 0.25, 10.0 * di + 1.25])
    c = [40, 80]
    for r in range(4):
        di, ci = r // 2, c[r % 2]
        assert np.allclose(X[r], (f[(di, ci, "11")] + f[(di, ci, "12")]) / 2)
    assert Gr[0] == Gr[1] != Gr[2] == Gr[3]


def test_build_descs():
    d = G.build_descs(3)
    assert len(d) == 10 and [x[0] for x in d].count("param") == 3
    assert len(G.FIXED) == 7
    assert d[:7] == [("fixed", p, 0) for p in G.FIXED]
    assert all(x[0] == "param" for x in d[7:])
    assert G.build_descs(3) == d
    json.dumps(d)


def test_run_match_smoke():
    p = param_bot.sample(np.random.default_rng(0))
    desc = ("param", p, 0)
    _, k, _, mean, feats = G.run_match((desc, 0, {}, "11", 45))
    assert np.isfinite(mean) and len(feats) == 1 and feats[0][0] == 40
    x = feats[0][1]
    assert x.shape == (8,) and (x >= 0).all() and (x <= 1).all()
    _, k, _, mean, feats = G.run_match((desc, 1, {}, "11", 45))
    assert np.isfinite(mean) and feats == []


def test_label_sign_vs_calling_station():
    desc = ("fixed", "baseline-bots/baseline_calling_station.py", 0)
    _, _, _, mean, _ = G.run_match((desc, 0, {}, "11", 60))
    assert mean > 0
