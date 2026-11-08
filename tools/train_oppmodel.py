"""Fit the softmax profile selector and report held-out policy value.

  python tools/train_oppmodel.py --data data/opp_dataset.npz --out data/oppmodel.npz
"""
import argparse, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import oppmodel  # noqa: E402

TEMP = 0.25     # bb/deal temperature for soft targets
L2 = 1e-2
STEPS = 3000
LR = 0.3


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit(X, Y):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mu) / sd
    T = softmax_rows(Y / TEMP)
    K, F = Y.shape[1], X.shape[1]
    W, b = np.zeros((K, F)), np.zeros(K)
    for _ in range(STEPS):
        P = softmax_rows(Z @ W.T + b)
        G = (P - T) / len(Z)
        W -= LR * (G.T @ Z + L2 * W)
        b -= LR * G.sum(axis=0)
    return W, b, mu, sd


def policy_value(W, b, mu, sd, X, Y, min_conf):
    P = softmax_rows(((X - mu) / sd) @ W.T + b)
    use_default = P.max(axis=1) < min_conf
    P = np.where(use_default[:, None], np.eye(Y.shape[1])[0], P)      # profile 0 = default
    return float((P * Y).sum(axis=1).mean())


def group_folds(G, folds=5, seed=0):
    """Boolean test masks; each group is in exactly one test fold."""
    u = np.unique(G)
    np.random.default_rng(seed).shuffle(u)
    return [np.isin(G, part) for part in np.array_split(u, folds)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "opp_dataset.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "oppmodel.npz"))
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(argv)
    d = np.load(a.data)
    X, Y, G = d["X"], d["Y"], d["G"]
    K = Y.shape[1]
    tot = {"default": 0.0, "model": 0.0, "oracle": 0.0}
    for test in group_folds(G, a.folds, 0):
        W, b, mu, sd = fit(X[~test], Y[~test])
        n = test.sum()
        tot["default"] += Y[test][:, 0].sum()
        tot["model"] += policy_value(W, b, mu, sd, X[test], Y[test], oppmodel.MIN_CONF) * n
        tot["oracle"] += Y[test].max(axis=1).sum()
    res = {k: float(v / len(Y)) for k, v in tot.items()}
    print("held-out policy value (bb/deal, group CV over %d opponents):" % len(np.unique(G)))
    print("  always default : %+.3f" % res["default"])
    print("  model          : %+.3f   (gain %+.3f)" % (res["model"], res["model"] - res["default"]))
    print("  oracle best    : %+.3f   (upper bound)" % res["oracle"])
    W, b, mu, sd = fit(X, Y)
    if not oppmodel._valid(W, b, mu, sd, [{}] * K):
        raise SystemExit("trained weights failed oppmodel._valid; not saving")
    np.savez(a.out, W=W, b=b, mu=mu, sd=sd)
    print("saved %s: W%s, %d bytes" % (a.out, W.shape, os.path.getsize(a.out)))
    return res


if __name__ == "__main__":
    main()
