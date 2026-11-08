import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import train_oppmodel as T   # noqa: E402
import oppmodel              # noqa: E402


def synth(n, seed, signal=True, rows=4):
    rng = np.random.default_rng(seed)
    X = rng.random((n * rows, 8))
    G = np.repeat(np.arange(n), rows)
    if signal:
        best = (X[:, 4] > 0.5).astype(int)
        Y = np.stack([np.zeros(len(X)), np.where(best == 1, 1.0, -1.0)], axis=1)
    else:
        Y = np.stack([np.zeros(len(X)), rng.choice([-1.0, 1.0], len(X))], axis=1)
    Y = Y + rng.normal(0, 0.05, Y.shape)
    return X, Y, G


def test_softmax_rows():
    P = T.softmax_rows(np.array([[1e4, 0.0, -1e4], [0.0, 0.0, 0.0]]))
    assert np.isfinite(P).all() and np.allclose(P.sum(axis=1), 1)


def test_learnable():
    X, Y, _ = synth(200, 0)
    W, b, mu, sd = T.fit(X, Y)
    Xt, Yt, _ = synth(200, 1)
    model = T.policy_value(W, b, mu, sd, Xt, Yt, oppmodel.MIN_CONF)
    default, oracle = Yt[:, 0].mean(), Yt.max(axis=1).mean()
    print("learnable", default, model, oracle)
    assert model - default > 0.3
    assert oracle - model < 0.2


def test_no_signal():
    X, Y, _ = synth(200, 0, signal=False)
    W, b, mu, sd = T.fit(X, Y)
    Xt, Yt, _ = synth(200, 1, signal=False)
    gain = T.policy_value(W, b, mu, sd, Xt, Yt, oppmodel.MIN_CONF) - Yt[:, 0].mean()
    print("nosignal gain", gain)
    assert abs(gain) < 0.15


def test_group_folds():
    G = np.repeat(np.arange(37), 4)
    masks = T.group_folds(G, 5, 0)
    assert len(masks) == 5
    assert (np.sum(masks, axis=0) == 1).all()
    for m in masks:
        assert m.any() and not (set(G[m]) & set(G[~m]))


def test_fit_valid():
    X, Y, _ = synth(50, 0)
    W, b, mu, sd = T.fit(X, Y)
    assert oppmodel._valid(W, b, mu, sd, [{}] * 2)


def test_main(tmp_path):
    X, Y, G = synth(60, 0)
    d, o = tmp_path / "d.npz", tmp_path / "o.npz"
    np.savez(d, X=X, Y=Y, G=G)
    r = T.main(["--data", str(d), "--out", str(o)])
    assert set(r) == {"default", "model", "oracle"}
    assert r["model"] > r["default"] and r["oracle"] >= r["model"] - 1e-9
    z = np.load(o)
    assert {"W", "b", "mu", "sd"} <= set(z.files)
