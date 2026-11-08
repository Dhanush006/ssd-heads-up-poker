"""Tests for oppmodel.py and the candidate bot. Run: python -m pytest test_oppmodel.py -q"""
import importlib.util
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import oppmodel  # noqa: E402
import strategy  # noqa: E402

DEFAULT = strategy.DEFAULT_PARAMS


def fake_model(hands=100, raises=(30, 20, 15, 10), acts=(100, 80, 60, 50), faced=40, folds=20):
    m = strategy.new_model()
    m["opp_act"] = list(acts)
    m["opp_raise"] = list(raises)
    m["faced"], m["fold_to_raise"] = faced, folds
    m["sd_raise_n"], m["sd_raise_pct"] = [5, 5], [3.5, 3.0]
    m["results"] = [0] * hands
    return m


def test_features_shape_and_range():
    x = oppmodel.features(fake_model(), 100)
    assert x.shape == (len(oppmodel.FEATURE_NAMES),)
    assert np.isfinite(x).all() and x.min() >= 0.0 and x.max() <= 1.0


def test_features_on_empty_model_use_priors():
    x = oppmodel.features(strategy.new_model(), 0)
    assert np.isfinite(x).all()
    assert abs(x[0] - 0.30) < 1e-9 and abs(x[4] - 0.35) < 1e-9


def test_blend_one_hot_default_profile_equals_default():
    out = oppmodel.blend_params(DEFAULT, [{}, {"bluff_freq": 0.5}], np.array([1.0, 0.0]))
    for k, v in DEFAULT.items():
        assert np.allclose(out[k], v), k


def test_blend_interpolates_scalars_and_lists():
    over = [{}, {"bluff_freq": 0.5, "bet_half": [0.4, 0.4, 0.4, 0.4]}]
    out = oppmodel.blend_params(DEFAULT, over, np.array([0.5, 0.5]))
    assert out["bluff_freq"] == pytest.approx(0.5 * DEFAULT["bluff_freq"] + 0.25)
    assert out["bet_half"][0] == pytest.approx(0.5 * DEFAULT["bet_half"][0] + 0.2)
    assert len(out["bet_half"]) == 4
    assert out["n_samples"] == DEFAULT["n_samples"]


def test_softmax_predict_sums_to_one():
    W = np.zeros((3, 8)); b = np.array([0.0, 1.0, 2.0])
    p = oppmodel.predict(np.zeros(8), W, b, np.zeros(8), np.ones(8))
    assert p.sum() == pytest.approx(1.0) and p.argmax() == 2


def test_advisor_returns_default_before_min_hands():
    adv = oppmodel.Advisor(DEFAULT)
    assert adv.params(fake_model(hands=5), 5) is DEFAULT


def test_advisor_falls_back_when_weights_missing(monkeypatch):
    monkeypatch.setattr(oppmodel, "_STATE", None)
    adv = oppmodel.Advisor(DEFAULT)
    assert adv.params(fake_model(), 100) is DEFAULT


def test_advisor_falls_back_on_exception(monkeypatch):
    K = 2
    state = (np.zeros((K, 8)), np.zeros(K), np.zeros(8), np.ones(8), [{}, {}])
    monkeypatch.setattr(oppmodel, "_STATE", state)
    adv = oppmodel.Advisor(DEFAULT)
    monkeypatch.setattr(oppmodel, "features", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert adv.params(fake_model(), 100) is DEFAULT


def test_advisor_low_confidence_uses_default(monkeypatch):
    K = 3
    state = (np.zeros((K, 8)), np.zeros(K), np.zeros(8), np.ones(8), [{}] * K)   # uniform probs
    monkeypatch.setattr(oppmodel, "_STATE", state)
    adv = oppmodel.Advisor(DEFAULT)
    assert adv.params(fake_model(), 100) is DEFAULT


def test_advisor_confident_blends_toward_profile(monkeypatch):
    K = 2
    state = (np.zeros((K, 8)), np.array([0.0, 10.0]), np.zeros(8), np.ones(8),
             [{}, {"bluff_freq": 0.5}])
    monkeypatch.setattr(oppmodel, "_STATE", state)
    adv = oppmodel.Advisor(DEFAULT)
    out = adv.params(fake_model(), 100)
    assert out is not DEFAULT and out["bluff_freq"] == pytest.approx(0.5, abs=1e-3)


def test_advisor_caches_between_refreshes(monkeypatch):
    K = 2
    state = (np.array([[5.0] + [0.0] * 7, [-5.0] + [0.0] * 7]), np.zeros(K), np.zeros(8), np.ones(8),
             [{}, {"bluff_freq": 0.5}])
    monkeypatch.setattr(oppmodel, "_STATE", state)
    adv = oppmodel.Advisor(DEFAULT)
    a = adv.params(fake_model(), 100)
    b = adv.params(fake_model(), 105)           # within REFRESH: same object
    assert a is b


def test_advisor_cost_is_tiny(monkeypatch):
    K = 4
    rng = np.random.default_rng(0)
    state = (rng.normal(size=(K, 8)), np.zeros(K), np.zeros(8), np.ones(8), [{}, {}, {}, {}])
    monkeypatch.setattr(oppmodel, "_STATE", state)
    m = fake_model()
    t = time.perf_counter()
    for i in range(1000):
        adv = oppmodel.Advisor(DEFAULT)          # worst case: recompute every call
        adv.params(m, 100 + i)
    per_call = (time.perf_counter() - t) / 1000
    assert per_call < 200e-6, f"{per_call * 1e6:.0f} µs per recompute (budget 200 µs)"


def test_no_heavy_imports():
    import subprocess
    code = "import sys, oppmodel; bad=[m for m in ('torch','sklearn','scipy') if m in sys.modules]; print(bad)"
    out = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {HERE!r}); {code}"],
                         capture_output=True, text=True).stdout.strip()
    assert out == "[]", f"oppmodel pulled in heavy modules: {out}"


def _state2(sd=None, ov=None):
    return (np.zeros((2, 8)), np.array([0.0, 10.0]), np.zeros(8),
            np.ones(8) if sd is None else sd, ov or [{}, {"bluff_freq": 0.5}])


def test_advisor_zero_sd_returns_default(monkeypatch):
    sd = np.ones(8); sd[0] = 0.0
    monkeypatch.setattr(oppmodel, "_STATE", _state2(sd))
    assert oppmodel.Advisor(DEFAULT).params(fake_model(), 100) is DEFAULT


def test_valid_rejects_bad_weights():
    K = 2
    ok = (np.zeros((K, 8)), np.zeros(K), np.zeros(8), np.ones(8), [{}, {"bluff_freq": 0.5}])
    assert oppmodel._valid(*ok)
    bad_sd = np.ones(8); bad_sd[3] = 0.0
    assert not oppmodel._valid(ok[0], ok[1], ok[2], bad_sd, ok[4])
    W = np.zeros((K, 8)); W[0, 0] = np.nan
    assert not oppmodel._valid(W, ok[1], ok[2], ok[3], ok[4])
    assert not oppmodel._valid(ok[0], np.zeros(3), ok[2], ok[3], ok[4])
    assert not oppmodel._valid(ok[0], ok[1], np.zeros(7), ok[3], ok[4])
    assert not oppmodel._valid(ok[0], ok[1], ok[2], ok[3], [{}, {"bluff_freq": float("inf")}])
    assert not oppmodel._valid(ok[0], ok[1], ok[2], ok[3], [{}, {"bet_half": [0.1, float("nan"), 0.1, 0.1]}])


def test_no_blend_keys_keep_default():
    d = dict(DEFAULT); d["prior_n"] = 7.0
    out = oppmodel.blend_params(d, [{}, {"prior_n": 99.0, "bluff_freq": 0.5}], np.array([0.0, 1.0]))
    assert out["prior_n"] == 7.0 and out["bluff_freq"] == pytest.approx(0.5)


def test_int_defaults_not_blended():
    d = {"bluff_freq": 0.1, "depth": 3}
    out = oppmodel.blend_params(d, [{}, {"depth": 9, "bluff_freq": 0.5}], np.array([0.5, 0.5]))
    assert out["depth"] == 3 and isinstance(out["depth"], int)


def test_failure_does_not_retry_every_hand(monkeypatch):
    monkeypatch.setattr(oppmodel, "_STATE", _state2())
    calls = []
    def boom(*a, **k):
        calls.append(1); raise RuntimeError("x")
    monkeypatch.setattr(oppmodel, "features", boom)
    adv = oppmodel.Advisor(DEFAULT)
    adv.params(fake_model(), 100)
    adv.params(fake_model(), 105)
    assert len(calls) == 1


def test_hands_going_backwards_recomputes(monkeypatch):
    monkeypatch.setattr(oppmodel, "_STATE", _state2())
    real = oppmodel.features
    calls = []
    monkeypatch.setattr(oppmodel, "features", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    adv = oppmodel.Advisor(DEFAULT)
    adv.params(fake_model(), 100)
    adv.params(fake_model(), 50)
    assert len(calls) == 2


def test_profiles_json_is_valid():
    import json, os
    import strategy
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "profiles.json")
    d = json.load(open(p))
    names, ovs = d["names"], d["overrides"]
    assert len(names) == len(ovs) >= 2
    assert ovs[0] == {}
    for ov in ovs:
        for k, v in ov.items():
            assert k in strategy.DEFAULT_PARAMS
            assert k not in oppmodel.NO_BLEND
            base = strategy.DEFAULT_PARAMS[k]
            if isinstance(base, list):
                assert isinstance(v, list) and len(v) == len(base)
