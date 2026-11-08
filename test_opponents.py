"""Tests for the sparring opponents in opponents/."""
import importlib.util
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "opponents"))

from game import Action, AgentConfig, Game, make_deal  # noqa: E402

HOLDOUTS = ["check_raiser", "three_bettor", "mixed_nit", "tilt", "exploiter"]
NEW = ["sizer", "river_bluffer", "trapper", "adaptive_caller", "nut_nit", "range_eq"]


def load_path(name):
    spec = importlib.util.spec_from_file_location("opp_" + name, os.path.join(HERE, "opponents", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(case):
    if isinstance(case, int):
        import param_bot
        return param_bot.make(param_bot.sample(np.random.default_rng(case)), seed=case)
    return load_path(case)


CASES = HOLDOUTS + NEW + [0, 1, 2]


@pytest.mark.parametrize("case", CASES, ids=[str(c) if isinstance(c, str) else f"param{c}" for c in CASES])
def test_bot_plays_legally(case):
    bot = build(case)
    mem = {}
    bot.new_half(AgentConfig(), mem)
    rng = random.Random(5)
    used = set()
    for hand in range(60):
        game = Game(make_deal("opp-test", hand), button_seat=hand % 2, hand_index=hand)
        my_seat = 0 if hand < 30 else 1
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == my_seat:
                action = bot.choose_action(obs, mem)
                assert action in obs.legal_actions
                used.add(action)
            else:
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < 0.3 else Action.CHECK_CALL
            game.apply(action)
        bot.on_hand_end(game.hand_result(my_seat), mem)
    assert len(used) >= 2, f"only used {used}"


def test_param_sample_ranges():
    import param_bot
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = param_bot.sample(rng)
        assert 0.55 <= p["raise_eq"] <= 0.85
        assert 0.0 <= p["bluff"] <= 0.40
        assert -0.10 <= p["call_margin"] <= 0.20
        assert 0.30 <= p["pf_floor"] <= 0.55


def test_exploiter_learns_fold_rate():
    bot = load_path("exploiter")
    mem = {}
    bot.new_half(AgentConfig(), mem)
    saw_probe = False
    for hand in range(40):
        game = Game(make_deal("fold-test", hand), button_seat=hand % 2, hand_index=hand)
        my_seat = hand % 2
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == my_seat:
                action = bot.choose_action(obs, mem)
                assert action in obs.legal_actions
                if action == Action.RAISE_HALF and obs.to_call == 0 and obs.street >= 1:
                    saw_probe = True
            else:
                action = Action.FOLD if Action.FOLD in obs.legal_actions else Action.CHECK_CALL
            game.apply(action)
        bot.on_hand_end(game.hand_result(my_seat), mem)
        assert mem["raised_last"] is False
    assert mem["folds"] > 0
    assert saw_probe


def test_tilt_tight_phase_legal():
    bot = load_path("tilt")
    mem = {}
    bot.new_half(AgentConfig(), mem)
    rng = random.Random(3)
    for h in range(20):
        game = Game(make_deal("tilt-test", h), button_seat=h % 2, hand_index=150 + h)
        my_seat = h % 2
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == my_seat:
                action = bot.choose_action(obs, mem)
                assert action in obs.legal_actions
            else:
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < 0.3 else Action.CHECK_CALL
            game.apply(action)
        bot.on_hand_end(game.hand_result(my_seat), mem)


def test_adaptive_caller_widens_vs_raiser():
    bot = load_path("adaptive_caller")
    mem = {}
    bot.new_half(AgentConfig(), mem)
    base = None
    for hand in range(40):
        game = Game(make_deal("adapt-test", hand), button_seat=hand % 2, hand_index=hand)
        my_seat = hand % 2
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == my_seat:
                action = bot.choose_action(obs, mem)
                assert action in obs.legal_actions
                if base is None:
                    base = mem["agg"]
            else:
                raises = [a for a in obs.legal_actions if a == Action.RAISE_POT]
                action = raises[0] if raises else Action.CHECK_CALL
            game.apply(action)
        bot.on_hand_end(game.hand_result(my_seat), mem)
    assert mem["agg"] > base
