"""
Tests for YOUR bot (agent.py). Run them from the bundle folder:

    python -m pytest -q

They talk to your bot the same way the judge does (through its web routes,
but without starting a real server), so a passing run means the judge can
play with your bot. Add your own tests at the bottom!
"""

import importlib.util
import os
import random
import sys

import pytest
from fastapi.testclient import TestClient

# Works whether this file sits next to agent.py or inside a tests/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = HERE if os.path.isfile(os.path.join(HERE, "game.py")) else os.path.dirname(HERE)
sys.path.insert(0, BUNDLE)  # so agent.py can import game.py and evaluator.py

from game import Action, AgentConfig, Game, NETWORK_GRACE_SECONDS, make_deal  # noqa: E402
from judge_engine import coerce_action  # noqa: E402  (reads answers exactly like the judge)

TOKEN = "test-token"
AUTH = {"X-Agent-Token": TOKEN}


def load_agent():
    """Load a fresh copy of agent.py (its own MEMORY and globals)."""
    spec = importlib.util.spec_from_file_location("agent_under_test", os.path.join(BUNDLE, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot():
    """(agent module, test client) with the tournament's token check switched on."""
    module = load_agent()
    module.TOKEN = TOKEN
    return module, TestClient(module.app)


def start_half(client, half=1, seat=0):
    """POST /new-half the way the judge does."""
    body = {"half": half, "seat": seat, "config": AgentConfig().to_dict()}
    return client.post("/new-half", json=body, headers=AUTH)


def play_hands(client, num_hands, seed="test", raise_rate=0.3):
    """Play hands where YOUR bot is seat 0 and a random opponent is seat 1.

    Every decision goes through /act exactly like in a real match, and
    finished hands are delivered with the next /act (like the judge does).
    Returns the list of every action your bot chose.
    """
    rng = random.Random(seed)
    pending = []      # finished hands waiting to be delivered
    chosen = []
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200, f"/act failed: {r.json()}"
                reply = r.json()
                assert reply.get("hand_end_errors", []) == [], \
                    f"on_hand_end crashed: {reply['hand_end_errors'][0]['traceback']}"
                action = coerce_action(reply["action"])
                assert action in obs.legal_actions, \
                    f"choose_action returned {reply['action']!r}; legal were {[a.name for a in obs.legal_actions]}"
                chosen.append(action)
            else:
                # Random opponent that raises sometimes, so your bot faces bets.
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)  # end of half
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    return chosen


# ---------------------------------------------------------------------------
# Your bot plays correctly
# ---------------------------------------------------------------------------

def test_names_are_set(bot):
    _, client = bot
    info = client.get("/").json()
    assert isinstance(info["participant"], str) and info["participant"]
    assert isinstance(info["agent_name"], str) and info["agent_name"]


def test_new_half_works(bot):
    _, client = bot
    r = start_half(client)
    assert r.status_code == 200, f"new_half crashed: {r.json()}"


def test_every_decision_is_legal(bot):
    # 200 hands with a random opponent: every choose_action answer must be legal.
    _, client = bot
    start_half(client)
    play_hands(client, 200)


def test_works_in_both_seats_and_both_halves(bot):
    _, client = bot
    for half, seat in ((1, 0), (2, 1)):
        assert start_half(client, half, seat).status_code == 200
        play_hands(client, 30, seed=f"half{half}")


def test_handles_aggressive_opponent(bot):
    # An opponent that raises whenever it can: your bot must cope with
    # big bets and all-ins.
    _, client = bot
    start_half(client)
    play_hands(client, 100, seed="aggressive", raise_rate=1.0)


def test_memory_is_fresh_each_half(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 10)
    start_half(client, half=2, seat=1)
    after_reset = dict(module.MEMORY)
    fresh = load_agent()
    fresh.new_half(AgentConfig(), fresh.MEMORY)
    assert after_reset.keys() == fresh.MEMORY.keys(), "memory should only hold what new_half() puts there"


# ---------------------------------------------------------------------------
# The server plumbing still works (only fails if you edited it)
# ---------------------------------------------------------------------------

def test_token_is_checked(bot):
    _, client = bot
    body = {"half": 1, "seat": 0, "config": {}}
    assert client.post("/new-half", json=body).status_code == 401
    assert client.post("/new-half", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_bad_token_is_refused_before_the_body_is_read(bot):
    # Without the right token the bot must answer 401 without parsing the body,
    # so junk sent by anyone else can't slow your bot down.
    _, client = bot
    junk = b"this is not JSON" * 1000
    assert client.post("/act", content=junk).status_code == 401
    assert client.post("/act", content=junk, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_malformed_body_is_rejected(bot):
    _, client = bot
    start_half(client)
    assert client.post("/act", content=b"not json", headers=AUTH).status_code == 400
    assert client.post("/act", json=[1, 2, 3], headers=AUTH).status_code == 400


def test_act_before_new_half_is_refused(bot):
    _, client = bot
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    assert client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH).status_code == 409


def test_crashes_are_reported_to_the_judge(bot):
    # If choose_action raises, the judge must get a 500 with your traceback.
    module, client = bot

    def broken(obs, memory):
        raise ValueError("oops")

    module.choose_action = broken
    start_half(client)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH)
    assert r.status_code == 500
    assert "ValueError: oops" in r.json()["error"] and "Traceback" in r.json()["traceback"]


def test_reads_port_from_environment():
    source = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert 'os.environ.get("PORT"' in source, "agent.py must read its port from $PORT"
    assert '0.0.0.0' in source, "agent.py must listen on 0.0.0.0 so the judge can reach it"


# ---------------------------------------------------------------------------
# Add your own tests below, e.g. "with two aces preflop my bot raises".
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Extra tests: resource limits, strategy behaviour, edge cases, opponent model
# ---------------------------------------------------------------------------

import re  # noqa: E402
import time  # noqa: E402
from math import comb  # noqa: E402

import numpy as np  # noqa: E402

from game import HandResult, Observation  # noqa: E402

# Card = rank * 4 + suit; rank 0 = six ... 8 = ace; suit 3 = spades.
ACE_S, KING_S, QUEEN_S, JACK_S, TEN_S = 35, 31, 27, 23, 19


def make_obs(**over):
    """A preflop observation (button, facing the small-blind completion), with overrides."""
    d = {
        "hand_index": 0, "street": "PREFLOP", "hole_cards": [1, 12, 17, 24, 28], "board": [],
        "pot": 3, "to_call": 1, "my_bet": 1, "opp_bet": 2, "my_stack": 79, "opp_stack": 78,
        "is_button": True, "legal_actions": ["FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT"],
        "raise_to": {"RAISE_HALF": 4, "RAISE_POT": 6}, "min_raise_to": 4,
        "raises_this_street": 0, "action_history": [], "big_blind": 2,
    }
    d.update(over)
    return Observation.from_dict(d)


def fresh_memory(module):
    memory = {}
    module.new_half(AgentConfig(), memory)
    return memory


def run_direct(module, hands, seed="direct", raise_rate=0.3, plumbing=0.0):
    """Play `hands` hands in-process against a random opponent.

    Returns (charged seconds, slowest single decision, number of decisions).
    Charged = per request: time in our code + `plumbing` (HTTP cost), minus the
    free 1 ms, never below 0 - the way the judge bills the 8 s bank.
    """
    rng = random.Random(seed)
    memory = fresh_memory(module)
    total = worst = 0.0
    decisions = 0
    for hand in range(hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                t = time.perf_counter()
                action = module.choose_action(obs, memory)
                dt = time.perf_counter() - t
                assert action in obs.legal_actions
                total += max(0.0, dt + plumbing - NETWORK_GRACE_SECONDS)
                worst = max(worst, dt)
                decisions += 1
            else:
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(action)
        t = time.perf_counter()
        module.on_hand_end(game.hand_result(0), memory)
        total += time.perf_counter() - t
    return total, worst, decisions


# --- time and size limits ---------------------------------------------------

def test_startup_is_well_under_15_seconds():
    t = time.perf_counter()
    load_agent()          # imports strategy, builds tables, calibrates speed
    assert time.perf_counter() - t < 10.0, "agent must answer within 15 s of starting"


def test_think_time_for_a_full_half_fits_the_8s_bank():
    module = load_agent()
    total, worst, decisions = run_direct(module, 1000)
    # Judge also charges HTTP plumbing; leave a wide margin below 8 s.
    assert total < 4.0, f"{total:.2f}s thinking over {decisions} decisions is too close to the 8 s bank"
    assert worst < 0.5, f"slowest decision {worst:.3f}s (hard limit is 1 s)"


def test_time_guard_survives_a_3x_slower_cpu(monkeypatch):
    import strategy
    module = load_agent()
    factor = 3.0
    real_equity = strategy.equity

    def slow_equity(*a, **k):
        t = time.perf_counter()
        r = real_equity(*a, **k)
        time.sleep((time.perf_counter() - t) * (factor - 1))
        return r

    monkeypatch.setattr(strategy, "equity", slow_equity)
    monkeypatch.setattr(strategy, "SPEED", factor * 0.8)
    # HTTP plumbing (~0.5 ms on a dev laptop) scales with the slow machine too.
    charged, worst, decisions = run_direct(module, 1000, seed="slow", plumbing=0.0005 * factor)
    assert charged < 7.0, f"charged ~{charged:.2f}s on a 3x slower CPU; the bank is 8 s"
    assert worst < 1.0


def test_out_of_time_bot_still_plays_fast_legal_moves():
    module = load_agent()
    memory = fresh_memory(module)
    memory["charged"] = 99.0   # pretend the bank is spent
    t = time.perf_counter()
    for street, board in (("PREFLOP", []), ("FLOP", [22, 0, 29]), ("RIVER", [7, 23, 15, 29, 9])):
        obs = make_obs(street=street, board=board, to_call=2, pot=6)
        assert module.choose_action(obs, memory) in obs.legal_actions
    assert time.perf_counter() - t < 0.2, "after the hard budget the bot must use cheap rules only"


def test_requirements_pin_server_versions_and_avoid_gpu_packages():
    text = open(os.path.join(BUNDLE, "requirements.txt"), encoding="utf-8").read().lower()
    assert re.search(r"^fastapi==", text, re.M) and re.search(r"^uvicorn==", text, re.M)
    for banned in ("cuda", "nvidia", "jax", "tensorflow"):
        assert banned not in text, f"{banned} is too big or useless on the CPU-only tournament image"


def test_files_are_where_the_tournament_expects_them_and_small():
    for name in ("agent.py", "requirements.txt"):
        assert os.path.isfile(os.path.join(BUNDLE, name)), f"{name} must be in the repo root"
    for root, dirs, files in os.walk(BUNDLE):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", "venv", ".venv")]
        for f in files:
            size = os.path.getsize(os.path.join(root, f))
            assert size < 100 * 1024 * 1024, f"{f} is over GitHub's 100 MB limit"


def test_bot_code_never_writes_files_or_opens_network():
    # Match rules: read-only files (only /tmp is writable) and no network.
    for name in ("agent.py", "strategy.py", "fasteval.py"):
        source = open(os.path.join(BUNDLE, name), encoding="utf-8").read()
        assert not re.search(r"open\([^)]*['\"][wa]b?\+?['\"]", source), f"{name} writes a file"
        for banned in ("requests.", "urllib.request", "socket.", "np.save", "to_csv"):
            assert banned not in source, f"{name} uses {banned}"


def test_preflop_tables_are_complete_and_valid():
    import strategy
    n = comb(36, 5)
    assert strategy.PF_EQ.shape == (n,) and strategy.PF_RANK.shape == (n,)
    for table in (strategy.PF_EQ, strategy.PF_RANK):
        assert np.isfinite(table).all() and table.min() >= 0.0 and table.max() <= 1.0
    best = strategy.preflop_equity([35, 34, 31, 30, 27])    # A A K K Q
    worst = strategy.preflop_equity([0, 1, 4, 8, 13])
    assert best > worst


# --- strategy behaviour -----------------------------------------------------

def test_decisions_are_deterministic_for_the_same_observation():
    module = load_agent()
    obs = make_obs(street="FLOP", board=[22, 0, 29], hole_cards=[4, 9, 16, 21, 27], to_call=0, pot=12)
    first = module.choose_action(obs, fresh_memory(module))
    second = module.choose_action(obs, fresh_memory(module))
    assert first == second


def test_premium_preflop_hand_does_not_fold_and_raises():
    module = load_agent()
    obs = make_obs(hole_cards=[ACE_S, 34, KING_S, 30, QUEEN_S])
    action = module.choose_action(obs, fresh_memory(module))
    assert action in (Action.RAISE_HALF, Action.RAISE_POT), f"AAKKQ preflop played {action.name}"


def test_trash_hand_folds_to_a_pot_sized_preflop_raise():
    module = load_agent()
    obs = make_obs(hole_cards=[0, 4, 8, 12, 32], pot=8, to_call=4, my_bet=2, opp_bet=6, is_button=False,
                   raises_this_street=1, raise_to={"RAISE_HALF": 12, "RAISE_POT": 18}, min_raise_to=10,
                   action_history=[{"street": "PREFLOP", "is_me": False, "action": "RAISE_POT",
                                    "amount": 5, "street_total": 6}])
    assert module.choose_action(obs, fresh_memory(module)) == Action.FOLD


def test_nut_flush_straight_on_river_never_folds():
    module = load_agent()
    hole = [ACE_S, KING_S, 0, 4, 8]
    board = [QUEEN_S, JACK_S, TEN_S, 12, 16]          # royal flush in spades
    obs = make_obs(street="RIVER", hole_cards=hole, board=board, pot=40, to_call=20, my_bet=0, opp_bet=20,
                   raise_to={"RAISE_HALF": 50, "RAISE_POT": 80}, min_raise_to=40, raises_this_street=1)
    assert module.choose_action(obs, fresh_memory(module)) != Action.FOLD


def test_worthless_river_hand_folds_to_a_pot_bet():
    module = load_agent()
    hole = [0, 5, 10, 15, 20]
    board = [3, 8, 13, 18, 23]
    obs = make_obs(street="RIVER", hole_cards=hole, board=board, pot=40, to_call=40, my_bet=0, opp_bet=40,
                   raise_to={"RAISE_HALF": 100, "RAISE_POT": 160}, min_raise_to=80, raises_this_street=1)
    action = module.choose_action(obs, fresh_memory(module))
    assert action in (Action.FOLD, Action.CHECK_CALL)   # must not raise with nothing


def test_never_folds_when_checking_is_free():
    module = load_agent()
    for street, board in (("FLOP", [22, 0, 29]), ("TURN", [22, 0, 29, 7]), ("RIVER", [22, 0, 29, 7, 10])):
        obs = make_obs(street=street, board=board, hole_cards=[1, 6, 11, 16, 21], to_call=0,
                       legal_actions=["CHECK_CALL", "RAISE_HALF", "RAISE_POT"], pot=12, my_bet=0, opp_bet=0)
        assert module.choose_action(obs, fresh_memory(module)) != Action.FOLD


# --- edge cases -------------------------------------------------------------

def test_facing_all_in_returns_fold_or_call():
    module = load_agent()
    obs = make_obs(street="FLOP", board=[22, 0, 29], hole_cards=[4, 9, 16, 21, 27], pot=134, to_call=26,
                   my_bet=36, opp_bet=62, my_stack=26, opp_stack=0,
                   legal_actions=["FOLD", "CHECK_CALL"], raise_to={}, min_raise_to=88, raises_this_street=2)
    assert module.choose_action(obs, fresh_memory(module)) in (Action.FOLD, Action.CHECK_CALL)


def test_only_check_call_legal():
    module = load_agent()
    obs = make_obs(street="RIVER", board=[22, 0, 29, 7, 10], hole_cards=[1, 6, 11, 16, 21], to_call=0,
                   legal_actions=["CHECK_CALL"], raise_to={}, my_stack=0, opp_stack=10)
    assert module.choose_action(obs, fresh_memory(module)) == Action.CHECK_CALL


def test_short_stack_and_empty_history_do_not_crash():
    module = load_agent()
    obs = make_obs(my_stack=1, opp_stack=1, to_call=1, pot=3, raise_to={}, legal_actions=["FOLD", "CHECK_CALL"])
    assert module.choose_action(obs, fresh_memory(module)) in obs.legal_actions


def test_broken_strategy_falls_back_to_a_legal_move(monkeypatch):
    import strategy
    module = load_agent()
    memory = fresh_memory(module)

    def boom(*a, **k):
        raise RuntimeError("strategy exploded")

    monkeypatch.setattr(strategy, "decide", boom)
    obs = make_obs(street="FLOP", board=[22, 0, 29], hole_cards=[4, 9, 16, 21, 27], to_call=4, pot=12)
    assert module.choose_action(obs, memory) in obs.legal_actions


def test_illegal_strategy_answer_is_replaced(monkeypatch):
    import strategy
    module = load_agent()
    memory = fresh_memory(module)
    monkeypatch.setattr(strategy, "decide", lambda *a, **k: int(Action.RAISE_POT))
    obs = make_obs(legal_actions=["FOLD", "CHECK_CALL"], raise_to={})   # raises not allowed
    assert module.choose_action(obs, memory) in obs.legal_actions


def test_missing_strategy_module_still_plays(monkeypatch):
    module = load_agent()
    monkeypatch.setattr(module, "_strategy", None)
    memory = fresh_memory(module)
    assert memory["model"] == {}
    obs = make_obs()
    assert module.choose_action(obs, memory) in obs.legal_actions


def test_on_hand_end_never_raises_on_odd_results():
    module = load_agent()
    memory = fresh_memory(module)
    module.choose_action(make_obs(), memory)
    odd = [
        HandResult(0, 5, 10, False, (), None, "me"),                              # won by fold
        HandResult(1, -8, 16, True, (22, 0, 29, 7, 10), (4, 9, 16, 21, 27), "opp"),  # showdown
        HandResult(2, 0, 20, True, (22, 0, 29, 7, 10), (4, 9, 16, 21, 27), "split"),
        HandResult(99, 3, 6, False, (22, 0, 29), None, "me"),                      # hand we never saw
    ]
    for result in odd:
        module.on_hand_end(result, memory)
    assert memory.get("errors", 0) == 0, "on_hand_end swallowed an exception; it would be a real error"


def test_batched_results_are_delivered_with_one_act(bot):
    _, client = bot
    start_half(client)
    results = [HandResult(i, 2, 4, False, (), None, "me").to_dict() for i in range(5)]
    r = client.post("/act", json={"observation": make_obs().to_dict(), "hand_results": results}, headers=AUTH)
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []


def test_fuzz_many_seeds_never_errors(bot):
    _, client = bot
    start_half(client)
    for seed in range(5):
        play_hands(client, 40, seed=f"fuzz{seed}", raise_rate=0.2 * seed)


# --- opponent model ---------------------------------------------------------

def _hist(*items):
    return tuple(items)


def test_new_model_starts_empty():
    import strategy
    m = strategy.new_model()
    assert m["faced"] == 0 and m["fold_to_raise"] == 0 and m["results"] == []
    assert sum(m["opp_act"]) == 0


def test_model_counts_opponent_raises_per_street():
    import strategy
    from game import ObservedAction, Street
    m = strategy.new_model()
    history = _hist(
        ObservedAction(Street.PREFLOP, False, Action.RAISE_POT, 5, 6),
        ObservedAction(Street.PREFLOP, True, Action.CHECK_CALL, 4, 6),
        ObservedAction(Street.FLOP, False, Action.CHECK_CALL, 0, 0),
    )
    obs = make_obs(action_history=[a.to_dict() for a in history])
    strategy._process_history(m, obs)
    assert m["opp_act"][0] == 1 and m["opp_raise"][0] == 1 and m["opp_act"][1] == 1
    strategy._process_history(m, obs)     # same history again: nothing double counted
    assert m["opp_act"][0] == 1


def test_model_learns_fold_to_raise_and_rates_stay_in_range():
    import strategy
    m = strategy.new_model()
    P = strategy.DEFAULT_PARAMS
    base = strategy.fold_rate(m, P)
    for i in range(20):                       # we raise, opponent folds every time
        m["hand"] = i
        m["last_was_my_raise"] = True
        m["my_raise_street"] = 1
        strategy.on_result(m, HandResult(i, 4, 6, False, (22, 0, 29), None, "me"))
    assert m["faced"] == 20 and m["fold_to_raise"] == 20
    assert strategy.fold_rate(m, P) > base
    for street in range(4):
        assert 0.0 <= strategy.fold_rate(m, P, street) <= 1.0
        assert P["q_min"] <= strategy._raise_q(m, street, P) <= 1.0


def test_model_state_resets_between_halves():
    module = load_agent()
    memory = fresh_memory(module)
    module.choose_action(make_obs(), memory)
    module.on_hand_end(HandResult(0, 5, 10, False, (), None, "me"), memory)
    assert memory["model"]["results"]
    memory.clear()
    module.new_half(AgentConfig(), memory)
    assert memory["model"]["results"] == [] and memory["charged"] == 0.0


def test_shipped_raise_thresholds_are_raise_wider():
    # raise_wider passed all acceptance gates on 2026-11-07 (EXPERIMENTS.md); keep it pinned.
    import strategy
    assert strategy.DEFAULT_PARAMS["raise_eq"] == [0.53, 0.60, 0.64, 0.68]
