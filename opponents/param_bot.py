"""Parameterised random opponent for generating training data (never a hold-out bot).

Cannot be loaded via selfplay.load(path). Build it inside the worker process as
selfplay.Handle(param_bot.make(p, seed)); the SimpleNamespace of closures is not
picklable, so never pass it to a multiprocessing Pool.
"""
import random
from types import SimpleNamespace

from _common import A, eq_raw, pick


def make(p, seed=0):
    """Return a module-like object (new_half/choose_action/on_hand_end) with behaviour set by p.

    p keys: raise_eq, bluff, call_margin, pf_floor.
    """
    def new_half(config, memory):
        memory["r"] = random.Random(seed)

    def on_hand_end(result, memory):
        pass

    def choose_action(obs, memory):
        e = eq_raw(obs, 150)
        r = memory["r"].random()
        if e > p["raise_eq"]:
            return pick(obs, A.RAISE_POT)
        if obs.street >= 1 and r < p["bluff"]:
            return pick(obs, A.RAISE_HALF)
        if obs.to_call == 0:
            return A.CHECK_CALL
        if obs.street == 0 and e < p["pf_floor"]:
            return pick(obs, A.FOLD)
        price = obs.to_call / (obs.pot + obs.to_call)
        return A.CHECK_CALL if e >= price + p["call_margin"] else pick(obs, A.FOLD)

    return SimpleNamespace(new_half=new_half, choose_action=choose_action, on_hand_end=on_hand_end)


def sample(rng):
    """Draw one random opponent parameter set from a numpy Generator."""
    return {
        "raise_eq": float(rng.uniform(0.55, 0.85)),
        "bluff": float(rng.uniform(0.0, 0.40)),
        "call_margin": float(rng.uniform(-0.10, 0.20)),
        "pf_floor": float(rng.uniform(0.30, 0.55)),
    }
