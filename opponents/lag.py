"""Loose-aggressive: raises a lot with medium hands, bluffs, calls wide."""
from _common import A, eq_raw, pick


def new_half(config, memory):
    import random
    memory["r"] = random.Random(5)


def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    r = memory["r"].random()
    if e > 0.55 or r < 0.25:
        return pick(obs, A.RAISE_POT if e > 0.62 else A.RAISE_HALF)
    if obs.to_call == 0:
        return A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price - 0.08 else pick(obs, A.FOLD)
