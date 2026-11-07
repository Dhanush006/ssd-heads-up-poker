"""Solid equity bot that bluffs the river hard with its weakest hands."""
from _common import A, eq_raw, pick


def new_half(config, memory):
    import random
    memory["r"] = random.Random(13)


def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    r = memory["r"].random()
    if obs.street == 3 and e < 0.35 and r < 0.55:
        return pick(obs, A.RAISE_POT)
    if e > 0.68:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return pick(obs, A.RAISE_HALF) if e > 0.57 else A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price else pick(obs, A.FOLD)
