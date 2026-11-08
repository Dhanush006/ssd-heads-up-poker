"""Mixed-strategy nit: tight, but randomises thresholds and bluffs 15% when checked to."""
import random
from _common import A, eq_raw, pick


def new_half(config, memory):
    memory["r"] = random.Random(11)


def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    r = memory["r"]
    e = eq_raw(obs)
    cut = 0.72 + r.uniform(-0.06, 0.06)
    if e > cut:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return pick(obs, A.RAISE_HALF) if (obs.street >= 1 and r.random() < 0.15) else A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price + 0.10 + r.uniform(-0.05, 0.05) else pick(obs, A.FOLD)
