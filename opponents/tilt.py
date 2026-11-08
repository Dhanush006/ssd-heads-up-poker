"""Style-switcher: alternates loose-aggressive and tight-passive every 150 hands."""
import random
from _common import A, eq_raw, pick


def new_half(config, memory):
    memory["r"] = random.Random(7)


def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    if (obs.hand_index // 150) % 2 == 0:             # loose-aggressive
        if e > 0.55 or memory["r"].random() < 0.25:
            return pick(obs, A.RAISE_POT if e > 0.62 else A.RAISE_HALF)
        if obs.to_call == 0:
            return A.CHECK_CALL
        price = obs.to_call / (obs.pot + obs.to_call)
        return A.CHECK_CALL if e >= price - 0.08 else pick(obs, A.FOLD)
    if e > 0.78:                                      # tight-passive
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price + 0.15 else pick(obs, A.FOLD)
