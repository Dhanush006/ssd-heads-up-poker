"""River bluffer: straightforward early, bluffs the river 40% of the time when checked to with air."""
import random
from _common import A, eq_raw, pick


def new_half(config, memory):
    memory["rng"] = random.Random(21)


def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if obs.street == 3:
        if obs.to_call == 0:
            if e > 0.70:
                return pick(obs, A.RAISE_POT)
            rng = memory.setdefault("rng", random.Random(21))
            if e < 0.35 and rng.random() < 0.40:
                return pick(obs, A.RAISE_POT)
            return A.CHECK_CALL
        return A.CHECK_CALL if e >= price + 0.05 else pick(obs, A.FOLD)
    if e > 0.70:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return A.CHECK_CALL
    return A.CHECK_CALL if e >= price else pick(obs, A.FOLD)
