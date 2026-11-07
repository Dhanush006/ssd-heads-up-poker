"""Typical competitor: equity vs random hand + pot odds, value raises."""
from _common import A, eq_raw, pick


def new_half(config, memory): pass
def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    if e > 0.66:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return pick(obs, A.RAISE_HALF) if e > 0.58 else A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price else pick(obs, A.FOLD)
