"""Trapper: never raises preflop, flat-calls the flop with strong hands, springs the raise on turn and river."""
from _common import A, eq_raw, pick


def new_half(config, memory): pass
def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    if obs.street == 0:
        if obs.to_call <= 6:
            return A.CHECK_CALL if (obs.to_call == 0 or e > 0.40) else pick(obs, A.FOLD)
        return pick(obs, A.FOLD)
    if obs.street == 1 and e > 0.55:
        return A.CHECK_CALL
    if obs.street >= 2 and e > 0.68:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price else pick(obs, A.FOLD)
