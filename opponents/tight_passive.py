"""Nit: plays only strong hands, rarely raises, folds to pressure."""
from _common import A, eq_raw, pick


def new_half(config, memory): pass
def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    if e > 0.78:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    if e >= price + 0.15 or (obs.street == 0 and e > 0.52 and obs.to_call <= 4):
        return A.CHECK_CALL
    return pick(obs, A.FOLD)
