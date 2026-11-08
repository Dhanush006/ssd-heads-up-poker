"""Nut nit: only raises or calls bets with near-nut hands, never bluffs."""
from _common import A, eq_raw, pick


def new_half(config, memory): pass
def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    if e > 0.85:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return A.CHECK_CALL
    if e > 0.78 or (obs.street == 0 and obs.to_call <= 2 and e > 0.50):
        return A.CHECK_CALL
    return pick(obs, A.FOLD)
