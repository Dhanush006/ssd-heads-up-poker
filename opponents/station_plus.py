"""Calling station that value-raises only very strong hands; never folds unless hopeless."""
from _common import A, eq_raw, pick


def new_half(config, memory): pass
def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    if e > 0.75:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0 or e > 0.2:
        return A.CHECK_CALL
    return pick(obs, A.FOLD)
