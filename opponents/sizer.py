"""Sizer: bet-size mixer, pot raises with big equity, half-pot raises with medium equity."""
from _common import A, eq_raw, pick


def new_half(config, memory): pass
def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    if obs.to_call == 0:
        if e > 0.75:
            return pick(obs, A.RAISE_POT)
        if e > 0.60:
            return pick(obs, A.RAISE_HALF)
        return A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price + 0.02 else pick(obs, A.FOLD)
