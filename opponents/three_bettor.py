"""3-bettor: re-raises preflop raises with decent hands, otherwise straightforward."""
from _common import A, eq_raw, pick


def new_half(config, memory): pass
def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    if obs.street == 0 and obs.raises_this_street >= 1 and e > 0.45:
        return pick(obs, A.RAISE_POT)
    if e > 0.68:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price + 0.02 else pick(obs, A.FOLD)
