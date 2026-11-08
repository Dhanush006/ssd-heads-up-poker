"""Check-raiser: slow-plays strong hands, raises when the opponent bets."""
from _common import A, eq_raw, pick


def new_half(config, memory): pass
def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    if obs.to_call == 0:
        if obs.street >= 1 and e > 0.62:
            return A.CHECK_CALL          # trap
        return pick(obs, A.RAISE_HALF) if e > 0.70 else A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    if e > 0.62:
        return pick(obs, A.RAISE_POT)    # the check-raise
    return A.CHECK_CALL if e >= price + 0.03 else pick(obs, A.FOLD)
