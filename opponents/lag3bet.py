"""Pot-odds-correct loose-aggressive bot: 3-bets light preflop, bets often, semi-bluffs."""
from _common import A, eq_raw, pick


def new_half(config, memory):
    import random
    memory["r"] = random.Random(11)


def on_hand_end(result, memory): pass


def choose_action(obs, memory):
    e = eq_raw(obs)
    r = memory["r"].random()
    pre = obs.street == 0
    if obs.to_call == 0:
        if e > 0.55 or (e > 0.30 and r < 0.35) or (pre and e > 0.45):
            return pick(obs, A.RAISE_POT if e > 0.6 else A.RAISE_HALF)
        return A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    if (pre and e > 0.50 and obs.raises_this_street < 3) or e > 0.64 or (r < 0.12 and e > 0.35):
        return pick(obs, A.RAISE_POT)
    return A.CHECK_CALL if e >= price else pick(obs, A.FOLD)
