"""Range-equity bot: re-evaluates its hand against a narrowed range after the opponent raises."""
import numpy as np
from _common import A, eq_raw, pick
import strategy


def new_half(config, memory): pass
def on_hand_end(result, memory): pass


def _eq(obs):
    opp = [h for h in obs.action_history if not h.is_me and h.action in (A.RAISE_HALF, A.RAISE_POT)]
    if opp:
        q_pre = 0.4 if any(h.street == 0 for h in opp) else 1.0
        q_post = 0.4 if any(h.street >= 1 for h in opp) else 1.0
        rng = np.random.default_rng(obs.hand_index * 7 + len(obs.action_history))
        return strategy.equity(list(obs.hole_cards), list(obs.board), 300, q_pre, q_post, rng)[0]
    if len(obs.board) == 0:
        return strategy.preflop_equity(obs.hole_cards)
    return eq_raw(obs)


def choose_action(obs, memory):
    e = _eq(obs)
    if e > 0.66:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return pick(obs, A.RAISE_HALF) if e > 0.58 else A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price else pick(obs, A.FOLD)
