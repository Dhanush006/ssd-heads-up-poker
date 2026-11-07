"""Shared helpers for local practice opponents (harness only, not shipped logic)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import strategy
from game import Action

A = Action


def eq_raw(obs, n=400):
    rng = np.random.default_rng(obs.hand_index * 31 + len(obs.action_history))
    if len(obs.board) == 0:
        return strategy.preflop_equity(obs.hole_cards)
    return strategy.equity(list(obs.hole_cards), list(obs.board), n, 1.0, 1.0, rng)[1]


def pick(obs, want):
    legal = obs.legal_actions
    if want in legal:
        return want
    if want in (A.RAISE_POT, A.RAISE_HALF):
        return A.CHECK_CALL
    if want == A.FOLD and A.FOLD not in legal:
        return A.CHECK_CALL
    return A.CHECK_CALL
