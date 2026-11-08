"""Adaptive caller: tracks the opponent's raise frequency and calls wider against aggression, tighter against passivity."""
from _common import A, eq_raw, pick


def new_half(config, memory):
    memory.update(opp_raises=0, opp_acts=0, seen=0, hand=-1, agg=2 / 6)


def on_hand_end(result, memory): pass


def _update(obs, memory):
    if memory.get("hand") != obs.hand_index:
        memory["hand"] = obs.hand_index
        memory["seen"] = 0
    hist = obs.action_history
    for item in list(hist)[memory["seen"]:]:
        if not item.is_me:
            memory["opp_acts"] += 1
            if item.action in (A.RAISE_HALF, A.RAISE_POT):
                memory["opp_raises"] += 1
    memory["seen"] = len(hist)
    memory["agg"] = (memory["opp_raises"] + 2) / (memory["opp_acts"] + 6)


def choose_action(obs, memory):
    if "agg" not in memory:
        new_half(None, memory)
    _update(obs, memory)
    e = eq_raw(obs)
    margin = 0.08 - 0.20 * (memory["agg"] - 0.33)
    if e > 0.70:
        return pick(obs, A.RAISE_POT)
    if obs.to_call == 0:
        return pick(obs, A.RAISE_HALF) if e > 0.62 else A.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return A.CHECK_CALL if e >= price + margin else pick(obs, A.FOLD)
