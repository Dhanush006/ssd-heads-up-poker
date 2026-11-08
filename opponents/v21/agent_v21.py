"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "SSD"
AGENT_NAME = "ShortStackSage"


# ===== YOUR CODE HERE =====
import time as _time

try:  # numpy-based strategy; if anything here fails we still play safely
    import strategy_v21 as _strategy
except Exception:  # pragma: no cover
    _strategy = None

# Per-half think-time budget (judge's bank is 8 s; plumbing also costs time).
_SOFT_BUDGET = 3.5   # target charged seconds per half (spread over the hands)
_HARD_BUDGET = 6.0   # above this, cheap rules only for the rest of the half
_OVERHEAD = 0.0006   # est. per-request cost outside choose_action (JSON, HTTP), scaled by machine speed
_N_BASE = 700        # Monte Carlo samples per decision when time allows
_N_MIN = 60
_PARAMS = None       # strategy parameter override (None = strategy.DEFAULT_PARAMS)


def _safe_action(obs) -> Action:
    """A guaranteed-legal action: check if free, else call (never auto-fold)."""
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    return obs.legal_actions[0]


def _fallback(obs) -> Action:
    """Cheap rule set used if the strategy is unavailable or out of time."""
    legal = obs.legal_actions
    if obs.street == Street.RIVER and len(obs.board) == 5:
        cat = omaha_score(obs.hole_cards, obs.board) >> 20
        if cat >= 5 and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if cat >= 3 or obs.to_call == 0:
            return Action.CHECK_CALL
        return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
    if obs.to_call <= 2 * obs.big_blind or obs.to_call * 4 <= obs.pot:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL


def new_half(config: AgentConfig, memory: dict) -> None:
    """Start of a half: fresh opponent model."""
    try:
        memory["model"] = _strategy.new_model() if _strategy is not None else {}
    except Exception:
        memory["model"] = {}
    memory["think"] = 0.0
    memory["charged"] = 0.0
    memory["n"] = _N_BASE
    memory["slow"] = 1.0
    memory["errors"] = 0


def choose_action(obs: Observation, memory: dict) -> Action:
    """Robust wrapper: strategy -> legality check -> safe fallback.

    Time guard: we estimate what the judge charges (request time minus the
    free 1 ms) and shrink the Monte Carlo sample count whenever we are
    spending faster than _SOFT_BUDGET spread evenly over the half.
    """
    t0 = _time.perf_counter()
    action = None
    try:
        model = memory.get("model")
        charged = memory.get("charged", 0.0)
        if _strategy is not None and isinstance(model, dict) and model and charged < _HARD_BUDGET:
            n = memory.get("n", _N_BASE)
            allowed = _SOFT_BUDGET * min(1.0, (obs.hand_index + 1) / 1000.0) + 0.3
            if charged > allowed:
                n = max(_N_MIN, int(n * 0.7))
            elif charged < 0.6 * allowed and n < _N_BASE:
                n = min(_N_BASE, int(n * 1.15) + 1)
            memory["n"] = n
            a = _strategy.decide(obs, model, _PARAMS, n)
            action = Action(int(a))
        else:
            action = _fallback(obs)
    except Exception:
        memory["errors"] = memory.get("errors", 0) + 1
        try:
            action = _fallback(obs)
        except Exception:
            action = None
    if action is None or action not in obs.legal_actions:
        action = _safe_action(obs)
    dt = _time.perf_counter() - t0
    memory["think"] = memory.get("think", 0.0) + dt
    speed = getattr(_strategy, "SPEED", 2.0) if _strategy is not None else 2.0
    # live slowdown (CPU contention): actual vs expected decision time at this n
    n_used = memory.get("n", _N_BASE)
    expected = (0.00012 + 0.0000013 * n_used) * speed
    slow = 0.9 * memory.get("slow", 1.0) + 0.1 * min(10.0, max(0.5, dt / expected))
    memory["slow"] = slow
    memory["charged"] = memory.get("charged", 0.0) + max(0.0, dt + _OVERHEAD * speed * max(1.0, slow) - 0.001)
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Feed the opponent model; never raise."""
    try:
        model = memory.get("model")
        if _strategy is not None and isinstance(model, dict) and model:
            _strategy.on_result(model, result)
    except Exception:
        memory["errors"] = memory.get("errors", 0) + 1

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
