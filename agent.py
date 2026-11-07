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

import os
import random
import time
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, get_table, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "outliers"
AGENT_NAME = "outliers-bot"


# ===== YOUR CODE HERE =====
# Strategy v1: preflop hand-strength heuristic + Monte Carlo equity after the flop.

from itertools import combinations as _comb

_TABLE = get_table()
_PAIRS = tuple(_comb(range(5), 2))
_TRIPLES = tuple(_comb(range(5), 3))
_DECK = tuple(range(36))

TIME_SOFT_LIMIT = 5.0      # seconds of our own thinking per half before we cut down simulations
SIM_BUDGET = 0.0012        # seconds of simulation per decision (normal)
SIM_BUDGET_LOW = 0.0004    # seconds of simulation per decision once we are past the soft limit


def _best(hole_masks, board):
    """Best Omaha score for precomputed 2-card hole masks and a 5-card board."""
    b = [1 << c for c in board]
    bms = [b[i] | b[j] | b[k] for i, j, k in _TRIPLES]
    t = _TABLE
    return max(t[h | bm] for h in hole_masks for bm in bms)


def _hole_masks(hole):
    h = [1 << c for c in hole]
    return [h[i] | h[j] for i, j in _PAIRS]


def preflop_strength(hole) -> float:
    """Rough 0..~12 score of 5 hole cards. Higher is better."""
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    s = 0.0
    # pairs (count 2) are good, trips/quads waste cards
    for r in set(ranks):
        n = ranks.count(r)
        if n == 2:
            s += 1.5 + r / 4.0
        elif n >= 3:
            s += 0.5 + r / 8.0
    # suitedness: 2-3 of a suit is great, 4+ less so; ace/king of that suit is a bonus
    for su in set(suits):
        n = suits.count(su)
        if n >= 2:
            top = max(c >> 2 for c in hole if (c & 3) == su)
            s += (1.5 if n <= 3 else 0.7) + (0.8 if top == 8 else 0.4 if top == 7 else 0.0)
    # connectedness of distinct ranks
    d = sorted(set(ranks))
    for a, b in zip(d, d[1:]):
        gap = b - a
        if gap == 1:
            s += 0.8
        elif gap == 2:
            s += 0.4
    # high cards
    s += sum(ranks) / 10.0
    return s


def equity(obs: Observation, budget: float) -> float:
    """Monte Carlo win probability vs. a random opponent hand (ties count half)."""
    hole = obs.hole_cards
    board = list(obs.board)
    used = set(hole) | set(board)
    rest = [c for c in _DECK if c not in used]
    need = 5 - len(board)
    my_masks = _hole_masks(hole)
    rnd = random.random
    sample = random.sample
    deadline = time.perf_counter() + budget
    wins = 0.0
    n = 0
    fixed_mine = _best(my_masks, board) if need == 0 else None
    while True:
        cards = sample(rest, 5 + need)
        opp = cards[:5]
        full = board + cards[5:] if need else board
        mine = fixed_mine if fixed_mine is not None else _best(my_masks, full)
        theirs = _best(_hole_masks(opp), full)
        if mine > theirs:
            wins += 1.0
        elif mine == theirs:
            wins += 0.5
        n += 1
        if n >= 400 or (n >= 20 and time.perf_counter() > deadline):
            break
    return wins / n


def _raise(obs: Observation, prefer: Action) -> Action:
    if prefer in obs.legal_actions:
        return prefer
    for a in (Action.RAISE_POT, Action.RAISE_HALF):
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def _decide(obs: Observation, memory: dict) -> Action:
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal or Action.RAISE_HALF in legal
    to_call = obs.to_call
    pot_odds = to_call / (obs.pot + to_call) if to_call > 0 else 0.0

    if obs.street == Street.PREFLOP:
        s = preflop_strength(obs.hole_cards)
        if s >= 7.5 and can_raise and obs.raises_this_street < 3:
            return _raise(obs, Action.RAISE_POT)
        if s >= 5.5 and can_raise and obs.raises_this_street < 1:
            return _raise(obs, Action.RAISE_HALF)
        if to_call == 0:
            return Action.CHECK_CALL
        if s >= 5.0:
            return Action.CHECK_CALL
        if s >= 3.5 and to_call <= obs.big_blind:
            return Action.CHECK_CALL
        return Action.FOLD

    budget = SIM_BUDGET if memory.get("think", 0.0) < TIME_SOFT_LIMIT else SIM_BUDGET_LOW
    eq = equity(obs, budget)
    # equity is measured vs. a random hand; a raising opponent is usually stronger
    if to_call > 0 and obs.raises_this_street > 0:
        eq -= 0.05 * min(obs.raises_this_street, 3)

    if eq >= 0.80 and can_raise:
        return _raise(obs, Action.RAISE_POT)
    if eq >= 0.65 and can_raise and obs.raises_this_street < 2:
        return _raise(obs, Action.RAISE_HALF)
    if to_call == 0:
        return Action.CHECK_CALL
    if eq >= pot_odds + 0.05:
        return Action.CHECK_CALL
    return Action.FOLD


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["hands_played"] = 0
    memory["think"] = 0.0     # seconds spent inside choose_action this half


def choose_action(obs: Observation, memory: dict) -> Action:
    start = time.perf_counter()
    try:
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
    except Exception:
        action = Action.CHECK_CALL   # never crash: CHECK_CALL is always legal
    memory["think"] = memory.get("think", 0.0) + (time.perf_counter() - start)
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    memory["hands_played"] = memory.get("hands_played", 0) + 1

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
