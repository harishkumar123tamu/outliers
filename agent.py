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

import json
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
# Strategy v2:
#   * preflop: precomputed equity table (preflop_equity.json) for all 19,998 hand types
#   * postflop: Monte Carlo equity against the opponent's likely RANGE, built from how
#     they bet this hand and what they showed down earlier (rejection sampling)
#   * opponent model per half: aggression, fold-to-bet, bluff rate -> adjusts calling,
#     value-betting and bluffing

from itertools import combinations as _comb

_TABLE = get_table()
_PAIRS = tuple(_comb(range(5), 2))
_TRIPLES = tuple(_comb(range(5), 3))
_DECK = tuple(range(36))
_RCH = "6789TJQKA"
_SHIFT = 20  # score >> 20 = hand category (0 high card .. 8 straight flush)

_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(_HERE, "preflop_equity.json")) as _f:
        _PREFLOP = json.load(_f)
except Exception:
    _PREFLOP = {}

P = {
    "time_target": 3.0,      # seconds of thinking per half we plan to use (bank is 8 s)
    "time_hard": 5.0,
    "overhead": 0.00012,
    "budget": 0.0015,        # max seconds of simulation per decision
    "budget_min": 0.0002,
    # preflop equity thresholds (vs a random hand)
    "pf_open_pot": 0.56,
    "pf_open_half": 0.47,
    "pf_limp": 0.47,
    "pf_3bet": 0.60,
    "pf_4bet": 0.635,
    "pf_call_margin": 0.0,
    # postflop
    "val_pot": 0.72,
    "val_half": 0.60,
    "raise_vs_bet": 0.70,
    "cm1": 0.08,
    "cm2": 0.06,
    "cm3": 0.03,
    "range_on": 1.0,
    "prior_w": 6.0,          # pseudo-count weight of the postflop range priors
    "prior_w_pf": 8.0,
    "prev_exp": 0.3,         # evidence from earlier streets counts less
    "ref_n": 24,
    "learn_ref_n": 25,
    "use_np": 1.0,
    "rr_on": 1.0,
    "cm_rr": 0.0,
    "ftb_fix": 0.0,
    "sb_e": 0.30,
    "sb_pot": 0.0,
    "rb_e": 0.25,
    "rb_k": 4.0,
    "trap_th": 0.55,
    "trap_k": 0.0,
    "bc_k": 0.0,
    "cr_k": 0.0,
    "learn_ref_np": 200,
    "np_fixed": 0.00015,
    "np_min": 60,
    "np_max": 3000,
    "bluff_freq": 1.0,
}


# ---------- card helpers ----------

def _hole_masks(hole):
    h = [1 << c for c in hole]
    return [h[i] | h[j] for i, j in _PAIRS]


def _best(hole_masks, board):
    """Best Omaha score with a full 5-card board."""
    b = [1 << c for c in board]
    t = _TABLE
    bms = [b[i] | b[j] | b[k] for i, j, k in _TRIPLES]
    return max(t[h | bm] for h in hole_masks for bm in bms)


_TRIPLES_BY_LEN = {n: tuple(_comb(range(n), 3)) for n in (3, 4, 5)}


def _made(hole_masks, board):
    """Best Omaha score on a partial board (3 or 4 cards): 2 hole + 3 board."""
    b = [1 << c for c in board]
    t = _TABLE
    bms = [b[i] | b[j] | b[k] for i, j, k in _TRIPLES_BY_LEN[len(board)]]
    return max(t[h | bm] for h in hole_masks for bm in bms)


def preflop_key(hole) -> str:
    g = {}
    for c in hole:
        g.setdefault(c & 3, []).append(c >> 2)
    groups = sorted((tuple(sorted(v, reverse=True)) for v in g.values()),
                    key=lambda t: (-len(t), [-x for x in t]))
    return "|".join("".join(_RCH[r] for r in grp) for grp in groups)


def preflop_equity(hole) -> float:
    e = _PREFLOP.get(preflop_key(hole))
    if e is not None:
        return e
    # fallback (table missing): crude heuristic mapped to ~0.4..0.65
    ranks = [c >> 2 for c in hole]
    return 0.40 + sum(ranks) / 160.0


# ---------- preflop percentile of every 5-card hand ----------

_NB_PF = 10   # preflop strength bins (deciles vs random hands)
_NB = 5       # postflop strength bins (quintiles vs random hands on this board)
_FD_SCORE = 2 << _SHIFT   # a flush draw counts like a weak two pair


try:
    import numpy as _np
except Exception:          # numpy missing: fall back to the pure-Python simulator
    _np = None

_BINOM = [[0] * 37 for _ in range(6)]
for _i in range(6):
    for _c in range(37):
        if _i == 0:
            _BINOM[_i][_c] = 1
        elif _c >= _i:
            from math import comb as _mc
            _BINOM[_i][_c] = _mc(_c, _i)


def _colex(cards):
    c = sorted(cards)
    return _BINOM[1][c[0]] + _BINOM[2][c[1]] + _BINOM[3][c[2]] + _BINOM[4][c[3]] + _BINOM[5][c[4]]


def _build_tables():
    """Preflop decile of every 5-card hand (by colex index) and, with numpy,
    rank-key score tables for fast vectorised Omaha evaluation."""
    from bisect import bisect_right
    combos = list(_comb(range(36), 5))
    eqs = []
    if _PREFLOP:
        cache = {}
        for c in combos:
            k = preflop_key(c)
            e = cache.get(k)
            if e is None:
                e = cache[k] = _PREFLOP.get(k, 0.5)
            eqs.append(e)
        vals = sorted(eqs)
        n = len(vals)
        cuts = [vals[n * k // _NB_PF] for k in range(1, _NB_PF)]
        bmap = {e: bisect_right(cuts, e) for e in set(vals)}
        bins = [bmap[e] for e in eqs]
    else:
        bins = None
    pfbin = None
    nf = fl = None
    if _np is not None:
        arr = _np.array(combos, dtype=_np.int64)
        bt = _np.array(_BINOM, dtype=_np.int64)
        idx = bt[1][arr[:, 0]] + bt[2][arr[:, 1]] + bt[3][arr[:, 2]] + bt[4][arr[:, 3]] + bt[5][arr[:, 4]]
        if bins is not None:
            pfbin = _np.zeros(len(combos), dtype=_np.int8)
            pfbin[idx] = _np.array(bins, dtype=_np.int8)
        masks = (_np.int64(1) << arr).sum(axis=1)
        scores = _np.array([_TABLE[m] for m in masks.tolist()], dtype=_np.int32)
        r5 = _np.array([5 ** (c >> 2) for c in range(36)], dtype=_np.int64)
        keys = r5[arr].sum(axis=1)
        su = arr & 3
        flush = (su == su[:, :1]).all(axis=1)
        size = 5 * 5 ** 8 + 1
        nf = _np.zeros(size, dtype=_np.int32)
        fl = _np.zeros(size, dtype=_np.int32)
        nf[keys[~flush]] = scores[~flush]
        fl[keys[flush]] = scores[flush]
        pfd = None
    else:
        pfd = None
        if bins is not None:
            pfd = {}
            for c, b in zip(combos, bins):
                pfd[_colex(c)] = b
    return pfbin, pfd, nf, fl


try:
    _PF_ARR, _PF_DICT, _NF, _FL = _build_tables()
except Exception:
    _PF_ARR = _PF_DICT = _NF = _FL = None
    _np = None
_PF_OK = _PF_ARR is not None or bool(_PF_DICT)


def _pf_bin(cards):
    i = _colex(cards)
    if _PF_ARR is not None:
        return int(_PF_ARR[i])
    if _PF_DICT:
        return _PF_DICT.get(i)
    return None


if _np is not None:
    _R5 = _np.array([5 ** (c >> 2) for c in range(36)], dtype=_np.int32)
    _SU = _np.array([c & 3 for c in range(36)], dtype=_np.int8)
    _PI = _np.array([p[0] for p in _PAIRS]); _PJ = _np.array([p[1] for p in _PAIRS])
    _BT = _np.array(_BINOM, dtype=_np.int64)
    _RNG = _np.random.default_rng()
    _TRI_NP = {n: tuple(_np.array([t[k] for t in _TRIPLES_BY_LEN[n]]) for k in range(3)) for n in (3, 4, 5)}


def _np_pairs(h):
    """(..., 5) hole cards -> rank keys and suits (-1 = mixed) of the 10 two-card subsets."""
    a, b = h[..., _PI], h[..., _PJ]
    sa, sb = _SU[a], _SU[b]
    return _R5[a] + _R5[b], _np.where(sa == sb, sa, -1)


def _np_triples(bd):
    """(..., n) board -> rank keys and suits (-2 = mixed) of its three-card subsets."""
    ti, tj, tk = _TRI_NP[bd.shape[-1]]
    a, b, c = bd[..., ti], bd[..., tj], bd[..., tk]
    sa, sb, sc = _SU[a], _SU[b], _SU[c]
    return _R5[a] + _R5[b] + _R5[c], _np.where((sa == sb) & (sb == sc), sa, -2)


def _np_best(pk, ps, tk, ts):
    key = pk[..., :, None] + tk[..., None, :]
    sc = _np.where(ps[..., :, None] == ts[..., None, :], _FL[key], _NF[key])
    return sc.reshape(sc.shape[:-2] + (-1,)).max(-1)


def _np_strength(pk, ps, hole, board):
    """Opponent strength on a partial or full board (vectorised _strength)."""
    b = _np.array(board, dtype=_np.int64)
    tk, ts = _np_triples(b)
    s = _np_best(pk, ps, tk, ts)
    fds = _flush_draw_suits(board)
    if fds:
        hs = _SU[hole]
        has = _np.zeros(s.shape, dtype=bool)
        for su in fds:
            has |= (hs == su).sum(axis=-1) >= 2
        s = _np.where(has & (s < _FD_SCORE), _FD_SCORE, s)
    return s


def _np_pct_bins(s):
    srt = _np.sort(s)
    n = len(s)
    p = (_np.searchsorted(srt, s, "left") + _np.searchsorted(srt, s, "right")) / (2.0 * n)
    return _np.minimum((p * _NB).astype(_np.int64), _NB - 1)


def _np_pf_bins(hole):
    h = _np.sort(hole, axis=1)
    idx = _BT[1][h[:, 0]] + _BT[2][h[:, 1]] + _BT[3][h[:, 2]] + _BT[4][h[:, 3]] + _BT[5][h[:, 4]]
    return _PF_ARR[idx]


def _np_sample(rest, n, k):
    r = _np.array(rest, dtype=_np.int64)
    idx = _RNG.random((n, len(r))).argpartition(k, axis=1)[:, :k]
    return r[idx]


def equity_np(obs, memory, budget):
    hole = list(obs.hole_cards)
    board = list(obs.board)
    used = set(hole) | set(board)
    rest = [c for c in _DECK if c not in used]
    need = 5 - len(board)
    per = memory.get("np_per", 2.5e-6)
    n = int((budget - P["np_fixed"]) / per)
    n = max(int(P["np_min"]), min(int(P["np_max"]), n))
    t0 = time.perf_counter()
    C = _np_sample(rest, n, 5 + need)
    opp = C[:, :5]
    if need:
        full = _np.concatenate([_np.broadcast_to(_np.array(board, dtype=_np.int64), (n, len(board))), C[:, 5:]], axis=1)
    else:
        full = _np.array(board, dtype=_np.int64)
    tk, ts = _np_triples(full)
    pk, ps = _np_pairs(opp)
    mk, ms = _np_pairs(_np.array(hole, dtype=_np.int64))
    theirs = _np_best(pk, ps, tk, ts)
    mine = _np_best(mk, ms, tk, ts)
    res = (mine > theirs) + 0.5 * (mine == theirs)
    w = None
    if P["range_on"] > 0:
        pf_t, post_t = _range_tables(obs, memory)
        if pf_t is not None and _PF_ARR is not None:
            w = _np.array(pf_t)[_np_pf_bins(opp)]
        if post_t is not None:
            s = theirs if need == 0 else _np_strength(pk, ps, opp, board)
            pw = _np.array(post_t)[_np_pct_bins(s)]
            w = pw if w is None else w * pw
    if w is None:
        e = float(res.mean())
    else:
        tot = float(w.sum())
        e = float((w * res).sum() / tot) if tot > 0 else float(res.mean())
    dt = time.perf_counter() - t0
    memory["np_per"] = 0.9 * per + 0.1 * max(5e-7, (dt - P["np_fixed"]) / n)
    return e


# ---------- opponent model ----------

# Prior shapes (relative weights per strength bin, weakest first) for the hands an
# opponent shows down after each kind of action. Learned per half from showdowns.
_PRIOR = {
    "pfr": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    "pfc": [0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.05, 1.05, 1.1],
    "bet": [0.55, 0.6, 0.8, 1.0, 1.2],
    "call": [0.7, 0.85, 1.0, 1.05, 1.1],
    "check": [1.2, 1.15, 1.0, 0.85, 0.7],
    "raise": [0.3, 0.4, 0.6, 1.0, 1.7],
}
_TYPES = {3: "raise", 2: "bet", 1: "call", 0: "check"}


def _opp(memory):
    return memory["opp"]


def _rate(num, den, prior, weight):
    return (num + prior * weight) / (den + weight)


def opp_stats(memory):
    o = _opp(memory)
    return {
        "pf_raise": _rate(o["pf_raise"], o["pf_dec"], 0.35, 20),
        "agg": _rate(o["post_raise"], o["post_dec"], 0.35, 20),
        "fold_to_bet": _rate(o["fold_to_bet"], o["faced_bet"], 0.40, 15),
        "bet_free": _rate(o.get("free_bet", 0), o.get("free_dec", 0), 0.40, 20),
    }


def _new_hists():
    out = {}
    for k, v in _PRIOR.items():
        tot = sum(v)
        w = P["prior_w_pf"] if k.startswith("pf") else P["prior_w"]
        out[k] = [x * w / tot for x in v]
    return out


def _acc_table(h):
    m = max(h)
    return [x / m for x in h]


def _track_actions(obs: Observation, memory: dict):
    """Count opponent actions we have not seen yet in this hand and keep the
    opponent's line: per street 2 = bet/raise, 1 = call, 0 = check."""
    o = _opp(memory)
    if memory.get("cur_hand") != obs.hand_index:
        memory["cur_hand"] = obs.hand_index
        memory["seen"] = 0
        memory["hand_aggr"] = 0      # opponent bets/raises after the flop this hand
        memory["last_mine"] = None
        memory["line"] = {}
        memory["hole"] = tuple(obs.hole_cards)
    line = memory["line"]
    hist = obs.action_history
    for idx in range(memory["seen"], len(hist)):
        a = hist[idx]
        if a.is_me:
            continue
        raised = a.action in (Action.RAISE_HALF, Action.RAISE_POT)
        lv = 2 if raised else (1 if (a.action == Action.CHECK_CALL and a.amount > 0) else 0)
        if a.street != Street.PREFLOP:
            if P["ftb_fix"] > 0 and idx > 0:
                pv = hist[idx - 1]
                if pv.is_me and pv.street == a.street and pv.action in (Action.RAISE_HALF, Action.RAISE_POT):
                    o["faced_bet"] += 1      # they called or raised our bet (a fold is counted at hand end)
            prior = False
            for h in hist[:idx]:
                if h.street == a.street and h.action in (Action.RAISE_HALF, Action.RAISE_POT):
                    prior = True
                    break
            if raised and prior and P["rr_on"] > 0:
                lv = 3      # a raise over a bet: much stronger range than a bet
            if not prior:
                o["free_dec"] += 1
                o["free_bet"] += raised
        st = int(a.street)
        if a.action != Action.FOLD:
            line[st] = max(line.get(st, 0), lv)
        if a.street == Street.PREFLOP:
            o["pf_dec"] += 1
            o["pf_raise"] += raised
        else:
            o["post_dec"] += 1
            o["post_raise"] += raised
            if raised:
                memory["hand_aggr"] += 1
    memory["seen"] = len(hist)


def _strength(om, cards5, board, fds):
    """Opponent hand strength on a partial or full board; flush draws count as
    weak two pair on the flop and turn."""
    s = _made(om, board) if len(board) < 5 else _best(om, board)
    if fds and s < _FD_SCORE:
        for su in fds:
            k = 0
            for c in cards5:
                if c & 3 == su:
                    k += 1
            if k >= 2:
                return _FD_SCORE
    return s


def _flush_draw_suits(board):
    if len(board) >= 5:
        return ()
    cnt = [0, 0, 0, 0]
    for c in board:
        cnt[c & 3] += 1
    return tuple(su for su in range(4) if cnt[su] == 2)


def _pct_bin(ref, s, nb):
    from bisect import bisect_left, bisect_right
    k = len(ref)
    p = (bisect_left(ref, s) + bisect_right(ref, s)) / (2.0 * k)
    b = int(p * nb)
    return b if b < nb else nb - 1


# ---------- equity vs. range ----------

def _range_tables(obs: Observation, memory: dict):
    """Acceptance tables (per strength bin) from the opponent's line this hand."""
    hists = memory["hists"]
    line = memory.get("line", {})
    pf = None
    lv = line.get(0)
    if lv == 2:
        pf = _acc_table(hists["pfr"])
    elif lv == 1:
        pf = _acc_table(hists["pfc"])
    post = None
    cur = int(obs.street)
    for st in range(1, cur + 1):
        if st not in line:
            continue
        t = _acc_table(hists[_TYPES[line[st]]])
        if st < cur:
            t = [x ** P["prev_exp"] for x in t]
        post = t if post is None else [a * b for a, b in zip(post, t)]
    return pf, post


def equity_vs_range(obs: Observation, memory: dict, budget: float) -> float:
    if _np is not None and P["use_np"] > 0:
        return equity_np(obs, memory, budget)
    hole = obs.hole_cards
    board = list(obs.board)
    used = set(hole) | set(board)
    rest = [c for c in _DECK if c not in used]
    need = 5 - len(board)
    mine_m = _hole_masks(hole)
    if P["range_on"] > 0:
        pf_t, post_t = _range_tables(obs, memory)
    else:
        pf_t = post_t = None
    if not _PF_OK:
        pf_t = None
    fds = _flush_draw_suits(board)
    rnd = random.random
    sample = random.sample
    deadline = time.perf_counter() + budget
    fixed_mine = _best(mine_m, board) if need == 0 else None
    min_n = 8 if budget <= 0 else (25 if budget >= 0.0006 else 14)
    # reference strengths (random opponent hands) for percentiles on this board
    ref = None
    queue = []
    if post_t is not None:
        for _ in range(int(P["ref_n"])):
            cards = sample(rest, 5 + need)
            om = _hole_masks(cards[:5])
            queue.append((cards, om, _strength(om, cards[:5], board, fds)))
        ref = sorted(q[2] for q in queue)
    wins = 0.0
    n = 0
    tries = 0
    qi = 0
    hard = deadline + budget
    force = False
    while True:
        tries += 1
        if qi < len(queue):
            cards, om, s = queue[qi]
            qi += 1
            if not force:
                w = 1.0
                if pf_t is not None:
                    w = pf_t[_pf_bin(cards[:5])]
                w *= post_t[_pct_bin(ref, s, _NB)]
                if w < 1.0 and rnd() > w:
                    continue
        else:
            cards = sample(rest, 5 + need)
            om = _hole_masks(cards[:5])
            s = None
            if not force:
                if pf_t is not None:
                    w = pf_t[_pf_bin(cards[:5])]
                    if w < 1.0 and rnd() > w:
                        if tries & 31 == 0 and time.perf_counter() > hard:
                            force = True
                        continue
                if ref is not None:
                    s = _strength(om, cards[:5], board, fds)
                    w = post_t[_pct_bin(ref, s, _NB)]
                    if w < 1.0 and rnd() > w:
                        if tries & 31 == 0 and time.perf_counter() > hard:
                            force = True
                        continue
            if tries > 3000:
                force = True
        if need:
            full = board + cards[5:]
            mine = _best(mine_m, full)
            theirs = _best(om, full)
        else:
            mine = fixed_mine
            theirs = s if s is not None else _best(om, board)
        if mine > theirs:
            wins += 1.0
        elif mine == theirs:
            wins += 0.5
        n += 1
        if n >= 500 or (n >= min_n and time.perf_counter() > deadline):
            break
    return wins / n


# ---------- decisions ----------

def _pick(obs, *prefs):
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def _preflop(obs: Observation, memory: dict) -> Action:
    e = preflop_equity(obs.hole_cards)
    st = opp_stats(memory)
    to_call = obs.to_call
    r = obs.raises_this_street
    can_raise = Action.RAISE_POT in obs.legal_actions
    # a wide-raising opponent has a weaker range: loosen up against them
    loosen = max(-0.02, min(0.04, (st["pf_raise"] - 0.35) * 0.12))
    if r == 0:
        if to_call > 0:          # we are the button, first to act
            if e >= P["pf_open_pot"] and can_raise:
                return Action.RAISE_POT
            if e >= P["pf_open_half"] and can_raise:
                return _pick(obs, Action.RAISE_HALF, Action.RAISE_POT)
            if e >= P["pf_limp"] - loosen:
                return Action.CHECK_CALL
            return Action.FOLD
        # big blind, opponent limped
        if e >= P["pf_open_half"] + 0.01 and can_raise:
            return _pick(obs, Action.RAISE_POT)
        return Action.CHECK_CALL
    # facing a raise
    pot_odds = to_call / (obs.pot + to_call)
    if r == 1 and e >= P["pf_3bet"] - loosen and can_raise:
        return Action.RAISE_POT
    if r >= 2 and e >= P["pf_4bet"] - loosen and can_raise:
        return Action.RAISE_POT
    # preflop equity vs random overstates our edge vs a raiser; ask for a bit more
    need = pot_odds + 0.06 + 0.02 * (r - 1) - loosen + P["pf_call_margin"]
    if e >= need or to_call == 0:
        return Action.CHECK_CALL
    return Action.FOLD


def _budget(obs: Observation, memory: dict) -> float:
    """Simulation time for this decision: spread what is left of our target
    thinking time over the postflop decisions we still expect this half."""
    think = memory.get("think", 0.0)
    if think > P["time_hard"] * memory.get("bank", 8.0) / 8.0:
        return 0.0
    hands_left = max(1, memory.get("n_hands", 1000) - obs.hand_index)
    rate = (memory.get("post_n", 0) + 3.0) / (obs.hand_index + 2.0)   # postflop decisions per hand
    post_n = memory.get("post_n", 0)
    # measured non-simulation cost per postflop decision (preflop, bookkeeping, overruns)
    over = (think - memory.get("sim", 0.0) + 0.003) / (post_n + 3.0)
    per = (P["time_target"] * memory.get("bank", 8.0) / 8.0 - think) / (hands_left * rate) - max(P["overhead"], over)
    return max(P["budget_min"], min(P["budget"], per))


def _postflop(obs: Observation, memory: dict) -> Action:
    st = opp_stats(memory)
    budget = _budget(obs, memory)
    t0 = time.perf_counter()
    e = equity_vs_range(obs, memory, budget)
    memory["sim"] = memory.get("sim", 0.0) + min(time.perf_counter() - t0, budget)
    to_call = obs.to_call
    can_raise = Action.RAISE_POT in obs.legal_actions
    street = int(obs.street)
    if to_call == 0:
        # value-bet thinner against opponents who rarely fold
        shift = (st["fold_to_bet"] - 0.4) * 0.15
        if st["bet_free"] > P["trap_th"] and street < 3:
            shift += P["trap_k"] * (st["bet_free"] - P["trap_th"])
        if e >= P["val_pot"] + shift and can_raise:
            return Action.RAISE_POT
        if e >= P["val_half"] + shift and can_raise:
            return Action.RAISE_HALF
        # bluff / semi-bluff against opponents who fold too much
        if can_raise and street < 3 and st["fold_to_bet"] > 0.45 and e >= P["sb_e"]:
            if random.random() < P["bluff_freq"] * (st["fold_to_bet"] - 0.35) * 2.5:
                return Action.RAISE_POT if P["sb_pot"] > 0 else Action.RAISE_HALF
        if can_raise and street == 3 and st["fold_to_bet"] > 0.55 and e < P["rb_e"]:
            if random.random() < P["bluff_freq"] * (st["fold_to_bet"] - 0.45) * P["rb_k"]:
                return Action.RAISE_POT
        return Action.CHECK_CALL
    pot_odds = to_call / (obs.pot + to_call)
    rvb = P["raise_vs_bet"]
    if st["bet_free"] > P["trap_th"] and obs.raises_this_street <= 1:
        rvb -= P["cr_k"] * (st["bet_free"] - P["trap_th"])
    if e >= rvb and can_raise and obs.raises_this_street < 3:
        return Action.RAISE_POT
    margin = P["cm%d" % street] + P["cm_rr"] * max(0, obs.raises_this_street - 1)
    if st["bet_free"] > P["trap_th"] and obs.raises_this_street <= 1:
        margin -= P["bc_k"] * (st["bet_free"] - P["trap_th"])
    if e >= pot_odds + margin:
        return Action.CHECK_CALL
    return Action.FOLD


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["think"] = 0.0
    memory["post_n"] = 0
    memory["n_hands"] = 1000
    memory["bank"] = 8.0
    if config is not None:
        try:
            memory["n_hands"] = max(1, int(config.num_hands))
            memory["bank"] = float(config.time_bank_seconds)
        except Exception:
            pass
    memory["opp"] = {"pf_dec": 0, "pf_raise": 0, "post_dec": 0, "post_raise": 0,
                     "faced_bet": 0, "fold_to_bet": 0, "free_dec": 0, "free_bet": 0, "sd_aggr": 0, "sd_weak_aggr": 0}
    memory["cur_hand"] = None
    memory["hists"] = _new_hists()


def choose_action(obs: Observation, memory: dict) -> Action:
    start = time.perf_counter()
    action = Action.CHECK_CALL
    try:
        if "opp" not in memory:
            new_half(None, memory)
        _track_actions(obs, memory)
        if obs.street == Street.PREFLOP:
            action = _preflop(obs, memory)
        else:
            memory["post_n"] = memory.get("post_n", 0) + 1
            action = _postflop(obs, memory)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
        memory["last_mine"] = (action, int(obs.street))
    except Exception:
        action = Action.CHECK_CALL
    memory["think"] = memory.get("think", 0.0) + (time.perf_counter() - start)
    return action


def _learn_showdown(result: HandResult, memory: dict):
    line = memory.get("line", {})
    hists = memory["hists"]
    opp = tuple(result.opp_hole)
    om = _hole_masks(opp)
    lv = line.get(0)
    if lv in (1, 2) and _PF_OK:
        b = _pf_bin(opp)
        if b is not None:
            hists["pfr" if lv == 2 else "pfc"][b] += 1
    full = list(result.board)
    if len(full) < 5:
        return
    known = set(opp) | set(full) | set(memory.get("hole", ()))
    rest = [c for c in _DECK if c not in known]
    sample = random.sample
    last = memory.get("last_mine")
    max_st = max(list(line.keys()) + [last[1] if last else 0])
    if _np is not None and P["use_np"] > 0:
        R = _np_sample(rest, int(P["learn_ref_np"]), 5)
        rpk, rps = _np_pairs(R)
        oh = _np.array(opp, dtype=_np.int64)
        opk, ops = _np_pairs(oh)
        for st in range(1, min(3, max_st) + 1):
            board = full[:2 + st]
            ref = _np.sort(_np_strength(rpk, rps, R, board))
            s = _np_strength(opk, ops, oh, board)
            p = (_np.searchsorted(ref, s, "left") + _np.searchsorted(ref, s, "right")) / (2.0 * len(ref))
            b = min(int(p * _NB), _NB - 1)
            hists[_TYPES[line.get(st, 0)]][b] += 1
        return
    for st in range(1, min(3, max_st) + 1):
        board = full[:2 + st]
        fds = _flush_draw_suits(board)
        ref = []
        for _ in range(int(P["learn_ref_n"])):
            cards = sample(rest, 5)
            ref.append(_strength(_hole_masks(cards), cards, board, fds))
        ref.sort()
        b = _pct_bin(ref, _strength(om, opp, board, fds), _NB)
        hists[_TYPES[line.get(st, 0)]][b] += 1


def on_hand_end(result: HandResult, memory: dict) -> None:
    start = time.perf_counter()
    try:
        if "opp" not in memory:
            return
        o = memory["opp"]
        if memory.get("cur_hand") != result.hand_index:
            return  # we never acted in this hand
        last = memory.get("last_mine")
        if last and last[0] in (Action.RAISE_HALF, Action.RAISE_POT) and last[1] >= 1:
            o["faced_bet"] += 1
            if not result.went_to_showdown and result.winner == "me":
                o["fold_to_bet"] += 1
        if result.went_to_showdown and result.opp_hole:
            line = memory.setdefault("line", {})
            if last and last[0] in (Action.RAISE_HALF, Action.RAISE_POT):
                line[last[1]] = max(line.get(last[1], 0), 1)   # they called our last bet
            _learn_showdown(result, memory)
    except Exception:
        pass
    finally:
        memory["think"] = memory.get("think", 0.0) + (time.perf_counter() - start)

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
