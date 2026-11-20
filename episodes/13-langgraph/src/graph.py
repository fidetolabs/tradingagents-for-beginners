"""The graph. Nine nodes, one conditional edge, one loop.

HOW LANGGRAPH WORKS, IN THE THREE FACTS THIS AGENT RESTS ON
------------------------------------------------------------
1. You declare a shape. Nodes are plain functions `state -> partial state`;
   edges say what may follow what. `add_conditional_edges` hands the choice to a
   function at runtime, so the agent picks its own path -- but only along edges
   you wrote. The path it took is then a thing you read back rather than a thing
   you reconstruct from logs.

2. The run advances in SUPER-STEPS. Everything that can run in parallel runs,
   the returned partials are merged through each channel's reducer, and the
   whole state is written to a CHECKPOINT keyed by `thread_id`. With
   `durability="sync"` that write lands before the next super-step starts, which
   is what makes a resume possible after the process is gone.

3. The checkpointer is per thread; the STORE is not. A thread is one run. The
   store is namespaced key-value that outlives every thread, and it is the only
   reason the next run can begin from something the last run worked out.

Nodes here are deterministic arithmetic, not model calls. That is deliberate:
the subject is the runtime, and an LLM in the middle would make every number on
the frame a sample rather than a measurement.
"""
from __future__ import annotations

import os
import sys

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from . import instrument as log
from . import world
from .state import Research

NS = ("spy", "rules")
KEY = "breakout-window"
DEFAULT_WINDOW = 20
TUNE_STEP = 20
MAX_ATTEMPTS = 3
LOOKBACK = 500

CRASH_AT = os.environ.get("CRASH_AT") or None


def _maybe_die(node: str, state: Research) -> None:
    """A real process death, not an exception somebody can catch. No unwinding,
    no finally blocks, no flush the runtime had not already done."""
    if CRASH_AT and node == CRASH_AT and state.get("attempts", 0) == 1:
        log.ran("__killed__", at=node)
        sys.stderr.write(f"[kill] the process dies inside {node}\n")
        sys.stderr.flush()
        os._exit(137)


def recall(state: Research, *, store: BaseStore) -> dict:
    """Read the cross-thread store first. This is the self-improvement hinge: a
    new run does not start from the default, it starts from whatever the last
    run was allowed to leave behind."""
    log.ran("recall")
    item = store.get(NS, KEY)
    window = int(item.value["window"]) if item else DEFAULT_WINDOW
    return {"window": window, "attempts": 0,
            "rule_in": f"{KEY}={window}" + ("" if item else " (nothing remembered)"),
            "trail": ["recall"]}


def fetch(state: Research) -> dict:
    """THE ONLY NODE THAT TOUCHES THE WORLD, and the reason a fork is not a
    replay: it answers "the last 500 trading days as of now", and now moves."""
    log.ran("fetch")
    bars = world.latest(LOOKBACK)
    return {"asof": bars["asof"], "first": bars["first"], "bars": bars["bars"],
            "last_close": bars["last_close"], "slice": bars, "trail": ["fetch"]}


def features(state: Research) -> dict:
    log.ran("features", window=state["window"])
    import numpy as np

    c = np.array(state["slice"]["closes"])
    r = np.diff(np.log(c))
    return {"vol": round(float(r[-20:].std(ddof=1) * (252 ** 0.5)), 4), "trail": ["features"]}


def screen(state: Research) -> dict:
    """How many days the rule would have fired on, at the current window."""
    log.ran("screen", window=state["window"])
    import numpy as np

    high = np.array(state["slice"]["highs"])
    close = np.array(state["slice"]["closes"])
    w = state["window"]
    fired = sum(1 for i in range(w, len(close)) if close[i] > high[i - w:i].max())
    return {"signals": int(fired), "trail": ["screen"]}


def backtest(state: Research) -> dict:
    """The expensive node, and the one the process dies inside."""
    _maybe_die("backtest", state)
    log.ran("backtest", window=state["window"])
    return {"score": world.breakout_trades(state["slice"], state["window"]), "trail": ["backtest"]}


def judge(state: Research) -> dict:
    log.ran("judge", edge=state["score"]["edge"], attempts=state["attempts"])
    return {"trail": ["judge"]}


def route(state: Research) -> str:
    """The conditional edge. The agent picks what happens next, and it can only
    pick something the graph already allows."""
    if state["score"]["edge"] <= 0 and state["attempts"] < MAX_ATTEMPTS:
        return "tune"
    return "conclude"


def tune(state: Research) -> dict:
    log.ran("tune", frm=state["window"], to=state["window"] + TUNE_STEP)
    return {"window": state["window"] + TUNE_STEP,
            "attempts": state["attempts"] + 1, "trail": ["tune"]}


def conclude(state: Research) -> dict:
    log.ran("conclude")
    beat = state["score"]["edge"] > 0
    return {"verdict": ("the breakout beats holding on this slice" if beat
                        else "holding beats the breakout on this slice"),
            "trail": ["conclude"]}


def remember(state: Research, *, store: BaseStore) -> dict:
    """Write what this run worked out where the next run will read it."""
    log.ran("remember", window=state["window"])
    store.put(NS, KEY, {"window": state["window"], "asof": state["asof"],
                        "edge": state["score"]["edge"], "verdict": state["verdict"]})
    return {"rule_out": f"{KEY}={state['window']}", "trail": ["remember"]}


NODES = ["recall", "fetch", "features", "screen", "backtest", "judge", "tune", "conclude", "remember"]
EDGES = [("START", "recall"), ("recall", "fetch"), ("fetch", "features"), ("features", "screen"),
         ("screen", "backtest"), ("backtest", "judge"), ("judge", "tune"), ("judge", "conclude"),
         ("tune", "features"), ("conclude", "remember"), ("remember", "END")]


def build() -> StateGraph:
    g = StateGraph(Research)
    # NAMED FROM THE LIST, NOT FROM `fn.__name__`. A node's name is part of the
    # declared shape, so reading it off the function means anything that wraps a
    # node renames it: a test that instrumented all nine built a graph with one
    # node called `wrap`.
    for name in NODES:
        g.add_node(name, globals()[name])
    g.add_edge(START, "recall")
    g.add_edge("recall", "fetch")
    g.add_edge("fetch", "features")
    g.add_edge("features", "screen")
    g.add_edge("screen", "backtest")
    g.add_edge("backtest", "judge")
    g.add_conditional_edges("judge", route, {"tune": "tune", "conclude": "conclude"})
    g.add_edge("tune", "features")
    g.add_edge("conclude", "remember")
    g.add_edge("remember", END)
    return g
