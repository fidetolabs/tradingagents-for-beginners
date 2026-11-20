#!/usr/bin/env python
"""Run the experiment and write ../langgraph-data.json.

    .venv/bin/python run.py                # the whole thing
    .venv/bin/python run.py --snapshots    # re-download SPY and rebuild data/

FIVE PHASES, EACH IN ITS OWN PROCESS, because the claim being measured is about
processes dying. The parent orchestrates; every child opens its own checkpointer
and store off disk, which is the only way a resume proves anything.

  1  cold     thread A, a fresh run, killed with os._exit from inside `backtest`
  2  resume   thread A again, in a NEW process, invoke(None) from the checkpoint
  3  restart   thread B, the same work from the top, for the comparison
  4  fork     back to a checkpoint of thread A, change the rule, run it again
              -- against a market two trading days further on, which is the point
  5  control  the same fork with the rule left alone, so the world's share of
              the difference is measurable instead of assumed
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "checkpoints.sqlite")
STORE_DB = os.path.join(ROOT, "store.sqlite")
OUT = os.path.join(os.path.dirname(ROOT), "langgraph-data.json")
QUESTION = "does a Donchian breakout beat holding SPY on the last 500 trading days?"
THREAD_A, THREAD_B = "research-a", "research-b"
FORK_WINDOW = 55


@contextlib.contextmanager
def app():
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore
    from src.graph import build

    with SqliteSaver.from_conn_string(DB) as saver, SqliteStore.from_conn_string(STORE_DB) as store:
        store.setup()
        yield build().compile(checkpointer=saver, store=store), store


def cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def _slim(state: dict) -> dict:
    return {k: v for k, v in state.items() if k != "slice"}


def child_cold() -> dict:
    from src import instrument as log
    log.open_log("cold")
    with app() as (graph, _):
        graph.invoke({"question": QUESTION, "trail": []}, cfg(THREAD_A), durability="sync")
    return {}


def child_resume() -> dict:
    from src import instrument as log
    log.open_log("resume")
    with app() as (graph, store):
        out = graph.invoke(None, cfg(THREAD_A), durability="sync")
        item = store.get(("spy", "rules"), "breakout-window")
    return {"final": _slim(out), "store": item.value if item else None}


def child_restart() -> dict:
    from src import instrument as log
    log.open_log("restart")
    with app() as (graph, _):
        out = graph.invoke({"question": QUESTION, "trail": []}, cfg(THREAD_B), durability="sync")
    return {"final": _slim(out)}


def _fork(label: str, values: dict) -> dict:
    from src import instrument as log
    log.open_log(label)
    with app() as (graph, _):
        hist = list(graph.get_state_history(cfg(THREAD_A)))
        target = next(s for s in reversed(hist) if s.next == ("fetch",))
        forked = graph.update_state(target.config, values, as_node="recall")
        out = graph.invoke(None, forked, durability="sync")
    return {"final": _slim(out),
            "from_checkpoint": target.config["configurable"]["checkpoint_id"],
            "from_window": target.values.get("window")}


def child_fork() -> dict:
    return _fork("fork", {"window": FORK_WINDOW})


def child_fork_control() -> dict:
    """THE SAME FORK WITH NOTHING CHANGED.

    The fork above moved two things at once: the rule the agent was working with
    AND, because `fetch` runs again, the window of market it ran on. Reporting
    the difference between thread A and that fork as the rule's doing reports
    both and credits one. This changes nothing, so what it produces is the
    world's share, and the rule's share is what is left over."""
    return _fork("fork_control", {})


CHILDREN = {"cold": child_cold, "resume": child_resume, "restart": child_restart,
            "fork": child_fork, "fork_control": child_fork_control}


def spawn(phase: str, env_extra: dict | None = None):
    env = {**os.environ, **(env_extra or {})}
    p = subprocess.run([sys.executable, __file__, "--phase", phase],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    sys.stderr.write(p.stderr)
    payload = {}
    for line in p.stdout.splitlines():
        if line.startswith("__RESULT__"):
            payload = json.loads(line[len("__RESULT__"):])
        else:
            print(line)
    return p.returncode, payload


def checkpoint_report() -> dict:
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT thread_id, COUNT(*), SUM(LENGTH(checkpoint)), MAX(LENGTH(checkpoint)) "
                       "FROM checkpoints GROUP BY thread_id").fetchall()
    writes = con.execute("SELECT COUNT(*), SUM(LENGTH(value)) FROM writes").fetchone()
    con.close()
    return {"per_thread": [{"thread": r[0], "checkpoints": r[1], "bytes": r[2], "largest": r[3]} for r in rows],
            "writes": int(writes[0] or 0), "write_bytes": int(writes[1] or 0),
            "file_bytes": os.path.getsize(DB)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase")
    ap.add_argument("--snapshots", action="store_true", help="re-download SPY and rebuild data/")
    args = ap.parse_args()

    if args.phase:
        print("__RESULT__" + json.dumps(CHILDREN[args.phase]()))
        return 0

    from src import graph as G, instrument as log, world

    if args.snapshots:
        print("snapshots ", world.build_snapshots())
    for f in (DB, STORE_DB):
        if os.path.exists(f):
            os.remove(f)

    print(f"graph      {len(G.NODES)} nodes, {len(G.EDGES)} edges, 1 conditional edge, 1 loop")

    print("\n1  cold     a fresh run on thread A, and the process dies inside backtest")
    code, _ = spawn("cold", {"CRASH_AT": "backtest", "SPY_FILE": world.ASOF_RUN})
    cold = log.read("cold")
    killed = [e for e in cold if e["node"] == "__killed__"]
    ran_cold = [e for e in cold if e["node"] != "__killed__"]
    print(f"           exit {code}; {len(ran_cold)} nodes ran before it died "
          f"({' '.join(e['node'] for e in ran_cold)})")
    if not killed:
        raise SystemExit("the cold run was supposed to die and did not")

    print("\n2  resume   a NEW process picks thread A up from the last checkpoint")
    _, resumed = spawn("resume", {"SPY_FILE": world.ASOF_RUN})
    ran_resume = log.read("resume")
    print(f"           {len(ran_resume)} nodes ran ({' '.join(e['node'] for e in ran_resume)})")
    print(f"           verdict: {resumed['final']['verdict']}")

    print("\n3  restart  the same work from the top, on a fresh thread")
    _, restarted = spawn("restart", {"SPY_FILE": world.ASOF_RUN})
    ran_restart = log.read("restart")
    print(f"           {len(ran_restart)} nodes ran to reach the same answer")

    asof_before = resumed["final"]["asof"]
    # HOW MANY TRADING DAYS THE WORLD MOVED, COUNTED OFF THE FILES RATHER THAN
    # TYPED. The first cut of this said "two days" in three places, and when an
    # unfinished bar was dropped it became one and the sentences stayed wrong.
    import pandas as _pd
    _run = _pd.read_csv(world.ASOF_RUN, index_col=0, parse_dates=True)
    _fork = _pd.read_csv(world.ASOF_FORK, index_col=0, parse_dates=True)
    days_added = int(len(_fork) - len(_run))
    asof_after = _fork.index[-1].date().isoformat()
    plural = "day" if days_added == 1 else "days"
    print(f"\n   the gap  one download, read at two dates {days_added} trading {plural} apart.")
    print(f"            Thread A reads it as of {asof_before}; the fork reads it to {asof_after}")

    print("\n4  fork     back to the checkpoint after `recall`, change the rule, run forward")
    _, forked = spawn("fork", {"SPY_FILE": world.ASOF_FORK})
    ran_fork = log.read("fork")
    print(f"           {len(ran_fork)} nodes ran")

    print("\n5  control  the same fork with the rule left alone")
    _, control = spawn("fork_control", {"SPY_FILE": world.ASOF_FORK})
    ran_control = log.read("fork_control")
    print(f"           {len(ran_control)} nodes ran, window {control['final']['window']} "
          f"(thread A ended on {resumed['final']['window']})")

    ck = checkpoint_report()
    b = next(t for t in ck["per_thread"] if t["thread"] == THREAD_B)
    base = resumed["final"]["score"]["edge"]
    ctl = control["final"]["score"]["edge"]
    frk = forked["final"]["score"]["edge"]

    doc = {
        "meta": {"langgraph": _ver("langgraph"), "checkpointer": "SqliteSaver", "store": "SqliteStore",
                 "durability": "sync", "question": QUESTION, "nodes": G.NODES, "edges": G.EDGES,
                 "lookback": G.LOOKBACK, "llm": False},
        "cold": {"ran": [e["node"] for e in ran_cold], "died_in": killed[0]["at"], "exit": code},
        "resume": {"ran": [e["node"] for e in ran_resume], "final": resumed["final"], "store": resumed["store"]},
        "restart": {"ran": [e["node"] for e in ran_restart], "final": restarted["final"]},
        "fork": {"ran": [e["node"] for e in ran_fork], **forked},
        "fork_control": {"ran": [e["node"] for e in ran_control], **control},
        "checkpoints": ck,
        "attribution": {"thread_a_edge": base, "control_edge": ctl, "fork_edge": frk,
                        "thread_a_window": resumed["final"]["window"],
                        "control_window": control["final"]["window"],
                        "fork_window": forked["final"]["window"],
                        "naive_gain": round(frk - base, 2),
                        "world_share": round(ctl - base, 2),
                        "rule_share": round(frk - ctl, 2)},
        "cost": {"nodes": len(ran_restart), "checkpoints": b["checkpoints"],
                 "bytes": b["bytes"], "largest": b["largest"]},
        "pair": {"resume_nodes": len(ran_resume), "restart_nodes": len(ran_restart),
                 "saved_nodes": len(ran_restart) - len(ran_resume),
                 "asof_before": asof_before, "asof_after": forked["final"]["asof"],
                 "close_before": resumed["final"]["last_close"],
                 "close_after": forked["final"]["last_close"],
                 "days_added": days_added},
    }
    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=1)

    at, p, c = doc["attribution"], doc["pair"], doc["cost"]
    print("\nWHAT MOVED, AND WHY")
    print(f"  thread A          {at['thread_a_edge']:>10,.2f}   window {at['thread_a_window']}, as of {asof_before}")
    print(f"  the control fork  {at['control_edge']:>10,.2f}   same window, {days_added} more trading {plural}")
    print(f"  the fork          {at['fork_edge']:>10,.2f}   new window, {days_added} more trading {plural}")
    print(f"  the agent would credit the rule with {at['naive_gain']:,.2f}")
    print(f"  the {days_added} trading {plural} {'is' if days_added == 1 else 'are'} worth {at['world_share']:,.2f}; the rule is worth {at['rule_share']:,.2f}")
    print("\nTHE PAIR")
    print(f"  after the crash   {p['resume_nodes']} nodes from the checkpoint, "
          f"{p['restart_nodes']} from the top, {p['saved_nodes']} saved")
    print(f"  the fork          {p['asof_before']} at ${p['close_before']:,.2f} became "
          f"{p['asof_after']} at ${p['close_after']:,.2f}")
    print(f"  what it cost      one run is {c['nodes']} nodes, {c['checkpoints']} checkpoints "
          f"and {c['bytes']:,} bytes of state")
    print(f"\nwrote {OUT}")
    return 0


def _ver(pkg: str) -> str:
    import importlib.metadata as md
    return md.version(pkg)


if __name__ == "__main__":
    raise SystemExit(main())
