"""Runs both sessions, each in its own process, and writes mem0-data.json.

THE SESSIONS ARE SEPARATE PROCESSES BECAUSE THE CLAIM IS ABOUT SESSIONS. An
agent's session ends and everything it was holding goes with it; what the next
one knows is whatever reached the store. Two function calls in one process
would have shared a heap and measured nothing.

There is ONE COPY OF EVERY FIGURE. This file writes them and build-html.mjs
reads them, so no number on the frame was typed in by hand.

Run: uv run run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "mem0-data.json"


def phase(mod):
    r = subprocess.run([sys.executable, "-m", mod], cwd=HERE,
                       capture_output=True, text=True)
    keep = [l for l in r.stdout.splitlines() if l.strip()]
    for l in keep:
        print("  " + l)
    if r.returncode:
        sys.stderr.write(r.stderr[-3000:])
        raise SystemExit(f"{mod} exited {r.returncode}")
    return r


def recall_at_full(rows, store):
    """The threshold that catches every repeat, and what else it catches.

    THE THRESHOLD IS NOT A DIAL TURNED UNTIL A NUMBER LOOKED GOOD. A builder
    with a store of failures wants one thing from it: do not run something you
    already know fails. So the threshold is the lowest one that still catches
    every repeat -- full recall -- and the only question left is what else
    comes with it. That quantity is defined by the scores, not chosen.
    """
    top = {}
    for r in rows:
        hits = r[store]
        top[tuple(r["try"])] = hits[0]["score"] if hits else 0.0

    repeats = [r for r in rows if r["ran_before"] and r["verdict"] == "failed"]
    others = [r for r in rows if not (r["ran_before"] and r["verdict"] == "failed")]
    if not repeats:
        raise SystemExit("nothing in the proposals was a known failure; there is no lesson")

    cut = min(top[tuple(r["try"])] for r in repeats)
    blocked = [r for r in others if top[tuple(r["try"])] >= cut]
    return {
        "threshold": round(cut, 4),
        "caught": len(repeats),
        "blocked": len(blocked),
        "blocked_tries": [r["try"] for r in blocked],
        "runs": len(rows) - len(repeats) - len(blocked),
        "top": {" | ".join(map(str, k)): v for k, v in top.items()},
    }


def main():
    import src.mem as _mem
    _mem.check_embedder()
    print("phase 1 — session one, its own process")
    phase("src.session_one")
    print("phase 2 — session two, a new process with nothing in it")
    phase("src.session_two")

    one = json.loads((HERE / "runs" / "session_one.json").read_text())
    two = json.loads((HERE / "runs" / "session_two.json").read_text())

    import src.catalog as K
    import src.mem as mem
    import src.world as W

    rows = two["proposed"]
    scoped = recall_at_full(rows, mem.SCOPED)
    bare = recall_at_full(rows, mem.BARE)

    repeats = [r for r in rows if r["ran_before"] and r["verdict"] == "failed"]
    fresh = [r for r in rows if not r["ran_before"]]

    data = {
        "meta": {
            "mem0": __import__("importlib.metadata", fromlist=["version"]).version("mem0ai"),
            "embedder": mem.MODEL,
            "dims": mem.DIMS,
            "store": "qdrant, on disk",
            "llm_calls": one["llm_calls"] + two["llm_calls"],
            "industries": len(W.INDUSTRIES),
            "metrics": len(W.METRICS),
            "windows": len(W.WINDOWS),
            "keeps": W.KEEPS,
            "candidates": len(K.candidates()),
            "paths": W.PATHS,
            "bar": W.BAR,
            "pids": [one["pid"], two["pid"]],
        },
        "one": {
            "tries": len(one["tried"]),
            "failed": len(one["failed"]),
            "good": sum(1 for t in one["tried"] if t["verdict"] == "good"),
            "unclear": sum(1 for t in one["tried"] if t["verdict"] == "unclear"),
            "scoped_rows": one["scoped_rows"],
            "bare_rows": one["bare_rows"],
            "grid": one["tried"],
        },
        "two": {
            "proposed": len(rows),
            "repeats": len(repeats),
            "fresh": len(fresh),
            "fresh_good": sum(1 for r in fresh if r["verdict"] == "good"),
            "rows": rows,
        },
        "scoped": scoped,
        "bare": bare,
    }

    # ── the structure the lesson rests on, asserted here and again in the build
    same_metric_both_ways(W)
    if data["meta"]["llm_calls"]:
        raise SystemExit("a language model was reached, and the frame says none was")
    if len(one["bare_rows"]) >= len(one["scoped_rows"]):
        raise SystemExit("dropping the conditions merged nothing, so the shorter memory costs nothing")
    if bare["blocked"] <= scoped["blocked"]:
        raise SystemExit("the shorter memory blocked no more than the scoped one; there is no stress")
    mem.check_pin(data)

    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print()
    print(f"mem0 {data['meta']['mem0']}, {mem.MODEL} at {mem.DIMS} dims, {data['meta']['llm_calls']} model calls")
    print(f"session one: {data['one']['tries']} tries, {data['one']['failed']} failed")
    print(f"  written with their conditions: {len(one['scoped_rows'])} rows")
    print(f"  written without them:          {len(one['bare_rows'])} rows")
    print(f"session two: {data['two']['proposed']} put forward, {data['two']['repeats']} already failed, "
          f"{data['two']['fresh']} never tried ({data['two']['fresh_good']} of them good enough)")
    print(f"  with the conditions: catches {scoped['caught']}, also blocks {scoped['blocked']}, runs {scoped['runs']}")
    print(f"  without them:        catches {bare['caught']}, also blocks {bare['blocked']}, runs {bare['runs']}")
    print(f"wrote {OUT.name}")


def same_metric_both_ways(W):
    """THE SAME WAY OF PICKING PASSES IN ONE PLACE AND FAILS IN ANOTHER.

    This is the fact the whole reel rests on: if a metric were good or bad
    everywhere, a memory would not need to say where it was tried and the
    lesson would be false. So the world is refused if it does not contain one.
    """
    world = W.build()
    for m in W.METRICS:
        for k in W.KEEPS:
            got = {}
            for i in W.INDUSTRIES:
                for w in W.WINDOWS:
                    sc, se = W.score(world, m, k, i, w)
                    got[(i, w)] = W.verdict(sc, se)
            if "failed" in got.values() and "good" in got.values():
                return
    raise SystemExit("no way of picking both passes and fails in this world; the conditions would not matter")


if __name__ == "__main__":
    main()
