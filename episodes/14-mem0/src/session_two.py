"""SESSION TWO. A new process. It has nothing but the store.

It puts ten tries forward, asks the store about each one in its own words, and
records what came back. It does not decide anything here: the scores are the
measurement and `run.py` reads the decision off them, so the same scores answer
both versions of the question.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import catalog as K, mem, world as W

mem.hush()

HERE = Path(__file__).resolve().parent.parent


def main():
    world = W.build()
    plan = K.plan()
    m = mem.open_store()
    before = mem.calls(m)

    rows = []
    for c in plan["proposed"]:
        q = K.asked(c)
        sc, se = W.score(world, *c)
        rows.append({
            "try": list(c),
            "asked": q,
            "ran_before": c in plan["ran"],
            "verdict": W.verdict(sc, se),
            "score": round(sc, 4),
            "error": round(se, 4),
            mem.SCOPED: mem.recall(m, q, mem.SCOPED),
            mem.BARE: mem.recall(m, q, mem.BARE),
        })

    out = {"proposed": rows, "llm_calls": mem.calls(m) - before, "pid": os.getpid()}
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "session_two.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"session two (pid {out['pid']}): {len(rows)} tries put forward, "
          f"each asked of both stores, {out['llm_calls']} model calls")


if __name__ == "__main__":
    main()
