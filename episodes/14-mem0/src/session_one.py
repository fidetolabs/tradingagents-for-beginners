"""SESSION ONE. It runs the tries, and it writes down the ones that failed.

Its own process. Nothing it holds in memory survives it, which is the point:
the only thing session two gets is what reached the store.

Each failure is written twice, under two names, as the SAME four facts and then
as a fact with its conditions dropped. Writing both in one process is what
makes the comparison fair -- one world, one screen, one set of failures, and
the only thing that differs downstream is the sentence that was saved.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import catalog as K, mem, world as W

mem.hush()

HERE = Path(__file__).resolve().parent.parent


def main():
    world = W.build()
    plan = K.plan()
    tried = []
    for c in plan["ran"]:
        sc, se = W.score(world, *c)
        tried.append({"try": list(c), "score": round(sc, 4),
                      "error": round(se, 4), "verdict": W.verdict(sc, se)})

    failed = [t for t in tried if t["verdict"] == "failed"]
    if not failed:
        raise SystemExit("session one failed nothing, so there is nothing to remember")

    m = mem.open_store(reset=True)
    before = mem.calls(m)

    scoped, bare = [], []
    for t in failed:
        c = tuple(t["try"])
        scoped.append(K.written(c))
        bare.append(K.written_bare(c))
    # DROPPING THE CONDITIONS MERGES MEMORIES THAT WERE NOT THE SAME RUN. The
    # store holds one row a distinct sentence, so this is not a choice made
    # here -- it is what the shorter sentence does.
    bare_rows = sorted(set(bare))

    for s in scoped:
        mem.remember(m, s, mem.SCOPED)
    for b in bare_rows:
        mem.remember(m, b, mem.BARE)

    after = mem.calls(m)
    out = {
        "ran": [list(c) for c in plan["ran"]],
        "tried": tried,
        "failed": [t["try"] for t in failed],
        "bar": W.BAR,
        "paths": W.PATHS,
        "scoped_rows": scoped,
        "bare_rows": bare_rows,
        "llm_calls": after - before,
        "pid": __import__("os").getpid(),
    }
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "session_one.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"session one (pid {out['pid']}): {len(tried)} tries, {len(failed)} failed, "
          f"{len(scoped)} memories written with their conditions and "
          f"{len(bare_rows)} without, {out['llm_calls']} model calls")


if __name__ == "__main__":
    main()
