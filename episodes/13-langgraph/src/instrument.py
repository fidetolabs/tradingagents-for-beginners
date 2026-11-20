"""An out-of-band record of which node actually ran.

IT CANNOT LIVE IN THE STATE. The whole point of the experiment is a process that
dies, and on resume the restored state already carries everything the dead
process put in it, so counting executions off the state counts the crashed run's
work twice. One append-only file per process, flushed and fsynced per line,
because the process it is recording is about to be killed without warning.
"""
from __future__ import annotations

import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
_path: str | None = None


def open_log(label: str) -> str:
    global _path
    os.makedirs(RUNS, exist_ok=True)
    _path = os.path.join(RUNS, f"{label}.jsonl")
    open(_path, "w").close()
    return _path


def ran(node: str, **extra) -> None:
    if _path is None:
        return
    with open(_path, "a") as fh:
        fh.write(json.dumps({"node": node, "t": round(time.time(), 3), **extra}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read(label: str) -> list:
    path = os.path.join(RUNS, f"{label}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]
