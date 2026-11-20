"""The state the graph carries, which is also the thing the checkpointer saves."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Research(TypedDict, total=False):
    question: str
    window: int          # the parameter the agent is trying to improve
    attempts: int
    # what the world node brought back. THE SLICE LIVES IN THE STATE ON PURPOSE:
    # it is what makes a checkpoint worth something and also what makes one
    # expensive, and the run measures both.
    asof: str
    first: str
    bars: int
    last_close: float
    slice: dict
    vol: float
    signals: int
    score: dict
    verdict: str
    rule_in: str
    rule_out: str
    # every node that contributed to THIS state, which is not the same as every
    # node that ran: a crashed process contributed and then died
    trail: Annotated[list[str], operator.add]
