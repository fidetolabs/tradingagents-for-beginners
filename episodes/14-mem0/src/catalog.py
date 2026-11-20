"""The tries, and the two ways a try gets written down in English.

A CANDIDATE IS NOT A KEY. This is the part the reel is actually about. When a
session writes down what it tried, it writes a sentence; when the next session
asks whether it has tried something, it asks in its own words. The two
sentences are never the same string, which is why a log file keyed on text
finds nothing and a memory searched by meaning finds something.

So every candidate has two renderings, from two templates: the one a session
WRITES, and the one the next session ASKS. Nothing downstream ever compares
them as strings.

THE SCHEDULE IS DECLARED, NOT CHOSEN AFTER THE FACT. The candidate list is
shuffled once from the world's seed. Session one walks the first `RAN`; session
two starts `MOVED` further on, which is what an exploration schedule does, and
takes `PROPOSED`. The overlap and the novelty are then arithmetic, and which of
the overlapping ones failed is measured by the screen.
"""

import numpy as np

from .world import INDUSTRIES, KEEPS, METRICS, SEED, WINDOWS

RAN = 24          # tries session one ran
MOVED = 18        # how far the schedule had moved on by session two
PROPOSED = 10     # tries session two put forward


def candidates(seed=SEED):
    all_of_them = [(m, k, i, w) for m in METRICS for k in KEEPS
                   for i in INDUSTRIES for w in WINDOWS]
    g = np.random.default_rng(seed + 1)
    g.shuffle(all_of_them)
    return [tuple(c) for c in all_of_them]


def written(c):
    """How the session that ran it writes it down, with its conditions."""
    m, k, i, w = c
    return f"ranking on {m} and keeping {k} was not good enough, in {i}, {w}"


def written_bare(c):
    """THE SAME MEMORY WITH ITS CONDITIONS DROPPED.

    This is not a strawman. A memory layer extracts a fact from a run, and a
    fact is shorter than the run it came from -- that is what makes it a
    memory. What gets dropped first is the scope, because the scope reads as
    circumstance. The reel measures what that costs.
    """
    m, _k, _i, _w = c
    return f"ranking on {m} was not good enough"


def asked(c):
    """How the next session asks for it. Its own words, not the stored ones."""
    m, k, i, w = c
    return f"try {m} across {i} over {w}, holding the top {k}"


def plan():
    """Who ran what, and what the next session put forward."""
    cs = candidates()
    ran = cs[:RAN]
    proposed = cs[MOVED:MOVED + PROPOSED]
    overlap = [c for c in proposed if c in ran]
    fresh = [c for c in proposed if c not in ran]
    return {"ran": ran, "proposed": proposed, "overlap": overlap, "fresh": fresh}
