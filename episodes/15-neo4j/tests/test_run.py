"""The shapes and the figures the README talks about.

Two groups. The first needs nothing but Python and checks the corpus: that it
is the kind of corpus the lesson is about. The second needs a server up and
checks the figures the reel published, which is the point of the pin.

Run: uv run pytest tests -q        (with a server up, per README.md)
"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src import pin as PIN                     # noqa: E402
from src import world as W                     # noqa: E402

DATA = HERE.parent / "neo4j-data.json"


@pytest.fixture(scope="module")
def world():
    return W.build()


# ── the corpus, which needs no server ────────────────────────────────────────

def test_corpus_is_the_size_the_pin_says(world):
    assert len(world["companies"]) == PIN.CORPUS["companies"]
    assert len(world["filings"]) == PIN.CORPUS["filings"]
    assert len(world["events"]) == PIN.CORPUS["events"]
    assert len(W.mentions(world)) == PIN.CORPUS["mentions"]


def test_the_seed_gives_the_same_corpus_twice(world):
    again = W.build()
    assert [f.fid for f in again["filings"]] == [f.fid for f in world["filings"]]
    assert [f.phrases for f in again["filings"]] == [f.phrases for f in world["filings"]]


def test_one_event_is_worded_more_than_one_way(world):
    """The only reason a knowledge graph is worth building."""
    many = [e for e in world["events"] if len(e["phrases"]) > 1]
    assert many, "every event has exactly one phrasing; a text match could not miss anything"


def test_an_incident_is_named_by_a_handful_of_filings(world):
    """Not by half the corpus. An event everybody names is a theme."""
    from collections import Counter

    seen = Counter(m["event"] for m in W.mentions(world))
    incident = [seen[e["name"]] for e in world["events"] if e["kind"] == "incident"]
    median = sorted(incident)[len(incident) // 2]
    assert median <= len(world["filings"]) * 0.1, (
        f"the median incident is in {median} of {len(world['filings'])} filings")


def test_the_theme_is_the_most_named_event(world):
    from collections import Counter

    seen = Counter(m["event"] for m in W.mentions(world))
    theme = [e for e in world["events"] if e["kind"] == "theme"]
    assert len(theme) == 1
    biggest = max(seen[e["name"]] for e in world["events"] if e["kind"] == "incident")
    assert seen[theme[0]["name"]] > biggest


def test_the_anchor_is_the_filing_the_pin_names(world):
    a = W.anchor(world)
    assert a["fid"] == PIN.ANCHOR["fid"]
    assert a["event"] == PIN.ANCHOR["event"]
    assert a["phrase"] == PIN.ANCHOR["phrase"]


def test_the_anchors_event_is_also_worded_the_other_way(world):
    """Which is the whole of the first pair: the words miss those filings."""
    a = W.anchor(world)
    others = [m for m in W.mentions(world)
              if m["event"] == a["event"] and m["fid"] != a["fid"]]
    worded = {m["phrase"] for m in others}
    assert PIN.ANCHOR["other_phrase"] in worded
    assert len(others) == PIN.ANCHOR["degree"] - 1


# ── the figures, which need the server ───────────────────────────────────────

@pytest.fixture(scope="module")
def ran():
    """The last run's output. Skipped rather than failed when there is none,
    because the corpus tests above are worth having on a machine with no
    server. `uv run run.py` writes it."""
    if not DATA.exists():
        pytest.skip(f"{DATA.name} is not there; run `uv run run.py` first")
    return json.loads(DATA.read_text())


def test_the_run_prints_the_published_figures(ran):
    assert ran["pair"]["words"]["hits"] == PIN.FIGURES["words_hits"]
    assert ran["pair"]["words"]["found"] == PIN.FIGURES["words_found"]
    assert ran["pair"]["graph"]["hits"] == PIN.FIGURES["graph_hits"]
    assert ran["pair"]["graph"]["found"] == PIN.FIGURES["graph_found"]
    assert ran["theme"]["graph"]["hits"] == PIN.FIGURES["theme_graph_hits"]
    assert ran["theme"]["graph"]["found"] == PIN.FIGURES["theme_graph_found"]


def test_the_versions_are_the_pinned_ones(ran):
    assert ran["meta"]["neo4j"] == PIN.NEO4J
    assert ran["meta"]["driver"] == PIN.DRIVER


def test_the_planner_chose_the_pinned_plan(ran):
    """A db hit is a property of the plan, so the plan is part of the pin."""
    assert ran["plans"]["words"] == PIN.PLANS["words"]
    assert ran["plans"]["graph"] == PIN.PLANS["graph"]


def test_the_relationship_is_cheaper_and_finds_more(ran):
    p = ran["pair"]
    assert p["graph"]["hits"] < p["words"]["hits"]
    assert p["graph"]["found"] > p["words"]["found"]


def test_the_same_walk_costs_more_at_the_theme(ran):
    """The stress: a consequence of the shape, not a bug in it."""
    assert ran["theme"]["graph"]["hits"] > ran["pair"]["graph"]["hits"]
    assert ran["theme"]["graph"]["hits"] > ran["theme"]["words"]["hits"]
    # every OTHER company: both sides exclude the one doing the asking, because
    # the question is who else named it
    assert ran["theme"]["graph"]["found"] == ran["corpus"]["companies"] - 1


def test_both_sides_were_handed_the_same_words(ran):
    assert ran["pair"]["words"]["phrase"] == ran["pair"]["graph"]["phrase"]
    assert ran["pair"]["phrase"] == PIN.ANCHOR["phrase"]


def test_the_pin_refuses_a_figure_that_moved(ran):
    """The guard is the point of the pin, so it is tested rather than trusted."""
    versions = {"neo4j": PIN.NEO4J, "edition": PIN.EDITION, "driver": PIN.DRIVER}
    clean = PIN.check(versions, ran["corpus"], ran["pair"], ran["theme"])
    assert clean == []

    drifted = json.loads(json.dumps(ran["pair"]))
    drifted["graph"]["hits"] += 1
    moved = PIN.check(versions, ran["corpus"], drifted, ran["theme"])
    assert any("graph_hits" in m for m in moved)

    replanned = json.loads(json.dumps(ran["pair"]))
    replanned["graph"]["ops"] = ["ProduceResults", "NodeByLabelScan"]
    moved = PIN.check(versions, ran["corpus"], replanned, ran["theme"])
    assert any("plan.graph" in m for m in moved)
