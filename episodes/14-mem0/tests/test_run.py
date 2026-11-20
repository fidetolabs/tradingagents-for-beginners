"""What the reel claims, checked against what the run produced.

Each of these is a thing `build-html.mjs` draws. They are here as well as there
because the build reads a JSON file and these read the code that writes it, so
a change to the world that quietly removes the lesson fails here first.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src import catalog as K, mem, world as W  # noqa: E402

DATA = HERE.parent / "mem0-data.json"


@pytest.fixture(scope="module")
def data():
    if not DATA.exists():
        pytest.skip("run.py has not been run")
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def world():
    return W.build()


def test_the_world_is_the_same_every_time():
    a, b = W.build(), W.build()
    assert (a["quality"] == b["quality"]).all()
    assert (a["rets"] == b["rets"]).all()


def test_a_score_is_the_same_every_time(world):
    c = K.candidates()[0]
    assert W.score(world, *c) == W.score(world, *c)


def test_the_standard_error_is_small_against_the_bar(world):
    """The gap has to be several times the error on the gap, or the reel is
    reporting noise."""
    errs = [W.score(world, *c)[1] for c in K.candidates()[:40]]
    assert max(errs) < W.BAR / 4


def test_a_verdict_needs_two_standard_errors():
    assert W.verdict(W.BAR - 0.001, 0.05) == "unclear"
    assert W.verdict(W.BAR - 0.5, 0.01) == "failed"
    assert W.verdict(W.BAR + 0.5, 0.01) == "good"


def test_the_same_way_of_picking_passes_somewhere_and_fails_somewhere(world):
    """The reel rests on this. If a metric were good or bad everywhere, a
    memory would not need to say where it was tried."""
    found = False
    for m in W.METRICS:
        got = {W.verdict(*W.score(world, m, 20, i, w))
               for i in W.INDUSTRIES for w in W.WINDOWS}
        if {"failed", "good"} <= got:
            found = True
            break
    assert found


def test_the_candidate_list_is_stable():
    assert K.candidates() == K.candidates()
    assert len(set(K.candidates())) == len(K.candidates())


def test_what_is_written_is_not_what_is_asked():
    """Nothing downstream may compare the two as strings, so they must not be
    the same string."""
    for c in K.candidates()[:20]:
        assert K.written(c) != K.asked(c)
        assert K.written_bare(c) != K.asked(c)


def test_the_scope_is_the_tail_of_the_sentence():
    c = K.candidates()[0]
    assert K.written(c).index(", in ") > len(K.written_bare(c)) - 12


def test_dropping_the_scope_merges_memories():
    cs = K.candidates()[:24]
    assert len({K.written_bare(c) for c in cs}) < len({K.written(c) for c in cs})


def test_the_plan_overlaps_and_also_moves_on():
    p = K.plan()
    assert len(p["proposed"]) == K.PROPOSED
    assert p["overlap"] and p["fresh"]
    assert len(p["overlap"]) + len(p["fresh"]) == K.PROPOSED


def test_no_model_was_called(data):
    assert data["meta"]["llm_calls"] == 0


def test_the_sessions_were_separate_processes(data):
    assert data["meta"]["pids"][0] != data["meta"]["pids"][1]


def test_both_stores_catch_every_repeat(data):
    """The comparison is at equal recall, so the only thing left to differ is
    what else each one caught."""
    assert data["scoped"]["caught"] == data["bare"]["caught"] == data["two"]["repeats"]


def test_the_shorter_memory_blocks_more(data):
    assert data["bare"]["blocked"] > data["scoped"]["blocked"]
    assert data["scoped"]["runs"] > data["bare"]["runs"]


def test_every_blocked_try_was_one_it_had_not_failed(data):
    repeats = {tuple(r["try"]) for r in data["two"]["rows"]
               if r["ran_before"] and r["verdict"] == "failed"}
    for t in data["bare"]["blocked_tries"]:
        assert tuple(t) not in repeats


def test_the_new_tries_were_worth_running(data):
    """Blocking them has to cost something, or the stress is free."""
    assert data["two"]["fresh_good"] >= 1


def test_the_embedder_is_the_one_the_scores_came_from():
    """Every score here is a similarity out of one model, so the model is
    checked the way episode 13 checks its price series."""
    mem.check_embedder()


def test_the_run_matches_what_the_episode_published(data):
    """The pin, read here as well as in run.py: run.py refuses to write a
    mismatch, and this catches a mem0-data.json written by an older copy."""
    mem.check_pin(data)
