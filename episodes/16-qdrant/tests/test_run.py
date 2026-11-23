"""The shapes and the figures the README talks about.

Two groups. The first needs nothing but the cached filings and checks that the
corpus is the kind of corpus this lesson is about. The second reads the last
run's output and checks the figures the reel published, which is the point of
the pin.

Run: uv run pytest tests -q        (with a server up, per README.md)
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src import edgar as E                      # noqa: E402
from src import pin as PIN                      # noqa: E402

DATA = HERE.parent / "qdrant-data.json"


# ── the pin itself, which needs nothing ──────────────────────────────────────

def test_every_filing_is_pinned_by_accession_and_hash():
    for f in PIN.FILINGS:
        assert len(f.sha256) == 64, f"{f.accession} has no document hash"
        assert f.accession.count("-") == 2
    accessions = [f.accession for f in PIN.FILINGS]
    assert len(set(accessions)) == len(accessions), "a filing is pinned twice"


def test_the_corpus_spans_several_filers_and_several_years():
    """Both axes are load-bearing: the filter is a year, the pair counts filers."""
    assert len(PIN.TICKERS) >= 4
    assert len(PIN.YEARS) >= 3
    assert PIN.ASK_YEAR in PIN.YEARS
    by = Counter((f.ticker, f.year) for f in PIN.FILINGS)
    assert max(by.values()) == 1, "a filer has two filings for one year"
    assert len(PIN.FILINGS) == len(PIN.TICKERS) * len(PIN.YEARS), (
        "the grid is not square, so a year filter is not a clean quarter of it")


def test_the_filter_year_is_not_most_of_the_collection():
    """A filter that keeps nearly everything is not a filter."""
    assert len(PIN.YEARS) >= 3


def test_the_full_scan_threshold_keeps_the_search_approximate():
    """Left on its default a filtered search is read exactly and comes back
    perfect, which would make the last beat of the reel false."""
    assert PIN.FULL_SCAN_THRESHOLD_KB <= 10
    assert PIN.INDEXING_THRESHOLD_KB <= 1


def test_the_questions_are_not_quotations():
    """The first half of the reel is that a question need not use the filing's
    own words, so a question set made of quotations would prove nothing."""
    assert len(PIN.QUESTIONS) >= 8
    assert 0 <= PIN.ANCHOR < len(PIN.QUESTIONS)
    for q in PIN.QUESTIONS:
        assert q[0].islower(), f"{q!r} opens like a sentence lifted out of a filing"
        assert not q.endswith("."), f"{q!r} is punctuated like a quotation"


# ── the passages, which need the cached documents ────────────────────────────

@pytest.fixture(scope="module")
def anchor_text():
    f = PIN.FILINGS[0]
    path = E.CACHE / f"{f.accession}-{f.document}"
    if not path.exists():
        pytest.skip(f"{path.name} is not cached; run `uv run run.py` first")
    return E.text_of(path.read_bytes())


def test_a_passage_is_something_an_agent_could_quote(anchor_text):
    ps = E.passages(anchor_text)
    assert len(ps) > 100
    assert all(len(p) <= E.PASSAGE_CHARS for p in ps)
    assert all(len(p) >= E.PASSAGE_MIN for p in ps)
    assert all(p.count(".") >= 2 for p in ps), "a passage with no sentence in it"


def test_cutting_the_same_filing_twice_gives_the_same_passages(anchor_text):
    assert E.passages(anchor_text) == E.passages(anchor_text)


def test_table_residue_is_dropped():
    assert E.passages("2021 2022 2023 1,418 2,004 3,112 44.1 % 51.2 % 7.9 %") == []


# ── the figures, which need a run ────────────────────────────────────────────

@pytest.fixture(scope="module")
def ran():
    """The last run's output. Skipped rather than failed when there is none,
    because the tests above are worth having on a machine with no server."""
    if not DATA.exists():
        pytest.skip(f"{DATA.name} is not there; run `uv run run.py` first")
    return json.loads(DATA.read_text())


def test_the_run_prints_the_published_figures(ran):
    assert ran["corpus"]["passages"] == PIN.FIGURES["passages"]
    assert ran["corpus"]["year_passages"] == PIN.FIGURES["year_passages"]
    assert ran["pair"]["after"] == PIN.FIGURES["after"]
    assert ran["pair"]["inside"] == PIN.FIGURES["inside"]
    assert len(ran["anchor"]["shared"]) == PIN.FIGURES["shared_words"]
    assert len(ran["anchor"]["question_words"]) == PIN.FIGURES["question_words"]


def test_the_versions_are_the_pinned_ones(ran):
    assert ran["meta"]["qdrant"] == PIN.QDRANT_VERSION
    assert ran["meta"]["client"] == PIN.CLIENT_VERSION
    assert ran["meta"]["model_sha256"] == PIN.MODEL_SHA256
    assert ran["meta"]["model_revision"] == PIN.MODEL_REVISION


def test_the_filing_count_and_the_passage_count_agree(ran):
    assert sum(f["passages"] for f in ran["filings"]) == ran["corpus"]["passages"]
    assert len(ran["filings"]) == ran["corpus"]["filings"]
    assert len(ran["marks"]["year_at"]) == ran["corpus"]["year_passages"]


def test_the_filter_inside_leaves_more_to_quote(ran):
    p = ran["pair"]
    assert p["inside"]["cite"] > p["after"]["cite"]
    assert p["inside"]["filers"] > p["after"]["filers"]
    assert p["inside"]["cite"] == ran["meta"]["k"]


def test_filtering_afterwards_has_to_ask_for_more(ran):
    """The honest version: the first side can ask for more and keep filtering."""
    p = ran["pair"]["after"]
    assert p["asked_for"] > ran["meta"]["k"]
    assert p["thrown"] == p["asked_for"] - ran["meta"]["k"]


def test_the_question_is_not_in_the_filings_words(ran):
    """The vector is doing work a text match could not."""
    a = ran["anchor"]
    assert len(a["shared"]) * 2 < len(a["question_words"])
    assert set(a["shared"]) <= set(a["question_words"])


def test_the_kept_passages_are_a_subset_of_the_whole_search(ran):
    m = ran["marks"]
    assert set(m["kept"]) <= set(m["whole"])
    assert len(m["kept"]) == ran["pair"]["after"]["cite"]
    assert len(m["inside"]) == ran["pair"]["inside"]["cite"]


def test_the_declared_side_misses_nothing_on_any_build(ran):
    """Which is what makes the pair's figures the ones the index returns."""
    assert ran["stress"]["declared_missed"] == [0] * ran["stress"]["builds"]


def test_the_undeclared_side_misses_on_every_build(ran):
    """The stress: a consequence of an index built before the question exists."""
    assert ran["stress"]["undeclared_min"] > 0
    assert all(u > 0 for u in ran["stress"]["undeclared_missed"])
    assert PIN.UNDECLARED_MIN <= ran["stress"]["undeclared_min"]
    assert ran["stress"]["undeclared_max"] <= PIN.UNDECLARED_MAX


def test_the_miss_is_silent(ran):
    """The same count of passages comes back; it is a different set."""
    k = ran["meta"]["k"]
    for returned, filers, right in ran["stress"]["anchor_shape"]:
        assert returned == k, "fewer passages came back, so the response says something"
        assert filers >= 1
        assert right <= k


def test_the_size_of_the_miss_is_not_reproducible(ran):
    """Qdrant seeds no graph build, which is why the reel prints a range."""
    assert ran["stress"]["builds"] >= 5
    assert ran["stress"]["undeclared_sd"] > 0
    assert ran["stress"]["undeclared_max"] > ran["stress"]["undeclared_min"]


def test_the_frame_prints_no_two_headline_figures_the_same(ran):
    """A number printed twice is read as a typo."""
    printed = [
        ran["pair"]["after"]["cite"], ran["pair"]["after"]["filers"],
        ran["pair"]["inside"]["cite"], ran["pair"]["inside"]["filers"],
        ran["stress"]["declared_max"],
        ran["stress"]["undeclared_min"], ran["stress"]["undeclared_max"],
    ]
    assert len(set(printed)) == len(printed), f"{printed} repeats a figure"
