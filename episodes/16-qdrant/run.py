"""Load twenty-four 10-K filings into Qdrant and ask twelve questions of them.

    uv run run.py                  measure, assert, check the pin, write the data
    READING=1 uv run run.py        print everything and write nothing

Writes `../qdrant-data.json`. `build-html.mjs` reads that file, so there is one
copy of every figure on the frame.

WHAT THIS MEASURES. The agent may not claim anything it cannot quote, so all it
can say is what the search gave it. A question is about one year, which means
the search has two jobs: find the passages that mean what the question means,
and stay inside the filings the agent is allowed to quote. Where the second job
happens is the first half of the episode.

    filter after    ask for the ten nearest passages in the collection, then
                    throw away the ones from the wrong year
    filter inside   hand Qdrant the year and ask for ten

BOTH SIDES OF THAT PAIR ARE EXACT SEARCHES, so the pair is reproducible down to
the passage id. It is a fact about where a filter runs and not about any index.

Then the filter is pointed at a field that was never declared, which is the
second half. Qdrant builds its index before any question exists, so it can only
make a filtered search cheap for a restriction it was told to expect. On a
collection with no payload index the same question comes back with the same
count of passages from the same count of filers — and some of them are not the
nearest ones, and nothing says so.

AND THAT MISS IS NOT EVEN STABLE. Qdrant assigns graph levels at random and
exposes no seed, so the undeclared side misses a different number on every
build. The run rebuilds it `pin.BUILDS` times and publishes the range rather
than pretending one reading of it is the figure.
"""

from __future__ import annotations

import json
import os
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import ask, index, pin  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "qdrant-data.json"
URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
READING = bool(os.environ.get("READING"))

STOP = set(
    "a an the and or of to in on for with that this it its we our us is are was "
    "were be been by as at from not no if so than then which what who when where "
    "their they them he she all over up out but have has had would could will can "
    "may might do does did any some more most much many".split()
)


def content_words(s: str) -> set[str]:
    import re

    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


def main() -> None:
    client = index.connect(URL)
    print(f"qdrant {pin.QDRANT_VERSION} at {URL}, client {pin.CLIENT_VERSION}")

    rows = index.corpus()
    per_filing = Counter(r["fid"] for r in rows)
    per_year = Counter(r["year"] for r in rows)
    print(f"corpus  {len(rows)} passages from {len(pin.FILINGS)} filings, "
          f"{len(pin.TICKERS)} filers, {len(pin.YEARS)} years")

    model = index.embedder()
    V = index.vectors(model, rows)
    if V.shape != (len(rows), pin.VECTOR_SIZE):
        raise SystemExit(f"vectors are {V.shape} and the corpus is {len(rows)} passages")

    QV = [list(model.query_embed([q]))[0].tolist() for q in pin.QUESTIONS]
    K, YEAR = pin.TOP_K, pin.ASK_YEAR
    yflt = ask.year_filter(YEAR)

    index.load(client, pin.COLLECTION, rows, V, declare=True)
    info = client.get_collection(pin.COLLECTION)
    print(f"index   {info.indexed_vectors_count} of {info.points_count} vectors indexed, "
          f"status {info.status}")

    # ── the pair, exact on both sides ────────────────────────────────────
    pairs = []
    for qv in QV:
        inside = ask.exact_answer(client, qv, yflt, K)
        whole = ask.exact_answer(client, qv, None, K)
        kept = [h for h in whole if h.payload["year"] == YEAR]
        need, thrown = ask.asked_for_exact(client, qv, lambda p: p["year"] == YEAR, K, len(rows))
        pairs.append({
            "after": {"cite": len(kept),
                      "filers": len({h.payload["ticker"] for h in kept}),
                      "asked_for": need, "thrown": thrown},
            "inside": {"cite": len(inside),
                       "filers": len({h.payload["ticker"] for h in inside})},
            "ids": {"inside": [h.id for h in inside], "whole": [h.id for h in whole],
                    "kept": [h.id for h in kept]},
        })
    A = pairs[pin.ANCHOR]
    anchor_inside = ask.exact_answer(client, QV[pin.ANCHOR], yflt, K)
    best = anchor_inside[0]

    qw = content_words(pin.QUESTIONS[pin.ANCHOR])
    pw = content_words(best.payload["text"])
    shared = sorted(qw & pw)

    print(f"  the year {YEAR}: {per_year[YEAR]} passages of {len(rows)}")
    print(f"  the anchor question shares {len(shared)} of its {len(qw)} content words "
          f"with the passage it lands on: {shared}")
    print(f"  filter after  — {A['after']['cite']} to quote from {A['after']['filers']} filers, "
          f"and {A['after']['asked_for']} asked for to get {K}")
    print(f"  filter inside — {A['inside']['cite']} to quote from {A['inside']['filers']} filers")
    print(f"  over all {len(QV)} questions: after quotes "
          f"{[p['after']['cite'] for p in pairs]}")

    # ── the stress: the field nobody declared, over several builds ───────
    declared_miss, undeclared_miss, undeclared_shape = [], [], []
    missed_at = []
    for b in range(pin.BUILDS):
        index.load(client, pin.COLLECTION_UNDECLARED, rows, V, declare=False)
        d = u = 0
        gone = set()
        for j, qv in enumerate(QV):
            want = set(pairs[j]["ids"]["inside"])
            d += len(want - {h.id for h in ask.filter_inside(client, qv, yflt, K)})
            got = ask.filter_inside(client, qv, yflt, K, collection=pin.COLLECTION_UNDECLARED)
            u += len(want - {h.id for h in got})
            gone |= want - {h.id for h in got}
            if j == pin.ANCHOR:
                undeclared_shape.append(
                    (len(got), len({h.payload["ticker"] for h in got}),
                     len(want & {h.id for h in got}))
                )
        declared_miss.append(d)
        undeclared_miss.append(u)
        # WHICH ONES, NOT JUST HOW MANY. The reel draws them, and a different
        # scatter on every build is the whole of what this beat says.
        missed_at.append(sorted(gone))
        print(f"  build {b + 1} of {pin.BUILDS}: declared missed {d} of {K * len(QV)}, "
              f"undeclared missed {u}")

    total = K * len(QV)
    stress = {
        "questions": len(QV), "passages": total, "builds": pin.BUILDS,
        "declared_missed": declared_miss,
        "undeclared_missed": undeclared_miss,
        "declared_max": max(declared_miss),
        "undeclared_min": min(undeclared_miss),
        "undeclared_max": max(undeclared_miss),
        "undeclared_mean": round(stats.mean(undeclared_miss), 2),
        "undeclared_sd": round(stats.stdev(undeclared_miss), 2) if pin.BUILDS > 1 else 0.0,
        "anchor_shape": undeclared_shape,
        "missed_at": missed_at,
        # HOW OFTEN IT COMES BACK SHORT, which is the other way this fails and
        # the only way a caller could notice without checking the answer.
        "short_builds": sum(1 for s in undeclared_shape if s[0] < K),
        "silent_builds": sum(1 for s in undeclared_shape if s[0] == K and s[2] < K),
    }

    # ── the structure the lesson rests on ────────────────────────────────
    # NONE OF THESE TESTS WHETHER A FIGURE CAME OUT LARGE ENOUGH TO LOOK
    # GOOD. They test that the run contains a lesson at all.
    fail = []
    if A["inside"]["cite"] != K:
        fail.append(f"the year filter keeps fewer than {K} passages, so there is nothing to miss")
    if A["after"]["cite"] >= K:
        fail.append("every passage the search returned was from the right year, so "
                    "filtering afterwards cost nothing")
    if A["inside"]["cite"] <= A["after"]["cite"]:
        fail.append("the filter inside the search left the agent no more to quote than "
                    "filtering afterwards, so there is no pair")
    if A["inside"]["filers"] <= A["after"]["filers"]:
        fail.append("both sides reached the same number of filers")
    if A["after"]["asked_for"] <= K:
        fail.append(f"{K} passages were enough to get {K} past the filter")
    if any(p["after"]["cite"] >= K for p in pairs):
        fail.append("one of the questions loses nothing to filtering afterwards, so the "
                    "anchor is not representative")
    if len(shared) >= len(qw) / 2:
        fail.append("the question and the passage share half their words, so a text search "
                    "would have found it and the vector bought nothing")
    if stress["declared_max"] != 0:
        fail.append("the declared side missed a passage, so the pair's own figures are "
                    "not the ones the index reproduces")
    if stress["undeclared_min"] <= 0:
        fail.append("the undeclared side missed nothing on some build, so the shape is "
                    "not stressed")
    if not (pin.UNDECLARED_MIN <= stress["undeclared_min"]
            and stress["undeclared_max"] <= pin.UNDECLARED_MAX):
        fail.append(f"the undeclared side missed {stress['undeclared_min']} to "
                    f"{stress['undeclared_max']} of {total}, outside the "
                    f"{pin.UNDECLARED_MIN} to {pin.UNDECLARED_MAX} envelope src/pin.py names")
    if stress["undeclared_sd"] <= 0:
        fail.append("the undeclared side missed the same number on every build, so the "
                    "claim that the figure moves with the build is not what was measured")
    # THE MISS IS SILENT ON SOME BUILDS AND LOUD ON OTHERS, and an earlier
    # version of this assert demanded it always be silent. One build came back
    # with 98 of 120 missing and FEWER than ten passages for the anchor, which
    # is the response saying something. That is a truer finding than the one it
    # replaced: what you get depends on a build nobody can seed. So what is
    # asserted is that the silent case is real, not that it is the only case.
    if not any(s[0] == K and s[2] < K for s in undeclared_shape):
        fail.append("no build came back with a full ten of which some were the wrong ten, "
                    "so the silent miss the last beat is about never happened")
    if not info.indexed_vectors_count:
        fail.append("no vector is indexed, so every search was exact")
    if fail:
        raise SystemExit("this run holds no lesson:\n  - " + "\n  - ".join(fail))

    figures = {
        "passages": len(rows),
        "year_passages": per_year[YEAR],
        "shared_words": len(shared),
        "question_words": len(qw),
        "after": {k: A["after"][k] for k in ("cite", "filers", "asked_for", "thrown")},
        "inside": {k: A["inside"][k] for k in ("cite", "filers")},
    }
    if READING or pin.FIGURES is None:
        print("\nREADING — nothing written. src/pin.py FIGURES should read:\n")
        print(json.dumps(figures, indent=4, sort_keys=True))
        print(f"\nand the stress envelope held: {stress['undeclared_min']} to "
              f"{stress['undeclared_max']} of {total} over {pin.BUILDS} builds "
              f"(mean {stress['undeclared_mean']}, sd {stress['undeclared_sd']})")
        return

    if figures != pin.FIGURES:
        raise SystemExit(
            "this run does not reproduce the figures src/pin.py publishes:\n"
            f"  pinned {json.dumps(pin.FIGURES, sort_keys=True)}\n"
            f"  ran    {json.dumps(figures, sort_keys=True)}\n"
            "Both sides of the pair are exact searches, so this is not the graph moving. "
            "The corpus, the passages or the model did. Update src/pin.py and README.md "
            "together and say what moved."
        )

    data = {
        "meta": {
            "qdrant": pin.QDRANT_VERSION, "client": pin.CLIENT_VERSION,
            "fastembed": pin.FASTEMBED_VERSION, "model": pin.MODEL_NAME,
            "model_repo": pin.MODEL_REPO, "model_revision": pin.MODEL_REVISION,
            "model_sha256": pin.MODEL_SHA256, "dims": pin.VECTOR_SIZE,
            "distance": pin.DISTANCE, "m": pin.HNSW_M,
            "ef_construct": pin.HNSW_EF_CONSTRUCT, "hnsw_ef": pin.HNSW_EF,
            "indexing_threshold_kb": pin.INDEXING_THRESHOLD_KB,
            "full_scan_threshold_kb": pin.FULL_SCAN_THRESHOLD_KB,
            "k": K, "seed": pin.SEED,
        },
        "corpus": {
            "filings": len(pin.FILINGS), "filers": len(pin.TICKERS),
            "years": list(pin.YEARS), "passages": len(rows),
            "year_passages": per_year[YEAR], "ask_year": YEAR,
        },
        "filings": [
            {"fid": f.fid, "ticker": f.ticker, "year": f.year, "accession": f.accession,
             "sha256": f.sha256, "passages": per_filing[f.fid]}
            for f in pin.FILINGS
        ],
        "anchor": {
            "question": pin.QUESTIONS[pin.ANCHOR],
            "fid": best.payload["fid"],
            "accession": best.payload["accession"],
            "quote": best.payload["text"],
            "score": round(best.score, 4),
            "question_words": sorted(qw),
            "shared": shared,
        },
        "pair": {"after": A["after"], "inside": A["inside"]},
        "stress": stress,
        # ONE COPY OF EVERY NUMBER. A mark's year and filing are derivable from
        # `filings[].passages`, which is the order the points went in, so
        # writing nine thousand of each here would be a second copy of the
        # same fact and a second thing to drift.
        "marks": {
            "year_at": [i for i, r in enumerate(rows) if r["year"] == YEAR],
            "inside": sorted(A["ids"]["inside"]),
            "whole": sorted(A["ids"]["whole"]),
            "kept": sorted(A["ids"]["kept"]),
        },
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
