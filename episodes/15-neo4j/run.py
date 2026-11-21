"""Builds the graph, asks the one question both ways, and writes neo4j-data.json.

THERE IS ONE COPY OF EVERY FIGURE. This file writes them and `build-html.mjs`
reads them, so no number on the reel's frame was typed in by hand.

WHAT IS REAL HERE. A Neo4j server, a loaded graph of companies, filings and the
events their filings name, and Neo4j's own profiler counting the records each
query had to read. The corpus is generated from one seed and the reel never
says otherwise; the measurement is of the store.

Run: uv run run.py      (with a server up -- see README.md)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import graph as G                     # noqa: E402
from src import pin as PIN                     # noqa: E402
from src import world as W                     # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "neo4j-data.json"


def structure(world, corpus, degrees, pair, theme):
    """The structure the lesson needs, asserted. Never a result the reel wants.

    Each one of these is a thing the reel's drawing claims. None of them tests
    whether a figure came out large enough to look good: they test that the run
    contains a lesson at all.
    """
    bad = []
    incident = [d for d in degrees if d["kind"] == "incident"]
    themes = [d for d in degrees if d["kind"] == "theme"]

    if len(themes) != 1:
        bad.append("the corpus holds no single theme, so there is no hub to measure")
    if not incident:
        bad.append("the corpus holds no incidents")
    elif themes and themes[0]["degree"] <= max(d["degree"] for d in incident):
        bad.append("the theme is not the most-named event, so it is not a hub")
    # AN INCIDENT IS NAMED BY A HANDFUL OF FILINGS. An event half the corpus
    # names is a theme whatever it is called, and a corpus of those has no
    # specific event in it for the first half of the reel to be about.
    if incident:
        med = sorted(d["degree"] for d in incident)[len(incident) // 2]
        if med > corpus["filings"] * 0.1:
            bad.append(f"the median incident is in {med} of {corpus['filings']} filings; "
                       "nothing in this corpus is specific")

    # THE SAME EVENT IS WORDED MORE THAN ONE WAY, which is the only reason a
    # text match and a traversal can disagree. Without it the reel is false.
    if pair["words"]["found"] >= pair["graph"]["found"]:
        bad.append("matching the words found as much as the relationship did, "
                   "so resolving the event at write time bought nothing")
    # AND THE RELATIONSHIP IS THE CHEAPER WAY TO ASK, which is the claim the
    # first pair makes.
    if pair["graph"]["hits"] >= pair["words"]["hits"]:
        bad.append("the relationship read as many records as the scan, so there is no pair")
    # THE STRESS IS THE SAME QUERY COSTING MORE AT A HUB, and it has to cost
    # more than the thing it replaced or it is not a stress.
    if theme["graph"]["hits"] <= pair["graph"]["hits"]:
        bad.append("the walk at the theme is no dearer than the walk at an incident")
    if theme["graph"]["hits"] <= theme["words"]["hits"]:
        bad.append("the walk at the theme is still cheaper than the scan, so the "
                   "shape costs nothing")
    # AND THE ANSWER IT GIVES BACK THERE IS USELESS, which is the other half.
    # EVERY OTHER COMPANY, because the question is who ELSE named it and both
    # sides exclude the company doing the asking.
    if theme["graph"]["found"] < len(world["companies"]) - 1:
        bad.append("the walk at the theme does not return every other company, so the "
                   "answer is still an answer")
    # THE TWO SIDES ARE THE SAME QUESTION. Different inputs would make the pair
    # two questions with one label.
    if pair["words"]["phrase"] != pair["graph"]["phrase"]:
        bad.append("the two sides were handed different words")
    return bad


def marks(world, anchor, theme_target, pair, theme):
    """Which filing is which, so the reel draws the run rather than a sketch.

    `build-html.mjs` draws one mark a filing and one a company, and the sets
    below are what it lights. They are index lists into `order`, which is the
    order the corpus was written in, so the picture on the frame is the store's
    own layout and not an arrangement.
    """
    order = [f.fid for f in world["filings"]]
    tickers = [c["ticker"] for c in world["companies"]]
    at = {fid: i for i, fid in enumerate(order)}

    same, other, in_theme = [], [], []
    for f in world["filings"]:
        for name, phrase in zip(f.mentions, f.phrases):
            if name == theme_target["event"]:
                in_theme.append(at[f.fid])
            if name != anchor["event"] or f.fid == anchor["fid"]:
                continue
            (same if phrase == anchor["phrase"] else other).append(at[f.fid])

    anchor_filing = next(f for f in world["filings"] if f.fid == anchor["fid"])
    return {
        "order": order,
        "tickers": tickers,
        "anchor_at": at[anchor["fid"]],
        "anchor_same": sorted(same),
        "anchor_other": sorted(other),
        "theme_at": at[theme_target["fid"]],
        "theme_filings": sorted(set(in_theme)),
        # the anchor filing, opened up: what it names, in the words it used
        "anchor_phrases": list(anchor_filing.phrases),
        # the companies each side came back with, as index lists into `tickers`
        "words_companies": sorted(tickers.index(t) for t in pair["words"]["tickers"]),
        "graph_companies": sorted(tickers.index(t) for t in pair["graph"]["tickers"]),
        "theme_companies": sorted(tickers.index(t) for t in theme["graph"]["tickers"]),
    }


def main():
    world = W.build()
    anchor, theme_target = W.anchor(world), W.theme_filing(world)

    drv = G.driver()
    try:
        versions = G.versions(drv)
        print(f"neo4j {versions['neo4j']} {versions['edition']}, driver {versions['driver']}")
        corpus = G.load(drv, world)
        print(f"loaded {corpus['companies']} companies, {corpus['filings']} filings, "
              f"{corpus['events']} events, {corpus['mentions']} mentions")

        degrees = G.degrees(drv)
        pair = G.ask_both(drv, anchor)
        theme = G.ask_both(drv, theme_target)
    finally:
        drv.close()

    for side in (pair, theme):
        side["words"]["phrase"] = side["phrase"]
        side["graph"]["phrase"] = side["phrase"]

    bad = structure(world, corpus, degrees, pair, theme)
    if bad:
        raise SystemExit("the run holds no lesson:\n  " + "\n  ".join(bad))

    moved = PIN.check(versions, corpus, pair, theme)
    if moved:
        raise SystemExit(
            "this run does not print the figures the episode published, and it will not "
            "write over them:\n  " + "\n  ".join(moved) +
            "\n\nIf the change is deliberate, update src/pin.py and README.md together "
            "and say what moved.")

    data = {
        "meta": {
            "neo4j": versions["neo4j"], "edition": versions["edition"],
            "driver": versions["driver"], "seed": W.SEED,
            "years": W.YEARS, "sectors": len(W.SECTORS),
            "kinds": len(W.KINDS), "places": len(W.PLACES),
            "per_filing": list(W.EVENTS_PER_FILING),
        },
        "corpus": corpus,
        "degrees": {
            "theme": next(d["degree"] for d in degrees if d["kind"] == "theme"),
            "incident_max": max(d["degree"] for d in degrees if d["kind"] == "incident"),
            "incident_median": sorted(d["degree"] for d in degrees
                                      if d["kind"] == "incident")[
                len([d for d in degrees if d["kind"] == "incident"]) // 2],
            "anchor": next(d["degree"] for d in degrees if d["name"] == anchor["event"]),
        },
        "anchor": anchor,
        "other_phrase": PIN.ANCHOR["other_phrase"],
        "pair": pair,
        "theme": theme,
        "plans": {"words": pair["words"]["ops"], "graph": pair["graph"]["ops"]},
        "marks": marks(world, anchor, theme_target, pair, theme),
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n")

    print()
    print(f"the question is asked from {anchor['fid']}, about")
    print(f'  "{anchor["event"]}"')
    print(f'  which {data["degrees"]["anchor"]} filings name, and which others also word')
    print(f'  "{PIN.ANCHOR["other_phrase"]}"')
    print()
    print("                             records read   companies found")
    print(f"  as words on the filing       {pair['words']['hits']:>8}   {pair['words']['found']:>8}")
    print(f"  as a node and a relationship {pair['graph']['hits']:>8}   {pair['graph']['found']:>8}")
    print()
    print(f"the theme \"{theme['event']}\" is on {data['degrees']['theme']} filings, "
          f"against a median incident on {data['degrees']['incident_median']}")
    print(f"  the same relationship question {theme['graph']['hits']:>6}   "
          f"{theme['graph']['found']:>8}")
    print()
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
