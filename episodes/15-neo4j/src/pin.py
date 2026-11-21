"""What the published figures rest on, held here, and checked before anything is written.

EVERY EPISODE HAS FOUND ONE THING THAT WOULD HAVE CHANGED ITS NUMBERS IN
SILENCE. Episode 13's prices were rewritten by its vendor at every dividend.
Episode 14's scores came out of an embedding model a name alone does not fix,
so it pinned the repository file and the dimensions. This episode's figure is a
db-hit count, and there are three ways it moves without anybody touching a
number:

  1. THE SERVER. A db hit is the store's own count of records read, and which
     records a query reads is a decision Neo4j's planner makes. A different
     server version plans differently.
  2. THE DRIVER. It is what asks for the profile and what the counters come
     back through.
  3. THE PLAN ITSELF. This is the one nobody would think to check. The same
     query, the same data and the same server still read a different number of
     records if the planner picks a different operator -- an index seek instead
     of a label scan, or the expansions in the other order. The figure is a
     property of the chosen plan, so the CHOSEN PLAN is pinned beside the
     figure.

So the operator sequence of all three profiled queries is held here, in order,
and `run.py` refuses to write when one of them moves. A run that prints
different figures and says nothing is worse than one that stops.

A FOURTH THING, AND IT IS THE CORPUS. The figures are counts over a generated
corpus, so the seed, the counts and the chosen anchor are pinned too. Changing
`SEED` or `EVENTS_PER_FILING` in `world.py` is allowed; shipping the change
without re-reading the figures is not.
"""

# ── the versions this episode was measured on ────────────────────────────────
NEO4J = "5.26.33"
EDITION = "community"
DRIVER = "6.4.1"

# ── the corpus that came out of the seed ─────────────────────────────────────
CORPUS = {"companies": 120, "filings": 600, "events": 289, "mentions": 5308}

# ── the filing the question is asked from, and the event it names ────────────
ANCHOR = {
    "fid": "AAFB-2025",
    "event": "the insolvency of a glass supplier in Toledo",
    "phrase": "the insolvency of a glass supplier in Toledo",
    "other_phrase": "the failure of a glass supplier in Toledo",
    "degree": 35,
}
THEME = {"fid": "AAFB-2025", "event": "higher interest rates", "degree": 510}

# ── THE PUBLISHED FIGURES. These are the four on the reel's frames. ──────────
FIGURES = {
    # the same question, from the same filing, over the same records
    "words_hits": 1444, "words_found": 13,
    "graph_hits": 558, "graph_found": 28,
    # and the same relationship question, asked where the node is a theme
    "theme_graph_hits": 7700, "theme_graph_found": 119,
    # the scan at the theme, which is what the walk there is worse than
    "theme_words_hits": 1603, "theme_words_found": 99,
}

# ── THE PLANS. The figure is a property of these. ────────────────────────────
PLANS = {
    "words": ["ProduceResults", "OrderedDistinct", "Filter", "Expand(All)",
              "Filter", "NodeIndexScan"],
    "graph": ["ProduceResults", "Sort", "Distinct", "Filter", "Expand(All)",
              "Filter", "Expand(All)", "Filter", "Expand(All)", "NodeUniqueIndexSeek"],
}


def check(versions, corpus, pair, theme):
    """Refuses to write rather than printing figures the README does not have.

    Returns the list of everything that moved. An empty list is the only thing
    `run.py` writes on.
    """
    moved = []

    def cmp(what, want, got):
        if want != got:
            moved.append(f"{what}: pinned {want!r}, this run {got!r}")

    cmp("neo4j", NEO4J, versions["neo4j"])
    cmp("edition", EDITION, versions["edition"])
    cmp("driver", DRIVER, versions["driver"])
    for k, v in CORPUS.items():
        cmp(f"corpus.{k}", v, corpus[k])
    cmp("anchor.fid", ANCHOR["fid"], pair["fid"])
    cmp("anchor.event", ANCHOR["event"], pair["event"])
    cmp("anchor.phrase", ANCHOR["phrase"], pair["phrase"])
    cmp("theme.fid", THEME["fid"], theme["fid"])
    cmp("theme.event", THEME["event"], theme["event"])

    cmp("words_hits", FIGURES["words_hits"], pair["words"]["hits"])
    cmp("words_found", FIGURES["words_found"], pair["words"]["found"])
    cmp("graph_hits", FIGURES["graph_hits"], pair["graph"]["hits"])
    cmp("graph_found", FIGURES["graph_found"], pair["graph"]["found"])
    cmp("theme_graph_hits", FIGURES["theme_graph_hits"], theme["graph"]["hits"])
    cmp("theme_graph_found", FIGURES["theme_graph_found"], theme["graph"]["found"])
    cmp("theme_words_hits", FIGURES["theme_words_hits"], theme["words"]["hits"])
    cmp("theme_words_found", FIGURES["theme_words_found"], theme["words"]["found"])

    # THE PLAN, WHICH IS WHAT THE FIGURE IS A PROPERTY OF. Checked on both
    # anchors, because the same two queries run at both.
    cmp("plan.words", PLANS["words"], pair["words"]["ops"])
    cmp("plan.graph", PLANS["graph"], pair["graph"]["ops"])
    cmp("plan.words at the theme", PLANS["words"], theme["words"]["ops"])
    cmp("plan.graph at the theme", PLANS["graph"], theme["graph"]["ops"])
    return moved
