"""The graph: one Neo4j database holding the same facts two ways.

THE WHOLE EPISODE IS ONE DECISION, MADE AT WRITE TIME. A filing names an event
in its own words. You can keep those words on the filing, or you can decide
once that two wordings are the same event and store that decision as a
relationship. Both are in this database, built from one corpus in one load, so
the same question can be asked both ways against the same records.

  the words      (:Filing {phrases: [...]})          -- what the filing said
  the decision   (:Filing)-[:MENTIONS]->(:Event)     -- what it was about

Nothing else differs. `(:Company)-[:FILED]->(:Filing)` is a relationship on
both sides, because it is not what is under test.

WHAT IS MEASURED IS NEO4J'S OWN PROFILER. `PROFILE` returns a plan tree with a
`dbHits` on every operator -- the records that operator had to read -- and
`profiled()` below sums it. A db hit is the store's own count, not a stopwatch:
it is the same number on a slow machine and a fast one, which is why it is the
figure the reel prints.

AND IT IS A PROPERTY OF THE PLANNER, SO THE PLANNER IS PINNED TOO. The same
query over the same data reads a different number of records if the planner
picks a different operator -- an index seek instead of a label scan, a
different expand order. So `src/pin.py` holds the operator sequence beside the
figures, and the run refuses to write when either moves. That is this
episode's version of the thing every episode has had to find: the number rests
on something nobody would think to check.
"""

import os

# The server this run talks to. Overridable, because a reader's server is
# their own; the defaults are what the episode's README tells them to start.
URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "fidetolabs15")
DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


def driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def versions(drv):
    """The server and the driver, read back rather than assumed."""
    import neo4j

    with drv.session(database=DATABASE) as s:
        row = s.run("CALL dbms.components() YIELD name, versions, edition "
                    "RETURN versions[0] AS version, edition").single()
    return {"neo4j": row["version"], "edition": row["edition"], "driver": neo4j.__version__}


# ── loading ──────────────────────────────────────────────────────────────────
#
# WIPED AND REBUILT EVERY RUN. A graph that is added to twice has every
# relationship twice, and a db-hit count over it is a count over a store
# nobody can reproduce.

SCHEMA = [
    "CREATE CONSTRAINT company_ticker IF NOT EXISTS "
    "FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
    "CREATE CONSTRAINT filing_fid IF NOT EXISTS "
    "FOR (f:Filing) REQUIRE f.fid IS UNIQUE",
    "CREATE CONSTRAINT event_name IF NOT EXISTS "
    "FOR (e:Event) REQUIRE e.name IS UNIQUE",
]


def load(drv, world):
    """Writes the corpus, both ways, and returns what went in."""
    from src import world as W

    rows = W.mentions(world)
    with drv.session(database=DATABASE) as s:
        s.run("MATCH (n) DETACH DELETE n").consume()
        for stmt in SCHEMA:
            s.run(stmt).consume()
        s.run("UNWIND $rows AS r CREATE (:Company {ticker: r.ticker, sector: r.sector})",
              rows=world["companies"]).consume()
        s.run("UNWIND $rows AS r CREATE (:Event {name: r.name, kind: r.kind})",
              rows=[{"name": e["name"], "kind": e["kind"]} for e in world["events"]]).consume()
        # THE FILING CARRIES THE WORDS IT USED. That list is the whole of the
        # no-relationship representation, and it is what the first query reads.
        s.run("""
            UNWIND $rows AS r
            MATCH (c:Company {ticker: r.ticker})
            CREATE (f:Filing {fid: r.fid, year: r.year, phrases: r.phrases})
            CREATE (c)-[:FILED]->(f)
        """, rows=[{"fid": f.fid, "ticker": f.ticker, "year": f.year,
                    "phrases": list(f.phrases)} for f in world["filings"]]).consume()
        # AND THE DECISION, AS A RELATIONSHIP. One edge a mention, carrying the
        # words that mention used, so nothing is thrown away by resolving it.
        s.run("""
            UNWIND $rows AS r
            MATCH (f:Filing {fid: r.fid})
            MATCH (e:Event {name: r.event})
            CREATE (f)-[:MENTIONS {phrase: r.phrase}]->(e)
        """, rows=rows).consume()
        counts = s.run("""
            MATCH (c:Company) WITH count(c) AS companies
            MATCH (f:Filing) WITH companies, count(f) AS filings
            MATCH (e:Event) WITH companies, filings, count(e) AS events
            MATCH ()-[m:MENTIONS]->() RETURN companies, filings, events, count(m) AS mentions
        """).single()
    return dict(counts)


# ── the question, asked two ways ─────────────────────────────────────────────
#
# ONE QUESTION: starting from this filing, which other companies named the same
# event? It is the question a research agent asks all day, and it is the one a
# graph exists for.

ASK_WORDS = """
PROFILE
MATCH (o:Company)-[:FILED]->(g:Filing)
WHERE $phrase IN g.phrases AND o.ticker <> $ticker
RETURN DISTINCT o.ticker AS ticker
ORDER BY ticker
"""

# BOTH QUERIES GET THE SAME INPUT, and that is what makes the pair a pair: the
# filing, and the words that filing used. One matches those words against every
# filing's text. The other hands them to the relationship, which already knows
# what they were about, and walks out from there.
ASK_GRAPH = """
PROFILE
MATCH (:Filing {fid: $fid})-[:MENTIONS {phrase: $phrase}]->(e:Event)
MATCH (e)<-[:MENTIONS]-(g:Filing)<-[:FILED]-(o:Company)
WHERE o.ticker <> $ticker
RETURN DISTINCT o.ticker AS ticker
ORDER BY ticker
"""


def _walk(plan, hits, ops):
    hits[0] += plan.get("dbHits", 0) or 0
    ops.append(plan.get("operatorType", "?").split("@")[0])
    for child in plan.get("children", []) or []:
        _walk(child, hits, ops)


def profiled(drv, cypher, **params):
    """Runs a PROFILE'd query and returns its rows, its db hits and its plan.

    THE ROWS ARE DRAINED BEFORE THE PROFILE IS READ. A plan's counters are only
    complete once the query has finished, and a lazily-consumed result reports
    the hits of however much of it was pulled.
    """
    with drv.session(database=DATABASE) as s:
        res = s.run(cypher, **params)
        rows = [dict(r) for r in res]
        summary = res.consume()
    hits, ops = [0], []
    _walk(summary.profile, hits, ops)
    return {"rows": rows, "hits": hits[0], "ops": ops}


def ask_both(drv, target):
    """The same question, over the words and over the relationships."""
    args = {"fid": target["fid"], "phrase": target["phrase"], "ticker": target["ticker"]}
    words = profiled(drv, ASK_WORDS, **args)
    graph = profiled(drv, ASK_GRAPH, **args)
    return {
        "fid": target["fid"], "ticker": target["ticker"], "event": target["event"],
        "phrase": target["phrase"],
        "words": {"hits": words["hits"], "found": len(words["rows"]),
                  "tickers": [r["ticker"] for r in words["rows"]], "ops": words["ops"]},
        "graph": {"hits": graph["hits"], "found": len(graph["rows"]),
                  "tickers": [r["ticker"] for r in graph["rows"]], "ops": graph["ops"]},
    }


def degrees(drv):
    """How many filings name each event. The hub is whatever this says it is."""
    with drv.session(database=DATABASE) as s:
        rows = s.run("""
            MATCH (e:Event)<-[m:MENTIONS]-(:Filing)
            RETURN e.name AS name, e.kind AS kind, count(m) AS degree
            ORDER BY degree DESC
        """)
        return [dict(r) for r in rows]
