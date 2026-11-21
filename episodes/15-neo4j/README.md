# Episode 15: Neo4j, and the node everybody touches

The reel is on Instagram, on [@fidetolabs](https://www.instagram.com/fidetolabs/).
This folder is the code it was drawn from. Every figure on the frame came out of
one run of `run.py` against a real server.

## What the episode teaches

A research agent reads filings. A filing names what went wrong in whatever words
that company chose, and two companies hit by the same supplier problem do not
have to agree on the words. One writes *the insolvency of a glass supplier in
Toledo*; another writes *the failure of a glass supplier in Toledo*.

[Neo4j](https://github.com/neo4j/neo4j) is where you put the ruling that those
are one thing. You rule once, as the filing goes in, and store it as a
relationship to a shared node:

```
(:Company)-[:FILED]->(:Filing)-[:MENTIONS {phrase}]->(:Event)
```

The words stay on the filing as well, as a list, so the same question can be
asked both ways against the same records. That is the whole experiment.

Then the episode measures what the shape costs. A node is only worth travelling
through if the thing it names is narrow. One event in this corpus is a theme that
almost every filing names, and the trip that was cheap turns into a crawl over
most of the graph.

## Setup: this one needs a server

**Episodes 13 and 14 ran self-contained. This one does not.** Neo4j is a server
and it has to be up before `run.py` will do anything. Any of these gives you one.

```sh
# Docker, which is what the figures were measured against
docker run --rm -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/fidetolabs15 neo4j:5.26.33
```

- **Neo4j Desktop** — <https://neo4j.com/download/>. Make a local DBMS on
  **5.26.x**, set the password to `fidetolabs15`, start it.
- **The tarball** — <https://neo4j.com/deployment-center/>, community edition,
  5.26.33. It needs a Java 17 or 21 runtime, which is the only other thing you
  install. Then `bin/neo4j-admin dbms set-initial-password fidetolabs15` and
  `bin/neo4j start`.
- **Aura Free** — <https://neo4j.com/cloud/aura/>. It runs, and the figures will
  not match, because a db hit is a property of the server's planner and Aura is
  not on 5.26.33. The pin tells you so rather than printing different numbers
  quietly.

The connection is read from the environment and falls back to the local
defaults, so point it elsewhere like this:

```sh
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=fidetolabs15
```

## Run it

You also need [uv](https://docs.astral.sh/uv/). You don't need a Python: `uv`
reads `requires-python` from `pyproject.toml` and fetches one if yours is older
than 3.13. No API key.

```sh
cd episodes/15-neo4j
uv run run.py              # loads the graph, asks both ways, writes ../neo4j-data.json
uv run pytest tests -q     # 14 tests
```

**That order, not the other way round.** Seven of the tests read
`../neo4j-data.json`, so before `run.py` has written it they skip instead of
failing. The other seven need no server at all and check the corpus.

`run.py` wipes the database and rebuilds it, every time. A graph that is added to
twice holds every relationship twice, and a count of records read over that is a
count nobody can reproduce.

## The pair the reel publishes

One question, asked twice from the same filing with the same words, against the
same records: **which other companies named this?**

| how it is asked | records read | companies found |
| --- | --- | --- |
| matching the words on each filing | 1,444 | 13 |
| following the node and its relationships | **558** | **28** |

The question is asked from `AAFB-2025`, about *the insolvency of a glass supplier
in Toledo*, which 35 filings name. Fourteen of them use that wording and the
other twenty say *failure* instead. So the relationship is both cheaper to follow
and the only one of the two that reaches the filings that worded it differently.
Both sides exclude the company doing the asking, because the question is who
**else** named it.

## And the same question through a theme

The corpus holds one event every company writes down — *higher interest rates*,
on 510 of the 600 filings, against a median incident on 12.

| | records read | companies found |
| --- | --- | --- |
| the same relationship question, through that node | **7,700** | **119 of 119** |

Every other company in the corpus comes back, which answers nothing. That is not
a bug in Neo4j. Making a shared thing a node is what bought the cheap trip, and a
node everything connects to is what it costs. The word for it is **supernode**,
and the fix is the same ruling made more carefully: one node, one thing that went
wrong, one place.

## Records read, not seconds

Every figure above is Neo4j's own `PROFILE` output: a plan tree with a `dbHits`
on every operator, which is the records that operator had to touch. `profiled()`
in `src/graph.py` drains the rows first and then sums it, because a plan's
counters are only complete once the query has finished.

A db hit is the same number on a slow machine and a fast one. A stopwatch would
have measured the laptop the episode was built on.

## What the episode pins, and why the plan is the thing to pin

Episode 13 pins a price series, because its vendor rewrites history. Episode 14
pins an embedding model, because a model name alone doesn't fix which file loads.
Here the figure is a count of records read, and there is a third way it moves
without anybody touching a number.

**A db hit is a property of the plan the planner chose.** The same query, over the
same data, on the same server reads a different number of records if the planner
picks a different operator — an index seek instead of a label scan, or the
expansions in the other order. So `src/pin.py` holds the operator sequence of
both queries, in order, beside the figures:

```
words  ProduceResults OrderedDistinct Filter Expand(All) Filter NodeIndexScan
graph  ProduceResults Sort Distinct Filter Expand(All) Filter Expand(All)
       Filter Expand(All) NodeUniqueIndexSeek
```

It pins the server version and edition, the driver version, the corpus counts and
the filing the question is asked from as well. `run.py` calls `check()` before it
writes anything and refuses a run that misses any of it, naming what moved. Two
tests read the same pin. So a reader gets the numbers on the frame or an error,
never a third thing.

On a different server version expect both the plan and the figures to move. That
is the pin working, not breaking. Update `src/pin.py` and this README together
and say what moved.

## What else refuses to write

- the words found as much as the relationship did, so ruling at write time bought
  nothing
- the relationship read as many records as the scan, so there is no pair
- the trip through the theme is no dearer than the trip through an incident
- the trip through the theme is still cheaper than the scan
- the trip through the theme does not come back with every other company
- the theme is not the most-named event, so it is no hub
- the median incident is in more than a tenth of the filings, so nothing in the
  corpus is specific
- no filing names an event that other filings word differently, so a text match
  and a traversal could not have disagreed

None of those tests whether a figure came out big enough to look good. They test
that the run contains a lesson at all.

## The world is generated, and the reel never says otherwise

`src/world.py` builds the corpus from one seed: 120 companies, five years, 600
filings, and the 289 events those filings name, each with more than one wording.
Nothing here is presented as a measurement of a real market.

What has to be real is the **store** and the measurement of it — that a question
has a cost, that the cost is the planner's own count, and that two filings can
name one event in different words. That last one is the reason the whole lesson
holds, so `run.py` refuses a corpus without it.

## What is not committed

`run.py` writes `../neo4j-data.json`, which is the file the reel's build reads.
It is in the repo's `.gitignore`, along with `__pycache__` and the virtual
environment `uv` makes. The graph itself lives in your server, not in this
folder, so there is nothing else to leave behind.

## Files

| | |
| --- | --- |
| `src/world.py` | the companies, the filings, the events, and the words used for them |
| `src/graph.py` | the connection, the load, and the one question asked two ways |
| `src/pin.py` | the versions, the corpus, the figures and the plans they rest on |
| `run.py` | runs it, asserts the structure, checks the pin, writes the data |
