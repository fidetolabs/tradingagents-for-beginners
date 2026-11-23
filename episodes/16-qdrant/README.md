# the run under episode #16

Twenty-four 10-K documents from EDGAR, cut into 9,085 passages, loaded into one
Qdrant collection, and twelve research questions asked of it. `run.py` writes
`../qdrant-data.json`; `build-html.mjs` reads that file, so **there is one copy
of every figure on the frame.**

## Setup: you need a server, and local mode is not one

```sh
# Docker, if you have a container runtime
docker run --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.20.2
```

- **The binary** — <https://github.com/qdrant/qdrant/releases/tag/v1.20.2>.
  One file, no runtime to install. This is how the figures here were measured,
  because this machine had no container runtime:

  ```sh
  curl -L -o qdrant.tar.gz \
    https://github.com/qdrant/qdrant/releases/download/v1.20.2/qdrant-aarch64-apple-darwin.tar.gz
  # sha256 385b244e6283968b2be926358541b5bc19b9ada7ea9586e900684388bc912d5f
  tar xzf qdrant.tar.gz
  QDRANT__SERVICE__HOST=127.0.0.1 ./qdrant
  ```

- **Qdrant Cloud** — works, and expect the figures in the stress section to
  move. See the pin.

## Setup: EDGAR wants to know who is asking

The filings come from EDGAR, which asks for a real contact in the User-Agent.
It is yours, not this repo's, and there is no default: nothing downloads without
it.

```sh
export EDGAR_CONTACT="Your Name (you@example.com)"
export QDRANT_URL=http://localhost:6333
uv run run.py
uv run pytest tests -q
```

**`QdrantClient(path=…)` and `QdrantClient(":memory:")` are refused outright,
and that refusal is the most important line of code here.** Local mode is a
Python reimplementation in `qdrant_client/local/`, and its own source says so:
`_has_ignored_search_params` in `local/qdrant_local.py` lists `hnsw_ef`,
`quantization`, `acorn`, `exact` and `indexed_only` as parameters it ignores,
with the comment *"index behaviour that an exact brute-force search has no use
for"*, and `local/local_collection.py` scores a query by calling
`calculate_distance` over every vector in the collection. There is no graph in
it. The last beat of this reel is about what a graph gets wrong, so a run
against local mode would have reported a perfect score and taught the opposite
of what is true.

`run.py` rebuilds both collections from scratch, every time. A collection added
to twice holds every passage twice, and a figure over that is a figure nobody
can reproduce.

## What it does

**It fetches twenty-four real filings**, over your own `EDGAR_CONTACT`, one a
second, cached under `.cache/edgar/`. Six semiconductor filers — INTC, MU,
TXN, ADI, ON and QRVO — four years each, 2020 through 2023, by accession
number. One industry on purpose: the passages are then genuinely close to each
other in meaning, which is what gives a text match something to get wrong.

**No dataset is committed.** `src/pin.py` holds each accession and the sha256
of its primary document, and `src/edgar.py` refuses to continue if EDGAR hands
back anything else. An EDGAR accession points at an immutable archive, so this
is the one input in this series that cannot be rewritten underneath a run —
#13's prices were rewritten by a vendor at every dividend and #14's model name
resolved to a different file.

**It cuts each filing into quotable passages** on sentence boundaries, about
900 characters, dropping the table residue that has no sentence in it. 9,085
passages, 187 to 513 a filing.

**It puts the meaning and the facts on the same point.** The vector is what the
passage means; the payload is which filing, which filer, which year. That is
the whole architecture of the episode: the vector is how anything is found, and
the payload is how the search is held to what the agent may quote.

**Then it asks one question two ways.** The question is in a reader's own words,
not the filings': *we could not get enough substrate and assembly capacity from
our outside suppliers, so we were unable to ship everything our customers
ordered.* It shares **two of its twelve content words** with the passage it
lands on, which is the vector doing the work a text match could not.

The agent is asking about one year, 2022, which is 2,222 of the 9,085 passages.

|  | passages it can quote | filers they come from |
| --- | --- | --- |
| filter after the search | 3 | 2 |
| filter inside the search | **10** | **5** |

Both sides are **exact** searches, so this pair is a fact about where a filter
runs and nothing to do with any index. It reproduces down to the passage id.
And the first side does not have to give up: to get ten from 2022 by filtering
afterwards you must ask for **36** passages and throw **26** away.

## The stress, and why it is published as a range

**Then the same filter is pointed at a field nobody declared.** A second
collection holds the same points and the same vectors, with one difference: no
payload index. Qdrant builds its index before any question exists, so it can
only plan a cheap filtered search for a restriction it was told to expect.

What comes back is the same shape of answer. Ten passages, three to five filers,
and some of them are not the nearest ones — with nothing in the response saying
so. For a retrieval-grounded agent that is the dangerous failure: it quotes ten
real passages and they are the wrong ten.

**And the size of the miss is not reproducible, because the graph build is
not.** Qdrant assigns each point its level in the graph at random and exposes
no seed, so the same vectors build two different graphs. Over eight builds of
the same collection, the twelve questions missed:

```
declared      0  0  0  0  0  0  0  0   of 120
undeclared   32 24 37 35 27 43 49 64   of 120
```

So this episode publishes **0 of 120 against 24 to 64 over eight builds**, and
the frame prints the range with the build count beside it. A single figure here
would be a figure nobody could reproduce, and `src/pin.py` holds an envelope
rather than a number: `UNDECLARED_MIN` and `UNDECLARED_MAX`. A run that leaves
it stops.

**The declared side is 0 on every build**, which is what makes the pair's own
figures the ones the index actually returns.

## The pin

`src/pin.py` holds five things that could move a figure here without telling
you, and the first one would have made the whole reel fiction:

1. **local mode is not the engine** — refused in `src/index.py`
2. **the indexing threshold** — an unindexed segment is searched exactly, so
   it is set explicitly and the run waits for Qdrant's own `green`
3. **the embedding model is not the name you asked for** — `fastembed` turns
   `BAAI/bge-small-en-v1.5` into `model_optimized.onnx` from the Hugging Face
   repo `Qdrant/bge-small-en-v1.5-onnx-Q`, so the revision and the file's
   sha256 are pinned and the run hashes what it loaded
4. **the search parameters** — `m`, `ef_construct`, `hnsw_ef` and
   `full_scan_threshold`, all written down. `full_scan_threshold` matters most:
   left on its 20,000 KB default, a narrow filter is read exactly and comes back
   perfect
5. **the graph build is seedless** — the section above

On a different server version, expect the stress range to move and the pair not
to. Update `src/pin.py` and this README together and say what moved.

## What refuses to write

- the year filter keeps fewer than ten passages, so there is nothing to miss
- every passage the search returned was from the right year, so filtering
  afterwards cost nothing
- the filter inside the search left the agent no more to quote
- both sides reached the same number of filers
- ten passages were enough to get ten past the filter
- one of the twelve questions loses nothing to filtering afterwards, so the
  anchor is not representative
- the question and the passage share half their words, so a text search would
  have found it and the vector bought nothing
- the declared side missed a passage
- the undeclared side missed nothing on some build
- the undeclared side missed the same number on every build, so the claim that
  the figure moves with the build is not what was measured
- the undeclared side returned fewer passages, so the miss is not silent
- the undeclared range left the envelope
- no vector is indexed, so every search was exact

None of those tests whether a figure came out large enough to look good. They
test that the run contains a lesson at all. The first stress this episode tried
was refused by two of them: a filter narrowed to one filing cost no recall,
because the payload index was declared and Qdrant handled it. That refusal is
what found the real one.

## Files

| | |
| --- | --- |
| `src/pin.py` | the versions, the accessions and their hashes, the search parameters, the figures |
| `src/edgar.py` | the fetch, the hash check, the text, the passages |
| `src/index.py` | the model and its hash, the collection, the payload index, the wait for `green` |
| `src/ask.py` | the question asked two ways, and where the tenth match sits |
| `run.py` | runs it, asserts the structure, checks the pin, writes the data |
