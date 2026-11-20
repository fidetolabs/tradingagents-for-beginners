# Episode 14: Mem0 and what a memory drops

The reel is on Instagram, on [@fidetolabs](https://www.instagram.com/fidetolabs/).
This folder is the code it was drawn from. Every figure on the frame came out of
one run of `run.py`.

## What the episode teaches

[Mem0](https://github.com/mem0ai/mem0) gives an agent memory that outlives one
session. The use here is the plainest one a builder has: write down the
experiments that failed, so the next session doesn't run them again.

It works, and the episode measures what it costs. A memory is shorter than the
run that produced it, and the first thing a short sentence drops is the scope:
where the thing was tried, and when. Drop that and the memory still catches
every repeat, and it also blocks six tries it had never run.

**No language model is called anywhere in it.** `add(..., infer=False)` is Mem0's
own switch for storing a memory as written instead of having a model pull a fact
out of it, and `search` is embeddings and a vector store. So every score is a
function of the text and the embedder, and two runs give the same answer. A model
in the middle would make every number on the frame a sample instead of a
measurement.

## Run it

You need [uv](https://docs.astral.sh/uv/). You don't need a Python: `uv` reads
`requires-python` from `pyproject.toml` and fetches one if yours is older than
3.13. No API key.

```sh
cd episodes/14-mem0
uv run run.py              # both sessions, then writes ../mem0-data.json
uv run pytest tests -q     # 18 tests
```

**That order, not the other way round.** Six of the tests read
`../mem0-data.json`, so before `run.py` has written it they skip instead of
failing.

**The first run needs a network, and later ones don't.** `fastembed` downloads
the embedder on first use: five files, about 67 MB, from the Hugging Face repo
`Qdrant/bge-small-en-v1.5-onnx-Q`. It caches them under `~/.cache/huggingface`,
outside this repo, so nothing it writes can be committed and the second run is
offline.

**There is no fetch script, because there is no outside data.** The repo's rule
is that no dataset is committed and an episode ships a pinned fetcher instead.
This episode reads nothing from a vendor. The world it searches is generated
from one seed in `src/world.py`, and the reel says so on the frame. So the rule
doesn't bite here, and inventing a fetch step to match episode 13's shape would
be theatre.

## What you should see

```
phase 1 — session one, its own process
  session one (pid 69428): 24 tries, 12 failed, 12 memories written with their conditions and 4 without, 0 model calls
phase 2 — session two, a new process with nothing in it
  session two (pid 69432): 10 tries put forward, each asked of both stores, 0 model calls

mem0 2.2.1, BAAI/bge-small-en-v1.5 at 384 dims, 0 model calls
session one: 24 tries, 12 failed
  written with their conditions: 12 rows
  written without them:          4 rows
session two: 10 put forward, 3 already failed, 4 never tried (4 of them good enough)
  with the conditions: catches 3, also blocks 0, runs 7
  without them:        catches 3, also blocks 6, runs 1
wrote mem0-data.json
```

The two pids are the only thing that changes between runs. Everything else is
the same on every run, and `run.py` refuses to write a run where it isn't.

## The pair the reel publishes

Ten tries put forward. Three of them are things session one already ran and
already failed.

| the memory holds | runs it repeats | tries it still runs |
| --- | --- | --- |
| nothing | 3 | 10 |
| where and when it was tried | 0 | 7 |
| that dropped | 0 | 1 |

With no memory you run all ten and three are work you already did. With the
scoped memory you repeat nothing and still get to run seven. With the scope
dropped you repeat nothing and get to run one, and four of the six it blocked
were good enough to be worth running.

## The threshold is not a dial that was turned until a number looked good

A builder with a store of failures wants one thing from it: don't run something
you already know fails. So the threshold is the lowest one that still catches
every repeat, full recall, and the only question left is what else comes with it.
That quantity is defined by the scores rather than chosen.

At full recall the scoped store cuts at 0.9042 and the bare store at 0.359. The
shorter sentence has no scope to tell two tries apart on, so its scores don't
separate the repeats from anything else. One try it had never run scores
**higher** than a true repeat. That isn't a bug in Mem0. It's what a shorter
sentence does.

## What the episode pins, and why the model is the thing to pin

Episode 13 pins a price series, because its vendor rewrites history. Here the
scores are cosine similarities out of one embedding model, so the model is the
measurement. A different embedder is a different set of numbers.

`src/mem.py` holds the pin. The model name on its own doesn't pin it: `fastembed`
maps a name to a repository and a file, and another release could map the same
name somewhere else. So the pin holds the name, the repository, the file, the
dimension count and the `fastembed` version, and `check_embedder()` reads all of
them back off `fastembed` before the run starts.

`PIN` in the same file holds what the episode published: the try and failure
counts, the row counts, the repeats, and the two thresholds. `run.py` calls
`check_pin()` before it writes anything and refuses a run that misses any of it,
naming what moved. Two tests read the same pin. So a reader gets the numbers on
the frame or an error, never a third thing.

## What else refuses to write

- a language model was reached
- both sessions ran in one process
- dropping the conditions merged nothing
- the shorter memory blocked no more than the scoped one
- nothing put forward had already failed
- none of the new tries was good enough
- no way of picking both passes and fails in this world

## How a try is written down, and how it is asked for

A try is four things: what you rank companies on, how many of the top ones you
keep, which industry, and over which years.

Session one writes a failure as `ranking on cash yield and keeping 20 was not
good enough, in software, 2024 to 2026`. Session two asks for the same thing in
its own words: `try cash yield across software over 2024 to 2026, holding the top
20`. **The two are never the same string**, which is why a log file keyed on text
finds nothing and a memory searched by meaning finds something. One test checks
they stay different.

The short form drops the tail: `ranking on cash yield was not good enough`.
Written that way session one's twelve failures are no longer twelve different
sentences, so the store holds four.

## Why the sessions are separate processes

An agent's session ends and everything it held goes with it. What the next one
knows is whatever reached the store. `run.py` spawns both as child processes, so
session two starts with nothing but Qdrant on disk. Two function calls in one
process would have shared a heap and measured nothing. The two pids are in the
output and in `mem0-data.json`, and a test asserts they differ.

## The world is generated, and the reel never says otherwise

`src/world.py` builds it from one seed: quality per company per industry per
window, how well each metric sees that quality, how much it paid, and the
returns. Nothing here is presented as a measurement of a real market.

What has to be real is the search. A try has a score, a score can be under the
bar, and **the same way of picking passes in one place and fails in another**.
That last one is the reason a memory needs its scope at all, so `run.py` refuses
to write a world without one and a test checks it too.

## Files

| path | what it is |
| --- | --- |
| `pyproject.toml` | the dependencies, and `requires-python = ">=3.13"` |
| `uv.lock` | the exact versions the numbers came out of. Committed on purpose: the embedder version is the measurement |
| `run.py` | spawns both sessions, reads the decision off the scores, checks the pin, writes the data |
| `src/mem.py` | the Mem0 configuration, the model pin, the published numbers, and the call counter that must stay at zero |
| `src/world.py` | the companies, the metrics, the windows, and the screen |
| `src/catalog.py` | the tries, the schedule, and the two ways one gets written in English |
| `src/session_one.py` | runs the tries, writes the failures, its own process |
| `src/session_two.py` | puts ten forward and asks, a new process with nothing in it |
| `tests/` | 18 tests: the world and the scores repeat, a verdict needs two standard errors, what is written is not what is asked, dropping the scope merges memories, no model was called, the sessions had different pids, both stores catch every repeat, the shorter one blocks more, the embedder is the pinned one, and the run matches the pin |
| `runs/*.json` | what each session wrote down, from the run the numbers above came from. `run.py` reads these to build `mem0-data.json` |

`store/` is Qdrant on disk plus its history database, built by `run.py` and not
committed. `run.py` also writes `../mem0-data.json`, one level up, which is where
the reel's build reads it from. Both are in the repo's `.gitignore`, along with
the `.venv/` uv makes.

Running `run.py` overwrites `runs/*.json` with your own run. `git checkout
episodes/14-mem0/runs` puts the shipped ones back.
