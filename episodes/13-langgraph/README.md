# Episode 13: LangGraph checkpoints and replay

The reel is on Instagram, on [@fidetolabs](https://www.instagram.com/fidetolabs/).
This folder is the code it was drawn from. Everything on the frame came out of
one run of `run.py`.

## What the episode teaches

[LangGraph](https://github.com/langchain-ai/langgraph) writes the whole state of
a run to a checkpoint after every step. Two things follow from that. A run whose
process dies can carry on in a different process. And a run that finished can be
sent back to an earlier step to try a different rule.

The code does four things to show it. It answers one research question about SPY,
it gets killed halfway through on purpose, it resumes in a new process, and then
it forks back to an earlier step with the rule changed.

**No model is called anywhere in it.** The nodes are arithmetic. The subject is
the runtime, and an LLM in the middle would make every figure a sample instead of
a measurement.

## Run it

You need Python 3.13. You don't need an API key or a network connection: both SPY
files are already in `data/`.

```sh
cd episodes/13-langgraph
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q    # 14 tests
.venv/bin/python run.py                # the five phases
```

`run.py --snapshots` re-downloads SPY from Yahoo and rebuilds `data/` first. That
one needs a network, and it will change every number below, because it moves the
market the agent reads.

## What you should see

`run.py` prints the five phases and then the two numbers. Here is the run this
folder ships with:

```
graph      9 nodes, 11 edges, 1 conditional edge, 1 loop

1  cold     a fresh run on thread A, and the process dies inside backtest
           exit 137; 9 nodes ran before it died (recall fetch features screen backtest judge tune features screen)

2  resume   a NEW process picks thread A up from the last checkpoint
           14 nodes ran (backtest judge tune features screen backtest judge tune features screen backtest judge conclude remember)
           verdict: holding beats the breakout on this slice

3  restart  the same work from the top, on a fresh thread
           23 nodes ran to reach the same answer

   the gap  one download, read at two dates 1 trading day apart.
            Thread A reads it as of 2026-11-16; the fork reads it to 2026-11-17

4  fork     back to the checkpoint after `recall`, change the rule, run forward
           22 nodes ran

5  control  the same fork with the rule left alone
           22 nodes ran, window 80 (thread A ended on 80)

WHAT MOVED, AND WHY
  thread A           -1,798.34   window 80, as of 2026-11-16
  the control fork   -1,663.88   same window, 1 more trading day
  the fork           -1,158.53   new window, 1 more trading day
  the agent would credit the rule with 639.81
  the 1 trading day is worth 134.46; the rule is worth 505.35

THE PAIR
  after the crash   14 nodes from the checkpoint, 23 from the top, 9 saved
  the fork          2026-11-16 at $739.95 became 2026-11-17 at $732.44
  what it cost      one run is 23 nodes, 25 checkpoints and 454,928 bytes of state
```

The byte count moves by a few dozen between runs, because a checkpoint carries
timestamps of varying length. Every other figure is the same on every run.

## The two numbers the reel measured

**14 nodes from the checkpoint against 23 from the top.** Nine nodes of work
survived a process that no longer exists.

**The rule is worth 505.35 and the trading day is worth 134.46.** The fork gained
639.81 over its parent. An agent comparing a fork to its parent credits all of it
to the rule change, and a fifth of it is weather. The control fork is the only
reason anyone can split the two.

## How LangGraph works, in the three facts this agent rests on

**1. You declare a shape.** A node is a plain function, `state -> partial state`.
An edge says what may follow what. `add_conditional_edges` hands the choice to a
function at runtime, so the agent picks its own route, but only along edges you
wrote. The route it took is then something you read back instead of reconstructing
from logs.

Nine nodes, eleven edges, one conditional edge, one loop. The straight line runs
`recall -> fetch -> features -> screen -> backtest -> judge`. From `judge` the
conditional edge goes either to `tune`, which widens the parameter and sends the
run back to `features`, or to `conclude -> remember -> END`. `recall` reads
memory, `fetch` reads the market, `features`, `screen` and `backtest` do the
arithmetic, `judge` decides whether the result is good enough, `conclude` writes
the verdict, and `remember` saves what was learned.

**2. The run advances in super-steps, and every one is checkpointed.** Everything
that can run in parallel runs, the partials are merged through each channel's
reducer, and the whole state is written to a checkpoint keyed by `thread_id`.
With `durability="sync"` that write lands before the next super-step starts,
which is what makes a resume possible after the process is gone. No node here
runs beside another, so one super-step is one node. A finished run has 25
snapshots for 23 node executions: the input at step -1, step 0 with the input
merged and nothing run yet, then one a node.

**3. The checkpointer is per thread; the store is not.** A thread is one run.
`SqliteStore` is namespaced key-value that outlives every thread, and it is the
only reason the next run can begin from something the last one worked out. Here
`recall` reads `("spy", "rules") / breakout-window` and `remember` writes it.

## The five phases

| phase | what it does |
| --- | --- |
| 1 cold | thread A, a fresh run, killed with `os._exit(137)` from inside `backtest`. Not an exception somebody could catch: no unwinding, no `finally`, no flush the runtime hadn't already done |
| 2 resume | thread A again, in a **new process**, `invoke(None, config)` from the last checkpoint |
| 3 restart | thread B, the same question from the top, for the comparison |
| 4 fork | back to the checkpoint taken after `recall`, `update_state` to change the rule, forward from there |
| 5 control | the same fork with the rule left alone |

Each phase runs in its own process, because the claim being measured is about
processes dying. `run.py` is the parent and spawns all five.

## The part that is a trap

A fork re-runs `fetch`, and `fetch` asks the world a question. **A checkpoint
saves the agent, not the market.**

Thread A ran against the file as of 2026-11-16, with SPY at $739.95. The fork read
it to 2026-11-17 at $732.44, so the same node, from the same checkpoint, answered
differently. The fork moved two things at once, and only one of them was the
agent's idea.

**The fix is a build instruction.** Pin the world into the state: have the fetch
node record what it read, and make every downstream node take its data from
`state`. Or lift the fetch out of the graph and pass the slice in as input. Then a
fork replays the agent against the world the agent saw.

## What durability costs

One 23-node run is 25 checkpoints and about 455,000 bytes of state. The largest
single checkpoint is about 26 KB, because this agent keeps a 500-bar price slice
in the state. Checkpoints are written per super-step, so state size is multiplied
by the length of the run. Keep the slice out of the state and the same run costs
almost nothing. Keep it in and a resume is cheap, because the expensive thing was
already saved. That's the trade, either way round.

## The store makes runs non-independent

Thread B was meant to be the same question from the top. It isn't quite. Thread A
had already written `window=80` to the store, so thread B's `recall` started from
80 rather than the default 20. The node count is unaffected, because the tune loop
is capped either way, which is why the 14-against-23 comparison holds. The answer
is not unaffected. Memory that outlives a thread means no two runs are
independent. Clear the store, or namespace it per experiment.

## The two data files, and which part of one is reconstructed

`data/spy_2026-11-17.csv` is a live Yahoo download. `data/spy_2026-11-16.csv` is
**the same download truncated at 2026-11-16**, not a file captured that day. The
bars are real and unedited. What is reconstructed is the moment of capture, not
the prices. The claim the experiment makes, that a trading day arrived between
the run and the replay, is true either way, and naming the files makes the gap
reproducible next month.

**An unfinished bar is not a trading day.** Yahoo returns the session in progress
with a volume and no OHLC, and one NaN close turns every number downstream into
NaN without raising. The first run of this reported a fork that scored `nan` and
took the wrong branch out of `judge`, because `nan <= 0` is False. Incomplete rows
are dropped and counted, never filled.

## Files

| path | what it is |
| --- | --- |
| `run.py` | the five phases, and the parent that orchestrates them |
| `src/graph.py` | the `StateGraph`: nine nodes, the conditional edge, the loop |
| `src/world.py` | the only code that touches the market, and the two dated snapshots |
| `src/state.py` | the channels, and the reducer on `trail` |
| `src/instrument.py` | an append-only record of which node actually ran, kept out of the state because a crashed run's work is already in the restored state |
| `tests/` | 14 tests: the declared shape is the compiled shape, a checkpoint a super-step, a dead run resumes, a fork branches the same thread, the fork reads a world that moved, and neither snapshot carries an unfinished bar |
| `runs/*.jsonl` | the node-by-node log of the run the numbers above came from. One line a node execution, one file a phase |

`checkpoints.sqlite` and `store.sqlite` are built by `run.py` and are not
committed. `run.py` also writes `../langgraph-data.json`, one level up, which is
where the reel's build reads it from. Both are in the repo's `.gitignore`.

Running `run.py` overwrites `runs/*.jsonl` with your own run. `git checkout
episodes/13-langgraph/runs` puts the shipped ones back.
