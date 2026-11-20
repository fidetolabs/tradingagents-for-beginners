# TradingAgents for beginners

The code behind the reels on [@fidetolabs](https://www.instagram.com/fidetolabs/).
Each reel takes one real open-source tool, builds a small trading-research agent
with it, and measures something about it. This repo holds the agent, so you can
run it yourself and get the same numbers off your own machine. It's for anyone
who watched a reel and wants to see how the thing actually works.

The reels are a numbered series, *How To Build A Self-Improving AI Trading Agent
#n*, and the number in a folder name is the number in the reel's title.

## Layout

One folder an episode, under `episodes/<NN>-<tool>/`. Each folder stands alone and
carries its own `pyproject.toml`, `uv.lock`, tests and README. There is no shared
package and nothing to install at the root.

## Episodes

| # | tool | what it measures | folder |
| --- | --- | --- | --- |
| 13 | LangGraph | what a checkpoint is worth when the process dies, and how much of a fork's gain was the market moving | [`episodes/13-langgraph/`](episodes/13-langgraph/) |
| 14 | Mem0 | what a memory of failed experiments saves, and what it blocks once the short version drops the scope | [`episodes/14-mem0/`](episodes/14-mem0/) |

Earlier reels were built before this repo existed, so they aren't here. New rows
land as new reels ship.

## Running one

Every episode runs with [uv](https://docs.astral.sh/uv/), and that is the only
thing you install. Each folder carries a `pyproject.toml` and a committed
`uv.lock`, so `uv` builds the environment itself, down to fetching a Python if
yours is too old.

```sh
cd episodes/14-mem0
uv run run.py
uv run pytest tests -q
```

**No datasets are committed here.** An episode that reads a vendor ships a fetch
script with its date window pinned inside it, so you download the same bars the
reel was run on and get the same numbers. That is episode 13's first step:

```sh
cd episodes/13-langgraph
uv run fetch_data.py
uv run pytest tests -q
uv run run.py
```

An episode that reads nothing from outside has no fetcher. Read that episode's
README for the order its commands go in and what the output should say.

Whatever an episode's numbers rest on is pinned in its code, and the run refuses
to print a figure that doesn't match what the reel published. For episode 13 that
is a price series a vendor rewrites; for episode 14 it is the embedding model
every score comes out of.

## What this is not

The agents here are research code. They measure how a tool behaves, not what to
buy. Nothing in this repo is advice, and no number in it is a return you could
have had.

## Contributing

Fork the repo, make a branch, open a pull request into `main`. Fixes to an
existing episode are welcome. New episode folders follow the reels, so open an
issue before building one. [CONTRIBUTING.md](CONTRIBUTING.md) has the detail.

## License

MIT. See [LICENSE](LICENSE).
