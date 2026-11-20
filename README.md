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
carries its own requirements, fetcher, tests and README. There is no shared
package and nothing to install at the root.

## Episodes

| # | tool | what it measures | folder |
| --- | --- | --- | --- |
| 13 | LangGraph | what a checkpoint is worth when the process dies, and how much of a fork's gain was the market moving | [`episodes/13-langgraph/`](episodes/13-langgraph/) |

Earlier reels were built before this repo existed, so they aren't here. New rows
land as new reels ship.

## Fetch, then run

**No market data is committed here.** An episode ships a fetch script with its
date window pinned inside it, so you download the same bars the reel was run on
and get the same numbers. The fetch needs a network. Nothing after it does.

```sh
cd episodes/13-langgraph
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_data.py
.venv/bin/python -m pytest tests -q
.venv/bin/python run.py
```

The pattern is the same in every episode. Read that episode's README for what the
output should say.

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
