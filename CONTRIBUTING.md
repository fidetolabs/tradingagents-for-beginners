# Contributing

Fork, branch, pull request into `main`. That's the whole process.

## The data rule

**No datasets in this repo.** Every episode ships a fetch script with its window
pinned in it, so every reader downloads the same bars and gets the same numbers.
Market data in git goes stale, bloats the clone, and says nothing about where it
came from. A pinned fetcher says all of it in thirty lines.

What a fetcher owes a reader:

- **The window lives in the script**, as named constants at the top. Start date,
  end date, symbol. Not in a config file, not passed as an argument.
- **Ask for the series that doesn't move.** Yahoo's `auto_adjust=True` rewrites
  every past bar when a dividend is paid, so the same call returns different
  closes next quarter. Take raw bars and do the adjustment in the script from the
  actions paid up to your end date. Episode 13 has a worked version.
- **A pin, and a check against it.** Hold the bar count, the first and last date
  and a few closes the episode was measured on. Refuse to write files that miss
  it, and test the written files against it. A silent drift prints different
  numbers from the README, which is worse than a failure.
- **It runs as `uv run fetch_data.py`**, with no arguments and no setup step
  before it. The episode's `pyproject.toml` carries whatever it needs to reach
  the vendor.
- **The download goes in `.gitignore`**, so a reader's fetch can't be committed.

## Fixing an episode

Most useful kind of change. A command that fails on your Python, a pinned version
that's gone, a test that breaks, a README step that's missing. Open a pull request
with the fix and say which machine and Python version you hit it on.

Keep the numbers honest. If your change makes an episode print different figures,
update that episode's README in the same pull request and say what moved.

## Adding an episode

Episode folders follow the reels on
[@fidetolabs](https://www.instagram.com/fidetolabs/), so open an issue first.

A folder is `episodes/<NN>-<tool>/`, where `NN` is the reel's episode number and
`<tool>` is the open-source tool it uses. A good one holds:

- `pyproject.toml` with the dependencies pinned to exact versions and
  `requires-python`, and `uv.lock` committed beside it. The lock is what makes a
  reader's run resolve to the versions the episode was measured on instead of
  whatever is newest. There is no `requirements.txt`: one list, no second one to
  drift.
- A fetch script, per the data rule above.
- `run.py` that runs the whole thing and prints what happened. `uv run run.py`,
  no arguments needed, and no download: the fetch already happened.
- `src/`, the agent itself.
- `tests/` that assert the shapes and the numbers the README talks about.
- `README.md`: what the episode shows, the exact setup and run commands, the
  output to expect, and the numbers it measured.

A few more rules that aren't style:

- **Nothing generated gets committed.** No downloads, no databases, no caches, no
  `__pycache__`, no virtual environment. Add anything new of that kind to the
  root `.gitignore`.
- **No secrets.** No keys, no tokens, no `.env`, no email addresses. Check the
  diff before you push.
- **Every number in the README came out of the code.** If you can't print it, don't
  claim it.

## Prose

Short sentences. Contractions are fine. Say what happened and what to run. Skip
the wind-up.
