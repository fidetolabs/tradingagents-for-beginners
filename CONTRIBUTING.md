# Contributing

Fork, branch, pull request into `main`. That's the whole process.

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

- `requirements.txt`, with pinned versions, and a note saying what it was run with.
- `run.py` that runs the whole thing and prints what happened. One command, no
  arguments needed.
- `src/`, the agent itself.
- `tests/` that assert the shapes and the numbers the README talks about.
- `data/`, if the episode needs data. Commit the files it reads, so a run works
  with no network and no API key.
- `README.md`: what the episode shows, the exact setup and run commands, the
  output to expect, and the numbers it measured.

Rules that aren't style:

- **It runs offline.** No API key to get started. If the code can fetch live data,
  that's a flag, not the default path.
- **Nothing generated gets committed.** No databases, no caches, no
  `__pycache__`, no virtual environment. Add anything new of that kind to the
  root `.gitignore`.
- **No secrets.** No keys, no tokens, no `.env`, no email addresses. Check the
  diff before you push.
- **Every number in the README came out of the code.** If you can't print it, don't
  claim it.

## Prose

Short sentences. Contractions are fine. Say what happened and what to run. Skip
the wind-up.
