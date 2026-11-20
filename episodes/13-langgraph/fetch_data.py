#!/usr/bin/env python
"""Download the exact SPY bars this episode was run on.

    uv run fetch_data.py

THE WINDOW IS PINNED HERE AND NOWHERE ELSE. One download, two writes: the series
the fork reads, cut at FORK_CUT, and the series thread A reads, the same download
cut one trading day earlier at RUN_CUT. The experiment needs the agent's data
source to have advanced between a run and its replay, and that gap is these two
cuts.

WHY THIS DOES NOT ASK YAHOO FOR ADJUSTED BARS
---------------------------------------------
`auto_adjust=True` divides every past bar by the dividends paid since, so the
series is rewritten every quarter SPY pays one. A re-download next year returns
different closes for 2009 and the episode's figures move with no warning.

Raw bars never move, and they are a different series: on 2009-01-02 the raw
close is 92.96 against 67.64 adjusted, and the episode measured 427.16 on raw
where it published 505.35.

So this takes raw bars and does the adjustment here, from the dividends paid up
to FORK_CUT. Both inputs are frozen by the cut: a bar from 2009 does not change,
and no dividend is ever paid into a window that has already closed. The December
dividend changes what Yahoo would hand back and changes nothing here.

PIN holds what the result has to look like, and this script refuses to write
files that miss it.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import world

START = "2009-01-01"
FORK_CUT = "2026-11-17"      # the series the fork reads, and world.ASOF_FORK
RUN_CUT = "2026-11-16"       # one trading day earlier, and world.ASOF_RUN
OHLC = ["Open", "High", "Low", "Close"]

# WHAT THE SERIES HAS TO COME OUT AS. Yahoo's own floats wobble in the eighth
# decimal between downloads, which is why this is a tolerance and not equality.
# A dividend would move a 2009 close by about 0.25%, which is 2,500 times TOL.
TOL = 1e-5
PIN = {"first_date": "2009-01-02", "first_close": 67.6386260986,
       "run_bars": 4496, "run_close": 739.95,
       "fork_bars": 4497, "fork_close": 732.44}


def download() -> tuple[pd.DataFrame, pd.Series]:
    """Raw bars and the dividend record, straight from Yahoo."""
    import yfinance as yf

    raw = yf.download("SPY", start=START, interval="1d", auto_adjust=False,
                      actions=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise SystemExit("Yahoo returned nothing; refusing to invent a bar")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw.index.name = "Date"
    df = raw[OHLC + ["Volume"]].copy()
    # AN UNFINISHED BAR IS NOT A TRADING DAY. Yahoo returns the current session
    # with a volume and no OHLC, and one NaN close turns every number downstream
    # into NaN without raising anything: the first run of this reported a fork
    # that scored `nan` and took the wrong branch out of `judge`, because
    # `nan <= 0` is False. Incomplete rows are dropped and counted, never filled.
    bad = df[df[OHLC].isna().any(axis=1)]
    if len(bad):
        df = df.drop(bad.index)
    df.attrs["dropped"] = [d.date().isoformat() for d in bad.index]
    div = raw["Dividends"]
    return df, div[div > 0]


def adjust(df: pd.DataFrame, div: pd.Series, cut: str) -> pd.DataFrame:
    """Yahoo's own convention, applied here with the dividend list cut short.

    A bar is divided by every dividend paid after it, each as a fraction of the
    close the day before it went ex. Dividends after `cut` are not in the window
    the episode ran on, so they are dropped and the series stops moving."""
    out = df.loc[:cut].copy()
    ratio = pd.Series(1.0, index=out.index)
    for when, amount in div.loc[:cut].items():
        earlier = out.index[out.index < when]
        if not len(earlier):
            continue
        base = float(out.loc[earlier[-1], "Close"])
        ratio.loc[ratio.index < when] *= 1.0 - amount / base
    for col in OHLC:
        out[col] = out[col] * ratio
    return out


def check(fork: pd.DataFrame, run: pd.DataFrame) -> None:
    def near(got: float, want: float, what: str) -> None:
        if abs(got - want) > TOL * abs(want):
            raise SystemExit(
                f"{what}: got {got:.10f}, the episode ran on {want:.10f}.\n"
                f"The numbers in the README came off the second one, so they will "
                f"not come out of this. Open an issue with both figures.")

    if fork.index[-1].date().isoformat() != FORK_CUT:
        raise SystemExit(f"the download does not reach {FORK_CUT}; it ends "
                         f"{fork.index[-1].date()}")
    if len(fork) - len(run) != 1:
        raise SystemExit(f"{RUN_CUT} to {FORK_CUT} should be one trading day, got "
                         f"{len(fork) - len(run)}")
    for name, frame, bars in (("fork", fork, PIN["fork_bars"]), ("run", run, PIN["run_bars"])):
        if len(frame) != bars:
            raise SystemExit(f"the {name} series has {len(frame)} bars, the episode "
                             f"ran on {bars}")
    if fork.index[0].date().isoformat() != PIN["first_date"]:
        raise SystemExit(f"the series starts {fork.index[0].date()}, not {PIN['first_date']}")
    near(float(fork["Close"].iloc[0]), PIN["first_close"], "the 2009-01-02 close")
    near(float(fork["Close"].iloc[-1]), PIN["fork_close"], f"the {FORK_CUT} close")
    near(float(run["Close"].iloc[-1]), PIN["run_close"], f"the {RUN_CUT} close")


def main() -> int:
    df, div = download()
    fork = adjust(df, div, FORK_CUT)
    run = adjust(df, div, RUN_CUT)
    check(fork, run)

    os.makedirs(world.DATA, exist_ok=True)
    fork.to_csv(world.ASOF_FORK, float_format="%.10f")
    run.to_csv(world.ASOF_RUN, float_format="%.10f")

    print(f"downloaded  {len(df):,} raw bars, {df.index[0].date()} to {df.index[-1].date()}")
    if df.attrs["dropped"]:
        print(f"dropped     {df.attrs['dropped']} (no OHLC, so not a finished trading day)")
    print(f"dividends   {len(div.loc[:FORK_CUT])} paid up to {FORK_CUT}, last one "
          f"{div.loc[:FORK_CUT].index[-1].date()}; later ones ignored")
    print(f"wrote       {os.path.basename(world.ASOF_FORK)}  {len(fork):,} bars to "
          f"{fork.index[-1].date()}, close {float(fork['Close'].iloc[-1]):.2f}")
    print(f"wrote       {os.path.basename(world.ASOF_RUN)}   {len(run):,} bars to "
          f"{run.index[-1].date()}, close {float(run['Close'].iloc[-1]):.2f}")
    print(f"the pin     2009-01-02 close {float(fork['Close'].iloc[0]):.10f}, "
          f"the episode ran on {PIN['first_close']:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
