"""The world the agent reads, and the only node that touches it.

THE POINT OF THIS FILE IS THAT IT IS NOT DETERMINISTIC IN TIME. `latest()`
answers "what are the last N trading days, as of now", and the answer moves when
the market moves. Everything else in the agent is arithmetic on what this
returned. That split is the whole reel: a checkpoint saves the second half and
not the first.

TWO SNAPSHOTS, AND ONE OF THEM IS A RECONSTRUCTION
--------------------------------------------------
The experiment needs the agent's data source to have advanced between a run and
its replay. `spy_2026-11-17.csv` is a live Yahoo download. `spy_2026-11-16.csv`
is **the same download truncated at 2026-11-16** rather than a file captured on
that day, because the file captured on that day was deleted. The bars in it are
real and unedited; what is reconstructed is the moment of capture, not the
prices. Everything the reel claims about the gap between the two is a claim
about a real trading day arriving, which it did.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ASOF_RUN = os.path.join(DATA, "spy_2026-11-16.csv")
ASOF_FORK = os.path.join(DATA, "spy_2026-11-17.csv")
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download(start: str = "2009-01-01") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download("SPY", start=start, interval="1d", auto_adjust=True,
                      actions=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo returned nothing; refusing to invent a bar")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    # AN UNFINISHED BAR IS NOT A TRADING DAY. Yahoo returns the current session
    # with a volume and no OHLC, and one NaN close turns every number
    # downstream into NaN without raising anything: the first run of this
    # rebuild reported a fork that scored `nan` and took the wrong branch out of
    # `judge`, because `nan <= 0` is False. Incomplete rows are dropped and
    # counted, never filled.
    ohlc = ["Open", "High", "Low", "Close"]
    bad = df[df[ohlc].isna().any(axis=1)]
    if len(bad):
        df = df.drop(bad.index)
    df.attrs["dropped"] = [d.date().isoformat() for d in bad.index]
    return df


def build_snapshots(cut: str = "2026-11-16") -> dict:
    """Write both files from one download. The later one is the download; the
    earlier one is that download truncated, and the README says so."""
    df = download()
    os.makedirs(DATA, exist_ok=True)
    df.to_csv(ASOF_FORK, float_format="%.10f")
    df.loc[:cut].to_csv(ASOF_RUN, float_format="%.10f")
    return {"fork_ends": df.index[-1].date().isoformat(),
            "run_ends": df.loc[:cut].index[-1].date().isoformat(),
            "bars_added": int(len(df) - len(df.loc[:cut])),
            "incomplete_dropped": df.attrs.get("dropped", [])}


def live() -> str:
    """Which snapshot this process reads. run.py sets it per phase."""
    return os.environ.get("SPY_FILE") or ASOF_RUN


def latest(window: int = 500, path: str | None = None) -> dict:
    """The last `window` trading days, as of whatever the file holds now."""
    df = pd.read_csv(path or live(), index_col=0, parse_dates=True).tail(window)
    return {
        "asof": df.index[-1].date().isoformat(),
        "first": df.index[0].date().isoformat(),
        "bars": int(len(df)),
        "last_close": round(float(df["Close"].iloc[-1]), 2),
        "closes": [round(float(v), 4) for v in df["Close"]],
        "highs": [round(float(v), 4) for v in df["High"]],
        "lows": [round(float(v), 4) for v in df["Low"]],
        "opens": [round(float(v), 4) for v in df["Open"]],
    }


def breakout_trades(bars: dict, window: int) -> dict:
    """A Donchian breakout on the slice the agent was given. Signal at the close
    of T, filled at the open of T+1, 5bp a side. Arithmetic only."""
    close = np.array(bars["closes"])
    high = np.array(bars["highs"])
    low = np.array(bars["lows"])
    openp = np.array(bars["opens"])
    n = len(close)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    for i in range(window, n):
        hi[i] = high[i - window:i].max()
        lo[i] = low[i - max(1, window // 2):i].min()

    pos = np.zeros(n)
    held = 0.0
    for i in range(n):
        if not np.isnan(hi[i]) and close[i] > hi[i]:
            held = 1.0
        elif not np.isnan(lo[i]) and close[i] < lo[i]:
            held = 0.0
        pos[i] = held

    cash, shares, cost = 10_000.0, 0.0, 0.0005
    equity, entries = [], 0
    for i in range(1, n):
        want = pos[i - 1]
        if want >= 0.5 and shares == 0.0:
            notional = cash / (1 + cost)
            cash -= notional * (1 + cost)
            shares = notional / openp[i]
            entries += 1
        elif want < 0.5 and shares > 0.0:
            cash += shares * openp[i] * (1 - cost)
            shares = 0.0
            entries += 1
        equity.append(cash + shares * close[i])

    eq = np.array(equity)
    years = len(eq) / 252
    hold = 10_000 * close[-1] / close[1]
    return {
        "final": round(float(eq[-1]), 2),
        "cagr": round(float((eq[-1] / 10_000) ** (1 / years) - 1), 6),
        "hold_final": round(float(hold), 2),
        "trades": int(entries),
        "edge": round(float(eq[-1] - hold), 2),
    }
