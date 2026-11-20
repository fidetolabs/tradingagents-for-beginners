"""The arithmetic the nodes do, and the two snapshots they do it on."""
import numpy as np
import pandas as pd
import pytest

from src import world


def test_the_two_snapshots_differ_by_at_least_one_trading_day():
    a = pd.read_csv(world.ASOF_RUN, index_col=0, parse_dates=True)
    b = pd.read_csv(world.ASOF_FORK, index_col=0, parse_dates=True)
    assert len(b) > len(a)
    assert b.index[-1] > a.index[-1]
    assert a.index.equals(b.index[:len(a)]), "the earlier file is not a prefix of the later one"


def test_the_snapshots_are_the_ones_the_episode_measured():
    """The fetch is pinned, so the data it writes is checkable.

    Without this, a fetch that drifted writes a series that runs fine and
    prints different figures from the ones in the README. Yahoo's floats wobble
    in the eighth decimal between downloads; a dividend paid into the window
    would move a 2009 close by about 0.25%, which is 2,500 times this
    tolerance."""
    import fetch_data

    fork = pd.read_csv(world.ASOF_FORK, index_col=0, parse_dates=True)
    run = pd.read_csv(world.ASOF_RUN, index_col=0, parse_dates=True)
    pin, tol = fetch_data.PIN, fetch_data.TOL
    assert len(fork) == pin["fork_bars"]
    assert len(run) == pin["run_bars"]
    assert fork.index[0].date().isoformat() == pin["first_date"]
    assert fork.index[-1].date().isoformat() == fetch_data.FORK_CUT
    assert run.index[-1].date().isoformat() == fetch_data.RUN_CUT
    for got, want in ((fork["Close"].iloc[0], pin["first_close"]),
                      (fork["Close"].iloc[-1], pin["fork_close"]),
                      (run["Close"].iloc[-1], pin["run_close"])):
        assert float(got) == pytest.approx(want, rel=tol)


def test_neither_snapshot_carries_an_unfinished_bar():
    """A row with a volume and no close is the session in progress. One of them
    turned every number downstream into NaN and `nan <= 0` sent the run down
    the wrong branch without raising anything."""
    for p in (world.ASOF_RUN, world.ASOF_FORK):
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        assert not df[["Open", "High", "Low", "Close"]].isna().any().any()
        assert (df[["Open", "High", "Low", "Close"]] > 0).all().all()


def test_latest_reads_the_end_of_the_file():
    df = pd.read_csv(world.ASOF_RUN, index_col=0, parse_dates=True)
    got = world.latest(500, world.ASOF_RUN)
    assert got["asof"] == df.index[-1].date().isoformat()
    assert got["bars"] == 500
    assert got["last_close"] == round(float(df["Close"].iloc[-1]), 2)


def test_a_flat_market_never_fires_the_rule():
    flat = {"closes": [100.0] * 300, "highs": [100.0] * 300,
            "lows": [100.0] * 300, "opens": [100.0] * 300}
    s = world.breakout_trades(flat, 20)
    assert s["trades"] == 0
    assert s["final"] == pytest.approx(10_000.0)


def test_the_score_is_deterministic_and_finite():
    bars = world.latest(500, world.ASOF_RUN)
    a = world.breakout_trades(bars, 20)
    b = world.breakout_trades(bars, 20)
    assert a == b
    assert all(np.isfinite(v) for v in (a["final"], a["edge"], a["hold_final"]))
    assert a["trades"] > 0


def test_the_slice_is_the_window_the_agent_asked_for():
    for n in (120, 250, 500):
        assert world.latest(n, world.ASOF_RUN)["bars"] == n
