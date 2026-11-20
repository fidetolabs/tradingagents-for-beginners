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
