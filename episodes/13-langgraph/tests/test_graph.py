"""The runtime, and the claims the reel makes about it."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src import graph as G
from src import world

QUESTION = "test"


def compiled(**kw):
    return G.build().compile(**kw)


def cfg(t):
    return {"configurable": {"thread_id": t}}


@pytest.fixture(autouse=True)
def _pin_world(monkeypatch):
    monkeypatch.setenv("SPY_FILE", world.ASOF_RUN)


def test_the_declared_shape_is_the_compiled_shape():
    g = compiled().get_graph()
    assert {n for n in g.nodes} - {"__start__", "__end__"} == set(G.NODES)
    edges = {(e.source, e.target) for e in g.edges}
    for a, b in G.EDGES:
        a = "__start__" if a == "START" else a
        b = "__end__" if b == "END" else b
        assert (a, b) in edges, f"{a} -> {b} is declared and not compiled"


def test_the_agent_can_only_take_an_edge_that_exists():
    assert G.route({"score": {"edge": -1}, "attempts": 0}) == "tune"
    assert G.route({"score": {"edge": -1}, "attempts": G.MAX_ATTEMPTS}) == "conclude"
    assert G.route({"score": {"edge": 1}, "attempts": 0}) == "conclude"


def test_a_checkpoint_is_written_after_every_super_step():
    """No node here runs beside another, so one super-step is one node: the
    input checkpoint at step -1, step 0 with the input merged and nothing run
    yet, then one a node."""
    app = compiled(checkpointer=InMemorySaver(), store=InMemoryStore())
    out = app.invoke({"question": QUESTION, "trail": []}, cfg("t1"), durability="sync")
    hist = list(app.get_state_history(cfg("t1")))
    assert len(hist) == len(out["trail"]) + 2
    assert sorted(s.metadata["step"] for s in hist) == list(range(-1, len(out["trail"]) + 1))
    assert hist[0].next == ()
    assert hist[0].values["trail"] == out["trail"]


def test_a_dead_run_resumes_from_its_last_checkpoint(monkeypatch):
    saver = InMemorySaver()
    boom = {"n": 0}
    real = G.backtest

    def flaky(state):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("the process died")
        return real(state)

    monkeypatch.setattr(G, "backtest", flaky)
    app = G.build().compile(checkpointer=saver, store=InMemoryStore())
    with pytest.raises(RuntimeError):
        app.invoke({"question": QUESTION, "trail": []}, cfg("t2"), durability="sync")

    before = app.get_state(cfg("t2"))
    assert before.next == ("backtest",)
    assert before.values["trail"] == ["recall", "fetch", "features", "screen"]

    out = app.invoke(None, cfg("t2"), durability="sync")
    assert out["trail"][:4] == ["recall", "fetch", "features", "screen"]
    assert out["verdict"]


def test_resuming_runs_fewer_nodes_than_starting_over(monkeypatch):
    """The reel's first number, as a property rather than as a reading.

    The wrappers keep the real signatures: LangGraph decides whether to inject
    `store` by inspecting the node, so a `**kwargs` wrapper loses it."""
    ran = []

    def plain(name):
        fn = getattr(G, name)

        def w(state):
            ran.append(name)
            return fn(state)
        return w

    def with_store(name):
        fn = getattr(G, name)

        def w(state, *, store):
            ran.append(name)
            return fn(state, store=store)
        return w

    for name in G.NODES:
        monkeypatch.setattr(G, name, (with_store if name in ("recall", "remember") else plain)(name))

    app = G.build().compile(checkpointer=InMemorySaver(), store=InMemoryStore())
    app.invoke({"question": QUESTION, "trail": []}, cfg("whole"), durability="sync")
    whole = len(ran)
    assert whole == 6 + 5 * G.MAX_ATTEMPTS + 2

    ran.clear()
    app2 = G.build().compile(checkpointer=InMemorySaver(), store=InMemoryStore())
    app2.invoke({"question": QUESTION, "trail": []}, cfg("part"),
                durability="sync", interrupt_before=["backtest"])
    stopped = len(ran)
    ran.clear()
    app2.invoke(None, cfg("part"), durability="sync")
    assert stopped + len(ran) == whole
    assert len(ran) < whole, "resuming did not save any work"


def test_the_store_outlives_the_thread():
    store = InMemoryStore()
    app = compiled(checkpointer=InMemorySaver(), store=store)
    app.invoke({"question": QUESTION, "trail": []}, cfg("one"), durability="sync")
    kept = store.get(G.NS, G.KEY)
    assert kept is not None
    out = app.invoke({"question": QUESTION, "trail": []}, cfg("two"), durability="sync")
    assert str(kept.value["window"]) in out["rule_in"]


def test_a_fork_branches_the_same_thread():
    app = compiled(checkpointer=InMemorySaver(), store=InMemoryStore())
    app.invoke({"question": QUESTION, "trail": []}, cfg("f"), durability="sync")
    hist = list(app.get_state_history(cfg("f")))
    target = next(s for s in reversed(hist) if s.next == ("fetch",))
    forked = app.update_state(target.config, {"window": 55}, as_node="recall")
    assert forked["configurable"]["thread_id"] == "f"
    assert forked["configurable"]["checkpoint_id"] != target.config["configurable"]["checkpoint_id"]
    assert app.get_state(forked).values["window"] == 55


def test_the_fork_runs_fetch_again_and_fetch_reads_now(monkeypatch):
    """The stress, as a property: nothing in a checkpoint pins the world, so a
    node that asks the world a question asks it again."""
    app = compiled(checkpointer=InMemorySaver(), store=InMemoryStore())
    monkeypatch.setenv("SPY_FILE", world.ASOF_RUN)
    app.invoke({"question": QUESTION, "trail": []}, cfg("w"), durability="sync")
    hist = list(app.get_state_history(cfg("w")))
    target = next(s for s in reversed(hist) if s.next == ("fetch",))
    before = next(s for s in reversed(hist) if "asof" in s.values).values["asof"]

    monkeypatch.setenv("SPY_FILE", world.ASOF_FORK)
    forked = app.update_state(target.config, {}, as_node="recall")
    out = app.invoke(None, forked, durability="sync")
    assert out["asof"] != before, "the fork read the same world, so there is no lesson"
