"""Mem0, configured so that nothing here is a sample.

NO LANGUAGE MODEL IS CALLED. `add(..., infer=False)` is Mem0's own switch for
storing a memory as written instead of having a model extract a fact from it,
and `search` is embeddings and a vector store. So every score in this run is a
function of the text and the embedding model, and running it twice gives the
same answer twice. A model in the middle would make every number on the frame a
sample instead of a measurement, which is the same reason the reel before this
one called none either.

The LLM slot still has to be filled, because `Memory` builds one whether or not
it is used. It points at a local server and is never reached: `calls()` reads
the client's own counter back and the build refuses a run that moved it.

The embedder is `fastembed`, which is a small ONNX model on this machine. The
store is Qdrant on disk, so SESSION TWO IS A DIFFERENT PROCESS WITH NOTHING IN
ITS CONTEXT -- the only thing it has is what session one wrote down. That is the
claim the reel makes, and a shared process would not have measured it.
"""

import os
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "store"
DIMS = 384
MODEL = "BAAI/bge-small-en-v1.5"

# THE MODEL IS THE MEASUREMENT, SO IT IS PINNED LIKE ONE. Every score in this
# episode is a cosine similarity out of one embedder, so a different embedder
# is a different set of numbers. The name on its own does not pin it: fastembed
# maps a name to a repository and a file, and another release of fastembed
# could map the same name somewhere else. These three are what fastembed 0.9.1
# resolves MODEL to, and `check_embedder` reads them back.
SOURCE_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
MODEL_FILE = "model_optimized.onnx"
FASTEMBED = "0.9.1"

# WHAT THIS EPISODE MEASURED. run.py refuses to write a run that misses any of
# it, so a reader gets the numbers on the frame or an error, never a third
# thing. The counts are exact. The two thresholds are the lowest similarity
# that still catches every repeat, so they come straight off the embedder and
# carry a tolerance.
PIN = {
    "mem0ai": "2.2.1",
    "one_tries": 24, "one_failed": 12,
    "scoped_rows": 12, "bare_rows": 4,
    "proposed": 10, "repeats": 3, "fresh": 4, "fresh_good": 4,
    "caught": 3,
    "scoped_blocked": 0, "scoped_runs": 7, "scoped_threshold": 0.9042,
    "bare_blocked": 6, "bare_runs": 1, "bare_threshold": 0.3590,
}
THRESHOLD_TOL = 0.001

# the two ways the same four failures are written down
SCOPED, BARE = "scoped", "bare"


def open_store(reset=False):
    from mem0 import Memory

    if reset and STORE.exists():
        import shutil

        shutil.rmtree(STORE)
    STORE.mkdir(parents=True, exist_ok=True)
    cfg = {
        "llm": {"provider": "lmstudio", "config": {"model": "never-called"}},
        "embedder": {"provider": "fastembed", "config": {"model": MODEL}},
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "tries", "path": str(STORE / "qdrant"),
            "embedding_model_dims": DIMS}},
        "history_db_path": str(STORE / "history.db"),
    }
    return Memory.from_config(cfg)


def check_embedder():
    """What fastembed resolves MODEL to, read back from fastembed itself."""
    import importlib.metadata as md

    from fastembed import TextEmbedding

    got = md.version("fastembed")
    if got != FASTEMBED:
        raise SystemExit(f"fastembed {got} is installed, the episode ran on "
                         f"{FASTEMBED}. Run it with `uv run`, which uses uv.lock.")
    entry = next((e for e in TextEmbedding.list_supported_models()
                  if e.get("model") == MODEL), None)
    if entry is None:
        raise SystemExit(f"fastembed {got} does not serve {MODEL}")
    repo = (entry.get("sources") or {}).get("hf")
    for label, seen, want in (("repository", repo, SOURCE_REPO),
                              ("file", entry.get("model_file"), MODEL_FILE),
                              ("dims", entry.get("dim"), DIMS)):
        if seen != want:
            raise SystemExit(
                f"{MODEL} now resolves to {label} {seen}, and the episode "
                f"measured {want}. Every score here is a similarity out of that "
                f"model, so the numbers in the README will not come out.")


def check_pin(data):
    """Refuse a run that does not reproduce what the episode published."""
    d, one, two = data, data["one"], data["two"]
    counts = {
        "mem0ai": d["meta"]["mem0"],
        "one_tries": one["tries"], "one_failed": one["failed"],
        "scoped_rows": len(one["scoped_rows"]), "bare_rows": len(one["bare_rows"]),
        "proposed": two["proposed"], "repeats": two["repeats"],
        "fresh": two["fresh"], "fresh_good": two["fresh_good"],
        "caught": d["scoped"]["caught"],
        "scoped_blocked": d["scoped"]["blocked"], "scoped_runs": d["scoped"]["runs"],
        "bare_blocked": d["bare"]["blocked"], "bare_runs": d["bare"]["runs"],
    }
    wrong = [f"{k}: got {v}, the episode published {PIN[k]}"
             for k, v in counts.items() if v != PIN[k]]
    for k, got in (("scoped_threshold", d["scoped"]["threshold"]),
                   ("bare_threshold", d["bare"]["threshold"])):
        if abs(got - PIN[k]) > THRESHOLD_TOL:
            wrong.append(f"{k}: got {got}, the episode published {PIN[k]}")
    if wrong:
        raise SystemExit("this run does not match what the episode published:\n  "
                         + "\n  ".join(wrong)
                         + "\nThe README's numbers came off the published run. "
                           "Open an issue with both sets rather than editing it.")


def calls(m):
    """How many times the language model was reached. It must stay at zero."""
    client = getattr(m.llm, "client", None)
    return 0 if client is None else len(getattr(client, "_request_log", []) or [])


def remember(m, text, who):
    return m.add(text, user_id=who, infer=False)


def recall(m, query, who, top_k=4):
    r = m.search(query, filters={"user_id": who}, threshold=0.0, top_k=top_k)
    rows = r["results"] if isinstance(r, dict) else r
    return [{"memory": x["memory"], "score": round(float(x["score"]), 4)} for x in rows]


def hush():
    """fastembed and posthog both write to stderr; the run's own log is enough."""
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
