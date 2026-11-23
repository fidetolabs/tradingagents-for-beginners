"""The collection: one point a passage, with the filing beside it.

WHAT GOES IN A POINT. The vector is what the passage means and the payload is
what it is — which filing, which filer, which year. That split is the whole
architecture of this episode: the vector is how the search finds anything, and
the payload is how the search is restricted to what the agent is allowed to
cite. Both live on the same point, which is why the restriction can happen
inside the search instead of after it.

AND THE PAYLOAD FIELD IS DECLARED. `create_payload_index` is not an
optimisation here; it is what lets Qdrant plan a filtered search at all. Qdrant
needs the field indexed to know how many points a filter keeps, and that
estimate is what it uses to decide between walking the graph and reading the
matching points directly.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from . import edgar, pin

CACHE = Path(__file__).resolve().parent.parent / ".cache"


def embedder():
    """The model, and a check that it is the file `pin.py` names.

    #14 shipped with a model name that resolved to a different file, so the
    name is not the pin: the bytes are. fastembed puts the ONNX under its cache
    by Hugging Face repo and revision, so after loading there is exactly one
    file to hash.
    """
    from fastembed import TextEmbedding

    cache = CACHE / "fastembed"
    cache.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(pin.MODEL_NAME, cache_dir=str(cache))

    onnx = sorted(cache.rglob(pin.MODEL_FILE))
    if len(onnx) != 1:
        raise SystemExit(
            f"{len(onnx)} copies of {pin.MODEL_FILE} under {cache}; expected one. "
            f"Delete {cache} and run again."
        )
    got = hashlib.sha256(onnx[0].read_bytes()).hexdigest()
    if got != pin.MODEL_SHA256:
        raise SystemExit(
            f"fastembed loaded {onnx[0].name} with sha256 {got}, and src/pin.py names "
            f"{pin.MODEL_SHA256}.\nThe name {pin.MODEL_NAME} resolved to different bytes, "
            f"which is exactly what #14 shipped. Every vector in this run, and so every "
            f"figure, would be a different number."
        )
    if str(onnx[0]).find(pin.MODEL_REVISION) < 0:
        raise SystemExit(f"{onnx[0]} is not revision {pin.MODEL_REVISION}")
    return model


def corpus() -> list[dict]:
    """Every passage in every pinned filing, in filing order then document order."""
    out: list[dict] = []
    for f in pin.FILINGS:
        text = edgar.text_of(edgar.fetch(f))
        for i, p in enumerate(edgar.passages(text)):
            out.append({"fid": f.fid, "ticker": f.ticker, "year": f.year,
                        "accession": f.accession, "ordinal": i, "text": p})
    return out


def vectors(model, rows: list[dict]) -> np.ndarray:
    """Embed every passage, cached by the corpus's own hash.

    The cache is keyed on the text of every passage and on the model's hash, so
    it cannot survive a change to either. It exists because embedding the corpus
    takes minutes and the questions get asked many times while a reel is built.
    """
    key = hashlib.sha256(
        (pin.MODEL_SHA256 + "\n".join(r["text"] for r in rows)).encode()
    ).hexdigest()[:32]
    path = CACHE / f"vectors-{key}.npy"
    if path.exists():
        return np.load(path)
    V = np.array(list(model.embed([r["text"] for r in rows], batch_size=64)), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, V)
    return V


def connect(url: str) -> QdrantClient:
    """A server, and never local mode.

    `QdrantClient(path=…)` and `":memory:"` are a Python brute-force search with
    no graph in them — see note 1 in `src/pin.py`. This episode measures a
    search that is not exact, so a run against local mode would be measuring
    nothing, and it is refused rather than warned about.
    """
    if url.startswith(":memory:") or url.startswith("file:") or "://" not in url:
        raise SystemExit(
            f"{url!r} is local mode, which is an exact brute-force search with no HNSW "
            f"graph in it. Every recall figure in this episode would come back at 100% "
            f"by construction. Start a server and point QDRANT_URL at it — "
            f"agent/README.md has three ways."
        )
    c = QdrantClient(url=url, timeout=600)
    got = c.info().version
    if got != pin.QDRANT_VERSION:
        raise SystemExit(
            f"the server is Qdrant {got} and src/pin.py names {pin.QDRANT_VERSION}. "
            f"Recall is a property of this planner and this index, so expect every "
            f"figure to move. Update src/pin.py and README.md together and say what moved."
        )
    return c


def load(client: QdrantClient, name: str, rows: list[dict], V: np.ndarray,
         declare: bool = True) -> None:
    """Build a collection from scratch, every time.

    A collection that is added to twice holds every passage twice, and a recall
    figure over that is a figure nobody can reproduce.

    `declare` is the one difference between the episode's two collections: with
    it the fields the agent filters on get a payload index before the points go
    in, and without it Qdrant has nothing to plan a filtered search with. Same
    points, same vectors, same filter.
    """
    client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=pin.VECTOR_SIZE, distance=models.Distance[pin.DISTANCE.upper()]
        ),
        hnsw_config=models.HnswConfigDiff(
            m=pin.HNSW_M,
            ef_construct=pin.HNSW_EF_CONSTRUCT,
            full_scan_threshold=pin.FULL_SCAN_THRESHOLD_KB,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=pin.INDEXING_THRESHOLD_KB
        ),
    )
    # THE FIELDS THE AGENT FILTERS ON, DECLARED. Without these Qdrant cannot
    # estimate how many points a filter keeps, and a filtered search it cannot
    # plan is a filtered search it gets wrong.
    if declare:
        for field in ("year", "ticker", "fid"):
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.INTEGER
                if field == "year"
                else models.PayloadSchemaType.KEYWORD,
            )
    client.upload_collection(
        collection_name=name,
        vectors=V,
        payload=[{k: r[k] for k in ("fid", "ticker", "year", "accession", "ordinal", "text")}
                 for r in rows],
        ids=list(range(len(rows))),
        batch_size=256,
    )
    wait_indexed(client, name)


def wait_indexed(client: QdrantClient, name: str, limit: float = 600.0) -> None:
    """Wait for Qdrant's own word that the index is built.

    AN UNINDEXED SEGMENT IS SEARCHED EXACTLY, so a run that measures before the
    graph exists measures brute force through a real server and reports perfect
    recall. `green` is Qdrant's own status for a collection with nothing left to
    optimise.
    """
    t0 = time.time()
    while time.time() - t0 < limit:
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN and info.indexed_vectors_count:
            return
        time.sleep(1.0)
    info = client.get_collection(name)
    raise SystemExit(
        f"the collection is still {info.status} with "
        f"{info.indexed_vectors_count} of {info.points_count} vectors indexed after "
        f"{limit:.0f}s. An unindexed segment is searched exactly, so nothing measured "
        f"now would be an approximate search."
    )
