"""One question, asked three ways, and the right answer to compare them with.

THE RIGHT ANSWER IS QDRANT'S OWN. `search_params={"exact": true}` is a full scan
of the points the filter keeps, so it returns the true nearest passages by
definition. Every recall figure in this episode is measured against that, which
is why none of it needs anybody to judge whether a passage is a good one: the
question is only ever whether the search found the passages it should have.

THE THREE WAYS, and the difference between them is where the filter runs:

    after     ask for the k nearest passages in the whole collection, then keep
              the ones from the year the agent asked about
    inside    hand Qdrant the filter and ask for k

BOTH SIDES OF THE PAIR ARE EXACT, so the pair is a fact about where a filter
runs rather than about any index, and it reproduces down to the passage id. The
index only comes into it on the last beat, where `filter_inside` is pointed at
a collection whose filter field was never declared.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from . import pin


def year_filter(year: int) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="year", match=models.MatchValue(value=year))])


def fid_filter(fid: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="fid", match=models.MatchValue(value=fid))])


def _hits(client: QdrantClient, qv, *, limit: int, flt=None, exact=False, ef=None,
          collection=None):
    return client.query_points(
        collection_name=collection or pin.COLLECTION,
        query=qv,
        query_filter=flt,
        limit=limit,
        with_payload=True,
        search_params=models.SearchParams(
            hnsw_ef=(pin.HNSW_EF if ef is None else ef), exact=exact
        ),
    ).points


def exact_answer(client: QdrantClient, qv, flt, k: int) -> list:
    """The true k nearest passages among the ones the filter keeps."""
    return _hits(client, qv, limit=k, flt=flt, exact=True)


def filter_inside(client: QdrantClient, qv, flt, k: int, collection=None) -> list:
    """Hand the filter to Qdrant: the restriction happens during the search."""
    return _hits(client, qv, limit=k, flt=flt, collection=collection)


def asked_for_exact(client: QdrantClient, qv, keep, want: int, cap: int) -> tuple[int, int]:
    """How many passages you must ask for to get `want` past the filter.

    THE HONEST VERSION OF THE PAIR. The first side does not have to give up: it
    can ask for more and keep filtering. So this finds where the `want`-th
    matching passage sits in the true ranking, which is the price of doing it
    that way — how many passages the agent reads and how many it throws away.

    ORDERED BY SCORE AND THEN BY POINT ID, which is not fussiness. Qdrant
    searches each segment and merges, and the segment layout changes with every
    build, so two passages on the same score come back in either order and this
    figure moved by one between two runs of the same data. The pinned figures
    have to be reproducible, so the tie is broken here on something that cannot
    move. The scores are still Qdrant's own, out of an exact search.
    """
    got = _hits(client, qv, limit=min(cap, 400), exact=True)
    got = sorted(got, key=lambda h: (-h.score, h.id))
    n = 0
    for i, h in enumerate(got, start=1):
        if keep(h.payload):
            n += 1
            if n == want:
                return i, i - want
    return 0, 0
