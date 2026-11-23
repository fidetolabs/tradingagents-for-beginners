"""What this episode's figures rest on, and what would move them.

EVERY EPISODE HAS TO FIND THIS, and the thing it finds is different every time.
#13's prices were rewritten by a vendor at every dividend. #14's model name
resolved to a different file. #15's db hits were a property of the plan the
planner chose. Here there are four, and the first one is the one that would
have made every number in the reel fiction.

1. `qdrant-client`'s local mode IS NOT THE QDRANT ENGINE. `QdrantClient(path=…)`
   and `QdrantClient(":memory:")` run a Python reimplementation in
   `qdrant_client/local/`, and its own source says what it does: in
   `local/qdrant_local.py`, `_has_ignored_search_params` lists `hnsw_ef`,
   `quantization`, `acorn`, `exact` and `indexed_only` as parameters it ignores,
   with the comment *"index behaviour that an exact brute-force search has no
   use for"*, and `local/local_collection.py` scores a query by calling
   `calculate_distance` over every vector in the collection. There is no graph.
   So a recall figure measured in local mode is 100% by construction, and this
   episode is about a search that is not exact. It talks to a server.

2. THE INDEXING THRESHOLD IS WHY A SERVER IS NOT ENOUGH. Qdrant leaves a
   segment unindexed until it holds `indexing_threshold` kilobytes, and an
   unindexed segment is searched exactly. Below the default a collection
   answers perfectly and the reel would be measuring brute force again through
   a real server. It is set here and the run waits for the collection to say
   `green`, which is Qdrant's own word for the index being built.

3. THE EMBEDDING MODEL IS NOT THE NAME YOU ASKED FOR. `fastembed` takes
   `BAAI/bge-small-en-v1.5` and loads `model_optimized.onnx` out of the Hugging
   Face repo `Qdrant/bge-small-en-v1.5-onnx-Q` — a re-export, not the original
   weights. That is #14's hazard exactly, so the revision and the file's sha256
   are pinned and the run hashes what it actually loaded.

4. RECALL IS A PROPERTY OF THE SEARCH PARAMETERS. `hnsw_ef` sets how wide the
   walk is, `m` and `ef_construct` build the graph, and
   `full_scan_threshold` decides whether a filtered search walks the graph at
   all. All five are written down below, because a figure that moves when a
   default changes is a figure nobody can reproduce.

5. AND THE ONE THAT DECIDED THE SHAPE OF THIS EPISODE: THE GRAPH BUILD IS NOT
   DETERMINISTIC. Qdrant assigns each point its level in the graph at random
   and exposes no seed, so building the same collection from the same vectors
   twice gives two different graphs. The pair below is unaffected, because both
   its sides are exact searches. The stress is affected completely: the same
   twelve questions, on the same data, missed 22, 55, 63, 41, 41, 71, 41 and 30
   passages of 120 on eight builds. So the stress is NOT published as one
   figure. What is published
   is what holds on every build — the declared side misses nothing — and the
   range the undeclared side moved over, with the number of builds beside it.
   A reader who runs this will not see the same range. They will see one inside
   that envelope, and `run.py` refuses a run that leaves it.

AND THE FILINGS THEMSELVES ARE PINNED BY ACCESSION AND BY HASH — see
`src/edgar.py` for why an EDGAR accession is the one input in this series that
cannot be rewritten underneath a run.
"""

from __future__ import annotations

from dataclasses import dataclass

SEED = 20261122

# ── the server, the client, the model ────────────────────────────────────
QDRANT_VERSION = "1.20.2"
CLIENT_VERSION = "1.20.0"
FASTEMBED_VERSION = "0.9.1"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
MODEL_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
MODEL_FILE = "model_optimized.onnx"
MODEL_SHA256 = "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"
VECTOR_SIZE = 384
DISTANCE = "Cosine"

# ── the index, and the search ────────────────────────────────────────────
# TWO COLLECTIONS, ONE DIFFERENCE. Same points, same vectors, same filter. The
# only thing that changes is whether the field the filter uses was declared
# before the points went in. Anything else moving would make the stress a
# comparison of two runs rather than of one decision.
COLLECTION = "filings"
COLLECTION_UNDECLARED = "filings_undeclared"
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
# Kilobytes, not points. The default is 20,000 KB, which this corpus is under,
# so a collection left on the default is searched exactly and every recall
# figure comes back perfect. See note 2 above.
INDEXING_THRESHOLD_KB = 1
# Kilobytes again, and this one decides whether a FILTERED search walks the
# graph or reads the matching points directly: under it, Qdrant abandons the
# index and reads them. 10 is the smallest Qdrant accepts, and at 384 floats a
# point is about 1.5 KB, so ten kilobytes is about six passages — which means
# the filtered searches in this episode are graph walks and not scans. That is
# the honest setting for a reel about what a vector index does, and it is also
# the one a reader has to know: left on the 20,000 KB default, the narrow
# filter on the last beat is read exactly and comes back perfect.
FULL_SCAN_THRESHOLD_KB = 10
HNSW_EF = 128
TOP_K = 10

# ── the questions ────────────────────────────────────────────────────────
# TWELVE RESEARCH QUESTIONS, written as somebody would actually ask them: in
# their own words, not in the filings'. That is the point of the first half of
# the reel, and it is also what makes the question set a fair test — none of
# these is a quotation.
QUESTIONS: tuple[str, ...] = (
    "we could not get enough substrate and assembly capacity from our outside "
    "suppliers, so we were unable to ship everything our customers ordered",
    "a shortage of foundry wafer starts limited how much we could build",
    "freight and logistics costs rose sharply and we could not pass all of it on",
    "we wrote down inventory because demand fell faster than we could cut production",
    "a single plant makes most of one product line, so damage there would stop shipments",
    "export controls stopped us selling certain products to customers in China",
    "we depend on a small number of distributors for a large share of revenue",
    "the cost of building new capacity rose and the payback is years away",
    "we could not hire or keep enough engineers in a tight labour market",
    "a cyber attack on our systems interrupted manufacturing and order entry",
    "currency moves reduced reported revenue when we converted foreign sales",
    "an acquisition did not deliver the savings we expected and we recorded a charge",
)
# THE ONE THE REEL DRAWS. The pair on screen is this question's own, and the
# other eleven are what the stress is measured over, because one question's
# recall on one random graph is noise.
ANCHOR = 0
# The year the agent is asking about. Four years are in the collection, so this
# filter keeps about a quarter of the passages.
ASK_YEAR = 2022
# How many times the undeclared collection is rebuilt, because the figure moves
# with the build and one reading of it would be a figure nobody can reproduce.
BUILDS = 8

# THE ENVELOPE, NOT THE NUMBER. See note 5. A run inside this reproduces the
# claim; a run outside it means something moved that this file does not name.
UNDECLARED_MIN = 3          # it has to miss something, or there is no stress
UNDECLARED_MAX = 110        # and if it misses nearly all of them, something else is wrong


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: int
    accession: str
    year: int
    document: str
    sha256: str

    @property
    def fid(self) -> str:
        return f"{self.ticker}-{self.year}"


# TWENTY-FOUR 10-K DOCUMENTS, six filers over four years, all of them from one
# industry so that the passages are genuinely close to each other in meaning
# and a text search has something to get wrong. The years are the period the
# filing covers, which is the field the agent filters on.
FILINGS: tuple[Filing, ...] = (
    Filing("INTC", 50863, "0000050863-21-000010", 2020, "intc-20201226.htm",
           "29eec0e2d638e046c0f813486e845d680d6e9d7b0e7eef91937df0b5ae925f7b"),
    Filing("INTC", 50863, "0000050863-22-000007", 2021, "intc-20211225.htm",
           "4d306b7fd2702c23ef5c218b0ceb5af0e5ffa4481aabb269a5538b2f448d2244"),
    Filing("INTC", 50863, "0000050863-23-000006", 2022, "intc-20221231.htm",
           "bcb81dabed7395975f212b5958d50f04f0d77d70029a77a2518869b296969b28"),
    Filing("INTC", 50863, "0000050863-24-000010", 2023, "intc-20231230.htm",
           "725f64e156ffefe739cdbc42c12b1ccff244b7fc75895be848ab7dcba64dc3dd"),
    Filing("MU", 723125, "0000723125-20-000082", 2020, "mu-20200903.htm",
           "15cdc8f3759b4883fc66f11fafaa0029e57912beebc9e06eb13eb9060c1eefdf"),
    Filing("MU", 723125, "0000723125-21-000065", 2021, "mu-20210902.htm",
           "75319398c81d512adb453489c7696b1415e98f3693db09be4c6e254f153d1fcd"),
    Filing("MU", 723125, "0000723125-22-000048", 2022, "mu-20220901.htm",
           "511c052245a814b958ae332bf417c8747aa79aca672824a9cc314fbeaa5851d9"),
    Filing("MU", 723125, "0000723125-23-000054", 2023, "mu-20230831.htm",
           "e9a2dd8d64c3b22b0606fe40e8c3da4a7454eab285468918722b7e0d66000d54"),
    Filing("TXN", 97476, "0000097476-21-000006", 2020, "txn-20201231.htm",
           "72e90a329670a0696e10b515cee788fda72eb13358be9d1c276559d4a52038f6"),
    Filing("TXN", 97476, "0000097476-22-000009", 2021, "txn-20211231.htm",
           "01bb213519d320bdb5cd66f4f7347a6e19c503390f1c85f832fbafe3e42ef0db"),
    Filing("TXN", 97476, "0000097476-23-000007", 2022, "txn-20221231.htm",
           "b4efbc4f14028e95924ec7472af7f6d07d28576671d8d424ffceaea2a9406a9e"),
    Filing("TXN", 97476, "0000097476-24-000007", 2023, "txn-20231231.htm",
           "62bfbf754ebaaffda3296b1c19372e389486f4ed122706d55ec8a132bbe95f32"),
    Filing("ADI", 6281, "0000006281-20-000156", 2020, "adi-20201031.htm",
           "eaa35a45fa07326cbc8579371018e90a1c24ec8493ffac4c1fe09f27a95694b0"),
    Filing("ADI", 6281, "0000006281-21-000294", 2021, "adi-20211030.htm",
           "78f53e4c3b6e4d5fbb3adf6fc1832dc2edc4d17d8ebb1658c3d36e485410e3a6"),
    Filing("ADI", 6281, "0000006281-22-000250", 2022, "adi-20221029.htm",
           "6dcbac45d9a73f322b8f1ec5e3a7e8cabfd36e7b1b5d52e4779d38bb72fe1bf3"),
    Filing("ADI", 6281, "0000006281-23-000203", 2023, "adi-20231028.htm",
           "441df4485ee578f8fbcfb4e4d6fffdda46ef37f6c3d79f82107b2b88c36bc9a1"),
    Filing("ON", 1097864, "0001628280-21-002219", 2020, "on-20201231.htm",
           "d090f4d8f3d82394562ffa14f0a8f9bf972681fde75f964e95bc201e01c65b3d"),
    Filing("ON", 1097864, "0001628280-22-002416", 2021, "on-20211231.htm",
           "cc2c6797c3dc2bd42d44f8971bf75a2e014ca826aad678a6e11986ae8b5dfbbf"),
    Filing("ON", 1097864, "0001628280-23-002350", 2022, "on-20221231.htm",
           "90a3206be6163b58c81534e746f710da07257beb187dea881244653ea17b45f5"),
    Filing("ON", 1097864, "0001628280-24-003201", 2023, "on-20231231.htm",
           "f90f50ae453610c072db3a2703b9f3086440cf95554af7e893ee1cd72998c352"),
    Filing("QRVO", 1604778, "0001604778-20-000030", 2020, "qorvo2020032810-k.htm",
           "7590d7604663a452c4f5c9689173eaf85aad36f112e27991679d7e5c3bae1528"),
    Filing("QRVO", 1604778, "0001604778-21-000032", 2021, "rfmd-20210403.htm",
           "da5bdbaa9d25434d061f272d32693a66294553c8c09e98cfcdbfa154b9061d09"),
    Filing("QRVO", 1604778, "0001604778-22-000029", 2022, "rfmd-20220402.htm",
           "4cec5383acec856c02826095ce02a584ff88b45506c1b70115947216bc97b729"),
    Filing("QRVO", 1604778, "0001604778-23-000055", 2023, "rfmd-20230401.htm",
           "9e878b3ac1932da5300965ff622eef02d71f68abd9c3bd5ef9c4b8c8932363ad"),
)

YEARS = tuple(sorted({f.year for f in FILINGS}))
TICKERS = tuple(sorted({f.ticker for f in FILINGS}))

# ── the figures ──────────────────────────────────────────────────────────
# EVERY ONE OF THESE IS REPRODUCIBLE TO THE PASSAGE, because each comes out of
# an exact search over an immutable corpus. `run.py` refuses to write a run
# that does not produce them. Set FIGURES to None to take a reading on a new
# corpus, a new model or a new question.
#
# THE STRESS IS NOT IN HERE, and note 5 above says why: it moves with the graph
# build. What bounds it is UNDECLARED_MIN and UNDECLARED_MAX, and what the reel
# prints is the range this run measured, with the number of builds beside it.
FIGURES: dict | None = {
    "after": {"asked_for": 36, "cite": 3, "filers": 2, "thrown": 26},
    "inside": {"cite": 10, "filers": 5},
    "passages": 9085,
    "question_words": 12,
    "shared_words": 2,
    "year_passages": 2222,
}
