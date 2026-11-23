"""The filings, fetched from EDGAR and cut into passages.

NO DATASET IN THE REPO. `CONTRIBUTING.md` makes that law, and it is the right
law here for a second reason: a copy of twenty-four 10-K documents in git is a
hundred megabytes of somebody else's text. What is committed is the list of
accession numbers and the sha256 of each document, so a reader fetches the same
bytes and the run refuses to continue if EDGAR hands back anything else.

THE ACCESSION IS THE PIN. An EDGAR accession points at an immutable archive: the
document behind `0000050863-24-000010` is the same file this year and next. That
is what lets this episode publish a figure somebody else can reproduce, and it
is the answer to the question every episode has to ask about its own numbers.
#13's prices were rewritten by a vendor at every dividend; #14's model name
resolved to a different file; this one's inputs cannot move, and the hashes
prove it.
"""

from __future__ import annotations

import gzip
import hashlib
import html as htmllib
import os
import re
import time
import urllib.request
from pathlib import Path

from . import pin

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "edgar"

# EDGAR asks for a real contact in the User-Agent and throttles to ten requests
# a second. One request a second is polite and this runs twenty-four of them.
#
# THE CONTACT IS YOURS, NOT THIS REPO'S. EDGAR wants to reach whoever is making
# the requests, so it comes from your environment and there is no default to
# fall back on. A hard-coded address would send every reader's traffic out under
# one name, and `CONTRIBUTING.md` keeps addresses out of this repo anyway.
PAUSE = 1.0


def _ua() -> str:
    ua = os.environ.get("EDGAR_CONTACT", "").strip()
    if not ua:
        raise SystemExit(
            "EDGAR wants to know who is asking, so set EDGAR_CONTACT to your own\n"
            "name and email before the fetch:\n\n"
            '  export EDGAR_CONTACT="Your Name (you@example.com)"\n\n'
            "Nothing is downloaded without it. See https://www.sec.gov/os/webmaster-faq"
        )
    return ua


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _ua(), "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=180) as f:
        raw = f.read()
        if f.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return raw


def fetch(f: pin.Filing) -> bytes:
    """One filing's primary document, cached by accession and checked by hash."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{f.accession}-{f.document}"
    if not path.exists():
        url = (f"https://www.sec.gov/Archives/edgar/data/{f.cik}/"
               f"{f.accession.replace('-', '')}/{f.document}")
        path.write_bytes(_get(url))
        time.sleep(PAUSE)
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != f.sha256:
        raise SystemExit(
            f"{f.accession} came back as sha256 {got}, and src/pin.py names {f.sha256}.\n"
            f"Delete {path} and try again. If it keeps happening, EDGAR is serving a "
            f"different document and every figure in this episode rests on the old one."
        )
    return raw


def text_of(raw: bytes) -> str:
    """The document's words, with the markup taken out.

    A 10-K is one XBRL-inline HTML file: tables, hidden iXBRL headers, a few
    hundred kilobytes of style. What is wanted is the prose a person reads, so
    the tags come out and whitespace is collapsed. Nothing here tries to be a
    parser — the passages only have to be the same passages every run, and a
    deterministic strip over an immutable file is enough for that.
    """
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", s)
    # the iXBRL header block is machine metadata and holds no prose at all
    s = re.sub(r"(?is)<ix:header\b.*?</ix:header>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = htmllib.unescape(s)
    s = s.replace(" ", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", " - ").replace("–", "-")
    s = re.sub(r"[^\S\n]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return re.sub(r"\n{2,}", "\n", s).strip()


# A PASSAGE IS WHAT THE AGENT WOULD CITE, so it is sized to be quotable: long
# enough to carry a claim and its subject, short enough that a reader can check
# it against the filing. Sentence boundaries, not characters, because a passage
# cut mid-sentence is one nobody can quote.
PASSAGE_CHARS = 900
PASSAGE_MIN = 240


def passages(text: str) -> list[str]:
    """Cut the filing into quotable passages on sentence boundaries.

    A 10-K carries long runs of table residue — numbers, dates and single words
    left behind once the markup goes. Those are dropped, because a passage with
    no sentence in it is not something an agent can cite, and twenty thousand of
    them would be the collection.

    PASSAGE_CHARS IS A CEILING AND NOT A TARGET, and the first cut of this
    treated it as a target. It broke only where a sentence ended, so a run with
    no sentence end in it became one passage however long it ran: a chart panel
    out of Intel's 2020 filing came through at 9,845 characters, which is not
    something anybody could quote and is not what any of its neighbours in the
    collection look like. A run that long is not prose, so it goes.
    """
    out: list[str] = []
    buf = ""
    for sent in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        sent = sent.strip()
        if not sent or len(sent) > PASSAGE_CHARS:
            if buf:
                out.append(buf)
                buf = ""
            continue
        if len(buf) + len(sent) + 1 > PASSAGE_CHARS:
            out.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        out.append(buf)
    return [p for p in out if _is_prose(p)]


def _is_prose(p: str) -> bool:
    words = p.split()
    if len(words) < 40 or len(p) < PASSAGE_MIN:
        return False
    # table residue: mostly figures, or almost no sentence ends
    digits = sum(c.isdigit() for c in p)
    if digits > len(p) * 0.12:
        return False
    return p.count(".") >= 2
