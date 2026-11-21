"""The filing corpus, generated from one seed.

WHAT IS GENERATED AND WHAT IS MEASURED. The companies, their filings and the
wording inside those filings are built here from a seed. Nothing in this
episode is a reading of a real market, and the reel never says otherwise.

What is MEASURED is the store. Every figure the reel prints is Neo4j's own
profiler counting the records a query had to read, over this corpus, on a real
server. So the corpus has to be the kind of corpus the lesson is about, and
`run.py` refuses to write one that is not:

  a filing names an event IN ITS OWN WORDS, and two filings naming the same
    event do not have to agree on the words. That disagreement is the only
    reason a knowledge graph is worth building
  a specific incident is named by a HANDFUL of filings, not by half of them.
    An event every filing names is not an incident, it is a theme
  one event IS a theme, named by nearly everybody. That is what the second
    half of the reel measures, and it is the shape's own cost

THE EVENT IS A KIND OF THING THAT HAPPENED, AT A PLACE. "A fire at a tier-one
supplier plant in Kaohsiung" is one event; "a fire" is not. That is the grain a
risk factor is actually written at, and it is also what makes the event
specific enough to be worth a node -- which is the instruction the reel ends
on.
"""

from dataclasses import dataclass

SEED = 20261121

# the window the corpus covers, as named constants
YEARS = [2021, 2022, 2023, 2024, 2025]
COMPANIES = 120
SECTORS = ["semiconductors", "logistics", "specialty chemicals", "medical devices",
           "industrial software", "packaging"]

# How many events a filing names. A risk-factor section that names three is not
# a document anybody researches, and one that names forty is a dump.
EVENTS_PER_FILING = (6, 10)

# WHAT HAPPENED. Each kind carries the ways a filing writes it: `{0}` takes the
# place or the party. The first phrasing is the plain one, the rest are the ways
# the same thing gets written when somebody else writes it.
KINDS = [
    ("a fire", ["a fire at {0}", "fire damage at {0}", "production stopped at {0} after a fire"]),
    ("a flood", ["flooding at {0}", "flood damage at {0}"]),
    ("a strike", ["industrial action at {0}", "a labour stoppage at {0}", "a walkout at {0}"]),
    ("a cyber incident", ["a cyber incident at {0}", "unauthorised access at {0}"]),
    ("a recall", ["a product recall from {0}", "a field action on parts from {0}"]),
    ("a plant closure", ["the closure of {0}", "the shutdown of {0}"]),
    ("a shortage", ["a shortage of parts from {0}", "constrained supply from {0}"]),
    ("a contract loss", ["the loss of our contract with {0}", "non-renewal by {0}"]),
    ("a tariff", ["new duties on goods from {0}", "a tariff on imports from {0}"]),
    ("an export licence", ["export licensing for {0}", "a new licence requirement at {0}"]),
    ("a quality hold", ["a quality hold on parts from {0}", "parts from {0} held for inspection"]),
    ("a supplier failure", ["the insolvency of {0}", "the failure of {0}"]),
    ("a blockade", ["a blockade at {0}", "traffic held at {0}"]),
    ("a power cut", ["a power interruption at {0}", "loss of power at {0}"]),
    ("a water restriction", ["water restrictions at {0}", "limits on water use at {0}"]),
    ("a sole-source dependency", ["sole-source supply from {0}", "reliance on {0} as our only source"]),
]

# WHERE, OR WHO WITH. A site or a counterparty, named, because that is what
# makes an event one event.
PLACES = [
    "a tier-one supplier plant in Kaohsiung",
    "our packaging line in Guadalajara",
    "the port of Rotterdam",
    "a wafer fab in Hsinchu",
    "our distribution centre in Memphis",
    "a contract assembler in Penang",
    "the Suez Canal",
    "our plant in Dresden",
    "a resin supplier in Antwerp",
    "the port of Long Beach",
    "a rare-earth refiner in Baotou",
    "our testing site in Bangalore",
    "a logistics partner in Duisburg",
    "our largest hospital customer",
    "a solvent plant in Ludwigshafen",
    "the rail link through Suzhou",
    "our second-source foundry in Singapore",
    "a glass supplier in Toledo",
]

# THE THEME, AND IT IS WRITTEN EXACTLY LIKE EVERYTHING ABOVE. Nothing about it
# is special except how many filings name it, and the data decides that.
THEME = ("higher interest rates", [
    "higher interest rates",
    "a rising rate environment",
    "an increased cost of borrowing",
])
# the share of filings that name it. A theme is a theme because almost
# everybody writes it down.
THEME_SHARE = 0.86

# HOW TALKED-ABOUT AN INCIDENT IS, AND IT IS NOT FLAT. Some incidents are in
# everybody's filing and most are in a few. A flat corpus would make every
# event the same size and the hub a thing somebody arranged; a skewed one makes
# the hub the top of a distribution the data produced.
SKEW = 1.7


@dataclass(frozen=True)
class Filing:
    fid: str
    ticker: str
    year: int
    mentions: tuple        # the event names this filing names
    phrases: tuple         # the words it used for each, in the same order


def _rng():
    import random

    return random.Random(SEED)


def tickers():
    """Four-letter tickers, so a company on a frame is a name and not an index."""
    r = _rng()
    seen, out = set(), []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    while len(out) < COMPANIES:
        t = "".join(r.choice(alphabet) for _ in range(4))
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def catalogue():
    """Every event, as a kind that happened at a place, with its phrasings."""
    out = []
    for kind, forms in KINDS:
        for place in PLACES:
            out.append({
                "name": forms[0].format(place),
                "kind": "incident",
                "phrases": tuple(f.format(place) for f in forms),
            })
    out.append({"name": THEME[0], "kind": "theme", "phrases": tuple(THEME[1])})
    return out


def build():
    """The corpus. One dict, so the loader and the tests read the same thing."""
    r = _rng()
    names = tickers()
    companies = [{"ticker": t, "sector": SECTORS[i % len(SECTORS)]} for i, t in enumerate(names)]
    events = catalogue()
    incidents = [e for e in events if e["kind"] == "incident"]
    by_name = {e["name"]: e for e in events}

    # how often each incident gets written down, before anything is sampled
    weights = [1.0 / ((i + 1) ** (1.0 / SKEW)) for i in range(len(incidents))]
    order = list(range(len(incidents)))
    r.shuffle(order)
    weight = [weights[order.index(i)] for i in range(len(incidents))]

    filings = []
    for c in companies:
        for y in YEARS:
            lo, hi = EVENTS_PER_FILING
            k = r.randint(lo, hi)
            picked, seen = [], set()
            while len(picked) < k:
                e = r.choices(incidents, weights=weight, k=1)[0]
                if e["name"] in seen:
                    continue
                seen.add(e["name"])
                picked.append(e["name"])
            if r.random() < THEME_SHARE:
                picked.append(THEME[0])
            phr = tuple(r.choice(by_name[n]["phrases"]) for n in picked)
            filings.append(Filing(f"{c['ticker']}-{y}", c["ticker"], y, tuple(picked), phr))

    return {"companies": companies, "events": events, "filings": filings}


def mentions(world):
    """One row a mention: which filing, which event, and in whose words."""
    out = []
    for f in world["filings"]:
        for name, phrase in zip(f.mentions, f.phrases):
            out.append({"fid": f.fid, "ticker": f.ticker, "year": f.year,
                        "event": name, "phrase": phrase})
    return out


# THE QUESTION HAS TO BE ASKABLE, and these are the conditions under which it
# is. A text search and a traversal can only disagree where the same event is
# worded more than one way, so the anchor is chosen by that and not by the
# numbers it produces.
ANCHOR_OTHERS = 6          # other filings naming it, at least
ANCHOR_WORDINGS = 2        # ways they word it, at least


def anchor(world):
    """The company the reel asks about, and the filing it asks from.

    CHOSEN BY THE DATA. The latest filing, in identifier order, whose first
    incident mention is of an event that other filings both name and word
    differently. Picking a filing because its figures looked good is the thing
    this series refuses, so the rule is stated here and `run.py` refuses a
    corpus where nothing meets it.
    """
    rows = mentions(world)
    latest = max(YEARS)
    for f in sorted([f for f in world["filings"] if f.year == latest], key=lambda f: f.fid):
        for name, phrase in zip(f.mentions, f.phrases):
            if name == THEME[0]:
                continue
            others = [m for m in rows if m["event"] == name and m["fid"] != f.fid]
            worded = {m["phrase"] for m in others}
            if len(others) >= ANCHOR_OTHERS and len(worded) >= ANCHOR_WORDINGS and phrase in worded:
                return {"fid": f.fid, "ticker": f.ticker, "year": f.year,
                        "event": name, "phrase": phrase}
    raise SystemExit("no filing names an event that other filings word more than one way; "
                     "a text search and a traversal could not disagree, so there is no lesson")


def theme_filing(world):
    """The latest filing that names the theme, which is where the stress is asked."""
    latest = max(YEARS)
    for f in sorted([f for f in world["filings"] if f.year == latest], key=lambda f: f.fid):
        if THEME[0] in f.mentions:
            i = f.mentions.index(THEME[0])
            return {"fid": f.fid, "ticker": f.ticker, "year": f.year,
                    "event": THEME[0], "phrase": f.phrases[i]}
    raise SystemExit("no filing names the theme; there is no hub to measure")
