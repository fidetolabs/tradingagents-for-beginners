"""The world the agent searches, and the screen that scores one try.

GENERATED, AND DECLARED AS GENERATED. The reel never says a number here was
measured on a real market; it says what the memory layer does to a search. What
has to be real is the SEARCH -- that a try has a score, that a score can be
below the bar, and that the same way of picking can pass in one place and fail
in another. That last one is the whole reason a memory needs to carry where it
was tried, so it is asserted rather than assumed.

A try is two things the series already names -- what you rank companies on, and
how many of the top ones you keep -- plus the two conditions that make one try
different from another: which industry, and over which years.

Everything is seeded. The same seed gives the same numbers on every machine,
which is what lets `build-data.mjs` assert against them.
"""

import numpy as np

SEED = 20261120
INDUSTRIES = ["software", "retail", "banks", "energy", "health", "industrials"]
METRICS = ["revenue growth", "cash yield", "gross margin",
           "debt to assets", "earnings growth", "sales per asset"]
WINDOWS = ["2015 to 2017", "2018 to 2020", "2021 to 2023", "2024 to 2026"]
KEEPS = [10, 20, 30]

PER_INDUSTRY = 40          # companies in each one
MONTHS = 36                # a window is three years of monthly returns
PATHS = 240                # one history has a standard error several times the
                           # effect being measured, so every score is an average
BAR = 0.25                 # the number a try has to beat to be worth keeping


def build(seed=SEED):
    """Quality, how each metric sees it, how much it pays, and the returns."""
    g = np.random.default_rng(seed)
    nI, nW, nM, nC = len(INDUSTRIES), len(WINDOWS), len(METRICS), PER_INDUSTRY

    # what is actually good about a company, in one industry over one window
    quality = g.standard_normal((nI, nW, nC))

    # HOW WELL A METRIC SEES QUALITY, AND IT DEPENDS ON THE INDUSTRY. Cash
    # yield says something about a bank that it does not say about software.
    loading = g.uniform(0.05, 0.95, (nM, nI))

    # AND HOW MUCH BEING GOOD PAID, WHICH DEPENDS ON THE WINDOW TOO. This is
    # the second half of why a scope matters: a metric that saw quality
    # perfectly still earns nothing in a window where quality was not paid for.
    payoff = g.uniform(0.0, 0.9, (nI, nW))

    # the metric a screen actually reads: quality seen through the loading
    seen = np.empty((nM, nI, nW, nC))
    for m in range(nM):
        for i in range(nI):
            lo = loading[m, i]
            seen[m, i] = lo * quality[i] + np.sqrt(max(1e-9, 1 - lo * lo)) * g.standard_normal((nW, nC))

    # monthly returns: a drift that pays quality, plus noise
    drift = np.empty((nI, nW, nC))
    for i in range(nI):
        for w in range(nW):
            drift[i, w] = 0.004 * payoff[i, w] * quality[i, w]
    noise = g.normal(0.0, 0.055, (PATHS, nI, nW, nC, MONTHS))
    rets = drift[None, :, :, :, None] + noise

    return {"quality": quality, "loading": loading, "payoff": payoff,
            "seen": seen, "rets": rets}


def score(W, metric, keep, industry, window):
    """Run one try. Returns the information ratio and its standard error.

    Rank the industry's companies on the metric, keep the top `keep`, and
    measure the kept basket against the whole industry over the window. One
    number a path, averaged, so the standard error is reportable.
    """
    m, i, w = METRICS.index(metric), INDUSTRIES.index(industry), WINDOWS.index(window)
    order = np.argsort(-W["seen"][m, i, w])
    kept = order[:keep]
    r = W["rets"][:, i, w]                     # paths x companies x months
    excess = r[:, kept].mean(axis=1) - r.mean(axis=1)   # paths x months
    ir = excess.mean(axis=1) / excess.std(axis=1, ddof=1) * np.sqrt(12)
    return float(ir.mean()), float(ir.std(ddof=1) / np.sqrt(PATHS))


def verdict(sc, se):
    """Clearly under the bar, clearly over it, or too close to call.

    THE GAP HAS TO BE SEVERAL TIMES THE ERROR ON THE GAP or the reel is
    reporting noise, which is the format's own rule and the one that caught a
    single-path sweep returning a result that reversed when it was averaged.
    """
    if sc + 2 * se < BAR:
        return "failed"
    if sc - 2 * se > BAR:
        return "good"
    return "unclear"
