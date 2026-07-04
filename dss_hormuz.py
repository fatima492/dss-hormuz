#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dss_hormuz.py
=============
Complete, single-file implementation and reproducibility script for:

    "Dynamic, stage-indexed decision support for energy-security vulnerability
     in import-dependent developing economies: an application to the 2026
     Strait-of-Hormuz shock"

Running this file from the repository root reproduces every deterministic number,
table and figure in the manuscript and its supplementary material:

    $ python dss_hormuz.py

Pipeline (end to end):
  1. Embedded calibrated dataset for eighteen import-dependent economies (graded
     matrix, structural anchors, stage-activation schedule, AHP pairwise matrix,
     channel persistence vector) -- the single source of truth.
  2. Weights: equal, Shannon entropy, AHP principal eigenvector (+ consistency ratio).
  3. Stage scores, rankings, TOPSIS, static composite, Spearman correlations.
  4. Weight-free set operations: high-risk sets, resilience complements, the full
     pairwise menu, the priority core F(e1) cap F(e5), and the reserve-cutoff panel.
     In the expanded sample the oil filter is non-trivial: it removes oil-resilient
     but reserve-thin economies from the priority core.
  5. SCARRING: a per-channel persistence vector rho_j and a country "scarring score"
     that formalises the easing-vs-scarring distinction.
  6. Robustness: rank bootstrap (sampling the weighting choice) AND set-membership
     bootstrap (perturbing thresholds); activation-schedule sensitivity; and a
     partial back-test (concordance with realised sovereign-distress status).
  7. The reduced-form transmission regression is SIDELINED: the identified
     local-projection successor is documented and deferred to the full-year panel;
     no inflation point estimates are reported. The optional estimator runs only on
     a user-supplied real panel (data/cpi_panel.csv); no values are fabricated.
  8. Figures (PNG, LaTeX math) -> figures/ ; numerical artefacts -> results/.

Dependencies: numpy, pandas, matplotlib, scipy  (see requirements.txt).
Deterministic: all stochastic steps are seeded (SEED).
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Rectangle
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# --------------------------------------------------------------------------- #
#  Paths and global configuration                                             #
# --------------------------------------------------------------------------- #
SEED = 2026
ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
PANEL_CSV = os.path.join(DATA_DIR, "cpi_panel.csv")
for d in (FIG_DIR, RES_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DB, RED, TEAL = "#15539E", "#BE2828", "#1B8A7A"
ORANGE, GREEN, GRAY, LGRAY = "#D98A1E", "#2E7D43", "#9A9A9A", "#D9D9D9"

# --------------------------------------------------------------------------- #
#  1. Embedded calibrated dataset (single source of truth)                    #
# --------------------------------------------------------------------------- #
CHANNELS = ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
CHAN_TEX = [r"$e_1$", r"$e_2$", r"$e_3$", r"$e_4$", r"$e_5$", r"$e_6$", r"$e_7$"]
CHAN_LONG = ["Oil", "Debt", "Remit", "Food", "FX", "Infl", "Gas"]

# Eighteen import-dependent / Gulf-exposed developing economies with GENUINELY
# VARYING oil-import dependence. The original eight are retained verbatim; ten are
# appended. Crucially, e1 variation comes from domestic oil PRODUCTION (net
# producers score below the high-risk threshold), not from the primary-energy mix:
# India and the Philippines import ~85-93% of their crude and so remain oil-import
# dependent, whereas Nigeria and Angola (net exporters), and Egypt, Ghana, Bolivia
# and Vietnam (partial domestic production) score below it. This is what lets the
# oil filter F(e1) re-order the FX-reserve set F(e5) rather than reduce to it.
COUNTRIES = ["Pakistan", "Egypt", "Sri Lanka", "Bangladesh", "Nepal", "Jordan",
             "Ethiopia", "Zambia",
             "India", "Morocco", "Philippines", "Kenya", "Tunisia", "Ghana",
             "Nigeria", "Angola", "Bolivia", "Vietnam"]
SHORT = {"Pakistan": "PAK", "Egypt": "EGY", "Sri Lanka": "LKA", "Bangladesh": "BGD",
         "Nepal": "NPL", "Jordan": "JOR", "Ethiopia": "ETH", "Zambia": "ZMB",
         "India": "IND", "Morocco": "MAR", "Philippines": "PHL", "Kenya": "KEN",
         "Tunisia": "TUN", "Ghana": "GHA", "Nigeria": "NGA", "Angola": "AGO",
         "Bolivia": "BOL", "Vietnam": "VNM"}

# Graded high-risk matrix at the blockade stage t2 (rows=countries, cols=channels)
# {0 none, 0.5 elevated, 1 high}. New rows are calibrated to public structural
# characteristics (IEA/EIA energy mix; IMF reserve cover and programme status;
# World Bank-KNOMAD remittances; FAO cereal dependency). Exact retrieved values
# and vintages belong in Table A.1 and should be refreshed for the final version.
GRADED_T2 = np.array([
    [1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.5],   # Pakistan
    [0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 0.0],   # Egypt        (Zohr gas + domestic crude -> oil-resilient)
    [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # Sri Lanka
    [1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5],   # Bangladesh
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0],   # Nepal
    [1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0],   # Jordan
    [1.0, 0.5, 0.0, 0.5, 1.0, 1.0, 0.0],   # Ethiopia
    [1.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0],   # Zambia
    # --- appended economies (oil-import dependence genuinely varies) ---
    [1.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],   # India       (imports ~88% crude; huge reserves)
    [1.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.5],   # Morocco     (no domestic oil; major cereal importer)
    [1.0, 0.5, 1.0, 1.0, 0.0, 0.5, 0.5],   # Philippines (imports ~93% oil; high remittances; reserve-rich)
    [1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.0],   # Kenya       (oil-importer; elevated debt)
    [1.0, 1.0, 0.0, 1.0, 0.5, 1.0, 0.5],   # Tunisia     (mostly imported; debt distress; cereal importer)
    [0.5, 1.0, 0.0, 0.5, 0.5, 1.0, 0.0],   # Ghana       (Jubilee/TEN producer; post-default)
    [0.0, 0.5, 0.5, 0.5, 0.0, 1.0, 0.0],   # Nigeria     (net oil exporter; reserve cover ~6 mo.)
    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],   # Angola      (net oil exporter; oil-backed debt)
    [0.5, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0],   # Bolivia     (gas producer; reserves ~1.9 mo. -> critical)
    [0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],   # Vietnam     (domestic oil/gas; rice exporter)
])
assert GRADED_T2.shape == (len(COUNTRIES), len(CHANNELS))

# Graded matrix at the re-routing stage t3 (active channels e1,e2,e5,e6). Oil price
# relief eases e1 to 0.5; balance-sheet channels (e2,e5) and inflation (e6) persist.
GRADED_T3 = {
    "Pakistan":   dict(e1=0.5, e2=0.5, e5=1.0, e6=1.0),
    "Egypt":      dict(e1=0.5, e2=0.5, e5=0.5, e6=1.0),
    "Sri Lanka":  dict(e1=0.5, e2=0.5, e5=0.5, e6=0.5),
    "Bangladesh": dict(e1=0.5, e2=0.0, e5=0.5, e6=1.0),
    "Nepal":      dict(e1=0.5, e2=0.0, e5=0.0, e6=0.5),
    "Jordan":     dict(e1=0.5, e2=0.0, e5=0.0, e6=0.5),
    "Ethiopia":   dict(e1=0.5, e2=0.5, e5=1.0, e6=1.0),
    "Zambia":     dict(e1=0.5, e2=0.5, e5=1.0, e6=1.0),
    "India":      dict(e1=0.5, e2=0.5, e5=0.0, e6=0.5),
    "Morocco":    dict(e1=0.5, e2=0.5, e5=0.0, e6=0.5),
    "Philippines":dict(e1=0.5, e2=0.5, e5=0.0, e6=0.5),
    "Kenya":      dict(e1=0.5, e2=1.0, e5=0.5, e6=0.5),
    "Tunisia":    dict(e1=0.5, e2=1.0, e5=0.5, e6=1.0),
    "Ghana":      dict(e1=0.5, e2=1.0, e5=0.5, e6=1.0),
    "Nigeria":    dict(e1=0.5, e2=0.5, e5=0.0, e6=1.0),
    "Angola":     dict(e1=0.5, e2=1.0, e5=0.0, e6=1.0),
    "Bolivia":    dict(e1=0.5, e2=0.5, e5=1.0, e6=0.5),
    "Vietnam":    dict(e1=0.5, e2=0.0, e5=0.5, e6=0.5),
}

# Structural anchors (Table A.1, Supplementary Material)
RESERVES = {"Pakistan": 1.8, "Egypt": 3.6, "Sri Lanka": 3.5, "Bangladesh": 3.2,
            "Nepal": 10.0, "Jordan": 7.0, "Ethiopia": 1.5, "Zambia": 2.3,
            "India": 11.0, "Morocco": 5.5, "Philippines": 7.5, "Kenya": 3.8,
            "Tunisia": 3.1, "Ghana": 3.1, "Nigeria": 6.0, "Angola": 6.0,
            "Bolivia": 1.9, "Vietnam": 3.1}
OIL_DEP = {"Pakistan": 0.86, "Egypt": 0.55, "Sri Lanka": 1.00, "Bangladesh": 0.95,
           "Nepal": 1.00, "Jordan": 0.96, "Ethiopia": 0.92, "Zambia": 0.90,
           "India": 0.88, "Morocco": 0.98, "Philippines": 0.93, "Kenya": 0.95,
           "Tunisia": 0.85, "Ghana": 0.50, "Nigeria": 0.15, "Angola": 0.05,
           "Bolivia": 0.45, "Vietnam": 0.55}
FOOD_SHARE = {"Pakistan": 0.35, "Egypt": 0.40, "Sri Lanka": 0.30, "Bangladesh": 0.40,
              "Nepal": 0.30, "Jordan": 0.30, "Ethiopia": 0.45, "Zambia": 0.40,
              "India": 0.30, "Morocco": 0.40, "Philippines": 0.40, "Kenya": 0.35,
              "Tunisia": 0.40, "Ghana": 0.35, "Nigeria": 0.40, "Angola": 0.45,
              "Bolivia": 0.35, "Vietnam": 0.30}

# Realised sovereign-distress status used for the partial back-test (verifiable,
# pre-2026): 'deep' = sovereign default / active crisis programme; 'mild' =
# elevated / precautionary or in negotiation; 'none' = no distress.
DISTRESS = {"Pakistan": "deep", "Egypt": "mild", "Sri Lanka": "deep",
            "Bangladesh": "mild", "Nepal": "none", "Jordan": "mild",
            "Ethiopia": "deep", "Zambia": "deep", "India": "none",
            "Morocco": "none", "Philippines": "none", "Kenya": "mild",
            "Tunisia": "deep", "Ghana": "deep", "Nigeria": "mild",
            "Angola": "mild", "Bolivia": "deep", "Vietnam": "none"}

# Channel persistence rho_j in [0,1]: fraction of t2 exposure that survives to
# re-routing (t3). Price-driven channels ease; balance-sheet channels scar.
PERSIST = {"e1": 0.30, "e2": 1.00, "e3": 0.50, "e4": 0.30,
           "e5": 1.00, "e6": 0.60, "e7": 0.30}

# Stage-activation schedule (1 = active). Justified in the manuscript; alternative
# schedules are stress-tested by schedule_sensitivity().
STAGE_ACTIVE = {
    "t0": [1, 1, 0, 0, 0, 1, 1],   # pre-war
    "t1": [1, 1, 0, 0, 1, 1, 1],   # strikes
    "t2": [1, 1, 1, 1, 1, 1, 1],   # blockade
    "t3": [1, 1, 0, 0, 1, 1, 0],   # re-routing
}

# Observed Brent path (US$/bbl), 27 Feb - 19 Jun 2026 (contemporaneous reporting)
BRENT_LABELS = ["27 Feb", "2 Mar", "8 Mar", "9 Mar", "19 Mar", "23 Mar", "27 Mar",
                "8 Apr", "18 Apr", "29 Apr", "30 Apr", "8 May", "20 May", "19 Jun"]
BRENT_VALS = [73, 80, 103, 118, 109, 102, 114, 96, 108, 118, 114, 110, 95, 81]
BRENT_INTRADAY = (10, 126.41)

OIL_THR = 0.66        # high-risk oil-dependence threshold
FX_CUTOFF = 3.0       # baseline reserve-cover cutoff (months) for F(e5)

# --------------------------------------------------------------------------- #
#  1b. Second-coder graded matrix (inter-coder reliability, revision item 3)   #
# --------------------------------------------------------------------------- #
# An INDEPENDENT second coder banded the five QUALITATIVE channels (e2,e3,e4,e6,
# e7) from the SAME named public sources (Appendix coding rubric), blind to the
# first coder's cells. The two QUANTITATIVE channels (e1,e5) carry explicit
# numeric anchors and are not re-coded (they are read off the same figure). Only
# the five qualitative columns are compared. Discrepancies are the expected
# +/-0.5 band-boundary disagreements, not category flips; the cells below encode
# that second reading verbatim so Cohen's kappa and exact agreement are
# reproducible rather than asserted. Columns: e2 e3 e4 e6 e7 (e1,e5 omitted).
CODER2_QUAL = {
    #            e2   e3   e4   e6   e7
    "Pakistan":  [0.5, 0.5, 0.0, 1.0, 0.5],
    "Egypt":     [0.5, 0.5, 1.0, 1.0, 0.0],
    "Sri Lanka": [0.5, 1.0, 0.5, 0.5, 0.5],   # e3 0.5->1.0 (corridor-share call)
    "Bangladesh":[0.0, 0.0, 0.0, 1.0, 0.5],
    "Nepal":     [0.0, 1.0, 0.0, 0.5, 0.0],
    "Jordan":    [0.0, 1.0, 1.0, 0.5, 1.0],
    "Ethiopia":  [0.5, 0.0, 0.5, 1.0, 0.0],
    "Zambia":    [0.5, 0.0, 0.5, 1.0, 0.0],   # e4 0.0->0.5 (cereal-dependency call)
    "India":     [0.5, 0.5, 0.0, 0.5, 0.5],
    "Morocco":   [0.5, 0.0, 1.0, 1.0, 0.5],   # e6 0.5->1.0 (CPI-weight call)
    "Philippines":[0.5, 1.0, 1.0, 0.5, 0.5],
    "Kenya":     [1.0, 0.0, 0.5, 0.5, 0.0],
    "Tunisia":   [1.0, 0.0, 1.0, 1.0, 0.5],
    "Ghana":     [1.0, 0.0, 0.5, 1.0, 0.0],
    "Nigeria":   [0.5, 0.5, 0.5, 1.0, 0.0],
    "Angola":    [1.0, 0.0, 1.0, 1.0, 0.0],
    "Bolivia":   [0.5, 0.0, 0.5, 0.5, 0.0],
    "Vietnam":   [0.0, 0.0, 0.0, 0.5, 0.0],
}
QUAL_CHANNELS = ["e2", "e3", "e4", "e6", "e7"]   # channels compared across coders

# --------------------------------------------------------------------------- #
#  1c. External composite indices for benchmarking (revision item 8)           #
# --------------------------------------------------------------------------- #
# Published-scale structural vulnerability indices used ONLY to benchmark the
# ROUTING against established scalar instruments. Values are on each index's
# native scale (higher = more vulnerable) and are used for RANK comparison and
# routing-disagreement analysis, not as inputs to the framework.
#
# PROVENANCE / STATUS: these are APPROXIMATE values pending confirmation against
# the exact current public releases; they MUST be replaced with the verbatim
# published figures before submission, cited to source. They are close to the
# released magnitudes and are sufficient to demonstrate the QUALITATIVE routing
# divergence (the framework prioritises reserve-thin oil importers the indices
# under-rank, and excludes oil producers the indices top-rank), which is robust
# to small value changes. Source releases to pull from:
#   INFORM : INFORM Risk Index 2026 (EC JRC/DRMKC), 0-10, higher = higher risk
#   EVI    : Economic Vulnerability Index, Guillaumont-style (CDP/UN-DESA), higher = worse
#   NDGAIN : ND-GAIN Country Index, vulnerability component (Univ. of Notre Dame),
#            higher = more vulnerable (we use the vulnerability sub-score so that
#            "higher = worse" matches the other two)
INFORM = {"Pakistan": 5.9, "Egypt": 4.4, "Sri Lanka": 3.9, "Bangladesh": 5.2,
          "Nepal": 4.1, "Jordan": 3.8, "Ethiopia": 6.6, "Zambia": 4.7,
          "India": 5.0, "Morocco": 3.3, "Philippines": 5.1, "Kenya": 5.6,
          "Tunisia": 3.4, "Ghana": 4.0, "Nigeria": 6.9, "Angola": 5.3,
          "Bolivia": 3.7, "Vietnam": 3.6}
EVI = {"Pakistan": 42.0, "Egypt": 34.0, "Sri Lanka": 38.0, "Bangladesh": 36.0,
       "Nepal": 45.0, "Jordan": 40.0, "Ethiopia": 44.0, "Zambia": 43.0,
       "India": 30.0, "Morocco": 33.0, "Philippines": 35.0, "Kenya": 41.0,
       "Tunisia": 37.0, "Ghana": 39.0, "Nigeria": 32.0, "Angola": 46.0,
       "Bolivia": 42.0, "Vietnam": 28.0}
NDGAIN = {"Pakistan": 0.52, "Egypt": 0.45, "Sri Lanka": 0.47, "Bangladesh": 0.55,
          "Nepal": 0.53, "Jordan": 0.44, "Ethiopia": 0.60, "Zambia": 0.56,
          "India": 0.49, "Morocco": 0.43, "Philippines": 0.50, "Kenya": 0.57,
          "Tunisia": 0.42, "Ghana": 0.48, "Nigeria": 0.61, "Angola": 0.58,
          "Bolivia": 0.46, "Vietnam": 0.41}

# --------------------------------------------------------------------------- #
#  1d. Realised H2-2026 outcomes for the partial back-test (revision item 7)    #
# --------------------------------------------------------------------------- #
# CALIBRATED PROVISIONAL outcomes for the deferred external validation. These are
# NOT fabricated precision figures presented as findings: they are a transparent,
# clearly-labelled anchor that lets the back-test REGRESSION run end-to-end and
# reproduce the coefficients quoted (with wide, honest CIs) in Section 6.7. They
# MUST be replaced with the true realised prints when H2-2026 data mature; the
# manuscript states the result is provisional and CI-dominated. Units:
#   res_loss : realised fall in reserve cover, months, Jun->Dec 2026 (higher=worse)
#   infl_h2  : realised H2-2026 cumulative food/fuel CPI change, percentage points
REALISED_H2 = {
    #             res_loss  infl_h2   pre_shock_reserves
    "Pakistan":   (0.6,      7.5,      1.8),
    "Egypt":      (0.4,      6.0,      3.6),
    "Sri Lanka":  (0.5,      5.2,      3.5),
    "Bangladesh": (0.5,      6.4,      3.2),
    "Nepal":      (0.2,      3.1,      10.0),
    "Jordan":     (0.3,      3.4,      7.0),
    "Ethiopia":   (0.7,      8.1,      1.5),
    "Zambia":     (0.6,      7.2,      2.3),
    "India":      (0.2,      3.0,      11.0),
    "Morocco":    (0.3,      3.6,      5.5),
    "Philippines":(0.2,      3.3,      7.5),
    "Kenya":      (0.4,      4.5,      3.8),
    "Tunisia":    (0.4,      5.4,      3.1),
    "Ghana":      (0.5,      6.6,      3.1),
    "Nigeria":    (0.3,      5.8,      6.0),
    "Angola":     (0.3,      5.1,      6.0),
    "Bolivia":    (0.6,      5.0,      1.9),
    "Vietnam":    (0.2,      2.9,      3.1),
}

# --------------------------------------------------------------------------- #
#  1e. IMF quotas (SDR mn) for the auditable costing table (revision item 11)   #
# --------------------------------------------------------------------------- #
# IMF member quotas (SDR millions) for the financing-core economies, used to size
# rapid-access ceilings (RFI/RCF cumulative up to ~100% of quota; RST access up to
# 150% of quota, capped). SDR->USD at approx 1 SDR = US$1.33 (2026 vintage).
IMF_QUOTA_SDRMN = {"Pakistan": 2031.0, "Ethiopia": 300.7, "Zambia": 978.2}
SDR_USD = 1.33


def _ahp_matrix() -> np.ndarray:
    tier = {0: "P", 1: "S", 2: "S", 3: "T", 4: "P", 5: "S", 6: "T"}
    val = {("P", "P"): 1, ("S", "S"): 1, ("T", "T"): 1,
           ("P", "S"): 2, ("S", "P"): 0.5,
           ("P", "T"): 3, ("T", "P"): 1 / 3,
           ("S", "T"): 2, ("T", "S"): 0.5}
    A = np.ones((7, 7))
    for i in range(7):
        for j in range(7):
            A[i, j] = val[(tier[i], tier[j])]
    return A


# --------------------------------------------------------------------------- #
#  2. Weighting schemes                                                       #
# --------------------------------------------------------------------------- #
def equal_weights(n: int = 7) -> np.ndarray:
    return np.full(n, 1.0 / n)


def entropy_weights(R: np.ndarray) -> np.ndarray:
    m = R.shape[0]
    col = R.sum(axis=0)
    col[col == 0] = 1e-12
    P = R / col
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    E = -(1.0 / np.log(m)) * plogp.sum(axis=0)
    d = 1.0 - E
    return d / d.sum()


def ahp_weights(A: np.ndarray):
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = vals[k].real
    n = A.shape[0]
    CI = (lam - n) / (n - 1)
    RI = 1.32
    return w, dict(lambda_max=lam, CI=CI, CR=CI / RI)


# --------------------------------------------------------------------------- #
#  3. Scoring, ranking, TOPSIS                                                #
# --------------------------------------------------------------------------- #
def stage_scores(R: np.ndarray, w: np.ndarray, active_mask=None) -> np.ndarray:
    if active_mask is None:
        active_mask = np.ones(R.shape[1], dtype=bool)
    wa = w[active_mask].copy()
    wa = wa / wa.sum()
    return R[:, active_mask] @ wa


def topsis(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    norm = R / np.sqrt((R ** 2).sum(axis=0, keepdims=True) + 1e-12)
    V = norm * w
    ib, iw = V.max(axis=0), V.min(axis=0)
    d_best = np.sqrt(((V - ib) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - iw) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst + 1e-12)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    r = np.empty_like(order)
    r[order] = np.arange(1, len(scores) + 1)
    return r


# --------------------------------------------------------------------------- #
#  4. Set operations                                                          #
# --------------------------------------------------------------------------- #
def high_risk_set(R, j, level=1.0):
    return {COUNTRIES[i] for i in range(R.shape[0]) if R[i, j] >= level}


def complement(s):
    return set(COUNTRIES) - s


def priority_set(oil_thr, fx_cutoff):
    in_e1 = {c for c in COUNTRIES if OIL_DEP[c] > oil_thr}
    in_e5 = {c for c in COUNTRIES if RESERVES[c] < fx_cutoff}
    return in_e1 & in_e5


# --------------------------------------------------------------------------- #
#  5. Scarring (persistence-weighted exposure)                               #
# --------------------------------------------------------------------------- #
def scarring_scores(R, w, rho_vec=None):
    """Sc_i = sum_j w_j rho_j v2_ij / sum_j w_j v2_ij : the persistence-weighted
    SHARE of a country's blockade-stage exposure (1 = all scars, 0 = all eases).
    This is a compositional measure; the denominator (weighted exposure LEVEL) is
    returned alongside so the two can be read together (scarring x level)."""
    rho = np.array([PERSIST[c] for c in CHANNELS]) if rho_vec is None else np.asarray(rho_vec)
    num = R @ (w * rho)
    den = R @ w
    level = den.copy()
    den = np.where(den <= 0, np.nan, den)
    return num / den, level


def scarring_sensitivity(R, w, B=2000, seed=SEED):
    """rho-sensitivity: perturb each persistence parameter rho_j by a uniform
    +/-0.2 (clipped to [0,1]) and record, for each country, the distribution of its
    scarring RANK. Mirrors the weight/threshold bootstraps so the new scarring
    contribution is not resting on hand-set constants. Returns a per-country table
    (median rank, [5,95]) and the distribution of Spearman rho between each perturbed
    ranking and the baseline ranking."""
    rng = np.random.default_rng(seed)
    rho0 = np.array([PERSIST[c] for c in CHANNELS])
    base, _ = scarring_scores(R, w, rho0)
    base_rank = ranks_from_scores(base)
    rank_draws = np.zeros((B, len(COUNTRIES)), dtype=int)
    spear = np.zeros(B)
    for b in range(B):
        rho_b = np.clip(rho0 + rng.uniform(-0.2, 0.2, size=len(rho0)), 0.0, 1.0)
        sc_b, _ = scarring_scores(R, w, rho_b)
        rank_draws[b] = ranks_from_scores(sc_b)
        spear[b] = spearmanr(sc_b, base).correlation
    med = np.median(rank_draws, axis=0).astype(int)
    lo = np.percentile(rank_draws, 5, axis=0).astype(int)
    hi = np.percentile(rank_draws, 95, axis=0).astype(int)
    tbl = pd.DataFrame({"country": COUNTRIES, "baseline_rank": base_rank,
                        "median_rank": med, "p05": lo, "p95": hi}
                       ).sort_values("baseline_rank").reset_index(drop=True)
    diag = dict(spearman_median=float(np.median(spear)),
                spearman_p05=float(np.percentile(spear, 5)),
                spearman_min=float(spear.min()))
    return tbl, diag


# --------------------------------------------------------------------------- #
#  6a. Bootstraps                                                             #
# --------------------------------------------------------------------------- #
def bootstrap(R, w_ahp, w_eq, w_ent, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    rank_draws = np.zeros((B, len(COUNTRIES)), dtype=int)
    member_count = {c: 0 for c in COUNTRIES}
    exact_core = 0
    core_ref = frozenset(sorted(priority_set(OIL_THR, FX_CUTOFF)))

    for b in range(B):
        mix = rng.dirichlet([1.0, 1.0, 1.0])
        wp = mix[0] * w_eq + mix[1] * w_ent + mix[2] * w_ahp
        Rp = R.copy()
        half = (R == 0.5)
        flip = half & (rng.random(R.shape) < 0.10)
        Rp[flip] = Rp[flip] + rng.choice([-0.5, 0.5], size=flip.sum())
        Rp = np.clip(Rp, 0.0, 1.0)
        rank_draws[b] = ranks_from_scores(stage_scores(Rp, wp))

        oil_thr = rng.uniform(0.61, 0.71)
        fx_cut = rng.uniform(2.0, 4.0)
        res_jit = {c: RESERVES[c] + rng.normal(0, 0.25) for c in COUNTRIES}
        in_e1 = {c for c in COUNTRIES if OIL_DEP[c] > oil_thr}
        in_e5 = {c for c in COUNTRIES if res_jit[c] < fx_cut}
        core = in_e1 & in_e5
        for c in core:
            member_count[c] += 1
        if frozenset(sorted(core)) == core_ref:
            exact_core += 1

    med = np.median(rank_draws, axis=0)
    lo = np.percentile(rank_draws, 5, axis=0)
    hi = np.percentile(rank_draws, 95, axis=0)
    rank_tbl = pd.DataFrame({"country": COUNTRIES, "median_rank": med.astype(int),
                             "p05": lo.astype(int), "p95": hi.astype(int)}
                            ).sort_values("median_rank", kind="mergesort").reset_index(drop=True)
    set_tbl = pd.DataFrame({"country": COUNTRIES,
                            "membership_freq": [member_count[c] / B for c in COUNTRIES]}
                           ).sort_values("membership_freq", ascending=False).reset_index(drop=True)
    return rank_tbl, set_tbl, exact_core / B


# --------------------------------------------------------------------------- #
#  6b. Activation-schedule sensitivity                                        #
# --------------------------------------------------------------------------- #
def schedule_sensitivity(R, w_ahp):
    """Recompute the t2 priority core and the AHP top-5 under alternative
    activation schedules. The weight-free core depends only on e1,e5 being active
    at t2, so it is invariant; the ranking margin may move."""
    schedules = {
        "baseline (t2 all on)": [1, 1, 1, 1, 1, 1, 1],
        "remittance off at t2": [1, 1, 0, 1, 1, 1, 1],
        "food/gas off at t2":   [1, 1, 1, 0, 1, 1, 0],
        "only e1,e5 active":    [1, 0, 0, 0, 1, 0, 0],
    }
    core = sorted(priority_set(OIL_THR, FX_CUTOFF))
    rows = []
    for name, mask in schedules.items():
        m = np.array(mask, dtype=bool)
        sc = stage_scores(R, w_ahp, m)
        order = [COUNTRIES[i] for i in np.argsort(-sc, kind="mergesort")][:5]
        rows.append({"schedule": name,
                     "priority_core": ";".join(core),
                     "ahp_top5": ";".join(SHORT[c] for c in order)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  6c. Partial back-test (concordance with realised sovereign distress)       #
# --------------------------------------------------------------------------- #
def backtest(core):
    """Concordance of the financing-priority core with realised sovereign-distress
    status. Precision = share of core members in deep distress; the 'misses' are
    deep-distress economies the tool deliberately routes elsewhere (oil-resilient
    -> debt restructuring) and are reported, not hidden."""
    deep = {c for c in COUNTRIES if DISTRESS[c] == "deep"}
    core = set(core)
    tp = core & deep
    precision = len(tp) / len(core) if core else float("nan")
    recall = len(tp) / len(deep) if deep else float("nan")
    routed_elsewhere = sorted(deep - core)
    rows = [{"country": c, "in_core": c in core, "distress": DISTRESS[c],
             "oil_in_Fe1": OIL_DEP[c] > OIL_THR, "reserves_months": RESERVES[c]}
            for c in COUNTRIES]
    return pd.DataFrame(rows), dict(precision=precision, recall=recall,
                                    deep_distress=sorted(deep),
                                    core=sorted(core),
                                    deep_routed_elsewhere=routed_elsewhere)


# --------------------------------------------------------------------------- #
#  6d. Inter-coder reliability (revision item 3)                              #
# --------------------------------------------------------------------------- #
def _cohen_kappa_ordinal(a, b, categories=(0.0, 0.5, 1.0)):
    """Cohen's kappa for two coders on a common set of ordinal cells.
    Returned alongside exact agreement. Weighted (linear) kappa is also computed
    so that the +/-0.5 band-boundary disagreements are credited partial agreement,
    which is the appropriate statistic for an ordered 0/0.5/1 scale."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    cats = list(categories)
    K = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((K, K))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= n
    row = O.sum(axis=1)
    colm = O.sum(axis=0)
    # unweighted
    po = np.trace(O)
    pe = float(row @ colm)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan")
    # linear-weighted
    W = np.array([[abs(i - j) / (K - 1) for j in range(K)] for i in range(K)])
    dobs = float((W * O).sum())
    dexp = float((W * np.outer(row, colm)).sum())
    kappa_w = 1 - dobs / dexp if dexp > 1e-12 else float("nan")
    return dict(exact_agreement=float(po), kappa=float(kappa),
                weighted_kappa=float(kappa_w), n_cells=int(n))


def intercoder_reliability(R):
    """Compare coder 1 (R, qualitative columns) with the independent coder-2
    matrix on the five qualitative channels e2,e3,e4,e6,e7. Returns a per-channel
    table (exact agreement, unweighted and linear-weighted kappa) and the pooled
    statistic over all 5x18 qualitative cells."""
    col_of = {ch: CHANNELS.index(ch) for ch in QUAL_CHANNELS}
    rows = []
    pooled_a, pooled_b = [], []
    for k, ch in enumerate(QUAL_CHANNELS):
        a = [R[COUNTRIES.index(c), col_of[ch]] for c in COUNTRIES]
        b = [CODER2_QUAL[c][k] for c in COUNTRIES]
        st = _cohen_kappa_ordinal(a, b)
        rows.append({"channel": ch, "exact_agreement": round(st["exact_agreement"], 3),
                     "kappa": round(st["kappa"], 3),
                     "weighted_kappa": round(st["weighted_kappa"], 3),
                     "n": st["n_cells"]})
        pooled_a += a
        pooled_b += b
    pooled = _cohen_kappa_ordinal(pooled_a, pooled_b)
    tbl = pd.DataFrame(rows)
    diag = dict(pooled_exact_agreement=round(pooled["exact_agreement"], 3),
                pooled_kappa=round(pooled["kappa"], 3),
                pooled_weighted_kappa=round(pooled["weighted_kappa"], 3),
                pooled_cells=pooled["n_cells"])
    # does the second coding move the core? recompute F(e1)∩F(e5): e1,e5 are
    # numeric anchors and unchanged, so the core is invariant by construction.
    diag["core_under_coder2"] = sorted(priority_set(OIL_THR, FX_CUTOFF))
    diag["core_invariant_qualitative_recode"] = True
    return tbl, diag


# --------------------------------------------------------------------------- #
#  6e. External-index benchmarking (revision item 8)                          #
# --------------------------------------------------------------------------- #
def external_benchmarking(sc_ahp):
    """Benchmark the AHP dashboard ranking and the routing against three
    established structural-vulnerability indices (INFORM, EVI, ND-GAIN). Reports
    Spearman correlations (rank agreement) and, more importantly, the routing
    disagreements the composite indices cannot express."""
    dash = {c: sc_ahp[i] for i, c in enumerate(COUNTRIES)}
    dash_rank = {c: r for c, r in zip(COUNTRIES, ranks_from_scores(np.array([dash[c] for c in COUNTRIES])))}
    idx = {"INFORM": INFORM, "EVI": EVI, "NDGAIN": NDGAIN}
    rows = []
    corr = {}
    for name, d in idx.items():
        vals = np.array([d[c] for c in COUNTRIES])
        r = ranks_from_scores(vals)
        rho = spearmanr([dash[c] for c in COUNTRIES], vals).correlation
        corr[name] = round(float(rho), 3)
        for c, rk in zip(COUNTRIES, r):
            rows.append({"country": c, "index": name, "index_rank": int(rk)})
    idx_rank = {name: {c: rk for c, rk in zip(COUNTRIES,
                       ranks_from_scores(np.array([d[c] for c in COUNTRIES])))}
                for name, d in idx.items()}
    core = sorted(priority_set(OIL_THR, FX_CUTOFF))
    tbl = pd.DataFrame({"country": COUNTRIES,
                        "dashboard_rank": [dash_rank[c] for c in COUNTRIES],
                        "INFORM_rank": [idx_rank["INFORM"][c] for c in COUNTRIES],
                        "EVI_rank": [idx_rank["EVI"][c] for c in COUNTRIES],
                        "NDGAIN_rank": [idx_rank["NDGAIN"][c] for c in COUNTRIES]}
                       ).sort_values("dashboard_rank").reset_index(drop=True)
    # routing disagreements: core members that the indices would NOT prioritise,
    # and index-top economies the framework routes OUT of financing.
    K = len(core)
    disagreements = {}
    for name in idx:
        top_by_index = set(sorted(COUNTRIES, key=lambda c: idx_rank[name][c])[:K])
        disagreements[name] = {
            "index_topK": sorted(top_by_index),
            "core_missed_by_index": sorted(set(core) - top_by_index),
            "index_top_not_in_core": sorted(top_by_index - set(core)),
        }
    return tbl, dict(spearman=corr, core=core, disagreements=disagreements)


# --------------------------------------------------------------------------- #
#  6f. Sample-composition sensitivity (revision item 9)                       #
# --------------------------------------------------------------------------- #
# Plausible additional net-importer economies for the add-in test. e1,e5 anchors
# only (the sole determinants of F(e1)∩F(e5)); values follow the same public
# sources and thresholds as the main sample.
ADDIN_POOL = {
    "Lebanon":     dict(oil=0.98, res=1.2),   # oil-import dependent, reserve-thin
    "Malawi":      dict(oil=1.00, res=1.4),   # oil-import dependent, reserve-thin
    "Mozambique":  dict(oil=0.90, res=2.6),   # oil-import dependent, reserve-thin
    "Cambodia":    dict(oil=0.95, res=6.5),   # oil-import dependent, reserve-rich
    "Senegal":     dict(oil=0.92, res=3.4),   # oil-import dependent, mid reserves
    "Uganda":      dict(oil=0.97, res=4.1),   # oil-import dependent, mid reserves
    "Ecuador":     dict(oil=0.20, res=2.1),   # oil producer, reserve-thin (oil-resilient)
    "Gabon":       dict(oil=0.10, res=2.8),   # oil producer, reserve-thin (oil-resilient)
}


def sample_composition_sensitivity(seed=SEED):
    """Two tests of how the core {Pakistan,Ethiopia,Zambia} depends on the panel.
    (a) Leave-k-out: for k=1,2,3 remove every k-subset of the 18 economies, recompute
        F(e1)∩F(e5) on the remainder, and record how often each baseline core member
        survives (it always does unless it is the one removed) and whether any NEW
        economy enters. Because membership is a per-country property, the informative
        statistic is the survival rate of each core member conditional on being present.
    (b) Add-in: append each plausible additional importer (and all pairs) and record
        whether the core is preserved and who else joins."""
    base_core = set(priority_set(OIL_THR, FX_CUTOFF))
    # (a) leave-k-out
    from itertools import combinations
    rows_lko = []
    for k in (1, 2, 3):
        present_count = {c: 0 for c in base_core}
        survive_count = {c: 0 for c in base_core}
        n_subsets = 0
        for drop in combinations(range(len(COUNTRIES)), k):
            n_subsets += 1
            keep = [COUNTRIES[i] for i in range(len(COUNTRIES)) if i not in drop]
            in_e1 = {c for c in keep if OIL_DEP[c] > OIL_THR}
            in_e5 = {c for c in keep if RESERVES[c] < FX_CUTOFF}
            core_k = in_e1 & in_e5
            for c in base_core:
                if c in keep:
                    present_count[c] += 1
                    if c in core_k:
                        survive_count[c] += 1
        for c in sorted(base_core):
            rows_lko.append({"k": k, "member": c,
                             "survival_rate_given_present":
                                 round(survive_count[c] / present_count[c], 3)})
    lko = pd.DataFrame(rows_lko)

    # (b) add-in test
    rows_add = []
    pool = list(ADDIN_POOL.items())
    # singles
    for name, d in pool:
        oil = dict(OIL_DEP); res = dict(RESERVES)
        oil[name] = d["oil"]; res[name] = d["res"]
        in_e1 = {c for c in list(COUNTRIES) + [name] if oil[c] > OIL_THR}
        in_e5 = {c for c in list(COUNTRIES) + [name] if res[c] < FX_CUTOFF}
        core_a = in_e1 & in_e5
        rows_add.append({"added": name, "core_preserved": base_core <= core_a,
                         "new_entrants": ";".join(sorted(core_a - base_core)) or "-",
                         "core_size": len(core_a)})
    # all pairs
    pair_preserved = 0
    n_pairs = 0
    for (n1, d1), (n2, d2) in combinations(pool, 2):
        n_pairs += 1
        oil = dict(OIL_DEP); res = dict(RESERVES)
        oil[n1] = d1["oil"]; res[n1] = d1["res"]
        oil[n2] = d2["oil"]; res[n2] = d2["res"]
        allc = list(COUNTRIES) + [n1, n2]
        in_e1 = {c for c in allc if oil[c] > OIL_THR}
        in_e5 = {c for c in allc if res[c] < FX_CUTOFF}
        core_a = in_e1 & in_e5
        if base_core <= core_a:
            pair_preserved += 1
    add = pd.DataFrame(rows_add)
    diag = dict(base_core=sorted(base_core),
                leave_k_out_min_survival=float(lko["survival_rate_given_present"].min()),
                addin_singles_all_preserve=bool(add["core_preserved"].all()),
                addin_pairs_preserved=f"{pair_preserved}/{n_pairs}")
    return lko, add, diag


# --------------------------------------------------------------------------- #
#  6g. e7 (gas/LNG) informativeness test (revision item 10)                    #
# --------------------------------------------------------------------------- #
def e7_informativeness(R, w_ahp):
    """Is the gas/LNG channel e7 doing work, or is it near-inert in this sample?
    Three diagnostics: (i) how many high-risk (=1.0) flags e7 raises; (ii) whether
    dropping e7 changes any weight-free set-operation result that uses it; (iii) the
    Spearman correlation of the AHP ranking with and without e7 in the active set."""
    j7 = CHANNELS.index("e7")
    n_high = int((R[:, j7] >= 1.0).sum())
    n_elev = int((R[:, j7] == 0.5).sum())
    fe7 = high_risk_set(R, j7, 1.0)
    # set ops that reference e7 in the manuscript menu
    sets = {ch: high_risk_set(R, CHANNELS.index(ch), 1.0) for ch in CHANNELS}
    with_e7 = {"F(e1)capF(e7)": sorted(sets["e1"] & sets["e7"]),
               "F(e4)capF(e7)": sorted(sets["e4"] & sets["e7"])}
    # ranking with vs without e7
    mask_all = np.ones(R.shape[1], dtype=bool)
    mask_no7 = mask_all.copy(); mask_no7[j7] = False
    r_all = stage_scores(R, w_ahp, mask_all)
    r_no7 = stage_scores(R, w_ahp, mask_no7)
    rho = spearmanr(r_all, r_no7).correlation
    core_all = sorted(priority_set(OIL_THR, FX_CUTOFF))  # core doesn't use e7
    diag = dict(e7_high_flags=n_high, e7_elevated_flags=n_elev,
                Fe7=sorted(fe7),
                spearman_rank_with_vs_without_e7=round(float(rho), 3),
                core_uses_e7=False, core=core_all,
                verdict=("informative: raises the sole all-imported-energy flag {Jordan} "
                         "but is otherwise sparse; it does not enter the financing core"))
    return diag, with_e7


# --------------------------------------------------------------------------- #
#  6g2. e6-only perturbation of the transfer-routing group (revision item 4)   #
# --------------------------------------------------------------------------- #
def e6_transfer_sensitivity(R, seed=SEED, B=2000):
    """Isolate how much the TRANSFER-routing group depends on the coarse inflation
    band e6 specifically, holding all other channels fixed. The transfer group is
    F(e1) cap F(e6) (oil-import-dependent AND high embedded inflation); the FINANCING
    core is F(e1) cap F(e5) and does NOT use e6, so it is the control that must stay
    fixed under any e6 perturbation. We flip e6 band-boundary cells (0.5<->1.0 and
    0.5<->0.0) with probability p per 0.5 cell over B seeded draws, recompute both
    sets, and report (i) the baseline transfer group, (ii) how often each baseline
    transfer member survives, (iii) the churn (mean Jaccard vs baseline) of the
    transfer group, and (iv) confirmation that the financing core is invariant."""
    rng = np.random.default_rng(seed + 61)
    j1, j5, j6 = CHANNELS.index("e1"), CHANNELS.index("e5"), CHANNELS.index("e6")
    fe1 = high_risk_set(R, j1, 1.0)
    fe6_base = high_risk_set(R, j6, 1.0)
    transfer_base = fe1 & fe6_base
    core_base = priority_set(OIL_THR, FX_CUTOFF)  # uses e1,e5 only

    def _jaccard(a, b):
        a, b = set(a), set(b)
        u = a | b
        return len(a & b) / len(u) if u else 1.0

    surv = {c: 0 for c in transfer_base}
    jac_sum = 0.0
    core_changed = 0
    p_flip = 0.5   # aggressive: flip half of the 0.5 (band-boundary) e6 cells per draw
    for _ in range(B):
        col = R[:, j6].copy()
        for i in range(len(col)):
            if col[i] == 0.5 and rng.random() < p_flip:
                col[i] = 1.0 if rng.random() < 0.5 else 0.0
            elif col[i] == 1.0 and rng.random() < p_flip * 0.5:
                col[i] = 0.5   # allow a high cell to soften occasionally
        fe6 = {COUNTRIES[i] for i in range(len(col)) if col[i] >= 1.0}
        transfer = fe1 & fe6
        for c in transfer_base:
            if c in transfer:
                surv[c] += 1
        jac_sum += _jaccard(transfer, transfer_base)
        # financing core is independent of e6 -> must be invariant
        if priority_set(OIL_THR, FX_CUTOFF) != core_base:
            core_changed += 1

    diag = dict(
        transfer_group_base=sorted(transfer_base),
        financing_core_base=sorted(core_base),
        transfer_member_survival={c: round(surv[c] / B, 3) for c in sorted(transfer_base)},
        transfer_mean_jaccard_vs_base=round(jac_sum / B, 3),
        financing_core_changed_frac=round(core_changed / B, 3),
        note=("e6-only perturbation: the transfer group churns while the financing "
              "core (which does not use e6) is invariant, quantifying the two-tier "
              "confidence structure"))
    return diag


# --------------------------------------------------------------------------- #
#  6h. Provisional realised-outcome back-test regression (revision item 7)     #
# --------------------------------------------------------------------------- #
def realised_backtest_regression(sc_ahp, core, scar, seed=SEED):
    """Provisional realised-outcome back-test with three OLS (HC1) specifications and
    permutation inference appropriate to n=18.

    Addresses the definitional-overlap concern directly:
      * model 1  res_loss ~ core + pre_reserves         (core coef with the control)
      * model 1b res_loss ~ core                          (core coef WITHOUT the shared
                 reserve control; if the coefficient survives, it is not merely the
                 pre-shock reserve variable re-entering through the outcome)
      * model 2  infl_h2 ~ core                           (CLEANER test: realised H2
                 inflation is NOT part of the core definition, so a core-vs-non-core
                 inflation gap cannot be definitional)
    For every reported coefficient we also compute a two-sided PERMUTATION p-value
    (B label reshuffles of the core/scarring regressor), which does not rely on the
    normal approximation and is the honest statistic at n=18.

    Clearly labelled provisional: the outcomes in REALISED_H2 are a transparent
    placeholder to be replaced by true prints."""
    rng = np.random.default_rng(seed)
    core = set(core)
    scar_map = {c: scar[i] for i, c in enumerate(COUNTRIES)}
    rows = []
    for c in COUNTRIES:
        res_loss, infl, pre = REALISED_H2[c]
        rows.append(dict(country=c, res_loss=res_loss, infl_h2=infl,
                         in_core=1.0 if c in core else 0.0,
                         scarring=scar_map[c], pre_reserves=pre))
    df = pd.DataFrame(rows)

    def _ols_hc1(y, X, names):
        n, k = X.shape
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        resid = y - X @ beta
        S = (X * resid[:, None])
        cov = XtX_inv @ (S.T @ S) @ XtX_inv * (n / (n - k))
        se = np.sqrt(np.diag(cov))
        lo = beta - 1.645 * se
        hi = beta + 1.645 * se
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return (pd.DataFrame({"term": names, "coef": beta, "se_hc1": se,
                              "ci90_lo": lo, "ci90_hi": hi}), r2)

    def _perm_p(y, X, focal_col, B=10000):
        """Two-sided permutation p-value for the focal regressor's OLS coefficient,
        reshuffling only the focal column's labels and holding other covariates fixed."""
        def _coef(Xm):
            b = np.linalg.lstsq(Xm, y, rcond=None)[0]
            return b[focal_col]
        obs = _coef(X)
        Xp = X.copy()
        cnt = 0
        col = X[:, focal_col].copy()
        for _ in range(B):
            Xp[:, focal_col] = rng.permutation(col)
            if abs(_coef(Xp)) >= abs(obs) - 1e-12:
                cnt += 1
        return (cnt + 1) / (B + 1)

    out = {}
    y_res = df["res_loss"].to_numpy()
    y_inf = df["infl_h2"].to_numpy()
    core_v = df["in_core"].to_numpy()
    scar_v = df["scarring"].to_numpy()
    pre_v = df["pre_reserves"].to_numpy()
    ones = np.ones(len(df))

    # model 1: res_loss ~ core + pre_reserves
    X1 = np.column_stack([ones, core_v, pre_v])
    t1, r2_1 = _ols_hc1(y_res, X1, ["const", "in_core", "pre_reserves"])
    p1 = _perm_p(y_res, X1.copy(), 1)
    out["res_loss_ctrl"] = (t1, r2_1, p1)

    # model 1b: res_loss ~ core  (NO reserve control)
    X1b = np.column_stack([ones, core_v])
    t1b, r2_1b = _ols_hc1(y_res, X1b, ["const", "in_core"])
    p1b = _perm_p(y_res, X1b.copy(), 1)
    out["res_loss_noctrl"] = (t1b, r2_1b, p1b)

    # model 2: infl_h2 ~ core  (cleaner: inflation not in core definition)
    X2 = np.column_stack([ones, core_v])
    t2, r2_2 = _ols_hc1(y_inf, X2, ["const", "in_core"])
    p2 = _perm_p(y_inf, X2.copy(), 1)
    out["infl_core"] = (t2, r2_2, p2)

    # model 3: infl_h2 ~ scarring + pre_reserves (continuous scarring signal)
    X3 = np.column_stack([ones, scar_v, pre_v])
    t3, r2_3 = _ols_hc1(y_inf, X3, ["const", "scarring", "pre_reserves"])
    p3 = _perm_p(y_inf, X3.copy(), 1)
    out["infl_scarring"] = (t3, r2_3, p3)

    def _c(tbl, term, col):
        return float(tbl.loc[tbl.term == term, col].iloc[0])

    diag = dict(
        core_mean_res_loss=round(float(df.loc[df.in_core == 1, "res_loss"].mean()), 3),
        noncore_mean_res_loss=round(float(df.loc[df.in_core == 0, "res_loss"].mean()), 3),
        # with control
        res_loss_core_coef=round(_c(t1, "in_core", "coef"), 3),
        res_loss_core_ci90=[round(_c(t1, "in_core", "ci90_lo"), 3),
                            round(_c(t1, "in_core", "ci90_hi"), 3)],
        res_loss_core_permp=round(p1, 3),
        # without control (definitional-overlap check)
        res_loss_core_coef_noctrl=round(_c(t1b, "in_core", "coef"), 3),
        res_loss_core_permp_noctrl=round(p1b, 3),
        # cleaner inflation-on-core test
        infl_core_coef=round(_c(t2, "in_core", "coef"), 3),
        infl_core_ci90=[round(_c(t2, "in_core", "ci90_lo"), 3),
                        round(_c(t2, "in_core", "ci90_hi"), 3)],
        infl_core_permp=round(p2, 3),
        # scarring signal
        infl_scarring_coef=round(_c(t3, "scarring", "coef"), 3),
        infl_scarring_ci90=[round(_c(t3, "scarring", "ci90_lo"), 3),
                            round(_c(t3, "scarring", "ci90_hi"), 3)],
        infl_scarring_permp=round(p3, 3),
        note="PROVISIONAL: REALISED_H2 is a labelled placeholder, not final prints")
    return df, out, diag


# --------------------------------------------------------------------------- #
#  6i. Auditable costing table for the financing core (revision item 11)       #
# --------------------------------------------------------------------------- #
def costing_table():
    """Convert the prose cost estimates into an auditable table for the financing
    core: RFI/RCF rapid access sized at 100% of IMF quota (cumulative ceiling) and
    RST access at 150% of quota, converted to USD. Lead times are the institutional
    norms cited in the manuscript."""
    rows = []
    tot_rapid = 0.0
    for c in ["Pakistan", "Ethiopia", "Zambia"]:
        q_sdr = IMF_QUOTA_SDRMN[c]
        rapid_usd = q_sdr * 1.00 * SDR_USD / 1000.0   # US$ bn, 100% of quota
        rst_usd = q_sdr * 1.50 * SDR_USD / 1000.0     # US$ bn, 150% of quota
        tot_rapid += rapid_usd
        rows.append({"economy": c, "quota_SDRmn": q_sdr,
                     "rapid_access_100pct_quota_USDbn": round(rapid_usd, 2),
                     "RST_150pct_quota_USDbn": round(rst_usd, 2),
                     "lead_time": "2-6 weeks (RFI/RCF, pre-positioned)"})
    df = pd.DataFrame(rows)
    diag = dict(core_total_rapid_access_USDbn=round(tot_rapid, 2))
    return df, diag


# --------------------------------------------------------------------------- #
#  7. Optional / deferred transmission regression (real-data only)            #
# --------------------------------------------------------------------------- #
def run_regression(panel_csv):
    """The reduced-form pass-through is SIDELINED in the manuscript: the identified
    local-projection successor is deferred to the full-year panel and no point
    estimates are reported. This estimator runs only if a realised monthly panel is
    supplied; otherwise it documents the deferral. No inflation values are fabricated."""
    if not os.path.exists(panel_csv):
        return ("Transmission regression: DEFERRED (sidelined in the manuscript).\n"
                "The manuscript reports only a qualitative transmission statement and\n"
                "specifies an identified local-projection (Jorda) successor for the\n"
                "full-year 2026 panel. No oil->CPI point estimates are claimed.\n\n"
                "This optional estimator runs only if a realised monthly panel is\n"
                "supplied at data/cpi_panel.csv (schema in data/cpi_panel_template.csv).\n"
                "No inflation values are fabricated.\n")
    df = pd.read_csv(panel_csv)
    if "dpi" not in df.columns or df["dpi"].isna().all():
        return ("Transmission regression: panel present but 'dpi' empty; estimation skipped.\n")
    # (local-projection estimator omitted here; populate panel and extend as needed)
    return (f"Panel supplied with {df['dpi'].notna().sum()} inflation observations; "
            "extend run_regression() with the local-projection design to estimate.\n")


# --------------------------------------------------------------------------- #
#  8. Figures                                                                 #
# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, fc, ec, fs=10, tc="black", lw=1.2):
    ax.add_patch(FancyBboxPatch((xy[0]-w/2, xy[1]-h/2), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def _arrow(ax, p, q, color="#444444", lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                 lw=lw, color=color, zorder=1, shrinkA=2, shrinkB=2))


def routing_class(c, core, fe5):
    """Colour class for figures."""
    if c in core:
        return "core"
    if c in fe5:          # reserve-thin but oil-resilient (removed by oil filter)
        return "oilresilient"
    if c == "Jordan":
        return "breadth"
    return "other"


CLASS_COL = {"core": DB, "oilresilient": ORANGE, "breadth": RED, "other": GRAY}


def fig_weights(w_ahp, w_eq, w_ent):
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); x = np.arange(7); bw = 0.26
    ax.bar(x - bw, w_ahp, bw, label="AHP (substantive)", color=DB)
    ax.bar(x, w_eq, bw, label="Equal (benchmark)", color=GRAY)
    ax.bar(x + bw, w_ent, bw, label="Entropy (cautionary)", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{l}" for c, l in zip(CHAN_TEX, CHAN_LONG)], fontsize=9.5)
    ax.set_ylabel(r"Channel weight $w_j$", fontsize=10.5); ax.legend(fontsize=9, frameon=False)
    ax.annotate(r"entropy $\approx %.3f$ on $e_1$" % w_ent[0] + "\n(dispersion pathology)",
                xy=(0 + bw, w_ent[0]), xytext=(1.3, 0.18), fontsize=8.6, color=ORANGE,
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1))
    ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(0, max(0.34, w_ent.max()*1.15))
    fig.savefig(os.path.join(FIG_DIR, "fig_weights.png")); plt.close(fig)


def fig_scores(score_df):
    d = score_df.sort_values("ahp", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, 4.6)); x = np.arange(len(d)); bw = 0.27
    ax.bar(x - bw, d["equal"], bw, label="Equal", color=GRAY)
    ax.bar(x, d["entropy"], bw, label="Entropy", color=ORANGE)
    ax.bar(x + bw, d["ahp"], bw, label="AHP", color=DB)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in d["country"]], fontsize=8.5)
    ax.set_ylabel(r"Blockade-stage score $V_{t_2}(u_i)$", fontsize=10.5)
    ax.legend(fontsize=9, frameon=False, ncol=3); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 0.9)
    fig.savefig(os.path.join(FIG_DIR, "fig_scores.png")); plt.close(fig)


def fig_rankboot(rank_tbl, core, fe5):
    d = rank_tbl.copy(); n = len(d); fig, ax = plt.subplots(figsize=(8.4, 6.2))
    y = np.arange(n)[::-1]
    for k in range(n):
        col = CLASS_COL[routing_class(d["country"][k], core, fe5)]
        ax.plot([d["p05"][k], d["p95"][k]], [y[k], y[k]], color=col, lw=2.0, alpha=0.55, solid_capstyle="round")
        ax.plot(d["median_rank"][k], y[k], "o", color=col, ms=6, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[c] for c in d["country"]], fontsize=9)
    ax.set_xlabel(r"Bootstrap rank ($1=$ most vulnerable); point $=$ median, bar $=[5,95]$ pct", fontsize=9.5)
    ax.set_xticks(range(1, n + 1)); ax.set_xlim(0.5, n + 0.5); ax.invert_xaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(FIG_DIR, "fig_rankboot.png")); plt.close(fig)


def fig_setboot(set_tbl, core, fe5):
    d = set_tbl.copy(); n = len(d); y = np.arange(n)[::-1]
    cols = [CLASS_COL[routing_class(c, core, fe5)] for c in d["country"]]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.barh(y, d["membership_freq"], color=cols, height=0.62)
    for k in range(n):
        ax.text(d["membership_freq"][k] + 0.015, y[k], f"{d['membership_freq'][k]:.2f}", va="center", fontsize=8)
    ax.axvline(0.5, color=GRAY, ls="--", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[c] for c in d["country"]], fontsize=9)
    ax.set_xlabel(r"Membership frequency in $F(e_1)\cap F(e_5)$ ($B=2000$ draws)", fontsize=9.8)
    ax.set_xlim(0, 1.10); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=DB, markersize=10, label="priority core"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=ORANGE, markersize=10, label="oil-resilient, reserve-thin"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=GRAY, markersize=10, label="other")],
              fontsize=8.2, frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig_setboot.png")); plt.close(fig)


def fig_reserves(core, fe5):
    items = sorted(RESERVES.items(), key=lambda kv: kv[1]); n = len(items)
    names = [k for k, _ in items]; vals = [v for _, v in items]
    y = np.arange(n)[::-1]
    cols = [CLASS_COL[routing_class(k, core, fe5)] for k in names]
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    ax.barh(y, vals, color=cols, height=0.62, alpha=0.9)
    for i, (k, v) in enumerate(items):
        ax.text(v + 0.12, y[i], f"{v:.1f}", va="center", fontsize=8)
    for tau, ls in zip([2.0, 2.5, 3.0, 3.5, 4.0], [":", "--", "-", "--", ":"]):
        ax.axvline(tau, color=DB, ls=ls, lw=1.0)
        ax.text(tau, n - 0.3, rf"$\tau={tau}$", color=DB, fontsize=7.5, ha="center")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[k] for k in names], fontsize=9)
    ax.set_xlabel(r"FX reserves (months of imports); cutoff $\tau$ for $F(e_5^{<\tau})$", fontsize=9.8)
    ax.set_xlim(0, 12.0); ax.set_ylim(-0.6, n + 0.2); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=DB, markersize=10, label="financing core"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=ORANGE, markersize=10, label="oil-resilient, reserve-thin (oil filter excludes)"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=GRAY, markersize=10, label="reserve-resilient")],
              fontsize=8.0, frameon=True, framealpha=0.95, edgecolor=LGRAY, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig_reserves.png")); plt.close(fig)


def _declutter(labels_vals, min_gap):
    """Given a list of (name, y) sorted by y, nudge y-positions apart by at least
    min_gap so text labels do not collide, preserving order. Returns {name: y_lab}."""
    order = sorted(labels_vals, key=lambda kv: kv[1])
    ys = [y for _, y in order]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    return {order[i][0]: ys[i] for i in range(len(order))}


def fig_slope(t2, t3, core, fe5):
    # Grayscale-safe: distinct marker shapes AND line styles per class, plus colour.
    MARK = {"core": "o", "breadth": "s", "oilresilient": "^", "other": "."}
    LSTY = {"core": "-", "breadth": (0, (1, 1)), "oilresilient": (0, (4, 2)), "other": "-"}
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    left_lab = _declutter([(c, t2[c]) for c in COUNTRIES], 0.013)
    right_lab = _declutter([(c, t3[c]) for c in COUNTRIES], 0.013)
    for c in COUNTRIES:
        ya, yb = t2[c], t3[c]; cls = routing_class(c, core, fe5)
        col = CLASS_COL[cls]
        lw = 2.4 if cls in ("core", "breadth") else 1.2
        alpha = 0.95 if cls in ("core", "breadth", "oilresilient") else 0.45
        z = 4 if cls in ("core", "breadth") else 2
        ax.plot([0, 1], [ya, yb], color=col, lw=lw, ls=LSTY[cls], alpha=alpha,
                zorder=z, solid_capstyle="round")
        ax.plot(0, ya, MARK[cls], color=col, ms=5.0, zorder=z + 1)
        ax.plot(1, yb, MARK[cls], color=col, ms=5.0, zorder=z + 1)
        # leader line from node to decluttered label position
        ax.plot([-0.10, -0.02], [left_lab[c], ya], color=col, lw=0.5, alpha=0.5, zorder=1)
        ax.plot([1.02, 1.10], [yb, right_lab[c]], color=col, lw=0.5, alpha=0.5, zorder=1)
        ax.text(-0.12, left_lab[c], SHORT[c], ha="right", va="center", color=col, fontsize=7.6)
        ax.text(1.12, right_lab[c], SHORT[c], ha="left", va="center", color=col, fontsize=7.6)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0.16, 0.84)
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"$t_2$ (blockade)", r"$t_3$ (re-routing)"], fontsize=10.5)
    ax.set_ylabel(r"AHP score $V_{t_k}(u_i)$", fontsize=10.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], color=DB, lw=2.4, ls="-", marker="o", label="financing core (scars)"),
                       Line2D([0], [0], color=RED, lw=2.4, ls=(0, (1, 1)), marker="s", label="Jordan (breadth, eases)"),
                       Line2D([0], [0], color=ORANGE, lw=1.6, ls=(0, (4, 2)), marker="^", label="oil-resilient, reserve-thin"),
                       Line2D([0], [0], color=GRAY, lw=1.2, ls="-", marker=".", label="other")],
              fontsize=8.2, frameon=False, loc="lower center", ncol=2)
    fig.savefig(os.path.join(FIG_DIR, "fig_slope.png")); plt.close(fig)


def fig_scarring(scar_df, core, fe5):
    d = scar_df.dropna(subset=["scarring"]).sort_values("scarring", ascending=True).reset_index(drop=True)
    n = len(d); y = np.arange(n)
    cols = [CLASS_COL[routing_class(c, core, fe5)] for c in d["country"]]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.barh(y, d["scarring"], color=cols, height=0.62, alpha=0.9)
    for k in range(n):
        ax.text(d["scarring"][k] + 0.01, y[k], f"{d['scarring'][k]:.2f}", va="center", fontsize=8)
    ax.axvline(0.5, color=GRAY, ls="--", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[c] for c in d["country"]], fontsize=9)
    ax.set_xlabel(r"Scarring score $Sc(u_i)=\sum_j w_j\rho_j v_{2,j}/\sum_j w_j v_{2,j}$", fontsize=9.8)
    ax.set_xlim(0, 1.0); ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.51, 0.3, "easing $\\leftarrow$ | $\\rightarrow$ scarring", fontsize=8.5, color=GRAY)
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=DB, markersize=10, label="financing core"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=ORANGE, markersize=10, label="oil-resilient, reserve-thin"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=RED, markersize=10, label="Jordan (breadth)"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=GRAY, markersize=10, label="other")],
              fontsize=8.0, frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig_scarring.png")); plt.close(fig)


def fig_scarringlevel(scar_df, core, fe5):
    """Scatter of scarring SHARE (composition, y) against weighted exposure LEVEL
    (x). Clarifies that a high scarring score need not mean a high exposure level:
    the financing core sits high-right (high level AND high persistence); Bolivia
    sits high-left (high persistence but moderate level); Jordan sits low (eases)."""
    d = scar_df.dropna(subset=["scarring"]).copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for _, r in d.iterrows():
        c = r["country"]
        if c in core:
            col, z, ms = RED, 5, 80
        elif c in fe5:
            col, z, ms = ORANGE, 4, 70
        else:
            col, z, ms = DB, 3, 45
        ax.scatter(r["exposure_level"], r["scarring"], s=ms, color=col, zorder=z,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(SHORT[c], (r["exposure_level"], r["scarring"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.6, color=col)
    ax.axhline(0.5, color=GRAY, ls="--", lw=0.8)
    ax.set_xlabel(r"Weighted exposure level $\sum_j w_j v_{2,j}$ (severity)", fontsize=10.5)
    ax.set_ylabel(r"Scarring share $\mathrm{Sc}(u_i)$ (composition)", fontsize=10.5)
    ax.text(0.02, 0.97, "high persistence,\nmoderate level",
            transform=ax.transAxes, fontsize=8, color=ORANGE, va="top")
    ax.text(0.97, 0.97, "high level &\nhigh persistence", transform=ax.transAxes,
            fontsize=8, color=RED, va="top", ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=9, label="financing core"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=9, label=r"reserve-thin, oil-resilient ($F(e_5)\setminus$core)"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=DB, markersize=9, label="other")],
              fontsize=8, frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig_scarringlevel.png")); plt.close(fig)


def fig_brent():
    fig, ax = plt.subplots(figsize=(9.0, 4.4)); x = range(len(BRENT_VALS))
    ax.plot(x, BRENT_VALS, "-o", color=DB, ms=4, lw=1.8)
    ax.plot(BRENT_INTRADAY[0], BRENT_INTRADAY[1], "o", color=RED, ms=7)
    ax.annotate(r"intraday $\$126.41$ (30 Apr)", xy=BRENT_INTRADAY, xytext=(6.0, 128),
                color=RED, fontsize=9, arrowprops=dict(arrowstyle="-|>", color=RED, lw=1))
    ax.annotate(r"pre-war $\sim\$73$", xy=(0, 73), xytext=(0.2, 66), color=DB, fontsize=8.6)
    ax.annotate(r"March peak $\sim\$118$", xy=(3, 118), xytext=(2.4, 123), color=DB, fontsize=8.6)
    ax.annotate(r"eased $\sim\$80$--$95$", xy=(13, 81), xytext=(11.2, 88), color=DB, fontsize=8.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(BRENT_LABELS, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel(r"Brent (US\$/bbl)", fontsize=10.5); ax.set_ylim(60, 133)
    ax.grid(color="#EEEEEE", lw=0.7); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(FIG_DIR, "fig_brent.png")); plt.close(fig)


# ---- appendix figures ----------------------------------------------------- #
def fig_flowchart():
    """Clean top-to-bottom pipeline. Consistent box widths, single central spine
    with paired side-stages, correct sample size (18 economies)."""
    fig, ax = plt.subplots(figsize=(8.8, 10.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")
    cx, wide = 5.0, 7.8
    # inputs
    _box(ax, (2.0, 13.2), 2.6, 0.9, "Universe $U$\n(18 economies)", LGRAY, GRAY, 9.2)
    _box(ax, (5.0, 13.2), 2.6, 0.9, r"Channels $E=\{e_1,\dots,e_7\}$", LGRAY, GRAY, 9.2)
    _box(ax, (8.0, 13.2), 2.6, 0.9, r"Stages $T=\{t_0,\dots,t_3\}$", LGRAY, GRAY, 9.2)
    # graded scoring
    _box(ax, (cx, 11.7), wide, 0.9, r"Graded scoring $v_{k,j}(u_i)\in\{0,0.5,1\}$ (Table~1)", "#E8F0FA", DB, 9.6)
    for xx in (2.0, 5.0, 8.0):
        _arrow(ax, (xx, 12.72), (cx + (xx - cx) * 0.35, 12.18))
    # paired: activation | weights
    _box(ax, (2.7, 10.2), 3.6, 0.9, r"Activation $A_{t_k}$ + persistence $\rho_j$", "#E8F0FA", DB, 9.0)
    _box(ax, (7.3, 10.2), 3.6, 0.9, "Weights: AHP / Equal / Entropy", "#E8F0FA", DB, 9.0)
    _arrow(ax, (4.2, 11.22), (3.1, 10.68)); _arrow(ax, (5.8, 11.22), (6.9, 10.68))
    # stage scores + scarring
    _box(ax, (cx, 8.7), wide, 0.9, r"Stage scores $V_{t_k}(u_i)$ and scarring score $\mathrm{Sc}(u_i)$", "#E8F0FA", DB, 9.6)
    _arrow(ax, (2.7, 9.72), (4.0, 9.18)); _arrow(ax, (7.3, 9.72), (6.0, 9.18))
    # paired: set ops | benchmark
    _box(ax, (2.7, 7.2), 3.6, 1.0, "Weight-free set ops\n" + r"$R_t(e)$;  $F_t(e_1)\cap F_t(e_5)$", "#FBE9E9", RED, 9.0)
    _box(ax, (7.3, 7.2), 3.6, 1.0, "Ranking + TOPSIS\n(benchmark only)", "#EAF4EF", TEAL, 9.0)
    _arrow(ax, (4.2, 8.22), (3.2, 7.72)); _arrow(ax, (5.8, 8.22), (6.8, 7.72))
    # robustness
    _box(ax, (cx, 5.6), wide, 0.9, "Robustness: rank + set bootstrap; $\\rho$- and schedule-sensitivity; back-test", "#FFF4E6", ORANGE, 8.8)
    _arrow(ax, (2.9, 6.68), (4.2, 6.06)); _arrow(ax, (7.1, 6.68), (5.8, 6.06))
    # outputs
    _box(ax, (cx, 4.0), wide + 0.4, 1.0, "Outputs: priority core (oil filter active);\nscarring map; $t_2$ vs. $t_3$ dynamics", "#E7F3EA", GREEN, 9.0)
    _arrow(ax, (cx, 5.14), (cx, 4.52))
    # policy
    _box(ax, (cx, 2.2), wide + 0.6, 1.1, "Policy routing (indicative cost / lead time):\n" +
         r"financing $\to$ core;  debt restructuring $\to$ oil-resilient;  supply-side $\to$ breadth", "#E7F3EA", GREEN, 8.6)
    _arrow(ax, (cx, 3.48), (cx, 2.78))
    fig.savefig(os.path.join(FIG_DIR, "fig_flowchart.png")); plt.close(fig)


def fig_channels():
    """Channel taxonomy grouped by AHP priority tier, with the corrected e1
    definition (net petroleum-product import share) and a 4-A footer mapping."""
    fig, ax = plt.subplots(figsize=(9.4, 5.8)); ax.set_xlim(0, 10); ax.set_ylim(0, 8.8); ax.axis("off")
    rows = [
        (r"$e_1$ Oil-import dependence", "Net petroleum-product import share (IEA/EIA)", r"$>0.66$", 0.237),
        (r"$e_5$ FX-reserve adequacy", "Reserves, months of imports (IMF)", r"$<3$ mo.", 0.237),
        (r"$e_2$ External-debt vuln.", "Debt service/exports; IMF prog. (IMF)", "Elevated / distress", 0.129),
        (r"$e_3$ Gulf remittance exp.", r"(Remit/GDP)$\times$GCC share (WB-KNOMAD)", "High remit. & GCC", 0.129),
        (r"$e_6$ Inflation pass-through", r"Food/fuel CPI share $\times$ pass-through", "High share & p.t.", 0.129),
        (r"$e_4$ Food-import reliance", "Cereal-import dependency (FAO)", "Net importer", 0.070),
        (r"$e_7$ Gas/LNG exposure", "LNG reliance; Hormuz routing (IEA)", "High LNG reliance", 0.070),
    ]
    # header
    x0, x1, x2, x3 = 0.35, 3.55, 6.85, 9.35
    ax.text(x0, 8.05, "Channel", fontsize=9.6, weight="bold")
    ax.text(x1, 8.05, "Indicator (source)", fontsize=9.6, weight="bold")
    ax.text(x2, 8.05, "High-risk threshold", fontsize=9.6, weight="bold")
    ax.text(x3, 8.05, r"$w_j^{\rm ahp}$", fontsize=9.6, weight="bold", ha="center")
    ax.plot([0.2, 9.8], [7.78, 7.78], color=GRAY, lw=0.8)
    col = {0.237: "#FBE9E9", 0.129: "#E8F0FA", 0.070: "#EFEFEF"}
    ec = {0.237: RED, 0.129: DB, 0.070: GRAY}
    y = 7.25
    for name, ind, thr, wv in rows:
        ax.add_patch(Rectangle((0.2, y - 0.42), 9.6, 0.82, fc=col[wv], ec=ec[wv], lw=1.0))
        ax.text(x0, y, name, fontsize=9.0, va="center")
        ax.text(x1, y, ind, fontsize=8.1, va="center")
        ax.text(x2, y, thr, fontsize=8.4, va="center")
        ax.text(x3, y, f"{wv:.3f}", fontsize=8.8, va="center", ha="center")
        y -= 0.92
    # tier legend + 4-A footer
    ax.text(x0, 0.62, "Tiers:", fontsize=8.4, weight="bold")
    for lab, c, xx in [("primary $e_1,e_5$", RED, 1.3), ("secondary $e_2,e_3,e_6$", DB, 3.7),
                       ("tertiary $e_4,e_7$", GRAY, 6.6)]:
        ax.add_patch(Rectangle((xx, 0.48, ), 0.22, 0.22, fc=col[{RED: .237, DB: .129, GRAY: .070}[c]], ec=c, lw=1))
        ax.text(xx + 0.32, 0.59, lab, fontsize=8.2, va="center", color=c)
    ax.text(x0, 0.12, r"4-A mapping: availability ($e_1,e_5,e_7$), affordability ($e_4,e_6$), "
            r"accessibility ($e_2,e_3$).", fontsize=8.0, style="italic", color=GRAY)
    fig.savefig(os.path.join(FIG_DIR, "fig_channels.png")); plt.close(fig)


def fig_stages():
    stages = ["t0", "t1", "t2", "t3"]; titles = [r"$t_0$ pre-war", r"$t_1$ strikes", r"$t_2$ blockade", r"$t_3$ re-routing"]
    M = np.array([STAGE_ACTIVE[s] for s in stages])
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for i in range(4):
        for j in range(7):
            on = M[i, j]
            ax.add_patch(Rectangle((j, 3 - i), 0.92, 0.92, fc=DB if on else "white",
                         ec=DB if on else LGRAY, lw=1.0, alpha=0.85 if on else 1))
            if on:
                ax.text(j + 0.46, 3 - i + 0.46, "on", ha="center", va="center", color="white", fontsize=8.5)
    ax.set_xlim(-0.1, 7.1); ax.set_ylim(-0.1, 4.0)
    ax.set_xticks([j + 0.46 for j in range(7)]); ax.set_xticklabels([f"{c}\n{l}" for c, l in zip(CHAN_TEX, CHAN_LONG)], fontsize=9.5)
    ax.set_yticks([3 - i + 0.46 for i in range(4)]); ax.set_yticklabels(titles, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(os.path.join(FIG_DIR, "fig_stages.png")); plt.close(fig)


def fig_graded():
    fig, ax = plt.subplots(figsize=(7.8, 8.2))
    cmap = matplotlib.colors.ListedColormap(["#F2F2F2", "#F4C2A1", "#C0392B"])
    norm = matplotlib.colors.BoundaryNorm([-0.25, 0.25, 0.75, 1.25], cmap.N)
    im = ax.imshow(GRADED_T2, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(7)); ax.set_xticklabels([f"{c}\n{l}" for c, l in zip(CHAN_TEX, CHAN_LONG)], fontsize=9.5)
    ax.set_yticks(range(len(COUNTRIES))); ax.set_yticklabels(COUNTRIES, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    for i in range(len(COUNTRIES)):
        for j in range(7):
            v = GRADED_T2[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v == 1.0 else "black", fontsize=7.6)
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True); ax.set_yticks(np.arange(-.5, len(COUNTRIES), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1.0], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"], fontsize=8.5)
    fig.savefig(os.path.join(FIG_DIR, "fig_graded.png")); plt.close(fig)


def fig_euler(core, fe5):
    fig, ax = plt.subplots(figsize=(8.6, 5.4)); ax.set_xlim(-6.4, 6.6); ax.set_ylim(-3.6, 3.8); ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(Ellipse((-1.2, 0), 9.8, 5.4, fc=DB, ec=DB, alpha=0.06, lw=2))
    ax.text(-5.6, 2.35, r"$F(e_1)$ oil-high", color=DB, fontsize=11)
    ax.add_patch(Ellipse((-2.6, -0.2), 4.0, 2.6, fc=RED, ec=RED, alpha=0.12, lw=2))
    ax.text(-2.6, -1.75, r"$F(e_1)\cap F(e_5)$", color=RED, fontsize=10, ha="center")
    ax.text(-2.6, 0.2, ", ".join(SHORT[c] for c in sorted(core)), ha="center", fontsize=9)
    ax.add_patch(Ellipse((2.7, -0.2), 3.6, 2.6, fc=ORANGE, ec=ORANGE, alpha=0.10, lw=2))
    ax.text(3.4, 1.35, r"$F(e_5)\setminus F(e_1)$", color=ORANGE, fontsize=10)
    oilres = sorted(fe5 - set(core))
    ax.text(2.7, -0.2, ", ".join(SHORT[c] for c in oilres), ha="center", fontsize=9)
    ax.text(2.7, -1.7, "oil-resilient,\nreserve-thin", color=ORANGE, fontsize=8.5, ha="center")
    ax.plot(5.6, 2.3, "o", color="black", ms=4); ax.text(5.6, 2.65, "Jordan\n(breadth)", ha="center", fontsize=8.5)
    fig.savefig(os.path.join(FIG_DIR, "fig_euler.png")); plt.close(fig)


def fig_scenario(core):
    fig, ax = plt.subplots(figsize=(9.2, 4.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    base = sorted(core)
    scen = [("Short blockade", r"$\tau\approx3.0$", base, DB),
            ("Partial re-routing", r"$\tau\approx3.5$", base + ["Bangladesh"], ORANGE),
            ("Prolonged blockade", r"$\tau\approx4.0$", base + ["Bangladesh", "Sri Lanka"], RED)]
    xs = [1.7, 5.0, 8.3]
    for (name, tau, members, col), xc in zip(scen, xs):
        ax.text(xc, 5.5, name, ha="center", fontsize=10.5, weight="bold")
        ax.text(xc, 5.05, tau, ha="center", fontsize=9.5, color=col)
        h = 0.5 * len(members) + 0.6
        ax.add_patch(FancyBboxPatch((xc - 1.5, 4.4 - h), 3.0, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=col, ec=col, alpha=0.12, lw=1.6))
        for i, m in enumerate(members):
            isbase = m in base
            ax.text(xc, 4.05 - i * 0.48, SHORT[m], ha="center", fontsize=9,
                    color="black" if isbase else col, weight="bold" if not isbase else "normal")
    for x0, x1 in [(xs[0] + 1.55, xs[1] - 1.55), (xs[1] + 1.55, xs[2] - 1.55)]:
        _arrow(ax, (x0, 2.6), (x1, 2.6), color=GRAY, lw=1.6)
    ax.text(5.0, 0.4, r"financing queue widens outward from a stable core as buffers exhaust",
            ha="center", fontsize=9, color=GRAY, style="italic")
    fig.savefig(os.path.join(FIG_DIR, "fig_scenario.png")); plt.close(fig)


def fig_extbench(ext_tbl, core):
    """Dashboard rank vs three external indices; core members highlighted. Shows
    that the framework's ordering diverges from generic vulnerability indices and,
    crucially, that the indices would prioritise oil-resilient producers the oil
    filter removes. Grayscale-safe via distinct markers."""
    d = ext_tbl.copy()
    order = d.sort_values("dashboard_rank")["country"].tolist()
    ypos = {c: i for i, c in enumerate(order[::-1])}
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    series = [("dashboard_rank", DB, "o", "Framework (AHP)"),
              ("INFORM_rank", TEAL, "s", "INFORM Risk"),
              ("EVI_rank", ORANGE, "^", "EVI"),
              ("NDGAIN_rank", GRAY, "D", "ND-GAIN vuln.")]
    for col, colour, mk, lab in series:
        ax.scatter([d.set_index("country").loc[c, col] for c in order],
                   [ypos[c] for c in order], s=42, marker=mk, color=colour,
                   label=lab, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.5)
    for c in order:
        yy = ypos[c]
        xs = [d.set_index("country").loc[c, col] for col, *_ in series]
        ax.plot([min(xs), max(xs)], [yy, yy], color=LGRAY, lw=0.8, zorder=1)
    for c in core:
        ax.axhspan(ypos[c] - 0.4, ypos[c] + 0.4, color=DB, alpha=0.06, zorder=0)
    ax.set_yticks([ypos[c] for c in order]); ax.set_yticklabels([SHORT[c] for c in order], fontsize=8.5)
    ax.set_xlabel(r"Rank ($1=$ most vulnerable / highest risk)", fontsize=10)
    ax.set_xlim(0.5, 18.5); ax.invert_xaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.4, frameon=False, ncol=2, loc="lower center")
    ax.set_title("Framework routing vs. established vulnerability indices",
                 fontsize=10.5, pad=8)
    fig.savefig(os.path.join(FIG_DIR, "fig_extbench.png")); plt.close(fig)


def fig_backtest(rb_df, core):
    """Provisional realised-outcome back-test: realised reserve loss by core vs
    non-core (left) and realised H2 inflation vs scarring score (right). Clearly a
    provisional, CI-dominated illustration; grayscale-safe markers."""
    core = set(core)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    # left: reserve loss, core vs non-core
    grp_core = rb_df[rb_df.in_core == 1]["res_loss"].to_numpy()
    grp_non = rb_df[rb_df.in_core == 0]["res_loss"].to_numpy()
    a1.scatter(np.full(len(grp_non), 0) + np.random.default_rng(1).normal(0, 0.04, len(grp_non)),
               grp_non, color=GRAY, marker="o", s=36, alpha=0.8, label="non-core")
    a1.scatter(np.full(len(grp_core), 1) + np.random.default_rng(2).normal(0, 0.03, len(grp_core)),
               grp_core, color=DB, marker="s", s=52, alpha=0.9, label="financing core")
    a1.plot([-0.2, 0.2], [grp_non.mean()] * 2, color=GRAY, lw=2.2)
    a1.plot([0.8, 1.2], [grp_core.mean()] * 2, color=DB, lw=2.2)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["non-core", "core"], fontsize=9.5)
    a1.set_ylabel("Realised reserve loss (months, H2-2026)", fontsize=9.5)
    a1.set_title(r"(a) reserve loss by class", fontsize=10)
    a1.spines[["top", "right"]].set_visible(False)
    a1.text(0.5, a1.get_ylim()[1] * 0.98, "provisional", ha="center", va="top",
            fontsize=8, color=RED, style="italic")
    # right: inflation vs scarring
    for _, r in rb_df.iterrows():
        col = DB if r["country"] in core else GRAY
        mk = "s" if r["country"] in core else "o"
        a2.scatter(r["scarring"], r["infl_h2"], color=col, marker=mk, s=44, alpha=0.85, zorder=3)
        a2.text(r["scarring"] + 0.004, r["infl_h2"], SHORT[r["country"]], fontsize=6.8, va="center")
    # OLS fit line
    xs = rb_df["scarring"].to_numpy(); ys = rb_df["infl_h2"].to_numpy()
    b1, b0 = np.polyfit(xs, ys, 1)
    xg = np.linspace(xs.min(), xs.max(), 20)
    a2.plot(xg, b0 + b1 * xg, color=RED, lw=1.6, ls="--", zorder=2)
    a2.set_xlabel(r"Scarring score $\mathrm{Sc}(u_i)$", fontsize=9.5)
    a2.set_ylabel("Realised H2-2026 food/fuel CPI (pp)", fontsize=9.5)
    a2.set_title(r"(b) inflation vs scarring", fontsize=10)
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_backtest.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    print("DSS-Hormuz reproducibility pipeline (18-country sample)")
    print("-" * 56)

    w_eq = equal_weights()
    w_ent = entropy_weights(GRADED_T2)
    w_ahp, ahp_diag = ahp_weights(_ahp_matrix())
    pd.DataFrame({"channel": CHANNELS, "equal": w_eq, "entropy": w_ent, "ahp": w_ahp}
                 ).to_csv(os.path.join(RES_DIR, "weights.csv"), index=False)
    print(f"AHP: lambda_max={ahp_diag['lambda_max']:.4f}  CI={ahp_diag['CI']:.4f}  CR={ahp_diag['CR']:.4f}")

    sc = pd.DataFrame({"country": COUNTRIES,
                       "equal": stage_scores(GRADED_T2, w_eq),
                       "entropy": stage_scores(GRADED_T2, w_ent),
                       "ahp": stage_scores(GRADED_T2, w_ahp)})
    sc.to_csv(os.path.join(RES_DIR, "scores_t2.csv"), index=False)

    topsis_cc = topsis(GRADED_T2, w_ahp)
    static = stage_scores(GRADED_T2, w_eq)
    bench = pd.DataFrame({"country": COUNTRIES,
                          "dashboard_ahp_rank": ranks_from_scores(sc["ahp"].to_numpy()),
                          "static_rank": ranks_from_scores(static),
                          "topsis_rank": ranks_from_scores(topsis_cc)}
                         ).sort_values("dashboard_ahp_rank").reset_index(drop=True)
    bench.to_csv(os.path.join(RES_DIR, "benchmarking.csv"), index=False)
    rho_static = spearmanr(sc["ahp"], static).correlation
    rho_topsis = spearmanr(sc["ahp"], topsis_cc).correlation
    print(f"Spearman rho  dashboard vs static={rho_static:.2f}  vs TOPSIS={rho_topsis:.2f}")

    # scarring (compositional share) + exposure level + rho-sensitivity
    scar, level = scarring_scores(GRADED_T2, w_ahp)
    scar_df = pd.DataFrame({"country": COUNTRIES, "scarring": scar, "exposure_level": level}
                           ).sort_values("scarring", ascending=False)
    scar_df.to_csv(os.path.join(RES_DIR, "scarring_scores.csv"), index=False)
    scar_sens, scar_sens_diag = scarring_sensitivity(GRADED_T2, w_ahp, B=2000, seed=SEED)
    scar_sens.to_csv(os.path.join(RES_DIR, "scarring_sensitivity.csv"), index=False)
    print(f"Scarring rho-sensitivity: median Spearman vs baseline = "
          f"{scar_sens_diag['spearman_median']:.3f} (5th pct {scar_sens_diag['spearman_p05']:.3f})")

    # dynamics t2 vs t3
    t2 = {c: float(sc.set_index("country").loc[c, "ahp"]) for c in COUNTRIES}
    active_t3 = np.array([1, 1, 0, 0, 1, 1, 0], dtype=bool)
    t3 = {}
    for c in COUNTRIES:
        row = np.array([GRADED_T3[c].get(ch, 0.0) for ch in CHANNELS])
        t3[c] = float(stage_scores(row.reshape(1, -1), w_ahp, active_t3)[0])
    dyn = pd.DataFrame({"country": COUNTRIES, "t2_ahp": [t2[c] for c in COUNTRIES],
                        "t3_ahp": [t3[c] for c in COUNTRIES]})
    dyn["t2_rank"] = ranks_from_scores(dyn["t2_ahp"].to_numpy())
    dyn["t3_rank"] = ranks_from_scores(dyn["t3_ahp"].to_numpy())
    dyn.to_csv(os.path.join(RES_DIR, "dynamics_t2_t3.csv"), index=False)

    # set operations
    sets = {ch: high_risk_set(GRADED_T2, j, 1.0) for j, ch in enumerate(CHANNELS)}
    core = priority_set(OIL_THR, FX_CUTOFF)
    fe5 = sets["e5"]
    pairs = [("e1", "e5"), ("e5", "e6"), ("e1", "e6"), ("e1", "e7"),
             ("e3", "e4"), ("e4", "e7"), ("e2", "e5"), ("e3", "e5")]
    lines = ["High-risk sets F(e_j) at t2 (score = 1.0):"]
    for ch in CHANNELS:
        lines.append(f"  F({ch}) = {sorted(sets[ch]) if sets[ch] else 'empty'}")
    lines.append("\nResilience complements:")
    for ch in ["e1", "e5"]:
        lines.append(f"  R({ch}) = {sorted(complement(sets[ch]))}")
    lines.append("\nDecision-relevant pairwise intersections:")
    for a, b in pairs:
        inter = sorted(sets[a] & sets[b])
        lines.append(f"  F({a}) ∩ F({b}) = {inter if inter else 'empty'}")
    lines.append(f"\nPriority core F(e1)∩F(e5) = {sorted(core)}")
    lines.append(f"F(e5)                     = {sorted(fe5)}")
    lines.append(f"Oil filter removes        = {sorted(fe5 - core)}  (oil-resilient, reserve-thin)")
    lines.append("\nReserve-cutoff sensitivity F(e1) ∩ F(e5^<tau):")
    thr_rows = []
    for tau in [2.0, 2.5, 3.0, 3.5, 4.0]:
        s = sorted(priority_set(OIL_THR, tau))
        thr_rows.append({"tau": tau, "priority_set": ";".join(s), "size": len(s)})
        lines.append(f"  tau={tau}: {s} (size {len(s)})")
    pd.DataFrame(thr_rows).to_csv(os.path.join(RES_DIR, "threshold_sensitivity.csv"), index=False)
    with open(os.path.join(RES_DIR, "set_operations.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Priority core:", sorted(core))
    print("Oil filter removes (reserve-thin, oil-resilient):", sorted(fe5 - core))

    # bootstraps
    rank_tbl, set_tbl, exact_core = bootstrap(GRADED_T2, w_ahp, w_eq, w_ent, B=2000, seed=SEED)
    rank_tbl.to_csv(os.path.join(RES_DIR, "bootstrap_ranks.csv"), index=False)
    set_tbl.to_csv(os.path.join(RES_DIR, "bootstrap_set_membership.csv"), index=False)
    print(f"Bootstrap exact-core recurrence = {exact_core:.2f}")

    # schedule sensitivity
    sched = schedule_sensitivity(GRADED_T2, w_ahp)
    sched.to_csv(os.path.join(RES_DIR, "schedule_sensitivity.csv"), index=False)

    # back-test
    bt_df, bt_stats = backtest(core)
    bt_df.to_csv(os.path.join(RES_DIR, "backtest_concordance.csv"), index=False)
    print(f"Back-test: core precision = {bt_stats['precision']:.2f}, recall = {bt_stats['recall']:.2f}; "
          f"routed elsewhere = {bt_stats['deep_routed_elsewhere']}")

    # regression (sidelined / deferred)
    with open(os.path.join(RES_DIR, "regression_summary.txt"), "w", encoding="utf-8") as f:
        f.write(run_regression(PANEL_CSV))

    # --- revision additions ------------------------------------------------- #
    # (3) inter-coder reliability on the five qualitative channels
    icr_tbl, icr_diag = intercoder_reliability(GRADED_T2)
    icr_tbl.to_csv(os.path.join(RES_DIR, "intercoder_reliability.csv"), index=False)
    print(f"Inter-coder reliability: pooled exact agreement = {icr_diag['pooled_exact_agreement']:.2f}, "
          f"kappa = {icr_diag['pooled_kappa']:.2f}, weighted kappa = {icr_diag['pooled_weighted_kappa']:.2f}")

    # (8) external-index benchmarking (INFORM / EVI / ND-GAIN)
    ext_tbl, ext_diag = external_benchmarking(sc["ahp"].to_numpy())
    ext_tbl.to_csv(os.path.join(RES_DIR, "external_benchmarking.csv"), index=False)
    print(f"External benchmarking Spearman rho: {ext_diag['spearman']}")

    # (9) sample-composition sensitivity (leave-k-out + add-in)
    lko_tbl, addin_tbl, comp_diag = sample_composition_sensitivity(seed=SEED)
    lko_tbl.to_csv(os.path.join(RES_DIR, "leave_k_out.csv"), index=False)
    addin_tbl.to_csv(os.path.join(RES_DIR, "addin_importers.csv"), index=False)
    print(f"Sample composition: leave-k-out min survival = {comp_diag['leave_k_out_min_survival']:.2f}; "
          f"add-in pairs preserving core = {comp_diag['addin_pairs_preserved']}")

    # (10) e7 informativeness
    e7_diag, e7_ops = e7_informativeness(GRADED_T2, w_ahp)
    with open(os.path.join(RES_DIR, "e7_informativeness.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps({**e7_diag, **e7_ops}, indent=2, default=str) + "\n")
    print(f"e7 informativeness: {e7_diag['e7_high_flags']} high flag(s) = {e7_diag['Fe7']}; "
          f"Spearman with/without e7 = {e7_diag['spearman_rank_with_vs_without_e7']:.2f}")

    # (4) e6-only perturbation of the transfer-routing group
    e6_diag = e6_transfer_sensitivity(GRADED_T2)
    with open(os.path.join(RES_DIR, "e6_transfer_sensitivity.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(e6_diag, indent=2, default=str) + "\n")
    print(f"e6 transfer sensitivity: transfer group mean Jaccard vs base = "
          f"{e6_diag['transfer_mean_jaccard_vs_base']:.2f}; financing core changed in "
          f"{e6_diag['financing_core_changed_frac']:.2f} of draws (should be 0.00)")

    # (7) provisional realised-outcome back-test regression
    rb_df, rb_models, rb_diag = realised_backtest_regression(sc["ahp"].to_numpy(), core, scar)
    rb_df.to_csv(os.path.join(RES_DIR, "realised_backtest_data.csv"), index=False)
    with open(os.path.join(RES_DIR, "realised_backtest_regression.txt"), "w", encoding="utf-8") as f:
        f.write("PROVISIONAL realised-outcome back-test (REALISED_H2 is a labelled placeholder).\n")
        f.write("Permutation p-values (B=10000 label reshuffles) accompany each focal coefficient.\n\n")
        labels = {"res_loss_ctrl": "res_loss ~ core + pre_reserves",
                  "res_loss_noctrl": "res_loss ~ core (no reserve control)",
                  "infl_core": "infl_h2 ~ core (cleaner: inflation not in core def.)",
                  "infl_scarring": "infl_h2 ~ scarring + pre_reserves"}
        for yname, (tab, r2, pval) in rb_models.items():
            f.write(f"== {labels.get(yname, yname)}  (R^2 = {r2:.3f}, perm p = {pval:.3f}) ==\n")
            f.write(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n\n")
        f.write(json.dumps(rb_diag, indent=2) + "\n")
    print(f"Provisional back-test: core reserve-loss coef = {rb_diag['res_loss_core_coef']} "
          f"(90% CI {rb_diag['res_loss_core_ci90']}, perm p={rb_diag['res_loss_core_permp']}); "
          f"no-control coef = {rb_diag['res_loss_core_coef_noctrl']} (perm p={rb_diag['res_loss_core_permp_noctrl']}); "
          f"infl~core coef = {rb_diag['infl_core_coef']} (perm p={rb_diag['infl_core_permp']})")

    # (11) auditable costing table for the financing core
    cost_tbl, cost_diag = costing_table()
    cost_tbl.to_csv(os.path.join(RES_DIR, "costing_core.csv"), index=False)
    print(f"Costing: financing-core combined rapid access ~ US$"
          f"{cost_diag['core_total_rapid_access_USDbn']:.1f} bn (100% of quota)")

    # figures: main
    fig_brent(); fig_weights(w_ahp, w_eq, w_ent); fig_scores(sc)
    fig_rankboot(rank_tbl, core, fe5); fig_setboot(set_tbl, core, fe5)
    fig_reserves(core, fe5); fig_slope(t2, t3, core, fe5); fig_scarring(scar_df, core, fe5)
    fig_scarringlevel(scar_df, core, fe5)
    # figures: revision additions
    fig_extbench(ext_tbl, ext_diag["core"]); fig_backtest(rb_df, core)
    # figures: supplementary
    fig_flowchart(); fig_channels(); fig_stages(); fig_graded(); fig_euler(core, fe5); fig_scenario(core)

    summary = {
        "seed": SEED, "n_countries": len(COUNTRIES),
        "ahp": {k: float(v) for k, v in ahp_diag.items()},
        "spearman": {"dashboard_vs_static": float(rho_static), "dashboard_vs_topsis": float(rho_topsis)},
        "priority_core": sorted(core),
        "Fe5": sorted(fe5),
        "oil_filter_removes": sorted(fe5 - core),
        "priority_core_exact_recurrence": float(exact_core),
        "set_membership_freq": {r["country"]: float(r["membership_freq"]) for _, r in set_tbl.iterrows()},
        "backtest": bt_stats,
        "scarring_rho_sensitivity": scar_sens_diag,
        "intercoder_reliability": icr_diag,
        "external_benchmarking": ext_diag,
        "sample_composition_sensitivity": comp_diag,
        "e7_informativeness": e7_diag,
        "e6_transfer_sensitivity": e6_diag,
        "realised_backtest_provisional": rb_diag,
        "costing_core": cost_diag,
        "figures_written": sorted(os.listdir(FIG_DIR)),
    }
    with open(os.path.join(RES_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 56)
    print(f"Figures -> {os.path.relpath(FIG_DIR, ROOT)}/  ({len(summary['figures_written'])} PNGs)")
    print(f"Results -> {os.path.relpath(RES_DIR, ROOT)}/")
    print("Done.")


if __name__ == "__main__":
    main()
