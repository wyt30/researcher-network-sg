"""
Singapore Publication Network -- Node & Edge Extractor (v2)

Produces:
  nodes.csv  -- one row per unique author (publication_count, affiliations,
                country, name_variants)
  edges.csv  -- one row per unique author pair (source, target, weight)

Key design decisions
--------------------
Author normalization:
  Authors are grouped by (surname_lower, first_word_lower) so that
  "Foo, Roger" and "Foo, Roger S Y" merge into one node, while
  "Foo, Randy" stays separate. The canonical name (used in output) is the
  most-frequently-appearing exact variant across all papers. All merged
  variants are recorded in the 'name_variants' column for auditing.
  Single initials ("Foo, R") form their own ambiguous group and are NOT
  merged with full-name groups ("Foo, Roger").

Affiliation parsing:
  authors field      -- semicolon-separated names
  affiliations field -- semicolon-separated slots, positionally aligned to
                        authors; multiple institutions per author pipe-separated
"""

import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_FILE = Path(__file__).parent / "pubmed_singapore_20260502.csv"
NODES_FILE = Path(__file__).parent / "nodes.csv"
EDGES_FILE = Path(__file__).parent / "edges.csv"
CHUNKSIZE = 5_000


# ---------------------------------------------------------------------------
# Author name normalisation
# ---------------------------------------------------------------------------

def author_key(name: str) -> tuple[str, str] | None:
    """Return a grouping key (surname_lower, first_word_lower), or None.

    Groups "Foo, Roger" and "Foo, Roger S Y" -> ("foo", "roger").
    Keeps "Foo, R" as ("foo", "r"), separate from full-name groups.
    Returns None for consortium / group entries (no comma, or unparseable).
    """
    if "," not in name:
        return None
    surname, given_rest = name.split(",", 1)
    surname = surname.strip().lower()
    given_rest = given_rest.strip()
    if not surname or not given_rest:
        return None
    first_word = given_rest.split()[0].rstrip(".").lower()
    if not first_word or not first_word[0].isalpha():
        return None
    return (surname, first_word)


# ---------------------------------------------------------------------------
# Affiliation cleaning
# ---------------------------------------------------------------------------

_CLEAN_STEPS = [
    # Markdown email links: [user@host](mailto:user@host)
    re.compile(r"\[[\w.@+%\-]+\]\(mailto:[^)]+\)", re.IGNORECASE),
    # "Electronic address: ..." (PubMed corresponding-author block)
    re.compile(r"[,;.]?\s*Electronic\s+address\s*:.*$", re.IGNORECASE | re.DOTALL),
    # Fax / Tel / Phone / Email blocks
    re.compile(r"[,;.]?\s*(?:fax|tel\.?|phone|email|e-mail)\s*:.*$",
               re.IGNORECASE | re.DOTALL),
    # Truncated "Tele" at end of string
    re.compile(r"[,;.]?\s*Tele[:\s]*$", re.IGNORECASE),
    # Bare email addresses
    re.compile(r"[\w.+%\-]+@[\w.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
    # URLs
    re.compile(r"https?://\S+", re.IGNORECASE),
    # Residual "(Name)" tags left after email stripping
    re.compile(r"\s*\([A-Z][a-z]+\)\s*$"),
]


def clean_affiliation(aff: str) -> str:
    for step in _CLEAN_STEPS:
        aff = step.sub("", aff)
    return aff.strip().rstrip(".,;").strip()


# ---------------------------------------------------------------------------
# Country detection (pattern matching against canonical country list)
# ---------------------------------------------------------------------------

_RAW_COUNTRY_PATTERNS: list[tuple[str, str]] = [
    # ---- China variants (most-specific first) ----
    (r"People'?s\s+Republic\s+of\s+China",   "China"),
    (r"P\.?\s*R\.?\s*China",                  "China"),
    (r"PR\s*China",                            "China"),
    # ---- Taiwan (before bare China) ----
    (r"Republic\s+of\s+China",                "Taiwan"),
    (r"R\.?O\.?C\.?",                         "Taiwan"),
    (r"\bTaiwan\b",                            "Taiwan"),
    # ---- Korea ----
    (r"Republic\s+of\s+Korea",                "South Korea"),
    (r"South\s+Korea",                        "South Korea"),
    (r"\bKorea\b",                             "South Korea"),
    # ---- Singapore ----
    (r"Republic\s+of\s+Singapore",            "Singapore"),
    (r"\bSingapore\b",                         "Singapore"),
    # ---- USA ----
    (r"\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|"
     r"LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|"
     r"OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\s+US(?:A)?\b",
                                               "United States"),
    (r"United\s+States\s+of\s+America",       "United States"),
    (r"The\s+United\s+States\s+of\s+America", "United States"),
    (r"United\s+States",                       "United States"),
    (r"\bU\.S\.A\.\b",                         "United States"),
    (r"\bUSA\b",                               "United States"),
    # ---- UK ----
    (r"United\s+Kingdom",                      "United Kingdom"),
    (r"\bU\.?K\.?\b",                          "United Kingdom"),
    (r"\bEngland\b",                           "United Kingdom"),
    (r"\bScotland\b",                          "United Kingdom"),
    (r"\bWales\b",                             "United Kingdom"),
    (r"\bNorthern\s+Ireland\b",                "United Kingdom"),
    # ---- Canada ----
    (r"\b(?:BC|ON|QC|AB|SK|MB|NS|NB|PE|NL)\s+Canada\b", "Canada"),
    (r"\bCanada\b",                            "Canada"),
    # ---- Netherlands ----
    (r"The\s+Netherlands",                     "Netherlands"),
    (r"\bNetherlands\b",                       "Netherlands"),
    # ---- Hong Kong ----
    (r"Hong\s+Kong",                           "Hong Kong"),
    (r"SAR\s+China",                           "Hong Kong"),
    # ---- Other multi-word ----
    (r"New\s+Zealand",                         "New Zealand"),
    (r"Saudi\s+Arabia",                        "Saudi Arabia"),
    (r"South\s+Africa",                        "South Africa"),
    (r"United\s+Arab\s+Emirates",              "United Arab Emirates"),
    (r"Russian\s+Federation",                  "Russia"),
    (r"Viet\s+Nam",                            "Vietnam"),
    (r"Lao\s+PDR",                             "Laos"),
    (r"Sri\s+Lanka",                           "Sri Lanka"),
    (r"Czech\s+Republic",                      "Czech Republic"),
    (r"Republic\s+of\s+Serbia",               "Serbia"),
    (r"Myanmar/Burma|Myanmar|Burma",           "Myanmar"),
    (r"Inner\s+Mongolia|Tibet|Xinjiang",       "China"),
    # ---- China (bare, after all variants) ----
    (r"\bChina\b",                             "China"),
    # ---- Brazil / Mexico (non-English spellings) ----
    (r"\bBrasil\b",                            "Brazil"),
    (r"\bBrazil\b",                            "Brazil"),
    (r"\bM[eé]xico\b",                         "Mexico"),
    # ---- Russia / Vietnam ----
    (r"\bRussia\b",                            "Russia"),
    (r"\bVietnam\b",                           "Vietnam"),
    # ---- Alphabetical single-word countries ----
    (r"\bAfghanistan\b",   "Afghanistan"),  (r"\bAlbania\b",       "Albania"),
    (r"\bAlgeria\b",       "Algeria"),      (r"\bArgentina\b",     "Argentina"),
    (r"\bAustralia\b",     "Australia"),    (r"\bAustria\b",       "Austria"),
    (r"\bBahrain\b",       "Bahrain"),      (r"\bBangladesh\b",    "Bangladesh"),
    (r"\bBelarus\b",       "Belarus"),      (r"\bBelgium\b",       "Belgium"),
    (r"\bBrunei\b",        "Brunei"),       (r"\bBulgaria\b",      "Bulgaria"),
    (r"\bCambodia\b",      "Cambodia"),     (r"\bCameroon\b",      "Cameroon"),
    (r"\bChile\b",         "Chile"),        (r"\bColombia\b",      "Colombia"),
    (r"\bCroatia\b",       "Croatia"),      (r"\bCyprus\b",        "Cyprus"),
    (r"\bDenmark\b",       "Denmark"),      (r"\bEcuador\b",       "Ecuador"),
    (r"\bEgypt\b",         "Egypt"),        (r"\bEthiopia\b",      "Ethiopia"),
    (r"\bFinland\b",       "Finland"),      (r"\bFrance\b",        "France"),
    (r"\bGermany\b",       "Germany"),      (r"\bGhana\b",         "Ghana"),
    (r"\bGreece\b",        "Greece"),       (r"\bHungary\b",       "Hungary"),
    (r"\bIceland\b",       "Iceland"),      (r"\bIndia\b",         "India"),
    (r"\bIndonesia\b",     "Indonesia"),    (r"\bIran\b",          "Iran"),
    (r"\bIraq\b",          "Iraq"),         (r"\bIreland\b",       "Ireland"),
    (r"\bIsrael\b",        "Israel"),       (r"\bItaly\b",         "Italy"),
    (r"\bJamaica\b",       "Jamaica"),      (r"\bJapan\b",         "Japan"),
    (r"\bJordan\b",        "Jordan"),       (r"\bKazakhstan\b",    "Kazakhstan"),
    (r"\bKenya\b",         "Kenya"),        (r"\bKosovo\b",        "Kosovo"),
    (r"\bKuwait\b",        "Kuwait"),       (r"\bLaos\b",          "Laos"),
    (r"\bLatvia\b",        "Latvia"),       (r"\bLebanon\b",       "Lebanon"),
    (r"\bLibya\b",         "Libya"),        (r"\bLithuania\b",     "Lithuania"),
    (r"\bLuxembourg\b",    "Luxembourg"),   (r"\bMalaysia\b",      "Malaysia"),
    (r"\bMalta\b",         "Malta"),        (r"\bMexico\b",        "Mexico"),
    (r"\bMoldova\b",       "Moldova"),      (r"\bMongolia\b",      "Mongolia"),
    (r"\bMorocco\b",       "Morocco"),      (r"\bMozambique\b",    "Mozambique"),
    (r"\bNepal\b",         "Nepal"),        (r"\bNigeria\b",       "Nigeria"),
    (r"\bNorway\b",        "Norway"),       (r"\bOman\b",          "Oman"),
    (r"\bPakistan\b",      "Pakistan"),     (r"\bPer[uú]\b",       "Peru"),
    (r"\bPhilippines\b",   "Philippines"),  (r"\bPoland\b",        "Poland"),
    (r"\bPortugal\b",      "Portugal"),     (r"\bQatar\b",         "Qatar"),
    (r"\bRomania\b",       "Romania"),      (r"\bSerbia\b",        "Serbia"),
    (r"\bSlovakia\b",      "Slovakia"),     (r"\bSpain\b",         "Spain"),
    (r"\bSudan\b",         "Sudan"),        (r"\bSweden\b",        "Sweden"),
    (r"\bSwitzerland\b",   "Switzerland"),  (r"\bTanzania\b",      "Tanzania"),
    (r"\bThailand\b",      "Thailand"),     (r"\bTunisia\b",       "Tunisia"),
    (r"\bTurkey\b",        "Turkey"),       (r"\bUganda\b",        "Uganda"),
    (r"\bUkraine\b",       "Ukraine"),      (r"\bUruguay\b",       "Uruguay"),
    (r"\bZambia\b",        "Zambia"),       (r"\bZimbabwe\b",      "Zimbabwe"),
]

COUNTRY_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), canonical)
    for pat, canonical in _RAW_COUNTRY_PATTERNS
]


def detect_country(aff: str) -> str | None:
    """Detect canonical country from a single cleaned affiliation string.

    Splits by comma and scans tokens from end to start (country is last).
    Falls back to full-string search for no-comma affiliations (common in
    East-Asian format: 'Institution City Country').
    """
    tokens = [t.strip() for t in aff.split(",") if t.strip()]
    for token in reversed(tokens):
        for pattern, canonical in COUNTRY_PATTERNS:
            if pattern.search(token):
                return canonical
    return None


# ---------------------------------------------------------------------------
# Accumulators (keyed by norm_key, not raw name string)
# ---------------------------------------------------------------------------

# node_info[norm_key] = {
#   'name_counts': Counter,           -- exact name string -> paper count
#   'pub_count':   int,
#   'affiliations': set[str],
#   'country_counts': defaultdict[str, int],
# }
node_info: dict[tuple, dict] = {}

# edge_counts[(norm_key_a, norm_key_b)] -> int  (keys always sorted)
edge_counts: defaultdict = defaultdict(int)


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def process_row(authors_str, affiliations_str) -> None:
    if not isinstance(authors_str, str) or not authors_str.strip():
        return

    all_tokens = [a.strip() for a in authors_str.split(";") if a.strip()]

    # Pair each token with its norm_key (None = consortium/invalid)
    keyed = [(tok, author_key(tok)) for tok in all_tokens]
    valid_keys = [k for _, k in keyed if k is not None]
    if not valid_keys:
        return

    aff_slots: list[str] = []
    if isinstance(affiliations_str, str) and affiliations_str.strip():
        aff_slots = [s.strip() for s in affiliations_str.split(";")]

    # --- Update nodes ---
    for i, (token, key) in enumerate(keyed):
        if key is None:
            continue  # skip consortia; preserve positional index for affiliations
        if key not in node_info:
            node_info[key] = {
                "name_counts": Counter(),
                "pub_count": 0,
                "affiliations": set(),
                "country_counts": defaultdict(int),
            }
        info = node_info[key]
        info["name_counts"][token] += 1
        info["pub_count"] += 1

        if i < len(aff_slots) and aff_slots[i]:
            for inst in aff_slots[i].split("|"):
                inst = clean_affiliation(inst.strip())
                if inst:
                    info["affiliations"].add(inst)
                    country = detect_country(inst)
                    if country:
                        info["country_counts"][country] += 1

    # --- Update edges ---
    # Use sorted unique keys to avoid duplicate/self-loop edges
    unique_keys = sorted(set(valid_keys))
    if len(unique_keys) > 1:
        for ka, kb in combinations(unique_keys, 2):
            edge_counts[(ka, kb)] += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        sys.exit(f"Input file not found: {INPUT_FILE}")

    print(f"Reading: {INPUT_FILE}")
    t0 = time.perf_counter()
    total_rows = 0

    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE, dtype=str,
                              low_memory=False):
        # Strip BOM and quote artefacts from column names (Windows CSV)
        chunk.columns = [c.strip().lstrip("﻿").strip('"')
                         for c in chunk.columns]
        for _, row in chunk.iterrows():
            process_row(row.get("authors"), row.get("affiliations"))
        total_rows += len(chunk)
        print(f"  processed {total_rows:,} rows ...", end="\r")

    print(f"\nDone reading {total_rows:,} rows in "
          f"{time.perf_counter() - t0:.1f}s")

    # Resolve canonical name for each norm_key = most common exact variant
    key_to_canonical: dict[tuple, str] = {
        key: info["name_counts"].most_common(1)[0][0]
        for key, info in node_info.items()
    }

    # --- Identify Singapore authors ---
    # An author is "Singapore" if Singapore is their most frequent country.
    singapore_keys: set[tuple] = set()
    for key, info in node_info.items():
        if not info["country_counts"]:
            continue
        top_country = max(info["country_counts"], key=info["country_counts"].get)
        if top_country == "Singapore":
            singapore_keys.add(key)

    print(f"Singapore authors identified: {len(singapore_keys):,} "
          f"(out of {len(node_info):,} total)")

    # --- Build nodes DataFrame (Singapore only) ---
    print("Building nodes ...")
    nodes_rows = []
    for key, info in node_info.items():
        if key not in singapore_keys:
            continue
        canonical = key_to_canonical[key]
        country = max(info["country_counts"], key=info["country_counts"].get)
        affiliations = "; ".join(sorted(info["affiliations"]))
        variants = sorted(info["name_counts"].keys())
        nodes_rows.append({
            "author":            canonical,
            "publication_count": info["pub_count"],
            "affiliations":      affiliations,
            "country":           country,
            "name_variants":     " | ".join(variants) if len(variants) > 1 else "",
        })

    nodes_df = pd.DataFrame(nodes_rows)
    nodes_df.sort_values("author", inplace=True)
    try:
        nodes_df.to_csv(NODES_FILE, index=False, encoding="utf-8-sig")
        print(f"  nodes.csv: {len(nodes_df):,} Singapore authors -> {NODES_FILE}")
    except PermissionError:
        alt = NODES_FILE.with_stem(NODES_FILE.stem + "_new")
        nodes_df.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"  nodes.csv is open in another app -> saved to {alt}")

    # --- Build edges DataFrame (both endpoints must be Singapore authors) ---
    print("Building edges ...")
    edges_rows = []
    for (ka, kb), weight in edge_counts.items():
        if ka not in singapore_keys or kb not in singapore_keys:
            continue
        src = key_to_canonical[ka]
        tgt = key_to_canonical[kb]
        if src > tgt:
            src, tgt = tgt, src
        edges_rows.append({"source": src, "target": tgt, "weight": weight})

    # Merge any edges that collapsed to the same canonical-name pair
    edges_df = (pd.DataFrame(edges_rows)
                  .groupby(["source", "target"], as_index=False)["weight"]
                  .sum())
    edges_df.sort_values(["source", "target"], inplace=True)
    try:
        edges_df.to_csv(EDGES_FILE, index=False, encoding="utf-8-sig")
        print(f"  edges.csv: {len(edges_df):,} Singapore-only pairs -> {EDGES_FILE}")
    except PermissionError:
        alt = EDGES_FILE.with_stem(EDGES_FILE.stem + "_new")
        edges_df.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"  edges.csv is open in another app -> saved to {alt}")

    elapsed = time.perf_counter() - t0
    print(f"\nFinished in {elapsed:.1f}s")

    # --- Diagnostics ---
    merged = nodes_df[nodes_df["name_variants"] != ""]
    print(f"\nAuthor groups with merged name variants: {len(merged):,}")
    print(f"(See 'name_variants' column in nodes.csv to audit merges)")


if __name__ == "__main__":
    main()
