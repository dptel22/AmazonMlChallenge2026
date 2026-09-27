# -*- coding: utf-8 -*-
"""Phase 1: normalization + offline transliteration.

No network access anywhere: transliteration is a hand-written deterministic
Unicode character-mapping table (Devanagari U+0900.., Gujarati U+0A80..),
not a lookup service.
"""
import re
import unicodedata

# --- hand-written transliteration table ---------------------------------------
# Built once at import from unicodedata.name() (stdlib, offline, deterministic):
# Indic scripts share Unicode letter names ("LETTER KA" = /k/ in every script),
# so one name->romanization map covers all 9 scripts present in the data.
_LETTERS = {
    "A": "a", "AA": "aa", "I": "i", "II": "ii", "U": "u", "UU": "uu",
    "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li", "VOCALIC LL": "li",
    "CANDRA E": "e", "SHORT E": "e", "E": "e", "EE": "e", "AI": "ai",
    "CANDRA O": "o", "SHORT O": "o", "O": "o", "OO": "o", "AU": "au",
    "SHORT A": "a", "CANDRA A": "a", "OE": "o", "OOE": "o", "AW": "o",
    "UE": "u", "UUE": "u",
    "KA": "k", "KHA": "kh", "GA": "g", "GHA": "gh", "NGA": "n",
    "CA": "ch", "CHA": "ch", "JA": "j", "JHA": "jh", "NYA": "n",
    "TTA": "t", "TTHA": "th", "DDA": "d", "DDHA": "dh", "NNA": "n", "NNNA": "n",
    "TA": "t", "THA": "th", "DA": "d", "DHA": "dh", "NA": "n",
    "PA": "p", "PHA": "ph", "BA": "b", "BHA": "bh", "MA": "m",
    "YA": "y", "RA": "r", "RRA": "r", "LA": "l", "LLA": "l", "LLLA": "l",
    "VA": "v", "SHA": "sh", "SSA": "sh", "SA": "s", "HA": "h",
    "QA": "q", "KHHA": "kh", "GHHA": "g", "ZA": "z", "DDDHA": "r", "RHA": "r",
    "FA": "f", "YYA": "y", "GGA": "g", "JJA": "j", "DDDA": "d", "BBA": "b",
    "ZHA": "l", "MARWARI DDA": "d", "HEAVY YA": "y", "GLOTTAL STOP": "",
    "AVAGRAHA": "", "OM": "",
}
_SIGNS = {
    "ANUSVARA": "n", "CANDRABINDU": "", "INVERTED CANDRABINDU": "",
    "VISARGA": "h", "NUKTA": "", "VIRAMA": "",
    "VOWEL SIGN AA": "aa", "VOWEL SIGN I": "i", "VOWEL SIGN II": "ii",
    "VOWEL SIGN U": "u", "VOWEL SIGN UU": "uu", "VOWEL SIGN VOCALIC R": "ri",
    "VOWEL SIGN VOCALIC RR": "ri", "VOWEL SIGN VOCALIC L": "li",
    "VOWEL SIGN VOCALIC LL": "li", "VOWEL SIGN E": "e",
    "VOWEL SIGN SHORT E": "e", "VOWEL SIGN CANDRA E": "e",
    "VOWEL SIGN AI": "ai", "VOWEL SIGN O": "o", "VOWEL SIGN SHORT O": "o",
    "VOWEL SIGN CANDRA O": "o", "VOWEL SIGN AU": "au",
    "VOWEL SIGN PRISHTHAMATRA E": "e", "VOWEL SIGN AW": "o",
    "VOWEL SIGN OE": "o", "VOWEL SIGN OOE": "o", "VOWEL SIGN UE": "u",
    "VOWEL SIGN UUE": "u", "VOWEL SIGN CANDRA LONG E": "e",
    "AU LENGTH MARK": "au",
}
_TR = {}
for _cp in range(0x0900, 0x0D80):  # all Indic blocks present in the data
    _n = unicodedata.name(chr(_cp), "")
    for _prefix, _tab in (("DEVANAGARI ", None), ("GUJARATI ", None),
                          ("TAMIL ", None), ("TELUGU ", None), ("KANNADA ", None),
                          ("MALAYALAM ", None), ("ORIYA ", None),
                          ("GURMUKHI ", None), ("BENGALI ", None)):
        if _n.startswith(_prefix):
            _rest = _n[len(_prefix):]
            if _rest.startswith("LETTER "):
                _t = _LETTERS.get(_rest[7:])
            elif _rest.startswith("VOWEL SIGN "):
                _t = _SIGNS.get(_rest, _SIGNS.get("VOWEL SIGN " + _rest[11:]))
            elif _rest.startswith("SIGN "):
                _t = _SIGNS.get(_rest[5:])
            elif _rest.startswith("DIGIT "):
                _t = {"ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3",
                      "FOUR": "4", "FIVE": "5", "SIX": "6", "SEVEN": "7",
                      "EIGHT": "8", "NINE": "9"}.get(_rest[6:])
            else:
                _t = " "  # danda, rupee sign, abbreviation sign, ...
            if _t is not None:
                _TR[_cp] = _t
            break


_TRANSLATE_TABLE = {**{cp: " " for cp in range(0x0900, 0x0D80)},
                    **_TR, 0x200C: None, 0x200D: None}


def transliterate(s):
    if s.isascii():
        return s
    return s.translate(_TRANSLATE_TABLE)


_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9]+")


def norm_text(s):
    """lowercase + transliterate + fold Latin diacritics + punct->space + collapse.

    Order matters: NFKC-compose, transliterate Indic->ASCII (needs composed input),
    THEN NFKD-decompose + strip combining marks so accented Latin folds to base
    letters (café->cafe) instead of being deleted by the alnum filter. France is
    untestable against train labels, so this path is guarded by a self-check below.
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = transliterate(s).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = _NONALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


# --- legal suffix stripping ---------------------------------------------------
_SUFFIXES = {"inc", "incorporated", "llc", "llp", "pvt", "ltd", "limited",
             "corp", "corporation", "company", "co", "gmbh", "plc", "trust",
             "private"}
_PUNCTY = set(".,[](){}'\"")


def strip_suffixes(name_norm):
    """Strip legal suffix tokens from both ends of an already-normalized name.

    Tokens are compared with wrapper punctuation (brackets/parens/quotes) removed,
    so "[LLP]" and "(Inc.)" strip like their bare forms.
    """
    toks = name_norm.split()
    while toks and (toks[0].strip(".,[](){}'\"") in _SUFFIXES or all(c in _PUNCTY for c in toks[0])):
        toks.pop(0)
    while toks and (toks[-1].strip(".,[](){}'\"") in _SUFFIXES or all(c in _PUNCTY for c in toks[-1])):
        toks.pop()
    return " ".join(toks)


def norm_name(s):
    """Return (normalized, suffix-stripped) name strings."""
    n = norm_text(s)
    return n, strip_suffixes(n)


def addr_tokens(s):
    """Loose address tokenization: no order/component assumptions."""
    return norm_text(s).split()


def street_number(addr):
    """Leading house/street number of a normalized address, '' if none."""
    for t in addr_tokens(addr):
        if t[0].isdigit():
            return t
        return ""
    return ""


# --- TF-IDF cosine pair scorer (cheap Levenshtein replacement) ----------------
# Word-unigram TF-IDF with L2-normalized rows: the dot product of two rows is
# their cosine. Fitted once per pool frame and cached on the frame's .attrs;
# query frames (s1 / row chunks) are transformed with the pool-fitted
# vectorizer so both sides share one vocabulary. Lazy sklearn import keeps
# this module importable without scikit-learn.

def _cos_corpus(frame, primary, fallback=None):
    """Text corpus for one TF-IDF fit/transform: `primary` column when present,
    else `fallback` (minimal frames carry name_s but not the derived name_t),
    else all-blank (degenerate corpus -> zero vectors). Pool and query frames
    resolve the same way, so both sides share one vocabulary convention."""
    for col in (primary, fallback):
        if col in frame.columns:
            return frame[col].fillna("").values
    return [""] * len(frame)


def _cos_fit(corpus):
    """Fit the TF-IDF vectorizer; refit without the df floor when a tiny corpus
    prunes the min_df=2 vocabulary empty, and give up (None) only when the
    corpus holds no tokens at all."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    def vec(min_df):
        return TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\S+",
                               min_df=min_df, sublinear_tf=True, dtype=np.float32)
    try:
        v = vec(2)
        return v, v.fit_transform(corpus)
    except ValueError:
        pass
    try:
        v = vec(1)
        return v, v.fit_transform(corpus)
    except ValueError:
        return None, None            # blank corpus (tiny slices)


def _cos_fit_frame(frame):
    vn, Xn = _cos_fit(_cos_corpus(frame, "name_t", "name_s"))
    va, Xa = _cos_fit(_cos_corpus(frame, "addr_n"))
    return {"vn": vn, "va": va, "Xn": Xn, "Xa": Xa}


def attach_cosine(pool):
    """Fit TF-IDF on the pool frame once; cache state on pool.attrs."""
    if "_cos" not in pool.attrs:
        pool.attrs["_cos"] = _cos_fit_frame(pool)


def frame_cos_rows(frame, pool):
    """Transform a query frame (s1 frame or row chunk) with the pool vectorizer.

    The cache is tagged with the owning frame's identity: pandas propagates
    attrs through iloc (verified on pandas 3.0), so a chunk sliced from a
    parent that already carries vectors would otherwise reuse the parent-
    sized matrix and score chunk-local positions against the wrong rows."""
    attach_cosine(pool)
    cached = frame.attrs.get("_cos")
    if cached is not None and cached.get("owner") == id(frame):
        return
    st = pool.attrs["_cos"]
    xn = (st["vn"].transform(_cos_corpus(frame, "name_t", "name_s"))
          if st["vn"] is not None else None)
    xa = (st["va"].transform(_cos_corpus(frame, "addr_n"))
          if st["va"] is not None else None)
    frame.attrs["_cos"] = {"owner": id(frame), "Xn": xn, "Xa": xa}


def cos_pair_scores(frame, pool, si, ci):
    """Row-wise (name_cos, addr_cos) for frame row si vs pool row ci.
    si/ci are positional integer indices/arrays into frame/pool respectively."""
    import numpy as np
    fs, ps = frame.attrs["_cos"], pool.attrs["_cos"]
    n = len(np.asarray(si).reshape(-1))
    if fs["Xn"] is not None and ps["Xn"] is not None:
        name = np.asarray(fs["Xn"][si].multiply(ps["Xn"][ci]).sum(axis=1)).ravel()
    else:
        name = np.zeros(n, dtype=np.float64)
    if fs["Xa"] is not None and ps["Xa"] is not None:
        addr = np.asarray(fs["Xa"][si].multiply(ps["Xa"][ci]).sum(axis=1)).ravel()
    else:
        addr = np.zeros(n, dtype=np.float64)
    return name.astype(np.float64), addr.astype(np.float64)


# --- Phase 1 GATE self-check ---------------------------------------------------
if __name__ == "__main__":
    # accented-Latin fold self-check (France is unseen in train; only guard we get)
    assert norm_text("Société Générale") == "societe generale", norm_text("Société Générale")
    assert norm_text("Café Crème SARL") == "cafe creme sarl", norm_text("Café Crème SARL")
    print("accented-Latin fold: OK (é/è/ç survive as base letters)")
    import sys, io
    import pandas as pd
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    BASE = r"student_resource/dataset"
    s1 = pd.read_csv(f"{BASE}/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{BASE}/train/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{BASE}/train/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(f"{BASE}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=200000)
    m = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    m = m[m.m.str.strip() != ""].head(30)
    n2 = s2.set_index("entity_id").to_dict("index")
    n3 = s3.set_index("entity_id").to_dict("index")
    r1 = s1.set_index("entity_id").to_dict("index")

    def jac(a, b):
        ta, tb = set(a.split()), set(b.split())
        return len(ta & tb) / len(ta | tb) if ta | tb else 1.0

    improved = 0
    for _, row in m.iterrows():
        rec = r1[row.source1_entity_id]
        cand = n2.get(row.m) or n3[row.m]
        n1s, n2s = rec["business_name"], cand["business_name"]
        a1s, a2s = rec["business_address"], cand["business_address"]
        raw_j = jac(n1s.lower(), n2s.lower())
        a = norm_name(n1s)[1]; b = norm_name(n2s)[1]
        n_j = jac(a, b)
        raw_a = jac(a1s.lower(), a2s.lower())
        n_a = jac(" ".join(addr_tokens(a1s)), " ".join(addr_tokens(a2s)))
        improved += (n_j + n_a) > (raw_j + raw_a) - 1e-9
        if raw_j < 0.05 and n_j > 0.5:
            print(f"  rescued: {n1s!r} <=> {n2s!r}\n    -> {a!r} <=> {b!r}  (jac {raw_j:.2f}->{n_j:.2f})")
    print(f"gate: {improved}/30 true pairs improved or equal after normalization")
    # spot examples: script-crossing pairs
    shown = 0
    for _, row in m.iterrows():
        rec = r1[row.source1_entity_id]
        cand = n2.get(row.m) or n3[row.m]
        n1s, n2s = rec["business_name"], cand["business_name"]
        if any(ord(c) > 0x7F for c in n1s + n2s) and shown < 5:
            print(f"  {n1s!r} <=> {n2s!r}")
            print(f"    -> {norm_name(n1s)[1]!r} <=> {norm_name(n2s)[1]!r}")
            shown += 1
