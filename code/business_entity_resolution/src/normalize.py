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


def transliterate(s):
    out = []
    for ch in s:
        cp = ord(ch)
        if cp < 0x80:
            out.append(ch)
        elif cp in (0x200C, 0x200D):  # ZWNJ/ZWJ
            pass
        else:
            t = _TR.get(cp)
            if t is not None:
                out.append(t)
            elif 0x0900 <= cp <= 0x0D7F:  # Indic char with no mapping -> word break
                out.append(" ")
            else:
                out.append(ch)  # accented Latin etc.: keep for the NFKD fold in norm_text
    return "".join(out)


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


# --- Phase 1 GATE self-check ---------------------------------------------------
if __name__ == "__main__":
    # accented-Latin fold self-check (France is unseen in train; only guard we get)
    assert norm_text("Société Générale") == "societe generale", norm_text("Société Générale")
    assert norm_text("Café Crème SARL") == "cafe creme sarl", norm_text("Café Crème SARL")
    print("accented-Latin fold: OK (é/è/ç survive as base letters)")
    import sys, io
    import pandas as pd
    from rapidfuzz.distance import Levenshtein
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
