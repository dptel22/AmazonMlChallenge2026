import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import normalize
from normalize import norm_name, norm_text, street_number, strip_suffixes, transliterate


def test_norm_text_folds_accents_and_collapses_noise():
    assert norm_text("Société Générale") == "societe generale"
    assert norm_text("Café Crème SARL") == "cafe creme sarl"
    assert norm_text("  ACME---  NORTH\t\tWEST!! ") == "acme north west"
    assert norm_text("") == ""


def test_all_nine_indic_scripts_transliterate_to_ascii():
    samples = [
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "ગુજરાતી વેપાર પ્રાઇવેટ લિમિટેડ",
        "தமிழ் வணிகம் லிமிடெட்",
        "తెలుగు వ్యాపారం లిమిటెడ్",
        "ಕನ್ನಡ ವ್ಯಾಪಾರ ಲಿಮಿಟೆಡ್",
        "বাংলা ব্যবসা লিমিটেড",
        "ਪੰਜਾਬੀ ਵਪਾਰ ਲਿਮਿਟੇਡ",
        "ଓଡ଼ିଆ ବ୍ୟବସାୟ ଲିମିଟେଡ",
        "മലയാളം ബിസിനസ് ലിമിറ്റഡ്",
    ]
    for sample in samples:
        normalized, suffix_stripped = norm_name(sample)
        assert normalized and suffix_stripped
        assert normalized.isascii() and suffix_stripped.isascii()
    assert "raam maarketing" in norm_name(samples[0])[0]
    assert "raaj" in norm_name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி")[0]
    assert "investments" in norm_name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி")[0]


def test_strip_suffixes_handles_both_ends_and_wrappers():
    assert strip_suffixes("b retail inc") == "b retail"
    assert strip_suffixes("corp x ltd") == "x"
    assert strip_suffixes("[llp]") == ""
    assert strip_suffixes("(inc)") == ""
    assert strip_suffixes("llp inc") == ""
    # Legal-token stripping is deliberately applied at both ends, so leading
    # Co is removed too; this records the current documented behavior.
    assert strip_suffixes("co hosting ltd") == "hosting"


def test_street_number_only_accepts_leading_numeric_token():
    assert street_number("1795 Westchester Drive, High Point, NC") == "1795"
    assert street_number("No 10 Enkay Square") == ""
    assert street_number("West 10 Enkay Square") == ""
    assert street_number("") == ""


def test_zwnj_is_removed_without_inserting_a_space():
    assert transliterate("ab\u200ccd") == "abcd"
    assert norm_text("ab\u200ccd") == "abcd"


def test_translate_fast_path_preserves_previous_character_mapping():
    sample = "ASCII acme café" + "".join(chr(cp) for cp in range(0x0900, 0x0D80)) + "x\u200cy\u200d"
    expected = []
    for ch in sample:
        cp = ord(ch)
        if cp < 0x80:
            expected.append(ch)
        elif cp in (0x200C, 0x200D):
            pass
        elif normalize._TR.get(cp) is not None:
            expected.append(normalize._TR[cp])
        elif 0x0900 <= cp <= 0x0D7F:
            expected.append(" ")
        else:
            expected.append(ch)
    assert transliterate(sample) == "".join(expected)
    assert transliterate("ASCII only") == "ASCII only"


def test_frame_cos_rows_rebuilds_attrs_cache_inherited_by_iloc():
    import numpy as np
    import pandas as pd

    pool = pd.DataFrame({
        "name_s": ["acme market", "other place"],
        "addr_n": ["1 main road", "2 oak street"],
    })
    parent = pd.DataFrame({
        "name_s": ["acme market", "other place"],
        "addr_n": ["1 main road", "2 oak street"],
    })
    normalize.attach_cosine(pool)
    normalize.frame_cos_rows(parent, pool)

    chunk = parent.iloc[1:2]
    assert chunk.attrs["_cos"]["owner"] == id(parent)
    normalize.frame_cos_rows(chunk, pool)

    assert chunk.attrs["_cos"]["owner"] == id(chunk)
    assert chunk.attrs["_cos"]["Xn"].shape[0] == 1
    name_cos, _ = normalize.cos_pair_scores(
        chunk, pool, np.array([0]), np.array([1]))
    assert name_cos[0] > 0.999


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
