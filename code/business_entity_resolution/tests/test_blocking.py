import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from blocking import candidates_chunk, make_allowed, record_keys


def _row(country="US", name_s="acme retail", name_t="acme retail", addr_t="main street", stno="10"):
    return {"country": country, "name_s": name_s, "name_t": name_t, "addr_t": addr_t, "stno": stno}


def test_record_keys_are_country_scoped_and_respect_allowed_families():
    allowed = {"t": {"t|US|acme", "t|US|retail"}, "a": {"a|US|main", "a|US|street"},
               "sn": {"sn|US|10"}, "fn": {"fn|US|acme retail"}}
    keys = record_keys(**_row(), allowed=allowed)
    assert "c3|US|acm" in keys
    assert "t|US|acme" in keys and "t|US|retail" in keys
    assert "a|US|main" in keys and "a|US|street" in keys
    assert "sn|US|10" in keys and "fn|US|acme retail" in keys
    assert all("|India|" not in key and "|France|" not in key for key in keys)
    assert record_keys(**_row(name_s="ab", name_t="ab", addr_t="xy", stno=""), allowed=allowed) == []


def test_make_allowed_applies_each_df_cap_inclusive():
    dfs = {
        "t": {"t|US|one": 1, "t|US|many": 2},
        "a": {"a|US|one": 1, "a|US|many": 2},
        "sn": {"sn|US|one": 1, "sn|US|many": 2},
        "fn": {"fn|US|one": 1, "fn|US|many": 2},
    }
    allowed = make_allowed(dfs, token_cap=1, addr_cap=1, sn_cap=1, fn_cap=1)
    for family in allowed:
        assert f"{family}|US|one" in allowed[family]
        assert f"{family}|US|many" not in allowed[family]


def test_candidates_chunk_returns_empty_for_unkeyed_rows_and_never_crosses_country():
    pool = pd.DataFrame([_row("US", "acme retail", "acme retail", "main street", "10"),
                         _row("India", "acme retail", "acme retail", "main street", "10")])
    index = {
        "c3|US|acm": np.array([0]), "c3|India|acm": np.array([1]),
        "t|US|acme": np.array([0]), "t|India|acme": np.array([1]),
    }
    allowed = {"t": {"t|US|acme"}, "a": set(), "sn": set(), "fn": set()}
    s1 = pd.DataFrame([_row("US"), _row("France", "", "", "", "")])
    result = candidates_chunk(s1, 0, index, allowed)
    assert result[0].tolist() == [0]
    assert result[1].size == 0
    assert all(i != 1 for i in result[0])


def test_blank_name_and_address_is_safe_and_emits_no_keys():
    allowed = {"t": set(), "a": set(), "sn": set(), "fn": set()}
    assert record_keys("US", "", "", "", "", allowed) == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
