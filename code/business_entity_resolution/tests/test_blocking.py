import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import blocking
from blocking import (
    _stable_topk,
    batch_topk_prune,
    candidates_chunk,
    make_allowed,
    proxy_topk_indices,
    record_keys,
)


def _row(country="US", name_s="acme retail", name_t="acme retail", addr_t="main street", stno="10"):
    return {"country": country, "name_s": name_s, "name_t": name_t, "addr_t": addr_t, "stno": stno}


def test_record_keys_are_country_scoped_and_respect_allowed_families():
    allowed = {"c3": {"c3|US|acm"}, "t": {"t|US|acme", "t|US|retail"}, "a": {"a|US|main", "a|US|street"},
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
        "c3": {"c3|US|one": 1, "c3|US|many": 2},
        "t": {"t|US|one": 1, "t|US|many": 2},
        "a": {"a|US|one": 1, "a|US|many": 2},
        "sn": {"sn|US|one": 1, "sn|US|many": 2},
        "fn": {"fn|US|one": 1, "fn|US|many": 2},
    }
    allowed = make_allowed(dfs, token_cap=1, addr_cap=1, sn_cap=1, fn_cap=1, c3_cap=1)
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
    allowed = {"c3": set(), "t": {"t|US|acme"}, "a": set(), "sn": set(), "fn": set()}
    s1 = pd.DataFrame([_row("US"), _row("France", "", "", "", "")])
    result = candidates_chunk(s1, 0, index, allowed)
    assert result[0].tolist() == [0]
    assert result[1].size == 0
    assert all(i != 1 for i in result[0])


def test_blank_name_and_address_is_safe_and_emits_no_keys():
    allowed = {"c3": set(), "t": set(), "a": set(), "sn": set(), "fn": set()}
    assert record_keys("US", "", "", "", "", allowed) == []


def test_candidate_prefilter_is_deterministic_and_prefers_stronger_block_evidence():
    pool = pd.DataFrame([_row(name_s="acme", name_t="acme"),
                         _row(name_s="acme", name_t="acme")])
    index = {"c3|US|acm": np.array([0]), "fn|US|acme": np.array([1])}
    allowed = {"c3": {"c3|US|acm"}, "t": set(), "a": set(), "sn": set(),
               "fn": {"fn|US|acme"}}
    s1 = pd.DataFrame([_row(name_s="acme", name_t="acme")])

    first = candidates_chunk(s1, 0, index, allowed, max_candidates=1)[0]
    second = candidates_chunk(s1, 0, index, allowed, max_candidates=1)[0]

    assert first.tolist() == [1]
    assert second.tolist() == first.tolist()


def test_empty_ordinary_and_highly_skewed_rows_respect_candidate_cap():
    posting = np.arange(5000, dtype=np.int64)
    index = {"t|US|acme": posting}
    allowed = {"c3": set(), "t": {"t|US|acme"}, "a": set(), "sn": set(), "fn": set()}
    rows = pd.DataFrame([_row("US", "", "", "", ""), _row(),
                         _row("US", "acme", "acme", "", "")])
    # The empty row emits nothing, while both ordinary/skewed rows are stable and bounded.
    first = candidates_chunk(rows, 0, index, allowed, max_candidates=37)
    second = candidates_chunk(rows, 0, index, allowed, max_candidates=37)
    assert first[0].size == 0
    assert len(first[1]) <= 37 and len(first[2]) == 37
    assert first[1].tolist() == second[1].tolist()
    assert first[2].tolist() == second[2].tolist()


def test_fast_topk_has_deterministic_id_ties_at_the_cutoff():
    ids = np.array([19, 4, 12, 2, 33], dtype=np.int64)
    scores = np.array([0.8, 0.9, 0.9, 0.9, 0.1])
    assert _stable_topk(ids, scores, 2).tolist() == [2, 4]
    assert _stable_topk(ids, scores, 20).tolist() == [2, 4, 12, 19, 33]


def test_batched_topk_matches_scalar_for_oversized_row_and_respects_pair_budget(monkeypatch):
    pool = pd.DataFrame({
        "name_s": ["acme", "acme inc", "acm", "other", "acme ltd"],
        "addr_n": ["10 main street"] * 5,
    })
    rows = pd.DataFrame({"name_s": ["acme"], "addr_n": ["10 main street"]})
    cands = [np.array([4, 3, 2, 1, 0], dtype=np.int64)]
    expected = proxy_topk_indices(rows.iloc[0], pool, cands[0], 2, "mean")
    seen_sizes = []
    original = blocking.cos_pair_scores

    def tracked_cos_pair_scores(frame, pool_, si, ci):
        seen_sizes.append(len(ci))
        return original(frame, pool_, si, ci)

    monkeypatch.setattr(blocking, "cos_pair_scores", tracked_cos_pair_scores)
    actual = batch_topk_prune(rows, pool, cands, 2, "mean", pair_budget=2)

    assert actual[0].tolist() == expected.tolist()
    assert seen_sizes and max(seen_sizes) <= 2


def test_generate_pairs_flushes_by_pair_budget_not_query_row_count():
    from blocking import generate_pairs

    s1 = pd.DataFrame([_row(), _row(), _row()])
    index = {"c3|US|acm": np.array([0, 1]), "t|US|acme": np.array([0, 1]),
             "t|US|retail": np.array([0, 1])}
    allowed = {"c3": {"c3|US|acm"}, "t": {"t|US|acme", "t|US|retail"},
               "a": set(), "sn": set(), "fn": set()}

    batches = list(generate_pairs(s1, index, allowed, pair_budget=3,
                                  max_candidates=10, chunk_rows=100))

    assert all(len(candidate_ids) <= 3 for _, candidate_ids, _ in batches)
    assert sum(len(source_ids) for source_ids, _, _ in batches) == 6


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
