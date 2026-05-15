from __future__ import annotations

from src.retrieval.mode_config import build_query_param, suggest_mode_from_question


def test_build_query_param_mix_top_k() -> None:
    p = build_query_param("mix", top_k=10)
    assert p.mode == "mix"
    assert p.top_k == 10


def test_suggest_mode_global_keyword() -> None:
    assert suggest_mode_from_question("請總結全書要點") == "global"
