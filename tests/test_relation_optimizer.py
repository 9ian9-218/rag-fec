from src.retrieval.relation_keywords import enhance_keywords_for_retrieval, extract_user_query_from_prompt
from src.retrieval.relation_optimizer import filter_relationships


def test_extract_user_query_from_prompt() -> None:
    q = extract_user_query_from_prompt("prefix\nUser Query: Polar与RM性能对比\n---\nOutput:")
    assert "Polar" in q


def test_enhance_keywords_filters_junk() -> None:
    hl, ll = enhance_keywords_for_retrieval(
        "Polar码与RM码在BEC上性能对比如何？",
        ["Role", "You", "性能"],
        ["Polar", "RM"],
    )
    assert "Role" not in hl
    assert any("性能" in x or "Polar" in x for x in hl + ll)


def test_filter_relationships_dedupe() -> None:
    rels = [
        {"src_id": "A", "tgt_id": "B", "description": "Polar outperforms RM on BEC"},
        {"src_id": "B", "tgt_id": "A", "description": "duplicate edge"},
        {"src_id": "X", "tgt_id": "Y", "description": "unrelated cooking recipe"},
    ]

    async def _run():
        return await filter_relationships(
            "Polar码与RM码在BEC上性能对比",
            rels,
            hl_keywords=["信道编码性能", "性能对比"],
            ll_keywords=["Polar", "RM", "BEC"],
        )

    import asyncio

    out = asyncio.run(_run())
    assert len(out) <= 2
    keys = {f"{r['src_id']}|{r['tgt_id']}" for r in out}
    assert len(keys) == len(out)
