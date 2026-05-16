from src.retrieval.keyword_fallback import fec_keyword_fallback


def test_fec_keyword_fallback_polar_rm() -> None:
    hl, ll = fec_keyword_fallback("Polar码与RM码在BEC上性能对比如何？")
    assert hl
    assert ll
    assert any("Polar" in x or "极化" in x or "BEC" in x for x in ll) or "Polar码" in "".join(ll)
