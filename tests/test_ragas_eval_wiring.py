"""RAGAS 评测链路回归：依赖可用性、判官会话头、"裁判失败"与"真 0 分"的区别。

背景：langchain-community 0.4.2 删除了 `chat_models.vertexai`，而 ragas 0.4.3 顶层
导入它 → `import ragas` 直接失败，评测静默返回 0 分；修好导入后，判官 client 又因缺少
`x-opencode-session` 头被网关切 400，指标同样全是 0。这些测试锁住这两点与失败可见性。
"""

from __future__ import annotations


def test_ragas_stack_importable() -> None:
    """锁住依赖链：ragas 与它需要的 langchain 模块必须能一起导入。"""
    import ragas  # noqa: F401
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
    from ragas.llms import llm_factory  # noqa: F401
    from ragas.metrics import context_precision, context_recall, faithfulness  # noqa: F401


def test_build_ragas_llm_injects_session_header(monkeypatch) -> None:
    """判官 client 必须带 x-opencode-session，否则网关切 400、所有指标变 0。"""
    import openai
    import ragas.llms

    import src.evaluation.ragas_metrics as rm

    captured: dict = {}
    factory_kwargs: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_llm_factory(model, client=None, **kwargs):
        factory_kwargs.update({"model": model, "client": client, **kwargs})
        return factory_kwargs

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(ragas.llms, "llm_factory", fake_llm_factory)

    rm.build_ragas_llm()

    headers = captured.get("default_headers") or {}
    assert headers.get("x-opencode-session"), f"未注入会话头: {headers}"
    # 判官应低温以保证评测可复现
    assert factory_kwargs.get("temperature") == 0
    assert int(factory_kwargs.get("max_tokens") or 0) >= 512


def test_compute_ragas_batch_flags_failed_rows(monkeypatch, caplog) -> None:
    """判官失败的行标 ragas_ok=False，并留下告警，不能伪装成真实的 0 分。"""
    import pandas as pd
    import ragas

    import src.evaluation.ragas_metrics as rm

    class FakeResult:
        def to_pandas(self):
            return pd.DataFrame(
                [
                    {"context_recall": 0.8, "context_precision": 0.9, "faithfulness": 0.7},
                    {
                        "context_recall": float("nan"),
                        "context_precision": float("nan"),
                        "faithfulness": float("nan"),
                    },
                ]
            )

    monkeypatch.setattr(ragas, "evaluate", lambda *a, **k: FakeResult())

    with caplog.at_level("WARNING"):
        scores = rm.compute_ragas_batch(
            [{"user_input": "q1"}, {"user_input": "q2"}], llm=object()
        )

    assert scores[0]["ragas_ok"] is True
    assert scores[0]["faithfulness"] == 0.7
    assert scores[1]["ragas_ok"] is False
    assert any("RAGAS 裁判" in r.getMessage() for r in caplog.records)


def test_report_excludes_failed_ragas_rows_from_means(monkeypatch) -> None:
    """报告均值只统计该指标真正算出结果的行，并给出逐指标计数。

    走真实 compute_ragas_batch（只桩掉 ragas.evaluate），确保标记逻辑不被测试假设替代。
    """
    import pandas as pd
    import ragas

    from src.evaluation import runner

    class FakeResult:
        def to_pandas(self):
            return pd.DataFrame(
                [
                    {"context_recall": 1.0, "context_precision": 1.0, "faithfulness": 0.6},
                    # 只有 faithfulness 失败：不能把它的缺失当成 0 分拉低均值
                    {
                        "context_recall": 1.0,
                        "context_precision": 1.0,
                        "faithfulness": float("nan"),
                    },
                ]
            )

    monkeypatch.setattr(ragas, "evaluate", lambda *a, **k: FakeResult())

    rows = [
        {"id": "a", "question": "q1", "reference": "r1", "prediction": "p1", "retrieved_context": "c1"},
        {"id": "b", "question": "q2", "reference": "r2", "prediction": "p2", "retrieved_context": "c2"},
    ]
    report = runner.build_report(rows, ragas_llm=True, ragas_llm_override=object(), max_detail_rows=10)
    r = report["ragas"]

    assert r["scored_rows"] == 2
    assert r["scored_by_metric"]["faithfulness"] == 1
    assert r["scored_by_metric"]["context_recall"] == 2
    assert r["faithfulness_mean"] == 0.6  # 失败行的缺失值未参与均值
    assert r["context_recall_mean"] == 1.0
