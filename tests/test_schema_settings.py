"""测试领域 schema：理工科科研文献通用实体类型及其注入 LightRAG 的指引。"""

from __future__ import annotations

import json

import pytest

from config.schema_defaults import SCHOLARLY_ENTITY_TYPES, build_entity_types_guidance
from config.settings import DomainSchemaSettings


def test_default_schema_covers_general_stem() -> None:
    names = [name for name, _ in SCHOLARLY_ENTITY_TYPES]
    # 通用科研要素
    for expected in (
        "ResearchProblem",
        "TheoryOrModel",
        "MethodOrAlgorithm",
        "TheoremOrLemma",
        "FormulaOrEquation",
        "CaseOrExample",
        "FigureOrTable",
        "Dataset",
        "Experiment",
        "Metric",
        "SystemOrTool",
        "MaterialOrSubstance",
        "ParameterOrVariable",
        "ApplicationDomain",
        "Publication",
    ):
        assert expected in names
    # 不再是 FEC 专用类型
    for fec_only in ("coding_scheme", "channel_model", "code_instance"):
        assert fec_only not in names
    # 类型标识唯一且为 CamelCase 英文标识
    assert len(names) == len(set(names))
    assert all(name[:1].isupper() and name.replace("_", "").isalnum() for name in names)


def test_default_guidance_format() -> None:
    guidance = build_entity_types_guidance()
    assert guidance.startswith("Classify each entity using one of the following types.")
    assert "use `Other`" in guidance
    assert "- TheoryOrModel: 理论、模型、假设、机制、框架、定律" in guidance
    # 说明为空时只输出类型标识
    assert build_entity_types_guidance([("CustomType", "")]).endswith("- CustomType")


def test_schema_settings_default_and_json_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCHEMA_ENTITY_TYPES_JSON", raising=False)
    monkeypatch.delenv("FEC_ENTITY_TYPES_JSON", raising=False)
    monkeypatch.delenv("SCHEMA_ENTITY_TYPES_GUIDANCE", raising=False)
    default_settings = DomainSchemaSettings()
    assert [n for n, _ in default_settings.resolve_entity_types()] == [
        n for n, _ in SCHOLARLY_ENTITY_TYPES
    ]
    assert default_settings.resolve_entity_types_guidance() == build_entity_types_guidance()

    # JSON 数组：只给类型名
    monkeypatch.setenv("SCHEMA_ENTITY_TYPES_JSON", json.dumps(["TheoryOrModel", "Dataset"]))
    as_list = DomainSchemaSettings()
    assert as_list.resolve_entity_types() == [("TheoryOrModel", ""), ("Dataset", "")]
    assert as_list.resolve_entity_types_guidance().endswith("- Dataset")

    # JSON 对象：类型名 -> 说明
    monkeypatch.setenv(
        "SCHEMA_ENTITY_TYPES_JSON",
        json.dumps({"TheoryOrModel": "理论或模型", "MethodOrAlgorithm": "方法或算法"}),
    )
    as_dict = DomainSchemaSettings()
    assert as_dict.resolve_entity_types() == [
        ("TheoryOrModel", "理论或模型"),
        ("MethodOrAlgorithm", "方法或算法"),
    ]

    # 非法 JSON 回退到默认 schema，而不是抛错
    monkeypatch.setenv("SCHEMA_ENTITY_TYPES_JSON", "{not json")
    assert DomainSchemaSettings().resolve_entity_types() == list(SCHOLARLY_ENTITY_TYPES)


def test_guidance_override_wins_and_legacy_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCHEMA_ENTITY_TYPES_GUIDANCE", "  - OnlyThisType: 自定义  ")
    monkeypatch.setenv("SCHEMA_ENTITY_TYPES_JSON", json.dumps(["Ignored"]))
    assert DomainSchemaSettings().resolve_entity_types_guidance() == "- OnlyThisType: 自定义"

    # 旧 FEC_ 变量名仍然可读（向后兼容）
    monkeypatch.delenv("SCHEMA_ENTITY_TYPES_GUIDANCE", raising=False)
    monkeypatch.delenv("SCHEMA_ENTITY_TYPES_JSON", raising=False)
    monkeypatch.setenv("FEC_ENTITY_TYPES_JSON", json.dumps(["LegacyType"]))
    assert DomainSchemaSettings().resolve_entity_types() == [("LegacyType", "")]


def test_lightrag_resolves_our_guidance() -> None:
    """确认当前安装的 LightRAG 真会把我们的 schema 注入抽取提示词。"""
    lightrag_prompt = pytest.importorskip("lightrag.prompt")

    addon_params = {
        "language": "Chinese",
        "entity_types_guidance": build_entity_types_guidance(),
        "entity_types": [name for name, _ in SCHOLARLY_ENTITY_TYPES],
    }
    profile = lightrag_prompt.resolve_entity_extraction_prompt_profile(addon_params, False)
    assert profile["entity_types_guidance"] == build_entity_types_guidance()
    assert "MethodOrAlgorithm" in profile["entity_types_guidance"]
