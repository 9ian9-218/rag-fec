"""理工科科研文献（论文 / 专著 / 技术报告）通用实体类型 schema。

设计目标：覆盖数学、物理、化学、生物、材料、计算机、电子、控制、机械等
理工科领域论文与科研书籍中的主要知识要素，不绑定任何单一子领域。

- 类型标识（如 ``MethodOrAlgorithm``）会写入图数据库节点的 ``entity_type``，
  因此保持英文 CamelCase 且不随语言设置翻译。
- 说明文字用于生成 LightRAG 的 ``entity_types_guidance``（注入实体抽取提示词），
  可由 ``SCHEMA_ENTITY_TYPES_JSON`` / ``SCHEMA_ENTITY_TYPES_GUIDANCE`` 覆写。
"""

from __future__ import annotations

from collections.abc import Sequence

GUIDANCE_HEADER: str = (
    "Classify each entity using one of the following types. "
    "If no type fits, use `Other`.\n"
    "Use the type identifier exactly as written (English CamelCase); do not translate it."
)

# (类型标识, 说明)
SCHOLARLY_ENTITY_TYPES: tuple[tuple[str, str], ...] = (
    ("Person", "研究者、作者、发明者、审稿人等个人"),
    ("Organization", "高校、研究所、实验室、企业、基金机构、学会等组织"),
    ("Publication", "论文、书籍、学位论文、标准、专利、技术报告"),
    ("ResearchProblem", "研究问题、研究目标、动机、待解决的挑战或开放问题"),
    ("TheoryOrModel", "理论、模型、假设、机制、框架、定律"),
    ("MethodOrAlgorithm", "方法、算法、技术、流程、协议、实验方案"),
    ("TheoremOrLemma", "定理、引理、推论、命题及其证明"),
    ("DefinitionOrAxiom", "定义、公理、约定、记号、前提假设"),
    ("FormulaOrEquation", "公式、方程、不等式、数学表达式"),
    ("CaseOrExample", "例题、案例、示例、练习与习题"),
    ("FigureOrTable", "图、表、插图、示意图等视觉资料"),
    ("Dataset", "数据集、语料库、样本集合、基准测试数据"),
    ("Experiment", "实验、仿真、测试、评测、案例分析"),
    ("Metric", "评价指标、度量，以及性能/误差/复杂度等量化结果"),
    ("SystemOrTool", "系统、软件、代码库、工具、平台、仪器、硬件设备"),
    ("MaterialOrSubstance", "材料、物质、化合物、元素、生物分子等自然对象"),
    ("ParameterOrVariable", "参数、变量、超参数、常数、阈值、符号"),
    ("ApplicationDomain", "应用场景、任务、领域、使用背景"),
)


def build_entity_types_guidance(
    entity_types: Sequence[tuple[str, str]] | None = None,
) -> str:
    """按 LightRAG 期望的格式生成 ``entity_types_guidance`` 文本。

    ``entity_types`` 为 ``(类型标识, 说明)`` 序列；说明为空时只输出类型标识。
    """
    items = tuple(entity_types) if entity_types is not None else SCHOLARLY_ENTITY_TYPES
    lines = [GUIDANCE_HEADER, ""]
    for name, description in items:
        desc = (description or "").strip()
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    return "\n".join(lines).rstrip()


__all__ = ["SCHOLARLY_ENTITY_TYPES", "GUIDANCE_HEADER", "build_entity_types_guidance"]
