"""拒答/误答评分：负样本集上「该拒答有没有拒答」的度量。

被 ``scripts/score_refusal.py``（单文件评分）与 ``runner.build_report``（整份评测报告）
共用，保证两处口径一致。

口径：
  - 负样本误答率 = (out_of_corpus + false_premise 中未拒答的题数) / 该两类题数
  - 正样本误拒率 = (in_corpus 中拒答的题数) / in_corpus 题数
  - partial（边界题）只报告拒答率，不计入上面两项
  - 某一类样本数为 0 时，对应比率返回 None（而非 0.0/1.0 的伪值）

限制：基于短语匹配，只能识别**显式拒答**。若答案一边声称材料未提及、一边又给出编造
内容，会被判为「已拒答」——这类行（拒答短语 + 长答案）会被标为 suspect 供人工复核。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# 显式拒答短语（覆盖系统内置措辞与常见变体）
REFUSAL_PATTERNS: tuple[str, ...] = (
    "無法找到足夠的信息",
    "无法找到足够的信息",
    "無法找到足夠",
    "无法找到足够",
    "沒有足夠信息",
    "没有足够信息",
    "信息不足",
    "資料不足",
    "资料不足",
    "材料不足",
    "檢索材料不足",
    "检索材料不足",
    "提供的材料不足",
    "不足以回答",
    "無法回答該問題",
    "无法回答该问题",
    "無法回答這個問題",
    "无法回答这个问题",
    "無法從提供的",
    "无法从提供的",
    "材料中未提及",
    "材料中沒有提及",
    "材料中没有提及",
    "材料未提供",
    "未提供相關",
    "未提供相关",
    "沒有相關信息",
    "没有相关信息",
    "未涵蓋",
    "未涵盖",
    "無法依據",
    "无法依据",
    "insufficient information",
    "not enough information",
    "cannot answer",
)

# 拒答短语出现但答案仍很长 → 可能「先声称不足再编造」，需人工复核
SUSPECT_CHARS = 400

NEGATIVE_LABELS: tuple[str, ...] = ("out_of_corpus", "false_premise")
POSITIVE_LABELS: tuple[str, ...] = ("in_corpus",)


def looks_refused(text: str) -> bool:
    """答案是否属于显式拒答（空答案视为未能回答）。"""
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    return any(p.lower() in low for p in REFUSAL_PATTERNS)


def score_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """按 label 汇总拒答率/误答率，返回 (summary, details)。"""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    for r in rows:
        pred = str(r.get("prediction") or "")
        refused = looks_refused(pred)
        label = str(r.get("label") or "unlabeled")
        rec = {
            "id": r.get("id"),
            "label": label,
            "expect": r.get("expect"),
            "refused": refused,
            "chars": len(pred),
            "suspect": refused and len(pred) > SUSPECT_CHARS,
            "excerpt": pred[:120].replace("\n", " "),
        }
        details.append(rec)
        by_label[label].append(rec)

    def rate(items: list[dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return sum(1 for i in items if i[key]) / len(items)

    negative = [d for lb in NEGATIVE_LABELS for d in by_label.get(lb, [])]
    positive = [d for lb in POSITIVE_LABELS for d in by_label.get(lb, [])]

    summary: dict[str, Any] = {
        "counts": {lb: len(items) for lb, items in sorted(by_label.items())},
        "refusal_rate_by_label": {
            lb: round(rate(items, "refused"), 4) for lb, items in sorted(by_label.items())
        },
        # 没有对应样本时返回 None，而不是 0/1 这种会被误读的伪值
        # （历史缺陷：只有 in_corpus 的评估集会报 negative_misanswer_rate=1.0）
        "negative_misanswer_rate": (
            round(1.0 - rate(negative, "refused"), 4) if negative else None
        ),
        "positive_false_refusal_rate": (
            round(rate(positive, "refused"), 4) if positive else None
        ),
        "suspect_rows": [d["id"] for d in details if d["suspect"]],
    }
    return summary, details


def has_refusal_labels(rows: list[dict[str, Any]]) -> bool:
    """行里是否带 label（只有负样本集才有），用于决定报告是否输出 refusal 区块。"""
    return any(str(r.get("label") or "").strip() for r in rows)
