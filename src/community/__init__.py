"""社区摘要（可选能力）：Louvain 社区划分 + 轻量摘要存储 + 查询期按需注入。

设计约束：
- 能力默认关闭（``COMMUNITY_ENABLED=false``），关闭时不构建索引、不调 LLM、不改检索行为；
- 社区数量按目标数自适应并有硬上限，避免维护负担随语料规模失控；
- 摘要仅作背景概述注入上下文，不进入引用列表。
"""

from src.community.builder import build_communities, refresh_communities
from src.community.injector import CommunityInjection, select_community_context
from src.community.partition import Community, PartitionResult, detect_communities
from src.community.store import (
    community_version,
    load_index,
    load_reports,
    reports_path,
)

__all__ = [
    "Community",
    "CommunityInjection",
    "PartitionResult",
    "build_communities",
    "community_version",
    "detect_communities",
    "load_index",
    "load_reports",
    "refresh_communities",
    "reports_path",
    "select_community_context",
]
