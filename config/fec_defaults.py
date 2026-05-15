"""FEC（差錯控制編碼）領域預設實體類型，與 rag-fec ``src/extraction.py`` 中 ``FEC_ENTITY_KIND_TYPES`` 對齊。"""

from __future__ import annotations

FEC_DEFAULT_ENTITY_TYPES: tuple[str, ...] = (
    "coding_paradigm",
    "coding_scheme",
    "encoding_methods",
    "decoding_methods",
    "code_instance",
    "modulation_and_demodulation_methods",
    "math_methods_or_math_interpretations",
    "channel_model",
    "channel_phenomena",
    "case_for_illustration",
    "image_asset",
    "table",
)

__all__ = ["FEC_DEFAULT_ENTITY_TYPES"]
