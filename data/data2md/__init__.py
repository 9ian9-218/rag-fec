"""PDF → Markdown（MinerU CLI）。

避免在 ``import data.data2md`` 時預先載入 ``convert``，以消除
``python -m data.data2md.convert`` 時的 runpy RuntimeWarning。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "convert_pdf",
    "convert_pdf_batch",
    "mineru_executable",
]


def __getattr__(name: str):
    if name in __all__:
        from . import convert as _convert

        return getattr(_convert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .convert import convert_pdf, convert_pdf_batch, mineru_executable
