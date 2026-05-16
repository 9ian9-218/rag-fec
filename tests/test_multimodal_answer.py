from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import BadRequestError

from config.settings import LLMSettings, ModelsSettings, MultimodalSettings, PathsSettings, Settings


def _minimal_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for k in (
        "OPENAI_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "LLM_MODEL_NAME",
        "LLM_API_KEY",
        "LLM_BASE_URL",
    ):
        monkeypatch.delenv(k, raising=False)
    return Settings(
        paths=PathsSettings(project_root=str(tmp_path), data_raw="data/raw"),
        models=ModelsSettings(dir="models", offline=True),
        llm=LLMSettings(api_key="sk-test", base_url="http://127.0.0.1:9/v1", model_name="text-model"),
        multimodal=MultimodalSettings(vision_model="vision-model"),
        openai_api_key="sk-test",
        openai_base_url="http://127.0.0.1:9/v1",
    )


@pytest.mark.asyncio
async def test_no_local_images_warns_and_text_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _minimal_settings(tmp_path, monkeypatch)
    bundle = {
        "status": "success",
        "data": {
            "chunks": [
                {
                    "content": "僅文字 ![](images/missing.png)",
                    "file_path": str(tmp_path / "data/raw/x.md"),
                }
            ],
            "entities": [],
            "relationships": [],
        },
    }
    calls: list[str] = []

    async def fake_text(*args: object, **kwargs: object) -> str:
        calls.append("text")
        st = kwargs.get("settings")
        m = kwargs.get("model")
        assert m == st.resolved_llm_model_name()  # type: ignore[union-attr]
        return "text-ok"

    with patch(
        "src.retrieval.multimodal_answer._chat_completion_text",
        new=AsyncMock(side_effect=fake_text),
    ):
        from src.retrieval.multimodal_answer import answer_with_retrieved_images

        out = await answer_with_retrieved_images(
            settings=s, question="Q?", bundle=bundle, history_messages=None
        )
    assert out == "text-ok"
    assert calls == ["text"]


@pytest.mark.asyncio
async def test_vision_bad_request_falls_back_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _minimal_settings(tmp_path, monkeypatch)
    img = tmp_path / "data/raw/images" / "a.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal jpeg-like

    bundle = {
        "status": "success",
        "data": {
            "chunks": [
                {
                    "content": "見圖 ![](images/a.jpg)",
                    "file_path": str(tmp_path / "data/raw/doc.md"),
                }
            ],
            "entities": [],
            "relationships": [],
        },
    }

    req = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    br = BadRequestError("no vision", response=resp, body=None)

    fake_create = AsyncMock(side_effect=br)
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = fake_create

    text_calls: list[str] = []

    async def fake_text(*args: object, **kwargs: object) -> str:
        text_calls.append("text")
        return "fallback"

    with (
        patch("src.retrieval.multimodal_answer._openai_client_and_base", return_value=(fake_client, "http://x")),
        patch(
            "src.retrieval.multimodal_answer._chat_completion_text",
            new=AsyncMock(side_effect=fake_text),
        ),
    ):
        from src.retrieval.multimodal_answer import answer_with_retrieved_images

        out = await answer_with_retrieved_images(
            settings=s, question="Q?", bundle=bundle, history_messages=None
        )
    assert out == "fallback"
    assert fake_create.await_count == 1
    assert text_calls == ["text"]
