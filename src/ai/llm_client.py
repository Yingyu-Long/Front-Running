"""OpenAI-compatible client for local RTX 5070 Ti Ollama / vLLM inference with robust JSON parsing."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (Qwen3 / DeepSeek-R1)."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Robustly extract a JSON object from LLM output (with or without markdown fences)."""
    cleaned = strip_thinking_tags(text)
    if not cleaned:
        return None

    # Direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fenced ```json ... ``` block
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Fallback: outermost { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


class LocalLLMClient:
    """Connects to local RTX 5070 Ti Ollama or vLLM server via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        api_key: str = "ollama",
        model: str = "qwen3:14b",
        timeout_sec: int = 180,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_sec = int(timeout_sec)
        self.temperature = float(temperature)
        self._client = (
            OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_sec)
            if OpenAI is not None
            else None
        )

    def check_health(self) -> dict[str, Any]:
        """Probe Ollama/vLLM server availability and return installed model list."""
        if requests is None:
            return {"online": False, "models": [], "base_url": self.base_url, "error": "缺少 requests 依赖"}
        models: list[str] = []
        # Try OpenAI-compatible /models endpoint first
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [str(m.get("id", "")) for m in (data.get("data") or []) if m.get("id")]
                return {"online": True, "models": models, "base_url": self.base_url}
        except Exception:
            pass

        # Fallback: Ollama native /api/tags
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        try:
            resp = requests.get(f"{root}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [str(m.get("name", "")) for m in (data.get("models") or []) if m.get("name")]
                return {"online": True, "models": models, "base_url": self.base_url}
        except Exception as exc:
            return {"online": False, "models": [], "base_url": self.base_url, "error": str(exc)}

        return {"online": False, "models": [], "base_url": self.base_url, "error": "无法连接本地推理服务"}

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request to local 5070 Ti LLM and return cleaned text."""
        if self._client is not None:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return strip_thinking_tags(content)

        # Direct HTTP fallback if openai SDK is missing
        if requests is None:
            raise RuntimeError("Neither openai nor requests package is installed.")
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return strip_thinking_tags(content)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1800,
    ) -> dict[str, Any]:
        """Request structured JSON output with automatic fallback parsing."""
        raw = self.chat_text(system_prompt, user_prompt, max_tokens=max_tokens)
        parsed = extract_json_object(raw)
        if parsed is not None:
            return parsed

        # One retry asking explicitly for strict JSON repair
        try:
            repair = self.chat_text(
                "你是 JSON 格式化助手。只输出合法的 JSON 对象，不要输出任何额外文字或解释。",
                f"请将以下投研结论整理为严格合法的 JSON 对象：\n\n{raw[:3500]}",
                max_tokens=max_tokens,
            )
            parsed2 = extract_json_object(repair)
            if parsed2 is not None:
                return parsed2
        except Exception as exc:
            logger.debug("JSON repair retry failed: %s", exc)

        return {"parse_error": True, "raw_text": raw}
