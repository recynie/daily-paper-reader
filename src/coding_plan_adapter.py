#!/usr/bin/env python
"""统一 coding plan adapter。

支持两类接入方式：
1. 作为纯后端模块 import 使用；
2. 作为 CLI 单独调用，直接向不同 provider 请求结构化 plan。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import requests


DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_SYSTEM_PROMPT = (
    "You are a retrieval planning assistant and can only return valid JSON. "
    "The response must be fully based on the current user input and must not reference prior conversation history."
)


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_provider_name(value: Any) -> str:
    key = _norm_text(value).lower()
    if not key:
        return ""
    if key in {"openai", "gpt", "chatgpt"}:
        return "openai"
    if key in {"deepseek"}:
        return "deepseek"
    if key in {"anthropic", "claude"}:
        return "anthropic"
    if key in {"gemini", "google", "google-ai", "googleai"}:
        return "gemini"
    if key in {"blt", "bltcy", "plato", "gptbest"}:
        return "blt"
    if key in {"siliconflow", "silicon-flow", "sflow"}:
        return "siliconflow"
    if key in {"cstcloud", "cst", "cst-cloud", "keji", "keji-yun"}:
        return "cstcloud"
    if key in {"openrouter"}:
        return "openrouter"
    if key in {"ollama", "local"}:
        return "ollama"
    if key in {"openai-compatible", "openai_compatible", "generic", "compat"}:
        return "openai-compatible"
    return key


def provider_family(provider: str) -> str:
    normalized = normalize_provider_name(provider)
    if normalized == "anthropic":
        return "anthropic"
    if normalized == "gemini":
        return "gemini"
    return "openai-compatible"


def infer_provider_name(config: Dict[str, Any] | None) -> str:
    data = config or {}
    explicit = normalize_provider_name(data.get("provider"))
    if explicit:
        return explicit

    base_url = _norm_text(data.get("base_url") or data.get("baseUrl")).lower()
    model = _norm_text(data.get("model")).lower()

    if "anthropic" in base_url or model.startswith("claude"):
        return "anthropic"
    if (
        "generativelanguage" in base_url
        or "googleapis.com" in base_url
        or model.startswith("gemini")
    ):
        return "gemini"
    if "openrouter" in base_url:
        return "openrouter"
    if "deepseek" in base_url or model.startswith("deepseek"):
        return "deepseek"
    if (
        "api.openai.com" in base_url
        or model.startswith("gpt-")
        or model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
    ):
        return "openai"
    if "bltcy" in base_url or "gptbest" in base_url:
        return "blt"
    if "siliconflow" in base_url:
        return "siliconflow"
    if "cstcloud" in base_url:
        return "cstcloud"
    if "localhost" in base_url or "ollama" in base_url:
        return "ollama"
    return "openai-compatible"


def resolve_default_base_url(provider: str) -> str:
    normalized = normalize_provider_name(provider)
    if normalized == "openai":
        return "https://api.openai.com/v1"
    if normalized == "deepseek":
        return "https://api.deepseek.com/v1"
    if normalized == "anthropic":
        return "https://api.anthropic.com/v1"
    if normalized == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta"
    return ""


def resolve_api_key(provider: str, explicit_api_key: str) -> str:
    if explicit_api_key:
        return explicit_api_key
    env_candidates = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "blt": "BLT_API_KEY",
        "siliconflow": "SILICONFLOW_API_KEY",
        "cstcloud": "CSTCLOUD_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_name = env_candidates.get(normalize_provider_name(provider), "")
    return _norm_text(os.getenv(env_name)) if env_name else ""


@dataclass
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: str = ""
    label: str = ""
    enabled: bool = True

    @property
    def family(self) -> str:
        return provider_family(self.provider)


@dataclass
class RequestDescriptor:
    provider: str
    model: str
    family: str
    url: str
    headers: Dict[str, str]
    payload: Dict[str, Any]


def normalize_provider_config(config: Dict[str, Any] | ProviderConfig) -> ProviderConfig:
    if isinstance(config, ProviderConfig):
        provider = infer_provider_name({"provider": config.provider, "base_url": config.base_url, "model": config.model})
        api_key = resolve_api_key(provider, _norm_text(config.api_key))
        base_url = _norm_text(config.base_url or resolve_default_base_url(provider)).rstrip("/")
        return ProviderConfig(
            provider=provider,
            model=_norm_text(config.model),
            api_key=api_key,
            base_url=base_url,
            label=_norm_text(config.label) or _norm_text(config.model) or provider,
            enabled=bool(config.enabled),
        )

    data = config or {}
    provider = infer_provider_name(data)
    base_url = _norm_text(data.get("base_url") or data.get("baseUrl") or resolve_default_base_url(provider)).rstrip("/")
    model = _norm_text(data.get("model"))
    api_key = resolve_api_key(provider, _norm_text(data.get("api_key") or data.get("apiKey")))
    label = _norm_text(data.get("label")) or model or provider
    enabled = data.get("enabled", True) is not False
    return ProviderConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        label=label,
        enabled=enabled,
    )


def build_openai_compatible_endpoints(base_url: str) -> List[str]:
    src = _norm_text(base_url).rstrip("/")
    if not src:
        return []
    out: List[str] = []

    def push_unique(value: str) -> None:
        text = _norm_text(value)
        if text and text not in out:
            out.append(text)

    if "/chat/completions" in src:
        push_unique(src)
        push_unique(re.sub(r"/chat/completions$", "/v1/chat/completions", src))
        return out
    if re.search(r"/v\d+(?:beta)?$", src):
        push_unique(f"{src}/chat/completions")
        return out
    push_unique(f"{src}/v1/chat/completions")
    push_unique(f"{src}/chat/completions")
    return out


def build_anthropic_endpoint(base_url: str) -> str:
    src = _norm_text(base_url).rstrip("/")
    if not src:
        return "https://api.anthropic.com/v1/messages"
    if src.endswith("/messages"):
        return src
    if re.search(r"/v\d+$", src):
        return f"{src}/messages"
    return f"{src}/v1/messages"


def build_gemini_endpoint(base_url: str, model: str, api_key: str) -> str:
    src = _norm_text(base_url).rstrip("/") or "https://generativelanguage.googleapis.com/v1beta"
    if ":generateContent" in src:
        return src if "key=" in src else f"{src}{'&' if '?' in src else '?'}key={api_key}"
    if "/models/" in src:
        return f"{src}/{model}:generateContent?key={api_key}"
    return f"{src}/models/{model}:generateContent?key={api_key}"


def build_request_descriptors(
    provider_config: Dict[str, Any] | ProviderConfig,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.1,
) -> List[RequestDescriptor]:
    config = normalize_provider_config(provider_config)
    if not config.api_key:
        raise ValueError("coding plan provider 缺少 api_key。")
    if not config.model:
        raise ValueError("coding plan provider 缺少 model。")
    if not _norm_text(prompt):
        raise ValueError("coding plan prompt 不能为空。")

    if config.family == "anthropic":
        return [
            RequestDescriptor(
                provider=config.provider,
                model=config.model,
                family=config.family,
                url=build_anthropic_endpoint(config.base_url),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": config.api_key,
                    "anthropic-version": "2023-06-01",
                },
                payload={
                    "model": config.model,
                    "max_tokens": 4096,
                    "temperature": temperature,
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                },
            )
        ]

    if config.family == "gemini":
        return [
            RequestDescriptor(
                provider=config.provider,
                model=config.model,
                family=config.family,
                url=build_gemini_endpoint(config.base_url, config.model, config.api_key),
                headers={
                    "Content-Type": "application/json",
                },
                payload={
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}],
                    },
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                        "responseMimeType": "application/json",
                    },
                },
            )
        ]

    endpoints = build_openai_compatible_endpoints(config.base_url or resolve_default_base_url(config.provider))
    if not endpoints:
        raise ValueError("coding plan provider 缺少 base_url。")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {config.api_key}",
        "x-api-key": config.api_key,
    }
    base_payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
    }
    descriptors: List[RequestDescriptor] = []
    for url in endpoints:
        descriptors.append(
            RequestDescriptor(
                provider=config.provider,
                model=config.model,
                family=config.family,
                url=url,
                headers=headers,
                payload={
                    **base_payload,
                    "response_format": {"type": "json_object"},
                },
            )
        )
        descriptors.append(
            RequestDescriptor(
                provider=config.provider,
                model=config.model,
                family=config.family,
                url=url,
                headers=headers,
                payload=dict(base_payload),
            )
        )
    return descriptors


def _normalize_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(_norm_text(item.get("text") or item.get("content") or item.get("output_text")))
        return "\n".join([x for x in parts if x])
    if isinstance(content, dict):
        return _norm_text(content.get("text") or content.get("content") or content.get("output_text"))
    return ""


def extract_content_from_response(provider: str, data: Dict[str, Any]) -> str:
    family = provider_family(provider)
    if family == "anthropic":
        content = data.get("content") or []
        if not isinstance(content, list):
            return ""
        return "\n".join([_norm_text(item.get("text")) for item in content if isinstance(item, dict) and _norm_text(item.get("text"))])

    if family == "gemini":
        candidates = data.get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            return ""
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content") if isinstance(first, dict) else {}
        parts = content.get("parts") if isinstance(content, dict) else []
        if not isinstance(parts, list):
            return ""
        return "\n".join([_norm_text(part.get("text")) for part in parts if isinstance(part, dict) and _norm_text(part.get("text"))])

    choices = data.get("choices") or []
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict):
            text = _normalize_openai_content(message.get("content"))
            if text:
                return text
    return _norm_text(data.get("output_text"))


def _load_json_lenient(text: str) -> Any:
    raw = _norm_text(text)
    if not raw:
        return {}
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    if fence_match:
        raw = _norm_text(fence_match.group(1))
    try:
        return json.loads(raw)
    except Exception:
        object_match = re.search(r"\{[\s\S]*\}", raw)
        if object_match:
            return json.loads(object_match.group(0))
        raise


def generate_coding_plan(
    prompt: str,
    provider_config: Dict[str, Any] | ProviderConfig,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = 0.1,
    request_func: Callable[..., requests.Response] = requests.post,
) -> Dict[str, Any]:
    descriptors = build_request_descriptors(
        provider_config,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
    )
    last_error: Exception | None = None
    normalized = normalize_provider_config(provider_config)

    for descriptor in descriptors:
        try:
            response = request_func(
                descriptor.url,
                headers=descriptor.headers,
                json=descriptor.payload,
                timeout=timeout_seconds,
            )
            text = response.text or ""
            if response.status_code >= 400:
                last_error = RuntimeError(f"HTTP {response.status_code} {text or response.reason}")
                continue
            data = response.json() if text else {}
            content = extract_content_from_response(normalized.provider, data if isinstance(data, dict) else {})
            if not content:
                last_error = RuntimeError("coding plan 响应缺少可解析内容。")
                continue
            parsed = _load_json_lenient(content)
            return {
                "provider": normalized.provider,
                "model": normalized.model,
                "family": normalized.family,
                "content": content,
                "parsed": parsed,
                "raw": data,
            }
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("coding plan 请求失败。")


class CodingPlanAdapter:
    """纯后端 adapter 对象封装。"""

    def __init__(self, provider_config: Dict[str, Any] | ProviderConfig):
        self.config = normalize_provider_config(provider_config)

    @classmethod
    def from_env(cls) -> "CodingPlanAdapter":
        provider = _norm_text(os.getenv("CODING_PLAN_PROVIDER"))
        model = _norm_text(os.getenv("CODING_PLAN_MODEL"))
        base_url = _norm_text(os.getenv("CODING_PLAN_BASE_URL"))
        api_key = _norm_text(os.getenv("CODING_PLAN_API_KEY"))
        return cls(
            {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
            }
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = 0.1,
        request_func: Callable[..., requests.Response] = requests.post,
    ) -> Dict[str, Any]:
        return generate_coding_plan(
            prompt,
            self.config,
            system_prompt=system_prompt,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            request_func=request_func,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="调用统一 coding plan adapter。")
    parser.add_argument("--prompt", type=str, default="", help="用户 prompt。为空时从 stdin 读取。")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="system prompt。")
    parser.add_argument("--provider", type=str, default="", help="provider 名称。")
    parser.add_argument("--model", type=str, default="", help="模型名称。")
    parser.add_argument("--base-url", type=str, default="", help="provider base url。")
    parser.add_argument("--api-key", type=str, default="", help="API Key。")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="请求超时秒数。")
    args = parser.parse_args()

    prompt = _norm_text(args.prompt)
    if not prompt:
        prompt = _norm_text(os.sys.stdin.read())
    if not prompt:
        raise SystemExit("缺少 prompt。可通过 --prompt 或 stdin 传入。")

    adapter = CodingPlanAdapter(
        {
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
        }
    )
    result = adapter.generate(
        prompt,
        system_prompt=args.system_prompt,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
