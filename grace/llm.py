"""LLM access through one OpenAI-compatible chat-completions interface, plus response parsing.

* All decoding parameters come from the configuration. ``"omit"`` means the parameter is not sent
  (the provider default applies); ``null`` stops the run until the value is supplied.
* Failures return ``None`` after a bounded number of retries; they are never turned into an empty
  prediction silently. The caller decides (``on_api_error``) and every failure is logged.
* Parsing: see :func:`parse_completion` and README ("Response parsing").
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import OMIT, GraceConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    model_id: str
    messages: List[Dict[str, str]]
    params: Dict[str, Any]


@dataclass
class LLMResponse:
    text: Optional[str]
    ok: bool
    attempts: int
    error: Optional[str] = None
    raw_meta: Dict[str, Any] = field(default_factory=dict)


def _optional_param(cfg: GraceConfig, key: str):
    value = cfg.require(key)
    return None if value == OMIT else value


def decoding_params(cfg: GraceConfig) -> Dict[str, Any]:
    params: Dict[str, Any] = {"max_tokens": int(cfg.require("max_new_tokens"))}
    for key, api_name in (("llm_temperature", "temperature"), ("llm_top_p", "top_p"), ("llm_seed", "seed")):
        value = _optional_param(cfg, key)
        if value is not None:
            params[api_name] = value
    return params


def build_messages(prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


Transport = Callable[[LLMRequest], Dict[str, Any]]


class OpenAICompatibleClient:
    """Sends prompts to any OpenAI-compatible endpoint. ``transport`` can be replaced in tests."""

    def __init__(self, base_url: str, api_key_env: str, max_retries: int = 3, backoff_seconds: float = 2.0,
                 timeout_seconds: float = 120.0, transport: Optional[Transport] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.transport = transport or self._openai_transport
        self.sleep = sleep
        self._client = None

    def _openai_transport(self, request: LLMRequest) -> Dict[str, Any]:
        if self._client is None:
            from openai import OpenAI

            key = os.getenv(self.api_key_env)
            if not key:
                raise RuntimeError(f"Environment variable {self.api_key_env} is not set")
            self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout_seconds)
        resp = self._client.chat.completions.create(model=request.model_id, messages=request.messages,
                                                    **request.params)
        choice = resp.choices[0]
        return {
            "text": choice.message.content,
            "finish_reason": getattr(choice, "finish_reason", None),
            "response_model": getattr(resp, "model", None),
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
        }

    def complete(self, request: LLMRequest) -> LLMResponse:
        last_error = None
        for attempt in range(1, self.max_retries + 2):
            try:
                result = self.transport(request)
                text = result.get("text") if isinstance(result, dict) else None
                if not isinstance(text, str):
                    raise ValueError("provider returned no text content")
                meta = {k: v for k, v in result.items() if k != "text"}
                return LLMResponse(text=text, ok=True, attempts=attempt, raw_meta=meta)
            except Exception as exc:  # network, rate limit, malformed response ...
                last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("LLM request failed (attempt %d/%d): %s", attempt, self.max_retries + 1, last_error)
                if attempt <= self.max_retries:
                    self.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        return LLMResponse(text=None, ok=False, attempts=self.max_retries + 1, error=last_error)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# --------------------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------------------

_FENCE = re.compile(r"```[A-Za-z0-9_+-]*\s*\n?(.*?)```", re.S)
_NOISE = [
    re.compile(r"^\s*```"),
    re.compile(r"^\s*(python|java|json)\s*:?\s*$", re.I),
    re.compile(r"^\s*('''|\"\"\")\s*$"),
    re.compile(r"^\s*(here is|the next (line|code) is|completion:)", re.I),
]


@dataclass
class ParsedCompletion:
    completed_code: str
    parse_status: str  # json | json_multiline_truncated | fallback_first_line | fallback_empty
    explanation: str = ""
    confidence_score: Any = None
    referenced_nodes: List[Any] = field(default_factory=list)
    parse_error: Optional[str] = None


def _json_objects(text: str):
    """Yield decoded JSON objects found in ``text`` (fenced blocks first, then any '{...}')."""
    decoder = json.JSONDecoder()
    sources = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for src in sources:
        for start in [i for i, ch in enumerate(src) if ch == "{"]:
            try:
                obj, _ = decoder.raw_decode(src[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _first_line(text: str) -> str:
    for line in text.split("\n"):
        if not line.strip() or any(p.match(line) for p in _NOISE):
            continue
        return line.rstrip()
    return ""


def parse_completion(text: str, failure_policy: str = "first_code_line") -> ParsedCompletion:
    """Extract ``completed_code`` from a model response.

    1. The first JSON object (inside a code fence or anywhere in the text) that has a
       ``completed_code`` key is used. If the value has several lines, only the first non-empty line
       is kept (status ``json_multiline_truncated``).
    2. Otherwise ``failure_policy`` applies: ``first_code_line`` keeps the first non-empty line that is
       not a code fence / language tag / boilerplate phrase; ``empty`` predicts "". The parse error is
       recorded in both cases.
    """
    if failure_policy not in ("first_code_line", "empty"):
        raise ValueError("parse_failure_policy must be 'first_code_line' or 'empty'")
    for obj in _json_objects(text or ""):
        if "completed_code" not in obj:
            continue
        code = obj.get("completed_code")
        code = "" if code is None else str(code)
        status = "json"
        non_empty = [ln for ln in code.split("\n") if ln.strip()]
        if len(non_empty) > 1:
            code, status = non_empty[0], "json_multiline_truncated"
        elif non_empty:
            code = non_empty[0]
        refs = obj.get("referenced_nodes", [])
        return ParsedCompletion(
            completed_code=code.rstrip("\r"),
            parse_status=status,
            explanation=str(obj.get("explanation", "")),
            confidence_score=obj.get("confidence_score"),
            referenced_nodes=refs if isinstance(refs, list) else [refs],
        )
    error = "no JSON object with a completed_code field"
    if failure_policy == "empty":
        return ParsedCompletion("", "fallback_empty", parse_error=error)
    return ParsedCompletion(_first_line(text or ""), "fallback_first_line", parse_error=error)
