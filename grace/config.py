"""Configuration loading.

A configuration file has the sections listed in :data:`FLAT_SECTIONS` plus ``llm``. The flat sections
are merged into one key space and each key remembers the section it came from. A key may appear in
only one of them. :data:`OVERRIDE_SECTIONS` (used by the test config) may replace keys defined
earlier. :meth:`GraceConfig.require` raises :class:`MissingParameter` for a value that is ``null``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "paper_reproduction.yaml"
DEV_CONFIG = CONFIG_DIR / "dev_smoke_test.yaml"

FLAT_SECTIONS = (
    "paper_reported",
    "method_hyperparameters",
    "implementation",
)

# Sections that may replace keys defined in FLAT_SECTIONS (used by the test config).
OVERRIDE_SECTIONS = ("test_overrides",)

# Sentinel accepted for optional decoding parameters: do not send the parameter at all.
OMIT = "omit"


class MissingParameter(ValueError):
    """Raised when a stage needs a configuration value that is not set."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    parent = raw.pop("inherits", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        raw = _deep_merge(_load_yaml(parent_path), raw)
    return raw


@dataclass
class GraceConfig:
    name: str
    values: Dict[str, Any]
    provenance: Dict[str, str]
    llm: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_paper_config(self) -> bool:
        return self.name == "paper_reproduction"

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.values:
            raise KeyError(f"Unknown configuration key: {key}")
        value = self.values[key]
        if value is None:
            raise MissingParameter(f"Configuration value '{key}' is not set; pass --set {key}=VALUE.")
        return value

    def require_all(self, keys: Iterable[str]) -> Dict[str, Any]:
        missing: List[str] = [k for k in keys if self.values.get(k) is None]
        if missing:
            raise MissingParameter(
                "Configuration values required for this stage are not set: "
                + ", ".join(missing)
                + ". Pass them with --set key=VALUE."
            )
        return {k: self.values[k] for k in keys}

    def set(self, key: str, value: Any) -> None:
        origin = self.provenance.get(key)
        if origin == "paper_reported" and self.values.get(key) != value:
            LOGGER.warning(
                "Overriding paper value %s=%r with %r.",
                key,
                self.values.get(key),
                value,
            )
        self.values[key] = value
        self.overrides[key] = value
        if key not in self.provenance:
            self.provenance[key] = "override"

    def resolve_model(self, name: str) -> Dict[str, Any]:
        """Resolve a model alias to ``{name, provider, model_id, base_url, api_key_env}``."""
        models = self.llm.get("models", {}) or {}
        canonical = None
        if name in models:
            canonical = name
        else:
            for key, spec in models.items():
                if name in (spec or {}).get("aliases", []) or []:
                    canonical = key
                    break
        if canonical is None:
            raise KeyError(f"Unknown model '{name}'. Add it to the llm.models section of the config.")
        spec = models[canonical] or {}
        provider_name = spec.get("provider")
        model_id = spec.get("model_id")
        if provider_name is None or model_id is None:
            raise MissingParameter(
                f"Model '{canonical}' needs both provider and model_id under llm.models."
            )
        provider = (self.llm.get("providers", {}) or {}).get(provider_name)
        if provider is None:
            raise KeyError(f"Provider '{provider_name}' is not defined under llm.providers.")
        return {
            "name": canonical,
            "provider": provider_name,
            "model_id": model_id,
            "base_url": provider.get("base_url"),
            "api_key_env": provider.get("api_key_env"),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "config_name": self.name,
            "source_path": self.source_path,
            "values": dict(self.values),
            "provenance": dict(self.provenance),
            "overrides": dict(self.overrides),
        }


def parse_override(text: str) -> tuple:
    if "=" not in text:
        raise ValueError(f"Override must look like key=value, got: {text}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_config(path: Optional[str] = None, overrides: Optional[Iterable[str]] = None) -> GraceConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    raw = _load_yaml(cfg_path)
    values: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    for section in FLAT_SECTIONS:
        for key, value in (raw.get(section) or {}).items():
            if key in values and provenance[key] != section:
                raise ValueError(f"Key '{key}' appears in both {provenance[key]} and {section}")
            values[key] = value
            provenance[key] = section
    for section in OVERRIDE_SECTIONS:
        for key, value in (raw.get(section) or {}).items():
            values[key] = value
            provenance[key] = section
    cfg = GraceConfig(
        name=str(raw.get("config_name", cfg_path.stem)),
        values=values,
        provenance=provenance,
        llm=raw.get("llm", {}) or {},
        source_path=str(cfg_path),
    )
    for item in overrides or []:
        key, value = parse_override(item)
        cfg.set(key, value)
    return cfg


def resolve_device(spec: Optional[str]) -> str:
    import torch

    if spec in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(spec)
