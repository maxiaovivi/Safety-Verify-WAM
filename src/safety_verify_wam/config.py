from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    updated = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected KEY=VALUE override, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        keys = [part for part in dotted_key.split(".") if part]
        if not keys:
            raise ValueError(f"Invalid override key: {override!r}")
        target: dict[str, Any] = updated
        for key in keys[:-1]:
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot assign below non-mapping config key: {dotted_key}")
            target = child
        target[keys[-1]] = yaml.safe_load(raw_value)
    return updated


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
