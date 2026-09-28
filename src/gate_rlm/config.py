"""Configuration loading: default.yaml <- experiment yaml <- frozen thresholds."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    experiment: Optional[str | Path] = None,
    thresholds: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load the default config, apply an experiment file, then frozen thresholds.

    ``thresholds`` is the JSON written by ``experiments/sweep.py``. Applying it last
    guarantees every test-split run uses exactly the thresholds chosen on validation.
    """
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if experiment:
        cfg = deep_merge(cfg, yaml.safe_load(Path(experiment).read_text()) or {})
    if thresholds:
        cfg = deep_merge(cfg, json.loads(Path(thresholds).read_text()))
    cfg.setdefault("name", Path(experiment).stem if experiment else "default")
    return cfg
