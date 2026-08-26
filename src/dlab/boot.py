from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(__file__).resolve().parent

_CFG: Dict[str, Any] | None = None


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "Missing PyYAML. Run 'uv sync' from the repository root."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _add_dll_dirs(cfg: Dict[str, Any]) -> None:
    """Register DLL directories on Windows."""
    if os.name != "nt":
        return
    for key in ("drivers_andor", "drivers_avaspec", "drivers_slm"):
        rel = (cfg.get("paths") or {}).get(key)
        if rel:
            p = (ROOT / rel).resolve()
            if p.exists():
                os.add_dll_directory(str(p))


def bootstrap(config_path: Path) -> Dict[str, Any]:
    """Load config and initialize DLL paths."""
    global _CFG
    _CFG = load_config(config_path)
    _add_dll_dirs(_CFG)
    return _CFG


def get_config() -> Dict[str, Any]:
    """Return loaded config, or load default if not bootstrapped."""
    if _CFG is not None:
        return _CFG
    default = ROOT / "config" / "config.yaml"
    return load_config(default) if default.exists() else {}
