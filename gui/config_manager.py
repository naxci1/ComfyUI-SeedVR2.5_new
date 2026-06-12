"""
SeedVR2 GUI – Centralized configuration manager.

Detects the installation ROOT_DIR dynamically at import time so that the
application works from any drive or folder name (Task 1 & 2 of the dynamic
workspace directive).

config.json is stored in ROOT_DIR and holds the active paths for every
sub-system (python_embeded, ffmpeg, models, seedvr2 script folder, and the
last-used input / output paths).

On first run the file is auto-created with sensible relative defaults.
On subsequent runs paths that no longer exist are re-verified against the
current ROOT_DIR so that a folder-move is handled automatically.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# ROOT_DIR detection
# ---------------------------------------------------------------------------

def _detect_root_dir() -> Path:
    """Return the installation root, regardless of launch method.

    * PyInstaller bundle  – parent of the frozen EXE.
    * Normal Python       – parent of the gui/ package (i.e. the repo root
                            that contains inference_cli.py).
    """
    if hasattr(sys, "_MEIPASS"):
        # Frozen bundle: the EXE lives in the root alongside config.json.
        return Path(sys.executable).resolve().parent

    # Development / editable install: this file is gui/config_manager.py,
    # so its parent is gui/, and its grandparent is the repo root.
    return Path(os.path.dirname(os.path.abspath(__file__))).parent


ROOT_DIR: Path = _detect_root_dir()
CONFIG_PATH: Path = ROOT_DIR / "config.json"

_IS_WIN: bool = sys.platform == "win32"

# ---------------------------------------------------------------------------
# Default relative paths (all expressed relative to ROOT_DIR)
# ---------------------------------------------------------------------------

DEFAULT_PATHS: dict[str, str] = {
    "python_exe": str(
        ROOT_DIR / "python_embeded" / ("python.exe" if _IS_WIN else "python")
    ),
    "ffmpeg_path": str(
        ROOT_DIR / "ffmpeg" / "bin" / ("ffmpeg.exe" if _IS_WIN else "ffmpeg")
    ),
    "models_dir": str(ROOT_DIR / "models" / "SEEDVR2"),
    "seedvr2_folder": str(ROOT_DIR),
    # Session I/O – populated at runtime
    "input_path": "",
    "input_mode": "File",
    "output_path": "",
}


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_config() -> dict[str, str]:
    """Return the active configuration dict.

    Behaviour
    ---------
    1. If config.json exists it is read and merged with *DEFAULT_PATHS*
       (the file wins for keys it contains).
    2. For the four system paths (python_exe, ffmpeg_path, models_dir,
       seedvr2_folder) the stored value is *re-verified*: if the path no
       longer exists but the corresponding ROOT_DIR-relative default does,
       the default is restored (handles folder moves).
    3. The (possibly updated) config is written back so the file stays
       current.
    """
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                saved: dict = json.load(fh)
        except Exception:
            saved = {}
    else:
        saved = {}

    # Start from defaults then overlay saved values.
    cfg: dict[str, str] = dict(DEFAULT_PATHS)
    cfg.update({k: str(v) for k, v in saved.items() if isinstance(v, str)})

    # Re-verify system paths after a potential folder move.
    for key in ("python_exe", "ffmpeg_path", "models_dir", "seedvr2_folder"):
        stored = cfg.get(key, "")
        if stored and not Path(stored).exists():
            default = DEFAULT_PATHS.get(key, "")
            if default and Path(default).exists():
                cfg[key] = default

    save_config(cfg)
    return cfg


def save_config(cfg: dict[str, str]) -> None:
    """Persist *cfg* to config.json in ROOT_DIR."""
    try:
        CONFIG_PATH.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        # Read-only filesystem or missing parent – silently skip.
        pass
