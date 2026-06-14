"""
SeedVR2 GUI – Centralized configuration manager.

Detects the installation ROOT_DIR dynamically at import time so that the
application works from any drive or folder name.  PyInstaller bundles are
handled via the ``sys.frozen`` flag; the executable path is used (with two
dirname levels) to locate the true installation root without ever relying on
a hard-coded ``dist`` subdirectory.

config.json is stored in ROOT_DIR and holds the active paths for every
sub-system (python_embeded, ffmpeg, models, seedvr2 script folder, and the
last-used input / output paths).

On first run the file is auto-created with sensible relative defaults.
On subsequent runs paths that no longer exist are re-verified – including a
one-level backtrack – so that a folder-move or a misdetected root is handled
automatically.  Any path that still cannot be resolved is added to the
module-level ``INVALID_PATHS`` set so that the UI can prompt the user.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# ROOT_DIR detection
# ---------------------------------------------------------------------------

if getattr(sys, "frozen", False):
    ROOT_DIR: Path = Path(os.path.dirname(os.path.abspath(sys.executable)))
else:
    # Development / editable install: this file is gui/config_manager.py,
    # so its parent is gui/, and its grandparent is the repo root.
    ROOT_DIR = Path(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

CONFIG_PATH: Path = ROOT_DIR / "config.json"
DEFAULT_TEMP_DIR: str = os.path.normpath(os.path.join(str(ROOT_DIR), "temp"))

_IS_WIN: bool = sys.platform == "win32"

# ---------------------------------------------------------------------------
# Default relative paths (all expressed relative to ROOT_DIR)
# ---------------------------------------------------------------------------

# Canonical relative sub-paths used for backtracking in validate_paths().
_REL_PYTHON: str = os.path.join(
    "python_embeded", "python.exe" if _IS_WIN else "python"
)
_REL_FFMPEG: str = os.path.join(
    "ffmpeg", "bin", "ffmpeg.exe" if _IS_WIN else "ffmpeg"
)
_REL_MODELS: str = os.path.join("models", "SEEDVR2")

DEFAULT_PATHS: dict[str, str] = {
    "python_exe": os.path.normpath(os.path.join(str(ROOT_DIR), _REL_PYTHON)),
    "ffmpeg_path": os.path.normpath(os.path.join(str(ROOT_DIR), _REL_FFMPEG)),
    "models_dir": os.path.normpath(os.path.join(str(ROOT_DIR), _REL_MODELS)),
    "seedvr2_folder": str(ROOT_DIR),
    "temp_dir": DEFAULT_TEMP_DIR,
    # Session I/O – populated at runtime
    "input_path": "",
    "input_mode": "File",
    "output_path": "",
}

# ---------------------------------------------------------------------------
# Path validation with backtracking
# ---------------------------------------------------------------------------

# Module-level set of config keys whose paths could not be resolved even
# after backtracking.  Populated by load_config() so the UI can warn users.
INVALID_PATHS: set[str] = set()

# Mapping of system-path config keys → relative sub-path from ROOT_DIR
# (None means the path IS ROOT_DIR itself).
_SYSTEM_RELATIVES: dict[str, str | None] = {
    "python_exe": _REL_PYTHON,
    "ffmpeg_path": _REL_FFMPEG,
    "models_dir": _REL_MODELS,
    "seedvr2_folder": None,
}


def validate_paths(cfg: dict[str, str]) -> set[str]:
    """Validate system paths in *cfg*, attempting one backtrack level if needed.

    For each system path key the function:

    1. Checks whether the stored path exists – if yes, it is kept as-is.
    2. If the path is missing it tries a one-level backtrack:
       ``os.path.normpath(os.path.join(str(ROOT_DIR), '..', relative_sub_path))``
       and updates *cfg* in-place when the backtracked path exists.
    3. If the path still cannot be resolved the key is added to the returned
       set so callers can surface an actionable warning to the user.

    Parameters
    ----------
    cfg:
        Configuration dict (modified in-place on successful backtrack).

    Returns
    -------
    set[str]
        Keys whose paths remain unresolvable after backtracking.
    """
    invalid: set[str] = set()

    for key, rel in _SYSTEM_RELATIVES.items():
        path_str = cfg.get(key, "")
        if not path_str:
            invalid.add(key)
            continue

        if Path(path_str).exists():
            continue

        # Attempt one-level backtrack from ROOT_DIR.
        if rel is not None:
            backtrack = Path(
                os.path.normpath(os.path.join(str(ROOT_DIR), "..", rel))
            )
        else:
            # seedvr2_folder IS ROOT_DIR – backtrack to its parent.
            backtrack = ROOT_DIR.parent

        if backtrack.exists():
            cfg[key] = str(backtrack)
        else:
            invalid.add(key)

    return invalid


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
    3. ``validate_paths()`` is called to apply one-level backtracking for
       any path that is still missing.  Unresolvable keys are written to the
       module-level ``INVALID_PATHS`` set.
    4. The (possibly updated) config is written back so the file stays
       current.
    """
    global INVALID_PATHS

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
    if "temp_dir" not in cfg or not str(cfg.get("temp_dir", "")).strip():
        cfg["temp_dir"] = DEFAULT_TEMP_DIR

    # Re-verify system paths after a potential folder move.
    for key in ("python_exe", "ffmpeg_path", "models_dir", "seedvr2_folder"):
        stored = cfg.get(key, "")
        if stored and not Path(stored).exists():
            default = DEFAULT_PATHS.get(key, "")
            if default and Path(default).exists():
                cfg[key] = default

    # Backtracking validation for any path still missing.
    INVALID_PATHS = validate_paths(cfg)

    # Ensure writable temp directory exists.
    try:
        os.makedirs(cfg.get("temp_dir", DEFAULT_TEMP_DIR), exist_ok=True)
    except OSError:
        cfg["temp_dir"] = DEFAULT_TEMP_DIR
        try:
            os.makedirs(cfg["temp_dir"], exist_ok=True)
        except OSError:
            pass

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
