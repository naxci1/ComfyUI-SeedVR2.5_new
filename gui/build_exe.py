#!/usr/bin/env python3
"""
build_exe.py – Build a 100% self-contained SeedVR2_GUI.exe with PyInstaller.

What gets bundled
-----------------
  • GUI code (styles.py, worker.py, main_window.py, app.py)
  • SeedVR2 repository files (inference_cli.py and every .py / data file at
    the repo root, excluding the gui/ folder itself)
  • The ComfyUI embedded Python environment (python_embeded/), including all
    PyTorch and CUDA DLLs so the EXE runs on machines with no prior setup

At runtime the EXE extracts everything to a temp folder (sys._MEIPASS) and
worker.py's resolve_paths() automatically locates the bundled Python
interpreter and inference_cli.py from there.

WARNING: Because the entire embedded Python + PyTorch + CUDA environment is
included, the final EXE will be large (typically 5–20 GB depending on the
installed packages).  Extraction on first launch may take a few minutes.

Usage
-----
    python gui/build_exe.py [options]

Options
-------
    --embedded-python  PATH   Path to the ComfyUI embedded Python directory.
                              Default: C:\\ComfyUI-yeni\\python_embeded
    --repo-root        PATH   Path to the SeedVR2 repo root (parent of gui/).
                              Default: parent of this script's directory.
    --output-dir       PATH   Directory for the built EXE.
                              Default: <repo_root>/dist
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults – edit these if your installation paths differ
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDED_PYTHON = Path(r"C:\ComfyUI-yeni\python_embeded")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_data(src: Path, dest: str) -> list[str]:
    """Return a ['--add-data', 'src;dest'] pair using the OS path separator."""
    return ["--add-data", f"{src}{os.pathsep}{dest}"]


def _collect_repo_files(repo_root: Path, gui_dir: Path) -> list[str]:
    """
    Return --add-data flags for every file/directory at the repo root that
    should be bundled, excluding the gui/ directory itself.
    """
    flags: list[str] = []

    # Top-level .py files (inference_cli.py and peers)
    for py_file in sorted(repo_root.glob("*.py")):
        flags += _add_data(py_file, ".")

    # Top-level directories that are likely needed by inference_cli.py,
    # but skip gui/ (bundled separately) and heavyweight data dirs.
    skip_dirs = {gui_dir.name, "dist", "build", "__pycache__", ".git", "models"}
    for child in sorted(repo_root.iterdir()):
        if child.is_dir() and child.name not in skip_dirs:
            flags += _add_data(child, child.name)

    # Top-level non-Python data files (e.g. configs, text files)
    for item in sorted(repo_root.iterdir()):
        if item.is_file() and item.suffix not in {".py", ".exe", ".bat", ".sh"}:
            flags += _add_data(item, ".")

    return flags


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build SeedVR2_GUI.exe")
    parser.add_argument(
        "--embedded-python",
        type=Path,
        default=DEFAULT_EMBEDDED_PYTHON,
        help="Path to the ComfyUI embedded Python directory",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the SeedVR2 repository root (parent of gui/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the built EXE",
    )
    args = parser.parse_args()

    gui_dir: Path = Path(__file__).resolve().parent
    repo_root: Path = args.repo_root or gui_dir.parent
    dist_dir: Path = args.output_dir or (repo_root / "dist")
    embedded_python: Path = args.embedded_python

    entry_point = str(gui_dir / "app.py")
    icon_path = gui_dir / "icon.ico"

    # ── Validate inputs ─────────────────────────────────────────────────
    if not embedded_python.exists():
        print(
            f"⚠  Embedded Python directory not found: {embedded_python}\n"
            "   The EXE will be built WITHOUT a bundled Python interpreter.\n"
            "   Users will need ComfyUI / Python installed on their machine.\n"
            "   Re-run with --embedded-python <path> to bundle Python.\n"
        )
        bundle_python = False
    else:
        bundle_python = True
        print(f"✔  Bundling embedded Python from: {embedded_python}")

    # ── Base PyInstaller command ─────────────────────────────────────────
    cmd: list[str] = [
        sys.executable, "-m", "PyInstaller",
        "--name", "SeedVR2_GUI",
        "--onefile",
        "--noconsole",
        # GUI modules
        *_add_data(gui_dir / "styles.py",      "gui"),
        *_add_data(gui_dir / "worker.py",       "gui"),
        *_add_data(gui_dir / "main_window.py",  "gui"),
        # SeedVR2 repository files (inference_cli.py + peers)
        *_collect_repo_files(repo_root, gui_dir),
        # Hidden imports required by PyQt6 + cv2
        "--hidden-import", "PyQt6",
        "--hidden-import", "PyQt6.QtWidgets",
        "--hidden-import", "PyQt6.QtCore",
        "--hidden-import", "PyQt6.QtGui",
        "--hidden-import", "cv2",
        # Output paths
        "--distpath", str(dist_dir),
        "--workpath", str(repo_root / "build"),
        "--specpath", str(repo_root),
    ]

    # ── Bundle the embedded Python environment ───────────────────────────
    if bundle_python:
        # The entire python_embeded directory is placed at python_embeded/
        # inside sys._MEIPASS so worker.py can call it as a subprocess.
        cmd += _add_data(embedded_python, "python_embeded")

        # Collect CUDA and PyTorch DLLs from the embedded environment so
        # they are available on target machines without a GPU driver install.
        # These flags tell PyInstaller to harvest binaries from the packages
        # installed in the embedded env (requires running this script with
        # that same Python, e.g.:  python_embeded\python.exe gui/build_exe.py)
        for pkg in ("torch", "torchvision", "torchaudio"):
            pkg_path = embedded_python / "Lib" / "site-packages" / pkg
            if pkg_path.exists():
                cmd += ["--collect-binaries", pkg]
                print(f"✔  Collecting binaries from: {pkg}")

    # ── Optional icon ────────────────────────────────────────────────────
    if icon_path.exists():
        cmd += ["--icon", str(icon_path)]

    # ── Entry point ──────────────────────────────────────────────────────
    cmd.append(entry_point)

    # ── Run PyInstaller ──────────────────────────────────────────────────
    print("=" * 68)
    print("Building SeedVR2_GUI.exe …")
    print("Command:", " ".join(cmd))
    print("=" * 68)

    result = subprocess.run(cmd, cwd=str(repo_root))

    print("=" * 68)
    if result.returncode == 0:
        exe_path = dist_dir / "SeedVR2_GUI.exe"
        print("✅  Build succeeded!")
        print(f"    Output: {exe_path}")
        if bundle_python:
            print()
            print("The EXE is fully self-contained – no installation required.")
            print("Copy SeedVR2_GUI.exe to any Windows machine and run it.")
            print()
            print("NOTE: On first launch the EXE extracts its contents to a")
            print("      temporary folder which may take a minute.")
        else:
            print()
            print("NOTE: Python was NOT bundled.  Users need ComfyUI installed")
            print("      and must select their SeedVR2 folder in the GUI.")
    else:
        print(f"❌  Build failed (exit code {result.returncode}).")
        print("    Make sure PyInstaller is installed: pip install pyinstaller")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
