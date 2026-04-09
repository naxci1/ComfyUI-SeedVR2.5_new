#!/usr/bin/env python3
"""
build_exe.py – Build a lightweight SeedVR2_GUI.exe with PyInstaller.

Strategy
--------
The EXE is a "remote controller" that the user points at their existing
ComfyUI / Python installation.  No Python interpreter, no PyTorch DLLs,
and no SeedVR2 repository files are bundled.  Only the GUI code itself is
included, making the EXE small enough to distribute easily.

At runtime the user selects:
  • Their Python executable (e.g. ComfyUI's python_embeded\\python.exe)
  • The SeedVR2 folder containing inference_cli.py
  • Their model directory

Both paths are saved via QSettings so the user only needs to configure
them once.

Usage
-----
    python gui/build_exe.py [options]

Options
-------
    --output-dir  PATH   Directory for the built EXE.
                         Default: <repo_root>/dist
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build SeedVR2_GUI.exe")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the built EXE",
    )
    args = parser.parse_args()

    gui_dir: Path = Path(__file__).resolve().parent
    repo_root: Path = gui_dir.parent
    dist_dir: Path = args.output_dir or (repo_root / "dist")

    entry_point = str(gui_dir / "app.py")
    icon_path = gui_dir / "icon.ico"

    # ── PyInstaller command ──────────────────────────────────────────────
    cmd: list[str] = [
        sys.executable, "-m", "PyInstaller",
        "--name", "SeedVR2_GUI",
        "--onefile",
        "--noconsole",
        # Bundle only the GUI stylesheet; worker/main_window are imported
        # from the frozen package so PyInstaller discovers them automatically.
        "--add-data", f"{gui_dir / 'styles.py'}{os.pathsep}gui",
        # Hidden imports required by PyQt6 + optional cv2 preview
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

    # Optional icon
    if icon_path.exists():
        cmd += ["--icon", str(icon_path)]

    cmd.append(entry_point)

    # ── Run ──────────────────────────────────────────────────────────────
    print("=" * 68)
    print("Building SeedVR2_GUI.exe (lightweight mode) …")
    print("Command:", " ".join(cmd))
    print("=" * 68)

    result = subprocess.run(cmd, cwd=str(repo_root))

    print("=" * 68)
    if result.returncode == 0:
        exe_path = dist_dir / "SeedVR2_GUI.exe"
        print("✅  Build succeeded!")
        print(f"    Output: {exe_path}")
        print()
        print("Distribution notes:")
        print("  1. Copy dist/SeedVR2_GUI.exe to the target Windows machine.")
        print("  2. On first run, set:")
        print(r"       • Python Executable  →  e.g. C:\ComfyUI-yeni\python_embeded\python.exe")
        print(r"       • SeedVR2 Folder     →  folder containing inference_cli.py")
        print("  3. Both paths are remembered automatically (QSettings).")
    else:
        print(f"❌  Build failed (exit code {result.returncode}).")
        print("    Make sure PyInstaller is installed: pip install pyinstaller")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
