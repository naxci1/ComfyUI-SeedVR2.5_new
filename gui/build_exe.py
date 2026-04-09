#!/usr/bin/env python3
"""
build_exe.py – Build a standalone SeedVR2_GUI.exe with PyInstaller.

Usage:
    python gui/build_exe.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    gui_dir = Path(__file__).resolve().parent
    repo_root = gui_dir.parent

    entry_point = str(gui_dir / "app.py")
    icon_path = gui_dir / "icon.ico"

    cmd: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name", "SeedVR2_GUI",
        "--onefile",
        "--windowed",
        "--add-data", f"{gui_dir / 'styles.py'}{os.pathsep}gui",
        "--add-data", f"{gui_dir / 'worker.py'}{os.pathsep}gui",
        "--hidden-import", "PyQt6",
        "--hidden-import", "PyQt6.QtWidgets",
        "--hidden-import", "PyQt6.QtCore",
        "--hidden-import", "PyQt6.QtGui",
        "--hidden-import", "cv2",
        "--distpath", str(repo_root / "dist"),
        "--workpath", str(repo_root / "build"),
        "--specpath", str(repo_root),
        entry_point,
    ]

    if icon_path.exists():
        # Insert icon flag right after "--windowed"
        windowed_idx = cmd.index("--windowed")
        cmd.insert(windowed_idx + 1, "--icon")
        cmd.insert(windowed_idx + 2, str(icon_path))

    print("=" * 60)
    print("Building SeedVR2_GUI.exe …")
    print("Command:", " ".join(cmd))
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(repo_root))

    print("=" * 60)
    if result.returncode == 0:
        exe_path = repo_root / "dist" / "SeedVR2_GUI.exe"
        print("✅  Build succeeded!")
        print(f"    Output: {exe_path}")
        print()
        print("Next steps:")
        print("  1. Copy dist/SeedVR2_GUI.exe to any Windows machine.")
        print("  2. Run SeedVR2_GUI.exe – no Python installation required.")
        print("  3. Set the Python Executable path in the bottom bar to your")
        print("     ComfyUI embedded Python if the default path differs.")
    else:
        print(f"❌  Build failed (exit code {result.returncode}).")
        print("    Make sure PyInstaller is installed: pip install pyinstaller")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
