#!/usr/bin/env python3
"""
build_exe.py – Build a standalone SeedVR2_GUI.exe with PyInstaller.

Two modes are supported
-----------------------
**Lightweight (default)**
    A small "remote controller" EXE that points at the user's existing
    ComfyUI / Python installation at runtime.  Only the GUI code is bundled.

**Portable (--python-embeded-dir + --seedvr2-dir)**
    A self-contained bundle that includes the ComfyUI embedded Python
    interpreter and the entire SeedVR2 source tree (excluding the ``models``
    sub-folder to keep the size manageable, typically 3-4 GB).  The user
    can run this EXE without any external Python installation.

Usage
-----
    # Lightweight (default):
    python gui/build_exe.py

    # Portable bundle:
    python gui/build_exe.py \\
        --python-embeded-dir "C:\\ComfyUI-yeni\\python_embeded" \\
        --seedvr2-dir        "C:\\ComfyUI-yeni\\custom_nodes\\seedvr2_videoupscaler"

Options
-------
    --output-dir          PATH   Directory for the built EXE.
                                 Default: <repo_root>/dist
    --python-embeded-dir  PATH   ComfyUI embedded Python directory to bundle.
    --seedvr2-dir         PATH   SeedVR2 source folder (inference_cli.py parent).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
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
    parser.add_argument(
        "--python-embeded-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the ComfyUI embedded Python directory to bundle (portable mode)",
    )
    parser.add_argument(
        "--seedvr2-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the SeedVR2 source folder (portable mode; models/ is excluded)",
    )
    args = parser.parse_args()

    portable: bool = bool(args.python_embeded_dir or args.seedvr2_dir)

    gui_dir: Path = Path(__file__).resolve().parent
    repo_root: Path = gui_dir.parent
    dist_dir: Path = args.output_dir or (repo_root / "dist")

    entry_point = str(gui_dir / "app.py")
    # Prefer icon.ico at the gui root; fall back to assets/icon.ico
    icon_path = gui_dir / "icon.ico"
    if not icon_path.exists():
        icon_path = gui_dir / "assets" / "icon.ico"

    # ── PyInstaller command ──────────────────────────────────────────────
    cmd: list[str] = [
        sys.executable, "-m", "PyInstaller",
        "--name", "SeedVR2_GUI",
        "--onefile",
        "--noconsole",
        "--noupx",  # disable UPX compression – avoids "This app can't run on your PC"
        # Bundle GUI Python modules (styles, settings window, split view)
        "--add-data", f"{gui_dir / 'styles.py'}{os.pathsep}gui",
        "--add-data", f"{gui_dir / 'settings_window.py'}{os.pathsep}gui",
        "--add-data", f"{gui_dir / 'split_view.py'}{os.pathsep}gui",
        # Core hidden imports (PyQt6 + multimedia for the comparison player)
        "--hidden-import", "PyQt6",
        "--hidden-import", "PyQt6.QtWidgets",
        "--hidden-import", "PyQt6.QtCore",
        "--hidden-import", "PyQt6.QtGui",
        "--hidden-import", "PyQt6.QtMultimedia",
        "--hidden-import", "PyQt6.QtMultimediaWidgets",
        # Exclude heavy ML / AI libraries – they live in the user's python_embeded
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "numpy",
        "--exclude-module", "cv2",
        "--exclude-module", "onnxruntime",
        "--exclude-module", "transformers",
        "--exclude-module", "safetensors",
        "--exclude-module", "diffusers",
        "--exclude-module", "accelerate",
        "--exclude-module", "PIL",
        "--exclude-module", "scipy",
        "--exclude-module", "sklearn",
        "--exclude-module", "matplotlib",
        # Output paths
        "--distpath", str(dist_dir),
        "--workpath", str(repo_root / "build"),
        "--specpath", str(repo_root),
    ]

    # ── Bundle assets directory (icons) ─────────────────────────────────
    assets_dir = gui_dir / "assets"
    if assets_dir.is_dir():
        cmd += ["--add-data", f"{assets_dir}{os.pathsep}assets"]

    # Bundle icon.ico at the bundle root so _resource_path("icon.ico") resolves
    if icon_path.exists():
        cmd += ["--add-data", f"{icon_path}{os.pathsep}."]

    # ── Portable mode: bundle Python + SeedVR2 source ───────────────────
    if args.python_embeded_dir:
        py_dir = args.python_embeded_dir.resolve()
        if not py_dir.is_dir():
            print(f"❌  --python-embeded-dir not found: {py_dir}")
            sys.exit(1)
        # Bundle as "python_embedded" inside the bundle root
        cmd += ["--add-data", f"{py_dir}{os.pathsep}python_embedded"]
        print(f"  Bundling Python: {py_dir}")

    if args.seedvr2_dir:
        sv_dir = args.seedvr2_dir.resolve()
        if not sv_dir.is_dir():
            print(f"❌  --seedvr2-dir not found: {sv_dir}")
            sys.exit(1)
        # Bundle source files; exclude the models sub-folder to keep size manageable
        models_dir = sv_dir / "models"
        if models_dir.is_dir():
            print(f"  Excluding models folder: {models_dir}")
        # Copy source tree to a temp location without models, then add-data
        tmp = Path(tempfile.mkdtemp(prefix="seedvr2_bundle_"))
        shutil.copytree(
            str(sv_dir),
            str(tmp / "seedvr2_src"),
            ignore=shutil.ignore_patterns("models", "__pycache__", "*.pyc"),
        )
        cmd += ["--add-data", f"{tmp / 'seedvr2_src'}{os.pathsep}seedvr2_src"]
        print(f"  Bundling SeedVR2 source (from {sv_dir})")

    # Optional icon
    if icon_path.exists():
        cmd += ["--icon", str(icon_path)]

    cmd.append(entry_point)

    # ── Run ──────────────────────────────────────────────────────────────
    mode_label = "portable all-in-one" if portable else "lightweight"
    print("=" * 68)
    print(f"Building SeedVR2_GUI.exe ({mode_label} mode) …")
    print("Command:", " ".join(cmd))
    print("=" * 68)

    result = subprocess.run(cmd, cwd=str(repo_root))

    print("=" * 68)
    if result.returncode == 0:
        exe_path = dist_dir / "SeedVR2_GUI.exe"
        print("✅  Build succeeded!")
        print(f"    Output: {exe_path}")
        print()
        if portable:
            print("Portable distribution notes:")
            print("  1. Copy dist/SeedVR2_GUI.exe to the target Windows machine.")
            print("  2. Run it directly – no external Python needed.")
            print("  3. Model files are NOT included; point the GUI at your models folder.")
        else:
            print("Lightweight distribution notes:")
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
