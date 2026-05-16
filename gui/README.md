# SeedVR2 GUI — Topaz-style Windows Wrapper

## Requirements
- Windows 10/11
- Python 3.10+ (or use ComfyUI embedded Python)
- ComfyUI with SeedVR2 models installed

## Quick Start

### 1. Install GUI dependencies
```
pip install -r gui/gui_requirements.txt
```

### 2. Launch the GUI
```
python gui/app.py
```

### 3. Build standalone .exe
```
python gui/build_exe.py
```
The .exe will be created in `dist/SeedVR2_GUI.exe`.

## Configuration
- **Python Executable**: defaults to `C:\ComfyUI-yeni\python_embeded\python.exe` (the real ComfyUI folder name). Change this in the bottom bar if your installation is at a different path.
- **SeedVR2 Folder**: the directory containing `inference_cli.py`.  Both paths are saved automatically via QSettings so you only need to configure them once.
- **Model Directory**: leave blank to use the default `models/SEEDVR2/` folder next to `inference_cli.py`.

## Notes
- The GUI wraps `inference_cli.py` and does not modify any core logic.
- All console output from the CLI is streamed in real-time.
- Dual progress bars show total and per-batch progress with elapsed/ETA status.
- Right-panel model/processing settings are persisted across launches.
- Drag-and-drop supports input files/folders directly onto the main window.
- Presets can be saved/loaded as JSON from the right panel.
- A job queue panel can enqueue multiple jobs for sequential execution.
- System tray integration keeps the app running in the background and shows completion notifications.
- Output settings now include container-aware codec choices, image-sequence export modes, and audio profiles with FFmpeg mapping metadata logged per run.
