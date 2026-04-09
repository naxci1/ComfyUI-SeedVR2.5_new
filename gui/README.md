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
- **Python Executable**: defaults to `C:\ComfyUI\python_embeded\python.exe`. Change this in the bottom bar if your ComfyUI is installed elsewhere.
- **Model Directory**: leave blank to use the default `models/SEEDVR2/` folder next to `inference_cli.py`.

## Notes
- The GUI wraps `inference_cli.py` and does not modify any core logic.
- All console output from the CLI is streamed in real-time.
