"""
Async worker that runs inference_cli.py as a subprocess and forwards its
stdout / stderr to the main thread via Qt signals.  The process is launched
with the *embedded* Python interpreter that ships with ComfyUI so that all
heavy AI dependencies (torch, safetensors, …) are already present.
"""

from __future__ import annotations

import subprocess
import os
import sys
import signal
import json
import gc
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

# Suppress the console window that would appear on Windows when launching
# a subprocess from a --noconsole PyInstaller bundle or a windowed app.
_CREATE_NO_WINDOW: int = 0x08000000 if sys.platform == "win32" else 0
_CREATE_NEW_PROCESS_GROUP: int = 0x00000200 if sys.platform == "win32" else 0


# ---------------------------------------------------------------------------
# Python executable path
# ---------------------------------------------------------------------------

# Default to the ComfyUI embedded Python on Windows.
# Note: ComfyUI historically ships its folder as "python_embeded" (single 'd') –
# this path reflects the real on-disk name.  The user can override it in the GUI.
DEFAULT_PYTHON_EXE = r"C:\ComfyUI-yeni\python_embeded\python.exe"


# ---------------------------------------------------------------------------
# Runtime path resolution
# ---------------------------------------------------------------------------

def resolve_paths(seedvr2_folder: str = "", python_exe: str = "") -> tuple[str, str]:
    """
    Resolve ``(python_exe, cli_script)`` for the current runtime context.

    When ``python_exe`` and ``seedvr2_folder`` are both supplied (the normal
    case when running the lightweight GUI EXE), they are used directly.

    Fallback order when values are missing
    ----------------------------------------
    1. **PyInstaller bundle** – ``sys._MEIPASS`` present (future full-bundle mode).
    2. **Development / editable install** – ``inference_cli.py`` adjacent to
       this file's parent directory.
    3. **Hardcoded defaults** – ``DEFAULT_PYTHON_EXE`` and repo root.
    """
    # ── Explicit user-supplied values (normal GUI usage) ────────────────
    if python_exe and seedvr2_folder:
        cli_script = str(Path(seedvr2_folder) / "inference_cli.py")
        return python_exe, cli_script

    # ── 1. PyInstaller bundle ───────────────────────────────────────────
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
        python_exe = str(base / "python_embedded" / "python.exe")
        cli_script = str(base / "inference_cli.py")
        return python_exe, cli_script

    # ── 2. User-supplied SeedVR2 folder only (no explicit python_exe) ───
    if seedvr2_folder:
        folder = Path(seedvr2_folder)
        cli_script = str(folder / "inference_cli.py")
        return python_exe or sys.executable, cli_script

    # ── 3. Development / editable install ──────────────────────────────
    repo_root = Path(__file__).resolve().parent.parent
    cli_candidate = repo_root / "inference_cli.py"
    if cli_candidate.exists():
        return sys.executable, str(cli_candidate)

    # ── 4. Hardcoded fallback ───────────────────────────────────────────
    return DEFAULT_PYTHON_EXE, str(repo_root / "inference_cli.py")


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class InferenceWorker(QObject):
    """
    Runs ``inference_cli.py`` in a subprocess from a dedicated QThread.

    Signals
    -------
    log_line(str)
        Emitted for every line written to stdout or stderr.
    progress_update(int, int)
        Emitted with (current_frame, total_frames) when a progress token is
        detected in the output.
    finished(bool, str)
        Emitted when the process exits.  (success, message)
    started_signal()
        Emitted once the subprocess has been launched.
    """

    log_line = pyqtSignal(str)
    # progress_update  → global/total progress  (frame or chunk token)
    # batch_progress_update → per-batch progress (batch token)
    progress_update = pyqtSignal(int, int)
    batch_progress_update = pyqtSignal(int, int)
    queue_status_update = pyqtSignal(str, int, int, int, int)
    finished = pyqtSignal(bool, str)
    started_signal = pyqtSignal()

    # Tokens that SeedVR2 prints which carry frame-progress information.
    # "step N/M" / "steps N/M" → per-step within a batch → batch bar (inner)
    # "batch N/M" / "frame N/M" / "chunk N/M" → overall batches → total bar (outer)
    _BATCH_TOKENS = ("step ", "steps: ", "steps ")
    _GLOBAL_TOKENS = ("batch ", "frame ", "chunk ")
    _QUEUE_STATUS_PREFIX = "__SEEDVR2_GUI_STATUS__|"

    def __init__(
        self,
        cli_script: str,
        args: List[str],
        python_exe: str = DEFAULT_PYTHON_EXE,
        env: Optional[dict] = None,
        postprocess_config: Optional[dict[str, Any]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._cli_script = cli_script
        self._args = args
        self._python_exe = python_exe
        self._env = env
        self._postprocess_config = postprocess_config or {}
        self._process: Optional[subprocess.Popen] = None
        self._abort = False

    @staticmethod
    def _aggressive_cleanup() -> None:
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _ensure_unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        idx = 1
        while True:
            probe = path.with_name(f"{path.stem}_{idx}{path.suffix}")
            if not probe.exists():
                return probe
            idx += 1

    def _resolve_existing_output_asset(self, configured_output_path: Path) -> Optional[Path]:
        """
        Resolve the actual rendered video file path when CLI normalizes extensions.

        Example mismatch:
        - GUI arg:  ...\\preview_xxx.png
        - CLI save: ...\\preview_xxx.mp4
        """
        # Force-check exact video counterparts first.
        video_counterparts = [
            configured_output_path.with_suffix(".mp4"),
            configured_output_path.with_suffix(".mov"),
        ]
        for candidate in video_counterparts:
            if candidate.exists():
                self.log_line.emit(
                    "ℹ  Video stabilizer output path synchronized: "
                    f"{configured_output_path} -> {candidate}\n"
                )
                return candidate

        if configured_output_path.exists() and configured_output_path.suffix.lower() in {".mp4", ".mov"}:
            return configured_output_path

        # Robust fallback: case-insensitive suffix match on same stem in directory.
        parent = configured_output_path.parent
        stem = configured_output_path.stem.lower()
        if parent.exists():
            for entry in parent.iterdir():
                if not entry.is_file():
                    continue
                if entry.stem.lower() != stem:
                    continue
                if entry.suffix.lower() in {".mp4", ".mov"}:
                    self.log_line.emit(
                        "ℹ  Video stabilizer output path synchronized: "
                        f"{configured_output_path} -> {entry}\n"
                    )
                    return entry
        return None

    def _run_video_stabilizer(self, env: dict[str, str]) -> bool:
        if not self._postprocess_config.get("enabled", False):
            return True
        output_path_raw = str(self._postprocess_config.get("output_path", "")).strip().strip('"').strip("'")
        if not output_path_raw:
            self.log_line.emit("⚠  Video stabilizer skipped: output path missing.\n")
            return True
        configured_output_path = Path(output_path_raw)
        output_path = self._resolve_existing_output_asset(configured_output_path)
        if output_path is None:
            self.log_line.emit(
                "⚠  Video stabilizer skipped: output not found "
                f"({configured_output_path}).\n"
            )
            return True

        video_args = [str(x) for x in self._postprocess_config.get("video_args", [])]
        audio_args = [str(x) for x in self._postprocess_config.get("audio_args", ["-c:a", "copy"])]
        if not video_args:
            self.log_line.emit(
                "⚠  Video stabilizer skipped: no user-selected FFmpeg video args available.\n"
            )
            return True

        stabilized_path = self._ensure_unique_path(
            output_path.with_name(f"{output_path.stem}_stabilized{output_path.suffix}")
        )

        self.log_line.emit("🎥  Video stabilizer: pass 1/2 (motion analysis)…")
        with tempfile.TemporaryDirectory(prefix="seedvr2_vidstab_") as tmp_dir:
            trf_name = "transforms.trf"
            pass1 = [
                "ffmpeg",
                "-y",
                "-i",
                str(output_path),
                "-vf",
                f"vidstabdetect=shakiness=5:accuracy=15:result={trf_name}",
                "-f",
                "null",
                "-",
            ]
            run1 = subprocess.run(
                pass1,
                cwd=tmp_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
            if run1.returncode != 0:
                self.log_line.emit(
                    f"❌  Video stabilizer pass 1 failed (exit {run1.returncode}).\n{run1.stdout[-2000:]}"
                )
                return False

            self.log_line.emit("🎥  Video stabilizer: pass 2/2 (transform + encode)…")
            pass2 = [
                "ffmpeg",
                "-y",
                "-i",
                str(output_path),
                "-vf",
                f"vidstabtransform=input={trf_name}:smoothing=30:optzoom=1",
                *video_args,
                *audio_args,
                str(stabilized_path),
            ]
            self.log_line.emit(
                "ℹ  FFmpeg pass 2 command: "
                f'ffmpeg -y -i "{output_path}" -vf '
                f'"vidstabtransform=input={trf_name}:smoothing=30:optzoom=1" '
                + " ".join(video_args + audio_args)
                + f' "{stabilized_path}"\n'
            )
            run2 = subprocess.run(
                pass2,
                cwd=tmp_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
            if run2.returncode != 0:
                self.log_line.emit(
                    f"❌  Video stabilizer pass 2 failed (exit {run2.returncode}).\n{run2.stdout[-2000:]}"
                )
                return False

        self.log_line.emit(f"✅  Stabilized output written: {stabilized_path}")
        return True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Entry-point called by QThread.started."""
        cmd = [self._python_exe, self._cli_script] + self._args
        self.log_line.emit(f"▶  {' '.join(cmd)}\n")

        env = os.environ.copy()
        # Ensure UTF-8 output from the child process on Windows
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONLEGACYWINDOWSFSENCODING"] = "1"
        # Blackwell / performance environment variables
        runtime_env = {
            "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync,max_split_size_mb:256,garbage_collection_threshold:0.6",
            "TORCH_CUDNN_BENCHMARK": "1",
            "CUDA_MODULE_LOADING": "LAZY",
            "TORCH_CUDNN_V8_API_ENABLED": "1",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "CUDA_CACHE_MAXSIZE": "4294967296",
            "NVIDIA_TF32_OVERRIDE": "1",
            "ATTENTION_BACKEND": "sageattention",
        }
        for key, value in runtime_env.items():
            os.environ[key] = value
            env[key] = value
        if self._env:
            env.update(self._env)

        # Pre-run: set float32 matmul precision for maximum Blackwell throughput
        try:
            subprocess.run(
                [
                    self._python_exe, "-c",
                    "import torch; torch.set_float32_matmul_precision('high')",
                ],
                env=env,
                check=False,
                creationflags=_CREATE_NO_WINDOW,
            )
        except FileNotFoundError:
            self.log_line.emit(
                "⚠  Pre-run matmul config skipped: Python executable not found yet.\n"
            )

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=_CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP,
                start_new_session=(sys.platform != "win32"),
            )
        except FileNotFoundError:
            self.log_line.emit(
                f"❌  Python executable not found: {self._python_exe}\n"
                "    Please check the Python Executable path in the settings.\n"
            )
            self.finished.emit(False, "Python executable not found.")
            return

        self.started_signal.emit()

        # ── Stream output line-by-line ──────────────────────────────────
        for raw_line in self._process.stdout:  # type: ignore[union-attr]
            if self._abort:
                self._process.terminate()
                break

            line = raw_line.rstrip("\n")
            if line.startswith(self._QUEUE_STATUS_PREFIX):
                try:
                    payload = json.loads(line[len(self._QUEUE_STATUS_PREFIX):])
                    self.queue_status_update.emit(
                        str(payload.get("file_path", "")),
                        int(payload.get("current", 0)),
                        int(payload.get("total", 0)),
                        int(payload.get("done", 0)),
                        int(payload.get("remaining", 0)),
                    )
                except Exception:
                    self.log_line.emit(line)
                continue

            self.log_line.emit(line)

            # ── Parse progress tokens ───────────────────────────────────
            lower = line.lower()
            for token, is_batch in (
                *((t, True) for t in self._BATCH_TOKENS),
                *((t, False) for t in self._GLOBAL_TOKENS),
            ):
                if token in lower:
                    idx = lower.find(token) + len(token)
                    rest = lower[idx:].split()[0]  # e.g. "3/12"
                    if "/" in rest:
                        try:
                            cur, tot = rest.split("/")
                            c, t = int(cur), int(tot)
                            if is_batch:
                                self.batch_progress_update.emit(c, t)
                            else:
                                self.progress_update.emit(c, t)
                        except ValueError:
                            pass
                    break

        # Give the process a short window to exit cleanly, then force-kill.
        try:
            self._process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self._process.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        creationflags=_CREATE_NO_WINDOW,
                    )
                else:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process.wait()
        rc = self._process.returncode

        if self._abort:
            self.log_line.emit("\n⏹  Processing cancelled by user.\n")
            self._aggressive_cleanup()
            self.finished.emit(False, "Cancelled.")
        elif rc == 0:
            stabilized_ok = self._run_video_stabilizer(env)
            self._aggressive_cleanup()
            if stabilized_ok:
                self.log_line.emit("\n✅  Processing completed successfully.\n")
                self.finished.emit(True, "Done.")
            else:
                self.finished.emit(False, "Video stabilization failed.")
        else:
            self.log_line.emit(f"\n❌  Process exited with code {rc}.\n")
            self._aggressive_cleanup()
            self.finished.emit(False, f"Exit code {rc}.")

    def request_abort(self) -> None:
        """Immediately force-kill the subprocess tree and signal the worker to stop."""
        self._abort = True
        # Use force_kill directly so the process is terminated even when it is stuck
        # inside a GPU/VAE loop that does not respond to SIGTERM.
        self.force_kill()

    def force_kill(self) -> None:
        """Forcefully terminate the subprocess tree."""
        self._abort = True
        if not self._process or self._process.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self._process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=_CREATE_NO_WINDOW,
                )
            else:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_worker_thread(
    cli_script: str,
    args: List[str],
    python_exe: str = DEFAULT_PYTHON_EXE,
    env: Optional[dict] = None,
    postprocess_config: Optional[dict[str, Any]] = None,
) -> tuple[QThread, InferenceWorker]:
    """
    Create a (QThread, InferenceWorker) pair ready to be started.

    Usage::

        thread, worker = create_worker_thread(script, args, python_exe)
        worker.log_line.connect(my_slot)
        worker.finished.connect(my_done_slot)
        thread.start()
    """
    thread = QThread()
    worker = InferenceWorker(
        cli_script,
        args,
        python_exe,
        env,
        postprocess_config=postprocess_config,
    )
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    return thread, worker
