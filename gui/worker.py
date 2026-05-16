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
from pathlib import Path
from typing import List, Optional

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
    finished = pyqtSignal(bool, str)
    started_signal = pyqtSignal()

    # Tokens that SeedVR2 prints which carry frame-progress information.
    # "step N/M" / "steps N/M" → per-step within a batch → batch bar (inner)
    # "batch N/M" / "frame N/M" / "chunk N/M" → overall batches → total bar (outer)
    _BATCH_TOKENS = ("step ", "steps: ", "steps ")
    _GLOBAL_TOKENS = ("batch ", "frame ", "chunk ")

    def __init__(
        self,
        cli_script: str,
        args: List[str],
        python_exe: str = DEFAULT_PYTHON_EXE,
        env: Optional[dict] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._cli_script = cli_script
        self._args = args
        self._python_exe = python_exe
        self._env = env
        self._process: Optional[subprocess.Popen] = None
        self._abort = False

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
        env["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
        env["CUDA_MODULE_LOADING"] = "LAZY"
        env["NVIDIA_TF32_OVERRIDE"] = "1"
        env["ATTENTION_BACKEND"] = "sageattention"
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

        self._process.wait()
        rc = self._process.returncode

        if self._abort:
            self.log_line.emit("\n⏹  Processing cancelled by user.\n")
            self.finished.emit(False, "Cancelled.")
        elif rc == 0:
            self.log_line.emit("\n✅  Processing completed successfully.\n")
            self.finished.emit(True, "Done.")
        else:
            self.log_line.emit(f"\n❌  Process exited with code {rc}.\n")
            self.finished.emit(False, f"Exit code {rc}.")

    def request_abort(self) -> None:
        """Ask the worker to terminate the subprocess gracefully."""
        self._abort = True
        if self._process and self._process.poll() is None:
            self._process.terminate()

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
    worker = InferenceWorker(cli_script, args, python_exe, env)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    return thread, worker
