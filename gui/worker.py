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
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal


# ---------------------------------------------------------------------------
# Python executable path
# ---------------------------------------------------------------------------

# Default to the ComfyUI embedded Python on Windows.
# The user can override this in the GUI settings.
DEFAULT_PYTHON_EXE = r"C:\ComfyUI\python_embedded\python.exe"


# ---------------------------------------------------------------------------
# Runtime path resolution
# ---------------------------------------------------------------------------

def resolve_paths(seedvr2_folder: str = "") -> tuple[str, str]:
    """
    Resolve ``(python_exe, cli_script)`` for the current runtime context.

    Resolution order
    ----------------
    1. **PyInstaller bundle** – when ``sys._MEIPASS`` exists the entire
       ``python_embeded`` directory and ``inference_cli.py`` were bundled
       inside the EXE and are available under ``sys._MEIPASS``.
    2. **User-supplied SeedVR2 folder** – the folder selected via the
       "SeedVR2 Folder" button in the GUI.  If it contains a
       ``python_embeded\\python.exe`` that interpreter is used; otherwise
       ``sys.executable`` (the Python that launched the GUI) is used.
    3. **Development / editable install** – look for ``inference_cli.py``
       one level above this file (i.e. the repository root).
    4. **Hardcoded fallback** – ``DEFAULT_PYTHON_EXE`` and the repository
       root derived from this file's location.
    """
    # ── 1. PyInstaller bundle ───────────────────────────────────────────
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
        python_exe = str(base / "python_embeded" / "python.exe")
        cli_script = str(base / "inference_cli.py")
        return python_exe, cli_script

    # ── 2. User-supplied SeedVR2 folder ────────────────────────────────
    if seedvr2_folder:
        folder = Path(seedvr2_folder)
        cli_script = str(folder / "inference_cli.py")
        embedded = folder / "python_embeded" / "python.exe"
        if embedded.exists():
            return str(embedded), cli_script
        return sys.executable, cli_script

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
    progress_update = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)
    started_signal = pyqtSignal()

    # Tokens that SeedVR2 prints which carry frame-progress information.
    # Example:  "Processing batch 3/12 …"  or  "Frame 45/300"
    _PROGRESS_TOKENS = ("batch ", "frame ", "chunk ")

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
        if self._env:
            env.update(self._env)

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
            )
        except FileNotFoundError:
            self.log_line.emit(
                f"❌  Python executable not found: {self._python_exe}\n"
                "    Please select your SeedVR2 installation folder.\n"
            )
            self.finished.emit(False, "Python executable not found.")
            return

        self.started_signal.emit()

        # ── Stream output line-by-line ──────────────────────────────────
        total_frames = 0
        current_frame = 0

        for raw_line in self._process.stdout:  # type: ignore[union-attr]
            if self._abort:
                self._process.terminate()
                break

            line = raw_line.rstrip("\n")
            self.log_line.emit(line)

            # ── Parse progress tokens ───────────────────────────────────
            lower = line.lower()
            for token in self._PROGRESS_TOKENS:
                if token in lower:
                    idx = lower.find(token) + len(token)
                    rest = lower[idx:].split()[0]  # e.g. "3/12"
                    if "/" in rest:
                        try:
                            cur, tot = rest.split("/")
                            current_frame = int(cur)
                            total_frames = int(tot)
                            self.progress_update.emit(current_frame, total_frames)
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
