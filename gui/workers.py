"""
PySide6 async worker that runs ``inference_cli.py`` as a subprocess and forwards
its stdout / stderr to the GUI thread via Qt signals.

This worker preserves the original signal *contract* so existing consumers
keep working:

    log_line(str)
    progress_update(int, int)
    batch_progress_update(int, int)
    queue_status_update(str, int, int, int, int)
    finished(bool, str)
    started_signal()
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Signal

# Suppress the console window that would appear on Windows when launching a
# subprocess from a windowed app.
_CREATE_NO_WINDOW: int = 0x08000000 if sys.platform == "win32" else 0
_CREATE_NEW_PROCESS_GROUP: int = 0x00000200 if sys.platform == "win32" else 0

try:
    from gui.config_manager import (
        load_config as _load_config,
        DEFAULT_PATHS as _DEFAULT_PATHS,
    )
except ImportError:  # pragma: no cover - direct-script execution fallback
    try:
        from config_manager import (  # type: ignore[no-redef]
            load_config as _load_config,
            DEFAULT_PATHS as _DEFAULT_PATHS,
        )
    except ImportError:
        _load_config = None  # type: ignore[assignment]
        _DEFAULT_PATHS = {}  # type: ignore[assignment]

try:
    if _load_config is not None:
        _cfg = _load_config()
        DEFAULT_PYTHON_EXE: str = _cfg.get("python_exe", "") or _DEFAULT_PATHS.get(
            "python_exe", sys.executable
        )
    else:
        DEFAULT_PYTHON_EXE = sys.executable
except Exception:
    DEFAULT_PYTHON_EXE = _DEFAULT_PATHS.get("python_exe", sys.executable)  # type: ignore[assignment]
finally:
    try:
        del _cfg
    except NameError:
        pass


def resolve_paths(seedvr2_folder: str = "", python_exe: str = "") -> tuple[str, str]:
    """Resolve ``(python_exe, cli_script)`` for the current runtime context."""
    if python_exe and seedvr2_folder:
        return python_exe, str(Path(seedvr2_folder) / "inference_cli.py")

    if hasattr(sys, "_MEIPASS"):
        try:
            from gui.config_manager import ROOT_DIR as _bundle_root
        except ImportError:
            from config_manager import ROOT_DIR as _bundle_root  # type: ignore[no-redef]
        python_exe = os.path.normpath(
            os.path.join(str(_bundle_root), "python_embeded", "python.exe")
        )
        return python_exe, str(Path(str(_bundle_root)) / "inference_cli.py")

    if seedvr2_folder:
        cli_script = str(Path(seedvr2_folder) / "inference_cli.py")
        return python_exe or sys.executable, cli_script

    repo_root = Path(__file__).resolve().parent.parent
    cli_candidate = repo_root / "inference_cli.py"
    if cli_candidate.exists():
        return sys.executable, str(cli_candidate)
    return DEFAULT_PYTHON_EXE, str(repo_root / "inference_cli.py")


class InferenceWorker(QObject):
    """Runs ``inference_cli.py`` in a subprocess from a dedicated QThread."""

    log_line = Signal(str)
    progress_update = Signal(int, int)
    batch_progress_update = Signal(int, int)
    queue_status_update = Signal(str, int, int, int, int)
    finished = Signal(bool, str)
    started_signal = Signal()

    _BATCH_TOKENS = ("step ", "steps: ", "steps ")
    _GLOBAL_TOKENS = ("batch ", "frame ", "chunk ")
    _QUEUE_STATUS_PREFIX = "__SEEDVR2_GUI_STATUS__|"

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
    def run(self) -> None:
        """Entry-point called by ``QThread.started``."""
        cmd = [self._python_exe, self._cli_script] + self._args
        self.log_line.emit(f"▶  {' '.join(cmd)}\n")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONLEGACYWINDOWSFSENCODING"] = "1"
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
                creationflags=_CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP,
                start_new_session=(sys.platform != "win32"),
            )
        except FileNotFoundError:
            self.log_line.emit(
                f"❌  Python executable not found: {self._python_exe}\n"
            )
            self.finished.emit(False, "Python executable not found.")
            return

        self.started_signal.emit()

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
            self._parse_progress(line)

        try:
            self._process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.force_kill()
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

    def _parse_progress(self, line: str) -> None:
        lower = line.lower()
        for token, is_batch in (
            *((t, True) for t in self._BATCH_TOKENS),
            *((t, False) for t in self._GLOBAL_TOKENS),
        ):
            if token in lower:
                idx = lower.find(token) + len(token)
                parts = lower[idx:].split()
                if not parts:
                    break
                rest = parts[0]
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

    def request_abort(self) -> None:
        """Force-kill the subprocess tree and signal the worker to stop."""
        self._abort = True
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


def create_worker_thread(
    cli_script: str,
    args: List[str],
    python_exe: str = DEFAULT_PYTHON_EXE,
    env: Optional[dict] = None,
) -> tuple[QThread, InferenceWorker]:
    """Create a ``(QThread, InferenceWorker)`` pair ready to be started."""
    thread = QThread()
    worker = InferenceWorker(cli_script, args, python_exe, env)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    return thread, worker
