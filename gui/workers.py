"""
PySide6 async worker that runs ``inference_cli.py`` as a subprocess and forwards
its stdout / stderr to the GUI thread via Qt signals.

This worker preserves the original signal *contract* so existing consumers keep
working, while also surfacing Auto Tune OOM retries to the GUI.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Signal

try:
    import winsound
except Exception:  # pragma: no cover - Windows only
    winsound = None  # type: ignore[assignment]

_CREATE_NO_WINDOW: int = 0x08000000 if sys.platform == "win32" else 0
_CREATE_NEW_PROCESS_GROUP: int = 0x00000200 if sys.platform == "win32" else 0
_AUTOTUNE_MAX_STAGES = 14
_OOM_LINE_RE = re.compile(r"cuda out of memory", re.IGNORECASE)
_STAGE_RE = re.compile(r"stage\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_BATCH_RE = re.compile(r"batch_size(?:=|→)(\d+)", re.IGNORECASE)

try:
    from gui.config_manager import (
        DEFAULT_PATHS as _DEFAULT_PATHS,
        load_config as _load_config,
    )
except ImportError:  # pragma: no cover - direct-script execution fallback
    try:
        from config_manager import (  # type: ignore[no-redef]
            DEFAULT_PATHS as _DEFAULT_PATHS,
            load_config as _load_config,
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
    phase_update = Signal(str, int, int)
    finished = Signal(bool, str)
    started_signal = Signal()
    oom_detected = Signal(int, int, int)

    _BATCH_TOKENS = ("step ", "steps: ", "steps ")
    _GLOBAL_TOKENS = ("batch ", "frame ", "chunk ")
    _QUEUE_STATUS_PREFIX = "__SEEDVR2_GUI_STATUS__|"
    _PHASE_STATUS_PREFIX = "__SEEDVR2_GUI_PHASE__|"

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
        self.retry_count = 0
        self.max_retries = _AUTOTUNE_MAX_STAGES
        self.alarm_enabled = True
        self._current_batch_size = self._extract_initial_batch_size(args)
        self._pending_oom = False

    def run(self) -> None:
        """Entry-point called by ``QThread.started``."""
        cmd = [self._python_exe, self._cli_script] + self._args
        self.log_line.emit(f"▶  {' '.join(cmd)}")

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
            self.log_line.emit(f"❌  Python executable not found: {self._python_exe}")
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
            if line.startswith(self._PHASE_STATUS_PREFIX):
                try:
                    payload = json.loads(line[len(self._PHASE_STATUS_PREFIX):])
                    self.phase_update.emit(
                        str(payload.get("phase_name", "")),
                        int(payload.get("phase_index", 0)),
                        int(payload.get("phase_total", 0)),
                    )
                except Exception:
                    self.log_line.emit(line)
                continue

            self._inspect_oom_output(line)
            self.log_line.emit(line)
            self._parse_progress(line)

        try:
            self._process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.force_kill()
            self._process.wait()
        rc = self._process.returncode

        if self._abort:
            self.log_line.emit("⏹  Processing cancelled by user.")
            self.finished.emit(False, "Cancelled.")
        elif rc == 0:
            self._play_success_sound()
            self.log_line.emit("✅  Processing completed successfully.")
            self.finished.emit(True, "Done.")
        else:
            self._play_failure_sound()
            self.log_line.emit(f"❌  Process exited with code {rc}.")
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
                        current, total = int(cur), int(tot)
                        if is_batch:
                            self.batch_progress_update.emit(current, total)
                        else:
                            self.progress_update.emit(current, total)
                    except ValueError:
                        pass
                break

    def _inspect_oom_output(self, line: str) -> None:
        lower = line.lower()
        if _OOM_LINE_RE.search(line):
            self._pending_oom = True
            self.retry_count = min(self.retry_count + 1, self.max_retries)
            suggested = self._suggest_batch_size(self.retry_count)
            self._current_batch_size = suggested
            self.oom_detected.emit(self.retry_count, self.max_retries, suggested)
            return

        if "auto tune: cuda oom caught" in lower or "critical memory error" in lower:
            stage_match = _STAGE_RE.search(line)
            if stage_match:
                try:
                    self.retry_count = max(self.retry_count, int(stage_match.group(1)))
                    self.max_retries = int(stage_match.group(2))
                except ValueError:
                    pass
            batch_match = _BATCH_RE.search(line)
            if batch_match:
                try:
                    self._current_batch_size = int(batch_match.group(1))
                except ValueError:
                    pass
            self._pending_oom = False

    def _extract_initial_batch_size(self, args: List[str]) -> int:
        try:
            idx = args.index("--batch_size")
            return max(1, int(args[idx + 1]))
        except (ValueError, IndexError):
            return 81

    def _suggest_batch_size(self, retry_count: int) -> int:
        current = max(1, self._current_batch_size)
        if retry_count <= 1:
            return 77 if current > 77 else max(1, round((current - 1) / 4) * 4 + 1)
        if retry_count == 4 and current > 45:
            return 45
        if retry_count >= 5:
            snapped = max(1, current - 4)
            return max(1, round((snapped - 1) / 4) * 4 + 1)
        return current

    def _play_success_sound(self) -> None:
        if not self.alarm_enabled or winsound is None:
            return
        try:
            winsound.PlaySound(
                "SystemAsterisk",
                winsound.SND_ALIAS | winsound.SND_ASYNC,
            )
        except Exception:
            pass

    def _play_failure_sound(self) -> None:
        if not self.alarm_enabled or winsound is None:
            return
        try:
            for _ in range(3):
                winsound.Beep(1000, 250)
        except Exception:
            pass

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
