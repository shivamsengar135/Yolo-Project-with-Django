import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


TASK_CHOICES = [
    "status",
    "validate",
    "report",
    "train",
    "infer_yolo",
    "infer_ssd",
    "export_onnx",
    "infer_onnx",
    "benchmark",
    "benchmark_onnx",
    "benchmark_ssd",
    "compare",
    "plot",
    "full_eval",
    "full_eval_fast",
    "quick_report",
    "publish_results",
]

DEVICE_CHOICES = ["auto", "cuda", "cpu"]
PRESET_CHOICES = ["default", "balanced", "minority_boost"]


class PipelineTaskRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._task_name = ""
        self._status = "Ready"
        self._logs: deque[str] = deque(maxlen=2500)
        self._last_update = time.time()
        self.project_root = Path(__file__).resolve().parent.parent
        venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        self.python_exe = venv_python if venv_python.exists() else Path(sys.executable)

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)
            self._last_update = time.time()

    def _build_command(
        self,
        task: str,
        *,
        device: str,
        source: str,
        epochs: str,
        frames: str,
        preset: str,
    ) -> list[str]:
        return [
            str(self.python_exe),
            "src/run_pipeline.py",
            task,
            "--device",
            device,
            "--source",
            source,
            "--epochs",
            epochs,
            "--frames",
            frames,
            "--preset",
            preset,
        ]

    def start(
        self,
        task: str,
        *,
        device: str,
        source: str,
        epochs: str,
        frames: str,
        preset: str,
    ) -> tuple[bool, str]:
        if task not in TASK_CHOICES:
            return False, f"Unknown task: {task}"
        if device not in DEVICE_CHOICES:
            return False, f"Unsupported device: {device}"
        if preset not in PRESET_CHOICES:
            return False, f"Unsupported preset: {preset}"

        command = self._build_command(
            task,
            device=device,
            source=source,
            epochs=epochs,
            frames=frames,
            preset=preset,
        )

        with self._lock:
            if self._process is not None:
                return False, "A task is already running. Stop it first."
            self._status = f"Running: {task}"
            self._task_name = task
            self._logs.append(f"Starting task: {task}")
            self._logs.append("Command: " + " ".join(command))
            self._logs.append(f"Working directory: {self.project_root}")
            self._logs.append(f"Python executable: {self.python_exe}")
            self._last_update = time.time()

        def worker() -> None:
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self._lock:
                    self._process = process

                assert process.stdout is not None
                for line in process.stdout:
                    self._append_log(line.rstrip())

                code = process.wait()
                if code == 0:
                    self._append_log("Task completed successfully.")
                    with self._lock:
                        self._status = "Ready"
                else:
                    self._append_log(f"Task failed with exit code {code}.")
                    with self._lock:
                        self._status = "Failed"
            except Exception as exc:
                self._append_log(f"Error: {exc}")
                with self._lock:
                    self._status = "Error"
            finally:
                with self._lock:
                    self._process = None
                    self._thread = None

        thread = threading.Thread(target=worker, daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()
        return True, "Task started."

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            process = self._process
            running = process is not None
        if not running or process is None:
            return False, "No running task to stop."
        process.terminate()
        self._append_log("Sent terminate signal to running task.")
        with self._lock:
            self._status = "Stopping..."
        return True, "Stop signal sent."

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            return {
                "running": process is not None,
                "task": self._task_name,
                "status": self._status,
                "logs": list(self._logs),
                "last_update": self._last_update,
            }


runner = PipelineTaskRunner()
