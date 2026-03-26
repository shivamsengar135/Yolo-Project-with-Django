import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


class ObjectDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Laptop Edge Object Detection")
        self.root.geometry("980x680")

        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            candidates = [exe_dir, exe_dir.parent]
            self.project_root = exe_dir
            for candidate in candidates:
                if (candidate / "src" / "run_pipeline.py").exists():
                    self.project_root = candidate
                    break
        else:
            self.project_root = Path(__file__).resolve().parent.parent

        venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        self.python_exe = venv_python if venv_python.exists() else Path(sys.executable)
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.device_var = tk.StringVar(value="auto")
        self.source_var = tk.StringVar(value="auto")
        self.epochs_var = tk.StringVar(value="40")
        self.frames_var = tk.StringVar(value="100")
        self.preset_var = tk.StringVar(value="minority_boost")

        self._build_ui()
        self._poll_logs()

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        title = ttk.Label(
            self.root,
            text="Object Detection Desktop App",
            font=("Segoe UI", 16, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 8))

        control_frame = ttk.LabelFrame(self.root, text="Settings")
        control_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        for i in range(8):
            control_frame.grid_columnconfigure(i, weight=1)

        ttk.Label(control_frame, text="Device").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Combobox(
            control_frame,
            textvariable=self.device_var,
            values=["auto", "cuda", "cpu"],
            state="readonly",
            width=10,
        ).grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(control_frame, text="Source").grid(row=0, column=2, padx=8, pady=8, sticky="w")
        ttk.Entry(control_frame, textvariable=self.source_var, width=10).grid(
            row=0, column=3, padx=8, pady=8, sticky="w"
        )

        ttk.Label(control_frame, text="Epochs").grid(row=0, column=4, padx=8, pady=8, sticky="w")
        ttk.Entry(control_frame, textvariable=self.epochs_var, width=10).grid(
            row=0, column=5, padx=8, pady=8, sticky="w"
        )

        ttk.Label(control_frame, text="Frames").grid(row=0, column=6, padx=8, pady=8, sticky="w")
        ttk.Entry(control_frame, textvariable=self.frames_var, width=10).grid(
            row=0, column=7, padx=8, pady=8, sticky="w"
        )

        ttk.Label(control_frame, text="Preset").grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Combobox(
            control_frame,
            textvariable=self.preset_var,
            values=["default", "balanced", "minority_boost"],
            state="readonly",
            width=16,
        ).grid(row=1, column=1, padx=8, pady=8, sticky="w")

        button_frame = ttk.LabelFrame(self.root, text="Actions")
        button_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 8))
        for col in range(4):
            button_frame.grid_columnconfigure(col, weight=1)
        button_frame.grid_rowconfigure(3, weight=1)

        actions = [
            ("Status", lambda: self.run_pipeline_task("status")),
            ("Quick Report", lambda: self.run_pipeline_task("quick_report")),
            ("Full Eval Fast", lambda: self.run_pipeline_task("full_eval_fast")),
            ("Full Eval", lambda: self.run_pipeline_task("full_eval")),
            ("Publish Results", lambda: self.run_pipeline_task("publish_results")),
            ("Infer YOLO", lambda: self.run_pipeline_task("infer_yolo")),
            ("Infer ONNX", lambda: self.run_pipeline_task("infer_onnx")),
            ("Infer SSD", lambda: self.run_pipeline_task("infer_ssd")),
            ("Stop Running Task", self.stop_process),
            ("Open Results Folder", self.open_results_folder),
            ("Open README", self.open_readme),
            ("Clear Log", self.clear_log),
        ]

        for i, (text, command) in enumerate(actions):
            row = i // 4
            col = i % 4
            ttk.Button(button_frame, text=text, command=command).grid(
                row=row, column=col, padx=8, pady=8, sticky="ew"
            )

        self.log_text = tk.Text(button_frame, wrap="word", height=18)
        self.log_text.grid(row=4, column=0, columnspan=4, sticky="nsew", padx=8, pady=(8, 8))
        self.log_text.configure(state="disabled")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").grid(
            row=3, column=0, sticky="ew", padx=12, pady=(0, 12)
        )

    def append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _poll_logs(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.append_log(line)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_logs)

    def _build_run_pipeline_cmd(self, task: str) -> list[str]:
        cmd = [
            str(self.python_exe),
            "src/run_pipeline.py",
            task,
            "--device",
            self.device_var.get().strip(),
            "--source",
            self.source_var.get().strip(),
            "--epochs",
            self.epochs_var.get().strip(),
            "--frames",
            self.frames_var.get().strip(),
            "--preset",
            self.preset_var.get().strip(),
        ]
        return cmd

    def run_pipeline_task(self, task: str) -> None:
        if self.process is not None:
            messagebox.showwarning("Task Running", "A task is already running. Stop it first.")
            return

        cmd = self._build_run_pipeline_cmd(task)
        self.status_var.set(f"Running: {task}")
        self.append_log(f"Starting task: {task}")
        self.append_log("Command: " + " ".join(cmd))
        self.append_log(f"Working directory: {self.project_root}")
        self.append_log(f"Python executable: {self.python_exe}")

        def worker() -> None:
            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.log_queue.put(line.rstrip())
                code = self.process.wait()
                if code == 0:
                    self.log_queue.put("Task completed successfully.")
                    self.status_var.set("Ready")
                else:
                    self.log_queue.put(f"Task failed with exit code {code}.")
                    self.status_var.set("Failed")
            except Exception as exc:
                self.log_queue.put(f"Error: {exc}")
                self.status_var.set("Error")
            finally:
                self.process = None

        threading.Thread(target=worker, daemon=True).start()

    def stop_process(self) -> None:
        if self.process is None:
            messagebox.showinfo("No Task", "No running task to stop.")
            return
        self.process.terminate()
        self.append_log("Sent terminate signal to running task.")
        self.status_var.set("Stopping...")

    def open_results_folder(self) -> None:
        results_path = self.project_root / "results"
        subprocess.Popen(["explorer", str(results_path)])

    def open_readme(self) -> None:
        readme_path = self.project_root / "README.md"
        subprocess.Popen(["explorer", str(readme_path)])


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = ObjectDetectionApp(root)
    app.append_log("GUI ready. Choose an action to begin.")
    root.mainloop()


if __name__ == "__main__":
    main()
