# Laptop Edge Object Detection

This project is a beginner-friendly object detection starter that runs fully on your laptop. It uses YOLO as the main transfer learning model, SSD Lite as a baseline, and includes webcam inference, ONNX export, and a simple benchmark script.

## Project structure

```text
trial1/
  data/
    dataset.yaml
    images/
    labels/
  exports/
  models/
    ssd/
    yolo/
  results/
  src/
    capture_dataset.py
    check_environment.py
    benchmark.py
    benchmark_onnx.py
    benchmark_ssd.py
    dataset_report.py
    download_dataset.py
    export_yolo_onnx.py
    full_eval.py
    generate_comparison_report.py
    infer_webcam_onnx.py
    infer_webcam_ssd.py
    infer_webcam_yolo.py
    plot_training_results.py
    publish_final_results.py
    project_status.py
    run_pipeline.py
    train_yolo.py
    validate_dataset.py
  requirements.txt
```

## Step 1: Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run as desktop app (no commands after setup)

After setup, just double-click one of these files in project root:

- `Open_Object_Detector_GUI.vbs` (recommended, no console window)
- `Launch_Object_Detector_GUI.bat`

This opens a GUI with buttons for:

- Full eval (fast and full)
- Quick report
- Live YOLO / ONNX / SSD inference
- Publish final results

## Run as Django website

From project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```

Open:

`http://127.0.0.1:8000/`

First time only, create an admin user if needed:

```powershell
.\.venv\Scripts\python.exe manage.py createsuperuser
```

Web app behavior:

- Users must register/login before using the dashboard.
- Each login session lasts 10 minutes.
- Countdown is shown in the header.
- After timeout, user is logged out and must login again.
- Live webcam detection runs inside the same page (`YOLO`, `ONNX`, `SSD`).

## Host on Render (Cloud)

This repo is now deployment-ready for Render with:

- `Procfile`
- `build.sh`
- Production-ready Django settings (env-based secret/debug/hosts/static/database)

### Render setup

1. Push this repo to GitHub.
2. In Render, create a new **Web Service** from this repo.
3. Use:
   - Build Command: `./build.sh`
   - Start Command: `gunicorn detector_site.wsgi --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300 --log-file -`
4. Add environment variables:
   - `DJANGO_SECRET_KEY` = strong random string
   - `DJANGO_DEBUG` = `False`
   - `DJANGO_ALLOWED_HOSTS` = `<your-render-hostname>`
   - `DJANGO_CSRF_TRUSTED_ORIGINS` = `https://<your-render-hostname>`
5. Deploy.

### Important note about webcam detection

Cloud hosting does not have access to your local laptop webcam for server-side `cv2.VideoCapture`.
The current live detection stream works best when running locally on your machine.
To make hosted detection use end-user webcams, the app must be changed to browser camera capture (`getUserMedia`) and send frames to backend inference endpoints.

## Build single .exe with custom icon

From project root, run:

```powershell
build_exe.bat
```

This generates:

`dist/ObjectDetectorApp.exe`

Then create a desktop shortcut (auto):

```powershell
create_desktop_shortcut.bat
```

After shortcut creation, right-click it and choose:

- `Pin to taskbar` or
- `Pin to Start`

Note: Keep the EXE inside this project folder so it can find `.venv` and `src`.

If PowerShell blocks activation on your machine, use the virtual environment directly:

```powershell
.\.venv\Scripts\python.exe src\check_environment.py
```

If you do want activation in the current PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```

You can quickly verify your environment with:

```powershell
.\.venv\Scripts\python.exe src\check_environment.py
.\.venv\Scripts\python.exe src\project_status.py
```

## Step 2: Prepare your dataset

Use 3 simple classes to begin:

- `bottle`
- `cup`
- `cell phone`

You can download a small labeled dataset from the internet with:

```powershell
.\.venv\Scripts\python.exe src\download_dataset.py --train-samples 120 --val-samples 30 --num-workers 1
```

This downloads a filtered COCO subset and exports it to YOLO format under `data/`.

Place images here:

- `data/images/train`
- `data/images/val`
- `data/images/test`

Place matching YOLO label `.txt` files here:

- `data/labels/train`
- `data/labels/val`
- `data/labels/test`

Each label line should follow:

```text
class_id x_center y_center width height
```

All values except `class_id` are normalized between 0 and 1.

If you want to collect images from your laptop webcam first, use:

```powershell
.\.venv\Scripts\python.exe src\capture_dataset.py --class-name bottle --split train --count 50
.\.venv\Scripts\python.exe src\capture_dataset.py --class-name cup --split train --count 50
.\.venv\Scripts\python.exe src\capture_dataset.py --class-name "cell_phone" --split train --count 50
```

Press `space` to save a frame and `q` to quit.
Use `--source auto` (default) to auto-select an available webcam.

Before training, validate your dataset:

```powershell
.\.venv\Scripts\python.exe src\validate_dataset.py --data data\dataset.yaml
.\.venv\Scripts\python.exe src\dataset_report.py --data data\dataset.yaml
```

## Step 3: Train YOLO with transfer learning

This starts from pretrained YOLO weights and fine-tunes them on your dataset.

```powershell
.\.venv\Scripts\python.exe src\train_yolo.py --data data\dataset.yaml --weights yolov8n.pt --epochs 30 --imgsz 640 --batch 8 --device auto
```

The best trained model will be saved inside `results/yolo_train/exp/weights/`.

To help weaker classes (for example `cell phone`), try:

```powershell
.\.venv\Scripts\python.exe src\train_yolo.py --data data\dataset.yaml --weights yolov8n.pt --epochs 40 --imgsz 640 --batch 8 --device auto --preset minority_boost
```

## Step 4: Run YOLO webcam detection

Using pretrained YOLO:

```powershell
.\.venv\Scripts\python.exe src\infer_webcam_yolo.py --model yolov8n.pt --imgsz 640 --conf 0.4
```

Using your trained model:

```powershell
.\.venv\Scripts\python.exe src\infer_webcam_yolo.py --model results\yolo_train\exp\weights\best.pt --imgsz 640 --conf 0.4
```

Press `q` to quit.

## Step 5: Run SSD baseline

This uses pretrained SSD Lite with COCO classes for comparison.

```powershell
.\.venv\Scripts\python.exe src\infer_webcam_ssd.py --device auto --threshold 0.5
```

## Step 6: Export YOLO to ONNX

```powershell
.\.venv\Scripts\python.exe src\export_yolo_onnx.py --model results\yolo_train\exp\weights\best.pt --imgsz 640
.\.venv\Scripts\python.exe src\infer_webcam_onnx.py --model results\yolo_train\exp\weights\best.onnx --imgsz 640 --conf 0.4
```

## Step 7: Benchmark webcam inference

```powershell
.\.venv\Scripts\python.exe src\benchmark.py --model results\yolo_train\exp\weights\best.pt --frames 100 --imgsz 640
.\.venv\Scripts\python.exe src\benchmark_onnx.py --model results\yolo_train\exp\weights\best.onnx --frames 100 --imgsz 640
.\.venv\Scripts\python.exe src\benchmark_ssd.py --frames 100 --device auto
.\.venv\Scripts\python.exe src\generate_comparison_report.py --metrics results\metrics.csv --output results\comparison_report.md
```

Results are appended to `results/metrics.csv`.

## Step 8: Plot training curves

```powershell
.\.venv\Scripts\python.exe src\plot_training_results.py --results-csv results\yolo_train\exp\results.csv --output results\training_curves.png
```

This generates a single image with metric and loss curves so you can inspect training quality quickly.

## Optional: Use one command runner

Instead of remembering every script, you can run:

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py status
.\.venv\Scripts\python.exe src\run_pipeline.py validate
.\.venv\Scripts\python.exe src\run_pipeline.py report
.\.venv\Scripts\python.exe src\run_pipeline.py train --epochs 30 --preset default
.\.venv\Scripts\python.exe src\run_pipeline.py export_onnx
.\.venv\Scripts\python.exe src\run_pipeline.py infer_yolo
.\.venv\Scripts\python.exe src\run_pipeline.py benchmark --frames 100
.\.venv\Scripts\python.exe src\run_pipeline.py benchmark_onnx --frames 100
.\.venv\Scripts\python.exe src\run_pipeline.py benchmark_ssd --frames 100
.\.venv\Scripts\python.exe src\run_pipeline.py compare
.\.venv\Scripts\python.exe src\run_pipeline.py full_eval --epochs 40 --frames 100 --preset minority_boost --device auto
.\.venv\Scripts\python.exe src\run_pipeline.py full_eval_fast --device auto
.\.venv\Scripts\python.exe src\run_pipeline.py quick_report --device auto
.\.venv\Scripts\python.exe src\run_pipeline.py publish_results
```

For ONNX inference with the runner, pass your ONNX path:

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py infer_onnx --model results\yolo_train\exp\weights\best.onnx
```

If you omit `--model`, the runner auto-resolves the latest trained `best.pt`/`best.onnx` from `runs/detect/...`.
If you omit `--device`, tasks use `--device auto`, which prefers GPU and falls back to CPU.
If you omit `--source`, tasks use `--source auto`, which scans webcam indexes and picks the first working one.

Note: ONNX GPU acceleration requires an ONNX Runtime build with CUDA provider support.

### One-command full evaluation

`full_eval` runs the complete sequence in one shot:

1. Train YOLO
2. Export ONNX
3. Benchmark PyTorch YOLO
4. Benchmark ONNX YOLO
5. Benchmark SSD Lite
6. Generate comparison report
7. Generate training curves
8. Write one consolidated summary file

Run:

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py full_eval --epochs 40 --frames 100 --preset minority_boost --device auto
```

### Fewer-commands workflow (recommended)

1. Fast end-to-end run:

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py full_eval_fast --device auto
```

2. Re-benchmark and regenerate report without retraining:

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py quick_report --device auto
```

3. Publish latest final results into a single markdown file and README section:

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py publish_results
```

Outputs are written under:

`results/full_eval/full_eval_<timestamp>/`

The single summary file is:

`results/full_eval/full_eval_<timestamp>/full_eval_summary.md`

## Tips for laptop-only edge deployment

- Start with `yolov8n.pt` because it is light enough for CPU testing.
- If FPS is low, drop `--imgsz` from `640` to `416` or `320`.
- Keep batch size at `1` during inference.
- Use ONNX export after training to test faster local inference paths.
- Record both accuracy and FPS so you can compare YOLO and SSD clearly.

## Suggested beginner workflow

1. Run `.\.venv\Scripts\python.exe src\check_environment.py`.
2. Run `.\.venv\Scripts\python.exe src\project_status.py`.
3. Capture images with `capture_dataset.py` or place your own images in the dataset folders.
4. Download the starter dataset or label all images in YOLO format.
5. Run `.\.venv\Scripts\python.exe src\validate_dataset.py`.
6. Inspect the dataset with `.\.venv\Scripts\python.exe src\dataset_report.py`.
7. Train with `.\.venv\Scripts\python.exe src\train_yolo.py`.
8. Test webcam detection with `infer_webcam_yolo.py`.
9. Compare with `infer_webcam_ssd.py`.
10. Export to ONNX, test `infer_webcam_onnx.py`, run both benchmarks, generate comparison report, and plot training curves.

## Recommended next milestone

1. Collect 100 to 200 images per class from your own laptop webcam.
2. Label them in YOLO format.
3. Train YOLO and note the first accuracy result.
4. Run webcam inference and benchmark it.
5. Compare the result with SSD Lite on the same laptop.

## Published Final Results

<!-- FINAL_RESULTS_START -->
# Final Results

Generated: `2026-03-26 14:37:32`
Latest full-eval run: `results\full_eval\full_eval_20260326_143242`

## Artifacts

- Full eval summary: `results\full_eval\full_eval_20260326_143242\full_eval_summary.md`
- Comparison report: `results\full_eval\full_eval_20260326_143242\comparison_report.md`
- Training curves: `results\full_eval\full_eval_20260326_143242\training_curves.png`
- Metrics CSV: `results\full_eval\full_eval_20260326_143242\metrics.csv`

## Best Benchmark Per Backend

| Backend | Model | FPS | Avg Latency (ms) | Frames | Image Size |
|---|---|---:|---:|---:|---:|
| onnx | C:\Users\shiva\Desktop\Training\Coding\Session_13\trial1\runs\detect\results\yolo_train\full_eval_20260326_143242\weights\best.onnx | 20.38 | 49.08 | 30 | 640 |
| pytorch | C:\Users\shiva\Desktop\Training\Coding\Session_13\trial1\runs\detect\results\yolo_train\full_eval_20260326_143242\weights\best.pt | 17.38 | 57.55 | 30 | 640 |
| ssd | ssdlite320_mobilenet_v3_large | 9.73 | 102.78 | 30 | 320 |

## Full Eval Summary Copy

# Full Evaluation Summary

Run name: `full_eval_20260326_143242`
Best PT model: `C:\Users\shiva\Desktop\Training\Coding\Session_13\trial1\runs\detect\results\yolo_train\full_eval_20260326_143242\weights\best.pt`
Best ONNX model: `C:\Users\shiva\Desktop\Training\Coding\Session_13\trial1\runs\detect\results\yolo_train\full_eval_20260326_143242\weights\best.onnx`
Training CSV: `C:\Users\shiva\Desktop\Training\Coding\Session_13\trial1\runs\detect\results\yolo_train\full_eval_20260326_143242\results.csv`
Benchmark CSV: `results\full_eval\full_eval_20260326_143242\metrics.csv`
Comparison report: `results\full_eval\full_eval_20260326_143242\comparison_report.md`
Training curves image: `results\full_eval\full_eval_20260326_143242\training_curves.png`

## Final Training Metrics

- epoch: `30`
- precision: `0.54837`
- recall: `0.3392`
- mAP50: `0.40207`
- mAP50-95: `0.25674`

## Benchmark Rows

| backend | model | fps | avg_latency_ms | frames | imgsz |
|---|---|---:|---:|---:|---:|
| pytorch | C:\Users\shiva\Desktop\Training\Coding\Session_13\trial1\runs\detect\results\yolo_train\full_eval_20260326_143242\weights\best.pt | 17.38 | 57.55 | 30 | 640 |
| onnx | C:\Users\shiva\Desktop\Training\Coding\Session_13\trial1\runs\detect\results\yolo_train\full_eval_20260326_143242\weights\best.onnx | 20.38 | 49.08 | 30 | 640 |
| ssd | ssdlite320_mobilenet_v3_large | 9.73 | 102.78 | 30 | 320 |

<!-- FINAL_RESULTS_END -->
