from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

from .forms import LoginForm, RegisterForm
from .runner import DEVICE_CHOICES, runner

SESSION_TIMEOUT_SECONDS = 600
_YOLO_MODEL = None
_YOLO_MODEL_PATH = None
_YOLO_MODEL_LOCK = threading.Lock()


def _clean_value(raw: Any, fallback: str) -> str:
    value = str(raw or "").strip()
    return value if value else fallback


def _resolve_web_yolo_model(model_arg: str) -> str:
    project_root = Path(__file__).resolve().parent.parent
    requested = Path(model_arg)
    if requested.exists():
        return str(requested)

    fallbacks = [
        project_root / "yolo26n.pt",
        project_root / "yolov8n.pt",
    ]
    for candidate in fallbacks:
        if candidate.exists():
            return str(candidate)
    return model_arg


def _get_web_yolo_model(model_path: str):
    global _YOLO_MODEL, _YOLO_MODEL_PATH
    with _YOLO_MODEL_LOCK:
        if _YOLO_MODEL is None or _YOLO_MODEL_PATH != model_path:
            from ultralytics import YOLO

            _YOLO_MODEL = YOLO(model_path)
            _YOLO_MODEL_PATH = model_path
        return _YOLO_MODEL


@require_GET
def root_redirect(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("dashboard")
    return redirect("login")


@require_GET
def login_view_get(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "detector_web/login.html", {"form": LoginForm()})


@require_POST
def login_view_post(request: HttpRequest) -> HttpResponse:
    form = LoginForm(request, data=request.POST)
    if not form.is_valid():
        return render(request, "detector_web/login.html", {"form": form})

    login(request, form.get_user())
    expires_at = int(time.time()) + SESSION_TIMEOUT_SECONDS
    request.session["login_expires_at"] = expires_at
    request.session.set_expiry(SESSION_TIMEOUT_SECONDS)
    return redirect("dashboard")


def login_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        return login_view_post(request)
    return login_view_get(request)


def register_view(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            expires_at = int(time.time()) + SESSION_TIMEOUT_SECONDS
            request.session["login_expires_at"] = expires_at
            request.session.set_expiry(SESSION_TIMEOUT_SECONDS)
            return redirect("dashboard")
    else:
        form = RegisterForm()

    return render(request, "detector_web/register.html", {"form": form})


@login_required
def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect("login")


@login_required
@require_GET
def dashboard(request: HttpRequest) -> HttpResponse:
    expires_at = int(request.session.get("login_expires_at", int(time.time())))
    context = {
        "device_choices": DEVICE_CHOICES,
        "stream_backends": ["yolo", "onnx", "ssd"],
        "default_values": {
            "device": "auto",
            "source": "auto",
            "backend": "yolo",
            "model": "yolov8n.pt",
            "imgsz": "640",
            "conf": "0.4",
        },
        "expires_at": expires_at,
    }
    return render(request, "detector_web/dashboard.html", context)


@login_required
@require_POST
def run_task(request: HttpRequest, task: str) -> JsonResponse:
    payload = request.POST
    ok, message = runner.start(
        task,
        device=_clean_value(payload.get("device"), "auto"),
        source=_clean_value(payload.get("source"), "auto"),
        epochs=_clean_value(payload.get("epochs"), "40"),
        frames=_clean_value(payload.get("frames"), "100"),
        preset=_clean_value(payload.get("preset"), "minority_boost"),
    )
    status = runner.status()
    return JsonResponse(
        {
            "ok": ok,
            "message": message,
            "status": status,
        },
        status=200 if ok else 400,
    )


@login_required
@require_POST
def stop_task(request: HttpRequest) -> JsonResponse:
    ok, message = runner.stop()
    status = runner.status()
    return JsonResponse(
        {
            "ok": ok,
            "message": message,
            "status": status,
        },
        status=200 if ok else 400,
    )


@login_required
@require_GET
def task_status(request: HttpRequest) -> JsonResponse:
    return JsonResponse(runner.status(), status=200)


@csrf_exempt
@login_required
@require_GET
def stream_detection(request: HttpRequest, backend: str) -> HttpResponse:
    model = _clean_value(request.GET.get("model"), "yolov8n.pt")
    source = _clean_value(request.GET.get("source"), "auto")
    device = _clean_value(request.GET.get("device"), "auto")
    imgsz = int(_clean_value(request.GET.get("imgsz"), "640"))
    conf = float(_clean_value(request.GET.get("conf"), "0.4"))

    try:
        # Lazy import avoids loading heavy inference dependencies for normal page requests.
        from .live_stream import live_service

        stream = live_service.stream(
            backend,
            model=model,
            source=source,
            device=device,
            imgsz=imgsz,
            conf=conf,
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "message": str(exc)}, status=400)

    return StreamingHttpResponse(
        stream,
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


@csrf_exempt
@login_required
@require_POST
def infer_frame(request: HttpRequest) -> HttpResponse:
    backend = _clean_value(request.POST.get("backend"), "yolo").lower()
    if backend != "yolo":
        return JsonResponse(
            {
                "ok": False,
                "message": "Hosted mode currently supports YOLO backend for browser camera inference.",
            },
            status=400,
        )

    frame_file = request.FILES.get("frame")
    if frame_file is None:
        return JsonResponse({"ok": False, "message": "No frame received."}, status=400)

    model_arg = _clean_value(request.POST.get("model"), "yolov8n.pt")
    model_path = _resolve_web_yolo_model(model_arg)
    imgsz = int(_clean_value(request.POST.get("imgsz"), "640"))
    conf = float(_clean_value(request.POST.get("conf"), "0.4"))
    device = _clean_value(request.POST.get("device"), "cpu")
    if device.lower() in {"auto", "cuda", "gpu"}:
        device = "cpu"

    try:
        import cv2
        import numpy as np

        raw = frame_file.read()
        np_arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JsonResponse({"ok": False, "message": "Invalid image frame."}, status=400)

        model = _get_web_yolo_model(model_path)
        results = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            iou=0.45,
            device=device,
            verbose=False,
        )
        annotated = results[0].plot()
        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return JsonResponse({"ok": False, "message": "Could not encode frame."}, status=500)
        return HttpResponse(buffer.tobytes(), content_type="image/jpeg")
    except Exception as exc:
        return JsonResponse({"ok": False, "message": str(exc)}, status=500)
