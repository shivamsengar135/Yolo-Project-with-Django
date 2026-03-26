from __future__ import annotations

import time
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


def _clean_value(raw: Any, fallback: str) -> str:
    value = str(raw or "").strip()
    return value if value else fallback


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
