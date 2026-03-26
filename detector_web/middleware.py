from __future__ import annotations

import time

from django.contrib.auth import logout
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect


class SessionTimeoutMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if request.user.is_authenticated:
            expires_at = request.session.get("login_expires_at")
            if expires_at is not None and time.time() >= float(expires_at):
                logout(request)
                if request.path.startswith("/stream/") or request.path.startswith("/status/"):
                    return HttpResponse(status=401)
                return redirect("login")
        return self.get_response(request)
