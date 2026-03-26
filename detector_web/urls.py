from django.urls import path

from . import views


urlpatterns = [
    path("", views.root_redirect, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("run/<str:task>/", views.run_task, name="run-task"),
    path("stop/", views.stop_task, name="stop-task"),
    path("status/", views.task_status, name="task-status"),
    path("stream/<str:backend>/", views.stream_detection, name="stream-detection"),
]
