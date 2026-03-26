let localCameraStream = null;
let captureVideo = null;
let captureCanvas = null;
let captureTimer = null;
let inferenceBusy = false;
let currentFrameUrl = null;

function setStatus(text, live = false) {
    const status = document.getElementById("stream-status");
    if (!status) {
        return;
    }
    status.textContent = text;
    status.classList.toggle("live", live);
}

function showEmpty(show) {
    const empty = document.getElementById("stream-empty");
    if (!empty) {
        return;
    }
    empty.style.display = show ? "block" : "none";
}

async function startStream() {
    const backendSelect = document.getElementById("stream-backend");
    if (backendSelect.value !== "yolo") {
        alert("Hosted browser-camera mode supports YOLO backend. Switching to YOLO.");
        backendSelect.value = "yolo";
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Your browser does not support webcam access.");
        return;
    }

    stopStream();
    setStatus("Requesting Camera...");

    try {
        localCameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false,
        });
    } catch (error) {
        setStatus("Camera Permission Denied");
        showEmpty(true);
        return;
    }

    captureVideo = document.createElement("video");
    captureVideo.srcObject = localCameraStream;
    captureVideo.autoplay = true;
    captureVideo.muted = true;
    captureVideo.playsInline = true;
    await captureVideo.play();

    captureCanvas = document.createElement("canvas");
    showEmpty(false);
    setStatus("Loading Model (wait a few seconds)...");
    captureTimer = setInterval(captureAndInfer, 1400);
}

async function captureAndInfer() {
    if (inferenceBusy || !captureVideo || !captureCanvas || !localCameraStream) {
        return;
    }
    if (captureVideo.videoWidth < 2 || captureVideo.videoHeight < 2) {
        return;
    }

    inferenceBusy = true;
    const ctx = captureCanvas.getContext("2d");
    captureCanvas.width = captureVideo.videoWidth;
    captureCanvas.height = captureVideo.videoHeight;
    ctx.drawImage(captureVideo, 0, 0, captureCanvas.width, captureCanvas.height);

    const blob = await new Promise((resolve) => {
        captureCanvas.toBlob(resolve, "image/jpeg", 0.85);
    });
    if (!blob) {
        inferenceBusy = false;
        return;
    }

    const model = (document.getElementById("stream-model").value || "yolov8n.pt").trim();
    const conf = document.getElementById("stream-conf").value || "0.4";
    const device = document.getElementById("device").value || "cpu";

    const form = new FormData();
    form.append("frame", blob, "frame.jpg");
    form.append("backend", "yolo");
    form.append("model", model);
    form.append("conf", conf);
    form.append("imgsz", "320");
    form.append("device", device);

    try {
        const response = await fetch("/infer/frame/", {
            method: "POST",
            body: form,
        });
        if (!response.ok) {
            const payload = await response.json().catch(() => ({ message: "Inference failed." }));
            setStatus(payload.message || "Inference Error");
            inferenceBusy = false;
            return;
        }

        const imgBlob = await response.blob();
        const frame = document.getElementById("stream-frame");
        const nextUrl = URL.createObjectURL(imgBlob);
        frame.src = nextUrl;
        if (currentFrameUrl) {
            URL.revokeObjectURL(currentFrameUrl);
        }
        currentFrameUrl = nextUrl;
        setStatus("Live", true);
    } catch (error) {
        setStatus("Network Error");
    } finally {
        inferenceBusy = false;
    }
}

function stopStream() {
    if (captureTimer) {
        clearInterval(captureTimer);
        captureTimer = null;
    }
    if (localCameraStream) {
        localCameraStream.getTracks().forEach((track) => track.stop());
        localCameraStream = null;
    }
    if (captureVideo) {
        captureVideo.srcObject = null;
        captureVideo = null;
    }
    captureCanvas = null;
    inferenceBusy = false;

    const frame = document.getElementById("stream-frame");
    frame.src = "";
    if (currentFrameUrl) {
        URL.revokeObjectURL(currentFrameUrl);
        currentFrameUrl = null;
    }
    setStatus("Idle");
    showEmpty(true);
}

function formatRemaining(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function startCountdown() {
    const root = document.body;
    const expiresAt = Number(root.dataset.expiresAt || "0");
    const countdown = document.getElementById("countdown");
    if (!countdown || !expiresAt) {
        return;
    }

    const tick = () => {
        const now = Math.floor(Date.now() / 1000);
        const remaining = Math.max(0, expiresAt - now);
        countdown.textContent = formatRemaining(remaining);
        if (remaining <= 0) {
            stopStream();
            window.location.href = "/logout/";
        }
    };

    tick();
    setInterval(tick, 1000);
}

function bindEvents() {
    const startBtn = document.getElementById("start-stream-btn");
    const stopBtn = document.getElementById("stop-stream-btn");
    if (startBtn) {
        startBtn.addEventListener("click", startStream);
    }
    if (stopBtn) {
        stopBtn.addEventListener("click", stopStream);
    }
    document.querySelectorAll(".preset-btn").forEach((button) => {
        button.addEventListener("click", () => {
            const backend = button.dataset.backend;
            const model = button.dataset.model;
            const conf = button.dataset.conf;
            document.getElementById("stream-backend").value = backend;
            document.getElementById("stream-conf").value = conf;
            if (backend === "ssd") {
                document.getElementById("stream-model").value = "ssd";
            } else {
                document.getElementById("stream-model").value = model;
            }
        });
    });
}

bindEvents();
startCountdown();
