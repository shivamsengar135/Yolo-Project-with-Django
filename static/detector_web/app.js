function startStream() {
    const backend = document.getElementById("stream-backend").value;
    const modelInput = document.getElementById("stream-model");
    const model = modelInput.value.trim() || "yolov8n.pt";
    const conf = document.getElementById("stream-conf").value || "0.4";
    const source = document.getElementById("source").value || "auto";
    const device = document.getElementById("device").value || "auto";
    const status = document.getElementById("stream-status");
    const empty = document.getElementById("stream-empty");

    const params = new URLSearchParams();
    params.set("model", backend === "ssd" ? "ssd" : model);
    params.set("imgsz", "640");
    params.set("conf", conf);
    params.set("source", source);
    params.set("device", device);
    params.set("_t", String(Date.now()));

    const frame = document.getElementById("stream-frame");
    if (status) {
        status.textContent = "Connecting...";
        status.classList.remove("live");
    }
    if (empty) {
        empty.style.display = "none";
    }
    frame.onload = () => {
        if (status) {
            status.textContent = "Live";
            status.classList.add("live");
        }
    };
    frame.onerror = () => {
        if (status) {
            status.textContent = "Stream Error";
            status.classList.remove("live");
        }
    };
    frame.src = `/stream/${backend}/?${params.toString()}`;
}

function stopStream() {
    const frame = document.getElementById("stream-frame");
    const status = document.getElementById("stream-status");
    const empty = document.getElementById("stream-empty");
    frame.src = "";
    if (status) {
        status.textContent = "Idle";
        status.classList.remove("live");
    }
    if (empty) {
        empty.style.display = "block";
    }
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
