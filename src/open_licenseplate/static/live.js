(() => {
  const root = document.querySelector("[data-live-runtime]");
  if (!root) {
    return;
  }

  const cameraSelect = document.querySelector("#live-camera");
  const startButton = document.querySelector("#live-start");
  const stopButton = document.querySelector("#live-stop");
  const message = document.querySelector("#live-message");
  const preview = document.querySelector("#live-preview");
  const previewEmpty = document.querySelector("#live-preview-empty");
  const snapshot = document.querySelector("#live-snapshot");

  const field = (name) => document.querySelector(`#live-${name}`);
  const setMessage = (value, tone = "") => {
    message.textContent = value;
    message.dataset.tone = tone;
  };
  const selectedCamera = () => cameraSelect.value;
  const formatNumber = (value, suffix = "") =>
    typeof value === "number" && Number.isFinite(value)
      ? `${value.toFixed(2)}${suffix}`
      : "Unavailable";

  const updateStatus = (status) => {
    const source = status.source || status.stream_metadata || {};
    const metrics = status.metrics || {};
    const state = status.state || "stopped";
    field("state").textContent = state.charAt(0).toUpperCase() + state.slice(1);
    document.querySelector("#live-preview-title").textContent =
      state === "streaming" ? "Preview is streaming" : `Preview is ${state}`;
    field("codec").textContent = source.codec || "Unavailable";
    field("resolution").textContent = source.resolution || "Unavailable";
    field("fps").textContent = formatNumber(source.nominal_fps);
    field("transport").textContent = source.transport
      ? String(source.transport).toUpperCase()
      : "Unavailable";
    field("pts").textContent =
      source.camera_pts_available === true
        ? "Available"
        : source.camera_pts_available === false
          ? "Unavailable"
          : "Unknown";
    field("reconnect").textContent = String(status.reconnect_attempt || 0);
    field("age").textContent = formatNumber(metrics.frame_age_seconds, " s");
    field("replaced").textContent = String(metrics.replaced_frames || 0);

    if (state === "streaming") {
      preview.hidden = false;
      previewEmpty.hidden = true;
      snapshot.hidden = false;
      snapshot.href = `/api/v1/cameras/${encodeURIComponent(selectedCamera())}/snapshot.jpg`;
      setMessage("The source is streaming. The preview uses the newest decoded frame.", "positive");
    } else {
      preview.hidden = true;
      previewEmpty.hidden = false;
      snapshot.hidden = true;
      if (status.last_error) {
        setMessage(status.last_error, "attention");
      } else if (state === "connecting" || state === "reconnecting") {
        setMessage("Connecting to the camera. The source is not ready yet.");
      } else if (state === "degraded") {
        setMessage("The stream is degraded. Reconnect will continue automatically.", "attention");
      } else {
        setMessage("Select a camera and start the preview.");
      }
    }
  };

  const request = async (path, options = {}) => {
    const response = await fetch(path, {
      cache: "no-store",
      ...options,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Camera request failed.");
    }
    return payload;
  };

  const refresh = async () => {
    if (!selectedCamera()) {
      return;
    }
    try {
      const status = await request(
        `/api/v1/cameras/${encodeURIComponent(selectedCamera())}/status`,
      );
      updateStatus(status);
      if (status.state === "streaming" && preview.src === "") {
        preview.src = `/api/v1/cameras/${encodeURIComponent(selectedCamera())}/preview.mjpeg`;
      }
    } catch (error) {
      setMessage(error.message, "attention");
    }
  };

  startButton.addEventListener("click", async () => {
    try {
      const cameraId = selectedCamera();
      const status = await request(`/api/v1/cameras/${encodeURIComponent(cameraId)}/start`, {
        method: "POST",
      });
      preview.src = `/api/v1/cameras/${encodeURIComponent(cameraId)}/preview.mjpeg`;
      updateStatus(status);
    } catch (error) {
      setMessage(error.message, "attention");
    }
  });

  stopButton.addEventListener("click", async () => {
    try {
      const status = await request(
        `/api/v1/cameras/${encodeURIComponent(selectedCamera())}/stop`,
        { method: "POST" },
      );
      preview.removeAttribute("src");
      updateStatus(status);
    } catch (error) {
      setMessage(error.message, "attention");
    }
  });

  cameraSelect.addEventListener("change", () => {
    preview.removeAttribute("src");
    refresh();
  });

  refresh();
  window.setInterval(refresh, 500);
})();
