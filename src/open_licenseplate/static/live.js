(() => {
  const root = document.querySelector("[data-live-runtime]");
  if (!root) {
    return;
  }

  const cameraSelect = document.querySelector("#live-camera");
  const modelSelect = document.querySelector("#live-model");
  const computeSelect = document.querySelector("#live-compute");
  const threshold = document.querySelector("#live-threshold");
  const thresholdValue = document.querySelector("#live-threshold-value");
  const startDetectionButton = document.querySelector("#live-start-detection");
  const stopDetectionButton = document.querySelector("#live-stop-detection");
  const startPreviewButton = document.querySelector("#live-start");
  const stopPreviewButton = document.querySelector("#live-stop");
  const message = document.querySelector("#live-message");
  const preview = document.querySelector("#live-preview");
  const previewEmpty = document.querySelector("#live-preview-empty");
  const snapshot = document.querySelector("#live-snapshot");
  const processedEmpty = document.querySelector("#live-processed-empty");
  const processedFrame = document.querySelector("#live-processed-frame");
  const processedPreview = document.querySelector("#live-processed-preview");
  const overlayEnabled = document.querySelector("#live-overlay-enabled");

  let displaySocket = null;
  let pendingHeader = null;
  let activeEpoch = null;
  let lastSequence = -1;
  let currentHeader = null;
  let currentObjectUrl = null;
  let overlayCanvas = null;
  let detectionActive = false;

  const field = (name) => document.querySelector(`#live-${name}`);
  const setMessage = (value, tone = "") => {
    message.textContent = value;
    message.dataset.tone = tone;
  };
  const selectedCamera = () => cameraSelect.value;
  const selectedModel = () => modelSelect.value;
  const formatNumber = (value, suffix = "") =>
    typeof value === "number" && Number.isFinite(value)
      ? `${value.toFixed(2)}${suffix}`
      : "Unavailable";
  const formatCount = (value) =>
    typeof value === "number" && Number.isFinite(value) ? String(Math.max(0, value)) : "0";

  const request = async (path, options = {}) => {
    const response = await fetch(path, {
      cache: "no-store",
      ...options,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Live request failed.");
    }
    return payload;
  };

  const setProcessedState = (state) => {
    const title = document.querySelector("#live-processed-title");
    const stateField = document.querySelector("#live-processed-state");
    const label = state === "stopped" ? "Idle" : state.charAt(0).toUpperCase() + state.slice(1);
    stateField.textContent = label;
    title.textContent = state === "running" ? "Detection is running" : `Detection is ${state}`;
  };

  const updateRawStatus = (status) => {
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
    field("replaced").textContent = formatCount(metrics.replaced_frames);

    if (state === "streaming" && preview.hasAttribute("src")) {
      preview.hidden = false;
      previewEmpty.hidden = true;
      snapshot.hidden = false;
      snapshot.href = `/api/v1/cameras/${encodeURIComponent(selectedCamera())}/snapshot.jpg`;
    } else if (state !== "streaming") {
      preview.hidden = true;
      previewEmpty.hidden = false;
      snapshot.hidden = true;
    }
  };

  const updateProcessedMetrics = (header) => {
    const metrics = header.metrics || {};
    document.querySelector("#live-processed-fps").textContent =
      formatNumber(metrics.processed_fps, " fps");
    document.querySelector("#live-processed-age").textContent =
      formatNumber(metrics.capture_age_ms, " ms");
    document.querySelector("#live-processed-preprocessing").textContent =
      formatNumber(metrics.preprocessing_ms, " ms");
    document.querySelector("#live-processed-prediction").textContent =
      formatNumber(metrics.prediction_ms, " ms");
    document.querySelector("#live-processed-p50").textContent =
      formatNumber(metrics.prediction_p50_ms, " ms");
    document.querySelector("#live-processed-p95").textContent =
      formatNumber(metrics.prediction_p95_ms, " ms");
    document.querySelector("#live-processed-postprocessing").textContent =
      formatNumber(metrics.postprocessing_ms, " ms");
    document.querySelector("#live-processed-e2e").textContent =
      formatNumber(metrics.end_to_end_ms, " ms");
    document.querySelector("#live-processed-source-replaced").textContent = formatCount(
      metrics.source_replacement_count ?? metrics.source_replaced_frames,
    );
    document.querySelector("#live-processed-inference-replaced").textContent = formatCount(
      metrics.inference_replacement_count ?? metrics.inference_replaced_frames,
    );
    document.querySelector("#live-processed-display-replaced").textContent = formatCount(
      metrics.display_replacement_count ?? metrics.display_replaced_frames,
    );
    document.querySelector("#live-processed-sequence").textContent = String(
      header.frame_sequence,
    );
  };

  const ensureOverlayCanvas = () => {
    if (overlayCanvas) {
      return overlayCanvas;
    }
    overlayCanvas = document.createElement("canvas");
    overlayCanvas.id = "live-overlay";
    overlayCanvas.setAttribute("aria-label", "Detection overlay");
    processedFrame.appendChild(overlayCanvas);
    return overlayCanvas;
  };

  const drawOverlay = () => {
    if (!currentHeader || !processedPreview.complete || !processedPreview.naturalWidth) {
      return;
    }
    const canvas = ensureOverlayCanvas();
    const bounds = processedPreview.getBoundingClientRect();
    const sourceWidth = Number(currentHeader.source_width);
    const sourceHeight = Number(currentHeader.source_height);
    const jpegWidth = Number(currentHeader.jpeg_width);
    const jpegHeight = Number(currentHeader.jpeg_height);
    if (
      ![sourceWidth, sourceHeight, jpegWidth, jpegHeight].every(
        (value) => Number.isFinite(value) && value > 0,
      )
    ) {
      return;
    }
    const devicePixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(bounds.width * devicePixelRatio));
    canvas.height = Math.max(1, Math.round(bounds.height * devicePixelRatio));
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
    if (!overlayEnabled.checked) {
      return;
    }
    context.lineWidth = 2;
    context.strokeStyle = "#f5ca68";
    context.fillStyle = "#f5ca68";
    context.font = "700 12px sans-serif";
    for (const detection of currentHeader.detections || []) {
      const box = detection.box_xyxy;
      if (!Array.isArray(box) || box.length !== 4) {
        continue;
      }
      const x1 = (Number(box[0]) / sourceWidth) * jpegWidth * (bounds.width / jpegWidth);
      const y1 = (Number(box[1]) / sourceHeight) * jpegHeight * (bounds.height / jpegHeight);
      const x2 = (Number(box[2]) / sourceWidth) * jpegWidth * (bounds.width / jpegWidth);
      const y2 = (Number(box[3]) / sourceHeight) * jpegHeight * (bounds.height / jpegHeight);
      if (![x1, y1, x2, y2].every((value) => Number.isFinite(value))) {
        continue;
      }
      context.strokeRect(x1, y1, Math.max(0, x2 - x1), Math.max(0, y2 - y1));
      const label = `${detection.label || "detection"} ${(
        Number(detection.confidence || 0) * 100
      ).toFixed(0)}%`;
      const textWidth = context.measureText(label).width + 8;
      context.fillRect(x1, Math.max(0, y1 - 18), textWidth, 18);
      context.fillStyle = "#172327";
      context.fillText(label, x1 + 4, Math.max(13, y1 - 5));
      context.fillStyle = "#f5ca68";
    }
  };

  const validateHeader = (header) => {
    if (
      !header ||
      header.type !== "frame_header" ||
      header.message_type !== "frame_header" ||
      header.protocol_version !== 1 ||
      typeof header.stream_epoch !== "string" ||
      !Number.isInteger(header.frame_sequence) ||
      !Number.isInteger(header.jpeg_byte_count) ||
      header.frame_sequence < 0 ||
      header.jpeg_byte_count <= 0 ||
      header.jpeg_byte_count > 4 * 1024 * 1024
    ) {
      throw new Error("The processed display protocol is invalid.");
    }
  };

  const closeDisplaySocket = () => {
    pendingHeader = null;
    if (displaySocket) {
      displaySocket.close(1000, "client stopped");
      displaySocket = null;
    }
  };

  const openDisplaySocket = () => {
    closeDisplaySocket();
    activeEpoch = null;
    lastSequence = -1;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    displaySocket = new WebSocket(
      `${protocol}//${window.location.host}/api/v1/live/ws`,
    );
    displaySocket.binaryType = "arraybuffer";
    displaySocket.onmessage = (event) => {
      if (typeof event.data === "string") {
        let header;
        try {
          header = JSON.parse(event.data);
          if (header.type === "state") {
            pendingHeader = null;
            setProcessedState(header.state || "stopped");
            return;
          }
          validateHeader(header);
        } catch (error) {
          pendingHeader = null;
          setMessage(error.message, "attention");
          displaySocket.close(1008, "invalid protocol");
          return;
        }
        pendingHeader = header;
        return;
      }
      if (!pendingHeader) {
        return;
      }
      const header = pendingHeader;
      pendingHeader = null;
      if (event.data.byteLength !== header.jpeg_byte_count) {
        setMessage("The processed JPEG did not match its metadata.", "attention");
        displaySocket.close(1008, "invalid frame pairing");
        return;
      }
      if (activeEpoch !== null && activeEpoch !== header.stream_epoch) {
        return;
      }
      if (activeEpoch !== header.stream_epoch) {
        activeEpoch = header.stream_epoch;
        lastSequence = -1;
      }
      if (header.frame_sequence <= lastSequence) {
        return;
      }
      lastSequence = header.frame_sequence;
      currentHeader = header;
      updateProcessedMetrics(header);
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const objectUrl = URL.createObjectURL(blob);
      const previousObjectUrl = currentObjectUrl;
      currentObjectUrl = objectUrl;
      processedPreview.onload = () => {
        if (currentObjectUrl !== objectUrl) {
          URL.revokeObjectURL(objectUrl);
          return;
        }
        if (previousObjectUrl) {
          URL.revokeObjectURL(previousObjectUrl);
        }
        processedEmpty.hidden = true;
        processedFrame.hidden = false;
        drawOverlay();
        URL.revokeObjectURL(objectUrl);
      };
      processedPreview.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        setMessage("The processed JPEG could not be displayed.", "attention");
      };
      processedPreview.src = objectUrl;
    };
    displaySocket.onerror = () => {
      setMessage("The synchronized processed preview is unavailable.", "attention");
    };
    displaySocket.onclose = () => {
      displaySocket = null;
      pendingHeader = null;
    };
  };

  const updateLiveStatus = (status) => {
    const state = status.state || "stopped";
    const metrics = status.metrics || {};
    detectionActive = ["starting", "warming", "running"].includes(state);
    setProcessedState(state);
    if (typeof status.confidence_threshold === "number") {
      threshold.value = String(status.confidence_threshold);
      thresholdValue.textContent = status.confidence_threshold.toFixed(2);
    }
    document.querySelector("#live-processed-fps").textContent =
      formatNumber(metrics.processed_fps, " fps");
    document.querySelector("#live-processed-age").textContent =
      formatNumber(metrics.capture_age_ms, " ms");
    document.querySelector("#live-processed-prediction").textContent =
      formatNumber(metrics.prediction_ms, " ms");
    document.querySelector("#live-processed-source-replaced").textContent = formatCount(
      metrics.source_replacement_count,
    );
    document.querySelector("#live-processed-inference-replaced").textContent = formatCount(
      metrics.inference_replacement_count,
    );
    document.querySelector("#live-processed-display-replaced").textContent = formatCount(
      metrics.display_replacement_count,
    );
    if (state === "failed" && status.failure) {
      setMessage(`Detection failed: ${status.failure.message}`, "attention");
    }
  };

  const refresh = async () => {
    if (selectedCamera()) {
      try {
        updateRawStatus(
          await request(`/api/v1/cameras/${encodeURIComponent(selectedCamera())}/status`),
        );
      } catch (error) {
        setMessage(error.message, "attention");
      }
    }
    try {
      updateLiveStatus(await request("/api/v1/live/state"));
    } catch (error) {
      setMessage(error.message, "attention");
    }
  };

  threshold.addEventListener("input", () => {
    thresholdValue.textContent = Number(threshold.value).toFixed(2);
  });

  threshold.addEventListener("change", async () => {
    thresholdValue.textContent = Number(threshold.value).toFixed(2);
    if (!detectionActive) {
      return;
    }
    try {
      await request("/api/v1/live/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confidence_threshold: Number(threshold.value) }),
      });
      setMessage("Confidence threshold updated without reloading the model.", "positive");
    } catch (error) {
      setMessage(error.message, "attention");
    }
  });

  startDetectionButton.addEventListener("click", async () => {
    try {
      const status = await request("/api/v1/live/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          camera_id: selectedCamera(),
          model_id: selectedModel(),
          confidence_threshold: Number(threshold.value),
          compute_units: computeSelect.value,
        }),
      });
      updateLiveStatus(status);
      detectionActive = true;
      setMessage("Starting detection. Waiting for the first synchronized processed frame.");
      openDisplaySocket();
    } catch (error) {
      setMessage(error.message, "attention");
    }
  });

  stopDetectionButton.addEventListener("click", async () => {
    try {
      const status = await request("/api/v1/live/stop", { method: "POST" });
      detectionActive = false;
      closeDisplaySocket();
      updateLiveStatus(status);
      if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
      }
      processedPreview.removeAttribute("src");
      currentHeader = null;
      if (overlayCanvas) {
        overlayCanvas.remove();
        overlayCanvas = null;
      }
      processedFrame.hidden = true;
      processedEmpty.hidden = false;
      setMessage("Detection stopped. Camera and model resources are released.", "positive");
    } catch (error) {
      setMessage(error.message, "attention");
    }
  });

  startPreviewButton.addEventListener("click", async () => {
    try {
      const cameraId = selectedCamera();
      const status = await request(`/api/v1/cameras/${encodeURIComponent(cameraId)}/start`, {
        method: "POST",
      });
      preview.src = `/api/v1/cameras/${encodeURIComponent(cameraId)}/preview.mjpeg`;
      updateRawStatus(status);
      setMessage(
        "The raw camera preview is streaming. It is separate from detection overlays.",
        "positive",
      );
    } catch (error) {
      setMessage(error.message, "attention");
    }
  });

  stopPreviewButton.addEventListener("click", async () => {
    try {
      const status = await request(
        `/api/v1/cameras/${encodeURIComponent(selectedCamera())}/stop`,
        { method: "POST" },
      );
      preview.removeAttribute("src");
      updateRawStatus(status);
    } catch (error) {
      setMessage(error.message, "attention");
    }
  });

  cameraSelect.addEventListener("change", () => {
    preview.removeAttribute("src");
    refresh();
  });

  overlayEnabled.addEventListener("change", drawOverlay);
  window.addEventListener("resize", drawOverlay);
  if (window.ResizeObserver) {
    new ResizeObserver(drawOverlay).observe(processedFrame);
  }

  refresh();
  window.setInterval(refresh, 500);
})();
