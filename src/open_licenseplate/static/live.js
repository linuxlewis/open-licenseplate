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
  let activeGeneration = null;
  let activeEpoch = null;
  let activeCaptureSession = null;
  const retiredProvenances = [];
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
  const maxDimension = 8192;
  const maxDetections = 256;
  const maxMetadataMetrics = 64;

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
    document.querySelector("#live-processed-epoch").textContent = header.stream_epoch;
    document.querySelector("#live-processed-session").textContent = header.capture_session_id;
    document.querySelector("#live-processed-active-tracks").textContent = formatCount(
      header.active_tracks?.length,
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
    context.setLineDash([6, 4]);
    context.strokeStyle = "#7dd3c7";
    context.fillStyle = "#7dd3c7";
    for (const track of currentHeader.active_tracks || []) {
      const box = track.last_box_xyxy;
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
      context.fillStyle = "#7dd3c7";
      context.fillText(`track ${track.track_id}`, x1 + 4, Math.max(13, y1 + 14));
    }
    context.setLineDash([]);
  };

  const validateState = (state) => {
    if (
      !state ||
      state.type !== "state" ||
      state.protocol_version !== 1 ||
      !["starting", "warming", "running", "stopping", "stopped", "failed", "shutdown"].includes(
        state.state,
      )
    ) {
      throw new Error("The processed display protocol is invalid.");
    }
  };

  const validateDetection = (detection, header) => {
    if (
      !detection ||
      !Array.isArray(detection.box_xyxy) ||
      detection.box_xyxy.length !== 4 ||
      typeof detection.label !== "string" ||
      detection.label.length === 0 ||
      typeof detection.model_id !== "string" ||
      detection.model_id !== header.model_id ||
      typeof detection.model_checksum !== "string" ||
      detection.model_checksum !== header.model_checksum ||
      !Number.isInteger(detection.frame_sequence) ||
      detection.frame_sequence !== header.frame_sequence ||
      typeof detection.confidence !== "number" ||
      !Number.isFinite(detection.confidence) ||
      detection.confidence < 0 ||
      detection.confidence > 1
    ) {
      throw new Error("The processed detection provenance is invalid.");
    }
    const [x1, y1, x2, y2] = detection.box_xyxy.map(Number);
    if (
      ![x1, y1, x2, y2].every(Number.isFinite) ||
      x1 < 0 ||
      y1 < 0 ||
      x1 > x2 ||
      y1 > y2 ||
      x2 > header.source_width ||
      y2 > header.source_height
    ) {
      throw new Error("The processed detection geometry is invalid.");
    }
  };

  const validateRoi = (roi, header) => {
    if (roi === null) {
      return;
    }
    if (
      !roi ||
      Object.keys(roi).sort().join(",") !== "height,width,x,y" ||
      !["x", "y", "width", "height"].every((key) => Number.isInteger(roi[key])) ||
      roi.x < 0 ||
      roi.y < 0 ||
      roi.width <= 0 ||
      roi.height <= 0 ||
      roi.x + roi.width > header.source_width ||
      roi.y + roi.height > header.source_height
    ) {
      throw new Error("The processed ROI is invalid.");
    }
  };

  const validateMetrics = (metrics) => {
    if (!metrics || typeof metrics !== "object" || Array.isArray(metrics)) {
      throw new Error("The processed metrics are invalid.");
    }
    const names = Object.keys(metrics);
    if (names.length > maxMetadataMetrics) {
      throw new Error("The processed metrics are too large.");
    }
    for (const value of Object.values(metrics)) {
      if (
        value !== null &&
        (typeof value !== "number" || !Number.isFinite(value) || typeof value === "boolean")
      ) {
        throw new Error("The processed metrics are invalid.");
      }
    }
  };

  const validateActiveTrack = (track, header) => {
    if (
      !track ||
      typeof track.camera_id !== "string" ||
      track.camera_id.length === 0 ||
      track.capture_session_id !== header.capture_session_id ||
      track.generation_number !== header.generation_number ||
      track.stream_epoch !== header.stream_epoch ||
      track.model_id !== header.model_id ||
      track.model_checksum !== header.model_checksum ||
      !["confirmed", "active"].includes(track.state) ||
      !Number.isInteger(track.track_id) ||
      track.track_id < 0 ||
      !Array.isArray(track.last_box_xyxy) ||
      track.last_box_xyxy.length !== 4 ||
      !Number.isInteger(track.last_frame_sequence) ||
      track.last_frame_sequence < 0 ||
      !Number.isInteger(track.observation_count) ||
      track.observation_count < 1 ||
      !["last_confidence", "maximum_confidence"].every(
        (name) =>
          typeof track[name] === "number" &&
          Number.isFinite(track[name]) &&
          track[name] >= 0 &&
          track[name] <= 1,
      ) ||
      !["first_seen_utc", "last_seen_utc"].every(
        (name) => typeof track[name] === "string" && track[name].length > 0,
      )
    ) {
      throw new Error("The processed active-track provenance is invalid.");
    }
    const [x1, y1, x2, y2] = track.last_box_xyxy.map(Number);
    if (
      ![x1, y1, x2, y2].every(Number.isFinite) ||
      x1 < 0 ||
      y1 < 0 ||
      x1 >= x2 ||
      y1 >= y2 ||
      x2 > header.source_width ||
      y2 > header.source_height
    ) {
      throw new Error("The processed active-track geometry is invalid.");
    }
  };

  const validateHeader = (header) => {
    if (
      !header ||
      typeof header !== "object" ||
      header.type !== "frame_header" ||
      header.message_type !== "frame_header" ||
      header.protocol_version !== 1 ||
      !Number.isInteger(header.generation_number) ||
      header.generation_number < 0 ||
      typeof header.camera_id !== "string" ||
      header.camera_id.length === 0 ||
      typeof header.model_id !== "string" ||
      header.model_id.length === 0 ||
      typeof header.model_checksum !== "string" ||
      header.model_checksum.length === 0 ||
      typeof header.capture_session_id !== "string" ||
      header.capture_session_id.length === 0 ||
      typeof header.stream_epoch !== "string" ||
      header.stream_epoch.length === 0 ||
      !Number.isInteger(header.frame_sequence) ||
      !Number.isInteger(header.jpeg_byte_count) ||
      header.frame_sequence < 0 ||
      header.jpeg_byte_count <= 0 ||
      header.jpeg_byte_count > 4 * 1024 * 1024
    ) {
      throw new Error("The processed display protocol is invalid.");
    }
    if (header.active_tracks === undefined) {
      header.active_tracks = [];
    }
    for (const name of ["source_width", "source_height", "jpeg_width", "jpeg_height"]) {
      if (
        !Number.isInteger(header[name]) ||
        header[name] <= 0 ||
        header[name] > maxDimension
      ) {
        throw new Error("The processed frame geometry is invalid.");
      }
    }
    if (
      typeof header.captured_at_utc !== "string" ||
      header.captured_at_utc.length === 0 ||
      header.capture_timestamp !== header.captured_at_utc
    ) {
      throw new Error("The processed frame timestamp is invalid.");
    }
    if (
      typeof header.confidence_threshold !== "number" ||
      !Number.isFinite(header.confidence_threshold) ||
      header.confidence_threshold < 0 ||
      header.confidence_threshold > 1 ||
      header.threshold !== header.confidence_threshold
    ) {
      throw new Error("The processed frame threshold is invalid.");
    }
    if (!Array.isArray(header.detections) || header.detections.length > maxDetections) {
      throw new Error("The processed detections are invalid.");
    }
    if (JSON.stringify(header.roi) !== JSON.stringify(header.region_of_interest)) {
      throw new Error("The processed ROI pairing is invalid.");
    }
    validateRoi(header.region_of_interest, header);
    validateMetrics(header.metrics);
    for (const detection of header.detections) {
      validateDetection(detection, header);
    }
    if (!Array.isArray(header.active_tracks) || header.active_tracks.length > maxDetections) {
      throw new Error("The processed active tracks are invalid.");
    }
    const trackIds = new Set();
    for (const track of header.active_tracks) {
      validateActiveTrack(track, header);
      if (trackIds.has(track.track_id)) {
        throw new Error("The processed active track IDs are invalid.");
      }
      trackIds.add(track.track_id);
    }
  };

  const protocolError = (error, socket = displaySocket) => {
    pendingHeader = null;
    setMessage(error, "attention");
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.close(1008, "invalid protocol");
    }
  };

  const acceptProvenance = (header) => {
    const provenance = {
      generation: header.generation_number,
      epoch: header.stream_epoch,
      captureSession: header.capture_session_id,
    };
    if (
      retiredProvenances.some(
        (retired) =>
          retired.generation === provenance.generation &&
          retired.epoch === provenance.epoch &&
          retired.captureSession === provenance.captureSession,
      )
    ) {
      return false;
    }
    if (activeEpoch === null) {
      activeGeneration = provenance.generation;
      activeEpoch = provenance.epoch;
      activeCaptureSession = provenance.captureSession;
      lastSequence = -1;
      return true;
    }
    if (
      activeGeneration === provenance.generation &&
      activeEpoch === provenance.epoch &&
      activeCaptureSession === provenance.captureSession
    ) {
      return true;
    }
    if (provenance.generation < activeGeneration) {
      return false;
    }
    if (
      provenance.epoch === activeEpoch ||
      provenance.captureSession === activeCaptureSession
    ) {
      throw new Error("The processed epoch and capture session boundary is invalid.");
    }
    if (
      provenance.generation === activeGeneration ||
      provenance.generation > activeGeneration
    ) {
      retiredProvenances.push({
        generation: activeGeneration,
        epoch: activeEpoch,
        captureSession: activeCaptureSession,
      });
      while (retiredProvenances.length > 16) {
        retiredProvenances.shift();
      }
      activeGeneration = provenance.generation;
      activeEpoch = provenance.epoch;
      activeCaptureSession = provenance.captureSession;
      lastSequence = -1;
      return true;
    }
    return false;
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
    activeGeneration = null;
    activeEpoch = null;
    activeCaptureSession = null;
    retiredProvenances.length = 0;
    lastSequence = -1;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(
      `${protocol}//${window.location.host}/api/v1/live/ws`,
    );
    displaySocket = socket;
    socket.binaryType = "arraybuffer";
    socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        let header;
        try {
          header = JSON.parse(event.data);
          if (header.type === "state") {
            if (pendingHeader !== null) {
              throw new Error("A state message interrupted a display unit.");
            }
            validateState(header);
            setProcessedState(header.state || "stopped");
            return;
          }
          if (pendingHeader !== null) {
            throw new Error("Two display headers arrived without a JPEG.");
          }
          validateHeader(header);
        } catch (error) {
          protocolError(error.message, socket);
          return;
        }
        pendingHeader = header;
        return;
      }
      if (!pendingHeader) {
        protocolError("A binary message arrived without a display header.", socket);
        return;
      }
      const header = pendingHeader;
      pendingHeader = null;
      if (event.data.byteLength !== header.jpeg_byte_count) {
        protocolError("The processed JPEG did not match its metadata.", socket);
        return;
      }
      let currentProvenance;
      try {
        currentProvenance = acceptProvenance(header);
      } catch (error) {
        protocolError(error.message, socket);
        return;
      }
      if (!currentProvenance) {
        return;
      }
      if (header.frame_sequence <= lastSequence) {
        return;
      }
      lastSequence = header.frame_sequence;
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
        if (
          processedPreview.naturalWidth !== header.jpeg_width ||
          processedPreview.naturalHeight !== header.jpeg_height
        ) {
          protocolError("The processed JPEG geometry does not match its metadata.", socket);
          URL.revokeObjectURL(objectUrl);
          return;
        }
        currentHeader = header;
        updateProcessedMetrics(header);
        processedEmpty.hidden = true;
        processedFrame.hidden = false;
        drawOverlay();
        URL.revokeObjectURL(objectUrl);
      };
      processedPreview.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        protocolError("The processed JPEG could not be displayed.", socket);
      };
      processedPreview.src = objectUrl;
    };
    socket.onerror = () => {
      setMessage("The synchronized processed preview is unavailable.", "attention");
    };
    socket.onclose = () => {
      if (displaySocket === socket) {
        displaySocket = null;
      }
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
