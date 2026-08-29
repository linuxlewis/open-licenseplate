const forms = document.querySelectorAll("[data-detection-form]");

for (const form of forms) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const button = form.querySelector("[data-detect-button]");
    const status = form.parentElement.querySelector("[data-detection-status]");
    const result = form.parentElement.querySelector("[data-detection-result]");
    const imageInput = form.querySelector("[data-image-input]");
    if (!(button instanceof HTMLButtonElement) || !(status instanceof HTMLElement)) {
      return;
    }
    if (!(imageInput instanceof HTMLInputElement) || !imageInput.files?.length) {
      status.textContent = "Select a JPEG or PNG image.";
      return;
    }

    button.disabled = true;
    status.textContent = "Running bounded image validation and detection...";
    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        credentials: "same-origin",
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Image detection failed.");
      }
      renderDetectionResult(form.parentElement, payload);
      if (payload.model_reload?.reloaded) {
        status.textContent = "Detection complete. The model reloaded for the new compute units.";
      } else {
        status.textContent = "Detection complete. The displayed boxes use source-image pixels.";
      }
      if (result instanceof HTMLElement) {
        result.hidden = false;
      }
    } catch (error) {
      if (result instanceof HTMLElement) {
        result.hidden = true;
      }
      status.textContent = error instanceof Error ? error.message : "Image detection failed.";
    } finally {
      button.disabled = false;
    }
  });
}

function renderDetectionResult(container, payload) {
  const image = container.querySelector("[data-detection-image]");
  const boxes = container.querySelector("[data-detection-boxes]");
  if (!(image instanceof HTMLImageElement) || !(boxes instanceof HTMLElement)) {
    return;
  }

  image.onload = () => {
    boxes.replaceChildren();
    for (const detection of payload.detections || []) {
      const [x1, y1, x2, y2] = detection.box_xyxy;
      const box = document.createElement("div");
      box.className = "detection-box";
      box.style.left = `${(x1 / payload.source_width) * 100}%`;
      box.style.top = `${(y1 / payload.source_height) * 100}%`;
      box.style.width = `${((x2 - x1) / payload.source_width) * 100}%`;
      box.style.height = `${((y2 - y1) / payload.source_height) * 100}%`;
      box.title = `${detection.label} ${(detection.confidence * 100).toFixed(1)}%`;
      const label = document.createElement("span");
      label.textContent = `${detection.label} ${(detection.confidence * 100).toFixed(1)}%`;
      box.append(label);
      boxes.append(box);
    }
  };
  image.src = `data:${payload.image_content_type};base64,${payload.image_base64}`;

  const metrics = {
    detections: String(payload.detections?.length || 0),
    model_checksum: payload.model_checksum,
    compute_units: payload.compute_units_display || payload.compute_units,
    preprocessing_ms: `${payload.timings.preprocessing_ms} ms`,
    inference_ms: `${payload.timings.inference_ms} ms`,
    postprocessing_ms: `${payload.timings.postprocessing_ms} ms`,
    total_ms: `${payload.timings.total_ms} ms`,
  };
  for (const [key, value] of Object.entries(metrics)) {
    const target = container.querySelector(`[data-metric="${key}"]`);
    if (target) {
      target.textContent = value;
    }
  }
}
