function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "Unknown";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const precision = unitIndex === 0 || value >= 100 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[unitIndex]}`;
}

function shortModelName(displayName) {
  return displayName
    .replace(/^YOLOv\d+\s+/i, "")
    .replace(/\s+License Plate Detector$/i, "");
}

function recommendationLabel(recommendation) {
  const labels = {
    fast_default: "Recommended",
    balanced: "Balanced",
    higher_capacity: "Higher capacity",
  };
  return labels[recommendation] || recommendation.replaceAll("_", " ");
}

function setText(element, value) {
  element.textContent = value;
}

function createDetailRow(label, value) {
  const row = document.createElement("div");
  row.className = "detail-row";
  const term = document.createElement("dt");
  const detail = document.createElement("dd");
  setText(term, label);
  setText(detail, value);
  row.append(term, detail);
  return row;
}

function createCatalogCard(entry, onInstall) {
  const card = document.createElement("article");
  card.className = "catalog-item";
  card.dataset.catalogCard = "";
  card.dataset.catalogId = entry.catalog_id;

  const heading = document.createElement("div");
  heading.className = "catalog-item-heading";
  const titleGroup = document.createElement("div");
  const name = document.createElement("h3");
  const recommendation = document.createElement("span");
  titleGroup.className = "catalog-title-group";
  name.className = "catalog-item-name";
  recommendation.className = "catalog-recommendation";
  if (entry.recommendation === "fast_default") {
    recommendation.classList.add("catalog-recommendation-recommended");
  }
  setText(name, shortModelName(entry.display_name));
  setText(recommendation, recommendationLabel(entry.recommendation));
  titleGroup.append(name);
  heading.append(titleGroup, recommendation);

  const details = document.createElement("dl");
  details.className = "detail-list catalog-details";
  details.append(
    createDetailRow("Size", formatBytes(entry.archive_size)),
    createDetailRow("License", entry.license),
  );

  const status = document.createElement("p");
  status.className = "catalog-item-status";
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  status.hidden = true;

  const footer = document.createElement("div");
  footer.className = "catalog-item-footer";
  const installed = document.createElement("span");
  installed.className = "catalog-installed";
  const action = document.createElement("button");
  action.className = "save-button catalog-install";
  action.type = "button";
  action.dataset.catalogInstall = "";
  action.addEventListener("click", () => onInstall(entry, card, action, installed, status));
  footer.append(installed, action);

  card.append(heading, details, footer, status);
  updateCatalogCard(card, entry, { installed, action, status });
  return card;
}

function updateCatalogCard(card, entry, elements) {
  const { installed, action, status } = elements;
  const isInstalled = entry.installed === true;
  setText(installed, isInstalled ? "Installed" : "Not installed");
  installed.classList.toggle("catalog-installed-true", isInstalled);
  action.hidden = isInstalled;
  action.disabled = isInstalled || entry.install_available === false;
  action.title = entry.install_available === false ? "Install is not available." : "";
  setText(action, "Install");
  card.classList.toggle("catalog-item-installed", isInstalled);
  if (isInstalled) {
    status.hidden = true;
    status.textContent = "";
  }
}

async function responsePayload(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

function errorMessage(payload, fallback) {
  const detail = typeof payload.detail === "string" ? payload.detail.trim() : "";
  return detail || fallback;
}

async function loadCatalog(root, previousStatus = "") {
  const status = root.querySelector("[data-catalog-status]");
  const list = root.querySelector("[data-catalog-list]");
  if (!(status instanceof HTMLElement) || !(list instanceof HTMLElement)) {
    return;
  }

  setText(status, previousStatus || "Loading models...");
  try {
    const response = await fetch("/api/v1/models/catalog", {
      credentials: "same-origin",
    });
    const payload = await responsePayload(response);
    if (!response.ok || !Array.isArray(payload.models)) {
      throw new Error(errorMessage(payload, "catalog loading failed"));
    }

    const cards = payload.models.map((entry) =>
      createCatalogCard(entry, installCatalogModel),
    );
    list.replaceChildren(...cards);
    list.hidden = cards.length === 0;
    setText(status, cards.length ? "" : "No predefined models are available.");
  } catch (error) {
    list.replaceChildren();
    list.hidden = true;
    setText(
      status,
      `Catalog could not be loaded. ${error instanceof Error ? error.message : "Refresh the page and try again."}`,
    );
  }
}

async function installCatalogModel(entry, card, action, installed, status) {
  if (card.dataset.installing === "true" || action.disabled) {
    return;
  }
  card.dataset.installing = "true";
  card.setAttribute("aria-busy", "true");
  action.disabled = true;
  setText(action, "Installing...");
  status.hidden = false;
  setText(status, "Installing...");

  try {
    const response = await fetch(
      `/api/v1/models/catalog/${encodeURIComponent(entry.catalog_id)}/install`,
      {
        method: "POST",
        credentials: "same-origin",
      },
    );
    const payload = await responsePayload(response);
    if (!response.ok) {
      throw new Error(errorMessage(payload, "the install request failed"));
    }
    const root = card.closest("[data-model-catalog]");
    if (!(root instanceof HTMLElement)) {
      throw new Error("the catalog could not be refreshed");
    }
    await loadCatalog(root, "Installed. Refreshing status...");
  } catch (error) {
    card.dataset.installing = "false";
    card.setAttribute("aria-busy", "false");
    action.disabled = false;
    setText(action, "Install");
    status.hidden = false;
    const detail = error instanceof Error ? error.message : "the install request failed";
    setText(status, `Install failed: ${detail}. Try again.`);
  }
}

function initializeCatalogRoots() {
  for (const root of document.querySelectorAll("[data-model-catalog]")) {
    if (root.dataset.catalogInitialized === "true") {
      continue;
    }
    root.dataset.catalogInitialized = "true";
    loadCatalog(root);
  }
}

initializeCatalogRoots();
document.addEventListener("htmx:afterSwap", initializeCatalogRoots);
