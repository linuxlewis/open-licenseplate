function setText(element, value) {
  element.textContent = value;
}

function createCatalogCard(entry, onInstall) {
  const card = document.createElement("article");
  card.className = "catalog-item";
  card.dataset.catalogCard = "";
  card.dataset.catalogId = entry.catalog_id;

  const heading = document.createElement("div");
  heading.className = "catalog-item-heading";
  const name = document.createElement("h3");
  const recommendation = document.createElement("span");
  name.className = "catalog-item-name";
  recommendation.className = "catalog-recommendation";
  setText(name, "YOLO license plate model");
  setText(recommendation, "Recommended");
  recommendation.classList.add("catalog-recommendation-recommended");
  heading.append(name, recommendation);

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

  card.append(heading, footer, status);
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

async function loadCatalog(root, previousStatus = "") {
  const status = root.querySelector("[data-catalog-status]");
  const list = root.querySelector("[data-catalog-list]");
  if (!(status instanceof HTMLElement) || !(list instanceof HTMLElement)) {
    return;
  }

  setText(status, previousStatus || "Loading model...");
  try {
    const response = await fetch("/api/v1/models/catalog", {
      credentials: "same-origin",
    });
    const payload = await responsePayload(response);
    if (!response.ok || !Array.isArray(payload.models)) {
      throw new Error("catalog request failed");
    }

    const entry = payload.models.find((model) => model.recommendation === "fast_default");
    if (!entry) {
      list.replaceChildren();
      list.hidden = true;
      setText(status, "No recommended model is available.");
      return;
    }

    list.replaceChildren(createCatalogCard(entry, installCatalogModel));
    list.hidden = false;
    setText(status, "");
  } catch {
    list.replaceChildren();
    list.hidden = true;
    setText(status, "Catalog unavailable. Refresh the page and try again.");
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
    if (!response.ok) {
      throw new Error("install request failed");
    }
    const root = card.closest("[data-model-catalog]");
    if (!(root instanceof HTMLElement)) {
      throw new Error("the catalog could not be refreshed");
    }
    await loadCatalog(root, "Refreshing...");
  } catch {
    card.dataset.installing = "false";
    card.setAttribute("aria-busy", "false");
    action.disabled = false;
    setText(action, "Install");
    status.hidden = false;
    setText(status, "Install failed. Try again.");
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
