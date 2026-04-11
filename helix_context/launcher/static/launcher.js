/*
 * Helix Launcher — tiny reactivity layer.
 *
 * Two jobs:
 *   1. Poll /api/state/panels every N ms and replace the panels content
 *   2. Wire Start/Restart/Stop buttons to POST /api/control/{action}
 *
 * No framework, no vendored library. Pure DOM.
 *
 * Note on HTML parsing: the panels partial is rendered server-side by
 * Jinja2 with autoescape enabled for html+xml. All interpolated values
 * are auto-escaped, so the returned HTML is trusted. We still use
 * DOMParser + replaceChildren (not innerHTML =) as defense in depth.
 */

(function () {
  "use strict";

  const panels = document.getElementById("panels");
  if (!panels) return;

  const pollUrl = panels.dataset.pollUrl || "/api/state/panels";
  const pollIntervalMs = parseInt(panels.dataset.pollIntervalMs || "2000", 10);
  const domParser = new DOMParser();

  let pollTimer = null;
  let inFlight = false;

  function swapPanelsHtml(htmlString) {
    const doc = domParser.parseFromString(htmlString, "text/html");
    const newNodes = Array.from(doc.body.childNodes);
    panels.replaceChildren(...newNodes);
  }

  async function fetchPanels() {
    if (inFlight) return;
    inFlight = true;
    try {
      const resp = await fetch(pollUrl, { headers: { Accept: "text/html" } });
      if (!resp.ok) {
        panels.dataset.stale = "true";
        return;
      }
      const html = await resp.text();
      swapPanelsHtml(html);
      delete panels.dataset.stale;
    } catch (err) {
      panels.dataset.stale = "true";
    } finally {
      inFlight = false;
    }
  }

  async function refreshControls() {
    try {
      const resp = await fetch("/api/state", { headers: { Accept: "application/json" } });
      if (!resp.ok) return;
      const state = await resp.json();
      const running = state?.helix?.running === true;

      const statusDot = document.querySelector(".status-dot");
      if (statusDot) {
        statusDot.classList.toggle("status-dot--running", running);
        statusDot.classList.toggle("status-dot--stopped", !running);
      }

      const statusLabel = document.querySelector(".status-label");
      if (statusLabel) {
        if (running) {
          statusLabel.textContent =
            "Running · pid " + state.helix.pid + " · port " + state.helix.port;
        } else {
          statusLabel.textContent = "Stopped";
        }
      }

      const btnStart = document.querySelector('[data-action="start"]');
      const btnRestart = document.querySelector('[data-action="restart"]');
      const btnStop = document.querySelector('[data-action="stop"]');
      if (btnStart) btnStart.disabled = running;
      if (btnRestart) btnRestart.disabled = !running;
      if (btnStop) btnStop.disabled = !running;
    } catch (err) {
      // ignore — next poll will retry
    }
  }

  function startPolling() {
    if (pollTimer !== null) return;
    fetchPanels();
    refreshControls();
    pollTimer = setInterval(() => {
      fetchPanels();
      refreshControls();
    }, pollIntervalMs);
  }

  function stopPolling() {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  // ── Control button wiring ──────────────────────────────────

  async function sendControl(action) {
    const btn = document.querySelector('[data-action="' + action + '"]');
    if (btn) btn.disabled = true;

    try {
      const resp = await fetch("/api/control/" + action, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        alert(action + " failed: " + (body.error || resp.statusText));
      }
    } catch (err) {
      alert(action + " failed: " + err);
    } finally {
      setTimeout(() => {
        fetchPanels();
        refreshControls();
      }, 500);
    }
  }

  document.addEventListener("click", function (evt) {
    const target = evt.target;
    if (!(target instanceof HTMLElement)) return;
    const action = target.dataset.action;
    if (!action) return;
    if (action === "start" || action === "stop" || action === "restart") {
      sendControl(action);
    }
  });

  // ── Visibility-aware polling (pause when tab is hidden) ────

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      stopPolling();
    } else {
      startPolling();
    }
  });

  // Kick it off.
  startPolling();
})();
