"use strict";

(() => {
  const storageKey = "alphalens-theme";
  const root = document.documentElement;

  function storedTheme() {
    try {
      return window.localStorage.getItem(storageKey);
    } catch (error) {
      return null;
    }
  }

  function preferredTheme() {
    const requested = new URLSearchParams(window.location.search).get("theme");

    if (["light", "dark"].includes(requested)) {
      return requested;
    }

    const saved = storedTheme();

    if (["light", "dark"].includes(saved)) {
      return saved;
    }

    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function renderToggle(theme) {
    const button = document.querySelector("#theme-toggle");
    const label = document.querySelector("#theme-label");

    if (!button || !label) {
      return;
    }

    const isDark = theme === "dark";
    button.setAttribute("aria-pressed", String(isDark));
    button.setAttribute(
      "aria-label",
      `Color theme: ${theme}. Switch to ${isDark ? "light" : "dark"} mode.`
    );
    label.textContent = isDark ? "Dark" : "Light";
  }

  function applyTheme(theme, persist = false) {
    root.dataset.theme = theme;
    renderToggle(theme);

    if (persist) {
      try {
        window.localStorage.setItem(storageKey, theme);
      } catch (error) {
        // A blocked storage API should not prevent the visual toggle.
      }
    }
  }

  applyTheme(preferredTheme());

  document.addEventListener("DOMContentLoaded", () => {
    renderToggle(root.dataset.theme);
    document.querySelector("#theme-toggle")?.addEventListener("click", () => {
      applyTheme(root.dataset.theme === "dark" ? "light" : "dark", true);
    });
  });
})();
