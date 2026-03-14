"use client";

import { useEffect } from "react";
import { loadSettings } from "@/app/lib/settings";

export function ThemeProvider() {
  useEffect(() => {
    const applyTheme = () => {
      const settings = loadSettings();
      let theme = settings.theme;

      if (theme === "system") {
        theme = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
      }

      document.documentElement.setAttribute("data-theme", theme);
    };

    // Apply on mount
    applyTheme();

    // Listen for cross-tab or within-app updates
    window.addEventListener("storage", applyTheme);
    window.addEventListener("repdetect:settings", applyTheme);

    // Also listen for OS-level theme changes if system preference is selected
    const matcher = window.matchMedia("(prefers-color-scheme: light)");
    matcher.addEventListener("change", applyTheme);

    return () => {
      window.removeEventListener("storage", applyTheme);
      window.removeEventListener("repdetect:settings", applyTheme);
      matcher.removeEventListener("change", applyTheme);
    };
  }, []);

  return null; // This component handles side-effects only
}
