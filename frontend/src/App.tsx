import { useEffect } from "react";
import { emit } from "@tauri-apps/api/event";
import { useMouseTracker } from "./hooks/useMouseTracker";
import "./App.css";

/**
 * Root of the full-screen overlay window ("main").
 *
 * Owns cursor polling and the `/hover` calls (`useMouseTracker`) but no
 * longer renders the hover card itself — that lives in its own small window
 * now (see `card.html` / `CardApp.tsx`), so Save clicks land reliably and
 * "main" can stay permanently full-screen and click-through (see
 * `src-tauri/src/lib.rs`). Hover data reaches the card window via a Tauri
 * event instead of a React prop.
 */
function App() {
  const { data, triggerPoint } = useMouseTracker();

  useEffect(() => {
    void emit("hover-data", { data, triggerPoint }).catch(() => {
      // Not running inside Tauri (e.g. plain browser dev) — ignore.
    });
  }, [data, triggerPoint]);

  return <div className="overlay" style={{ pointerEvents: "none" }} />;
}

export default App;
