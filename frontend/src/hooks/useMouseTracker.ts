import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { HoverResponse } from "../types/hover";

const HOVER_ENDPOINT = "http://127.0.0.1:8123/hover";
/** How often the real cursor position is sampled from the Rust side. */
const POLL_MS = 500;
/** Minimum distance between samples to count as the cursor having moved. */
const MIN_MOVE_PX = 20;
/** The cursor must sit still (no sample further than MIN_MOVE_PX away) this
 *  long before a /hover request is fired for it. */
const STILL_DURATION_MS = 600;
/**
 * Once the cursor drifts more than this far from the point that triggered the
 * last hover request, the card is stale — drop the data immediately so it
 * disappears instead of lingering over unrelated screen content.
 */
const MAX_DRIFT_PX = 50;

interface MouseTrackerState {
  data: HoverResponse | null;
  loading: boolean;
  error: string | null;
}

/** Screen-space position the hover card should track. */
export interface CursorPosition {
  x: number;
  y: number;
}

/**
 * Tracks the cursor and calls the Mirume backend's /hover endpoint once it
 * settles somewhere new.
 *
 * The overlay window is click-through (`set_ignore_cursor_events(true)`), so the
 * webview never receives `mousemove` events while the cursor is over another
 * app. Instead we sample the `get_cursor_position` Rust command every
 * {@link POLL_MS}ms for the real global cursor position (top-left-origin logical
 * screen points).
 *
 * Firing /hover is debounced, not polled: each sample that differs from the
 * previous one by more than {@link MIN_MOVE_PX} resets a
 * {@link STILL_DURATION_MS}ms timer, and only when that timer completes
 * uninterrupted — meaning the cursor has genuinely stopped somewhere — does a
 * /hover request go out. A cursor sweeping across the screen never triggers a
 * request; only a deliberate pause over one spot does.
 */
export function useMouseTracker(): MouseTrackerState & { cursor: CursorPosition } {
  const [state, setState] = useState<MouseTrackerState>({
    data: null,
    loading: false,
    error: null,
  });
  const [cursor, setCursor] = useState<CursorPosition>({ x: 0, y: 0 });

  const lastSampleRef = useRef<CursorPosition | null>(null);
  const lastCalledRef = useRef<CursorPosition | null>(null);
  const stillTimerRef = useRef<number | undefined>(undefined);
  const requestIdRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    const fireHover = async (position: CursorPosition) => {
      lastCalledRef.current = position;

      const requestId = ++requestIdRef.current;
      setState((prev) => ({ ...prev, loading: true, error: null }));
      try {
        const response = await fetch(HOVER_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ x: position.x, y: position.y }),
        });
        if (!response.ok) {
          throw new Error(`hover request failed: ${response.status}`);
        }
        const json: HoverResponse = await response.json();
        if (requestId !== requestIdRef.current) {
          return;
        }
        if (json.source === "placeholder") {
          // Backend found no real screen text — keep the card hidden.
          setState((prev) => ({ ...prev, data: null, loading: false, error: null }));
          return;
        }
        setState({ data: json, loading: false, error: null });
      } catch (err) {
        if (requestId === requestIdRef.current) {
          const message = err instanceof Error ? err.message : String(err);
          setState((prev) => ({ ...prev, loading: false, error: message }));
        }
      }
    };

    const poll = async () => {
      let position: CursorPosition;
      try {
        const [x, y] = await invoke<[number, number]>("get_cursor_position");
        position = { x, y };
      } catch {
        // Not running inside Tauri (e.g. plain browser dev) — nothing to poll.
        return;
      }
      if (cancelled) return;
      setCursor(position);

      const lastSample = lastSampleRef.current;
      const moved =
        !lastSample || Math.hypot(position.x - lastSample.x, position.y - lastSample.y) > MIN_MOVE_PX;
      if (!moved) {
        // Cursor is holding steady — leave any in-flight still-timer running
        // rather than restarting it on every sample.
        return;
      }
      lastSampleRef.current = position;

      // The cursor is actively moving again, so a previously shown card no
      // longer matches where it's pointing. Hide it once the drift from the
      // point that triggered it is large enough to be a different target.
      const lastCalled = lastCalledRef.current;
      if (lastCalled) {
        const drift = Math.hypot(position.x - lastCalled.x, position.y - lastCalled.y);
        if (drift > MAX_DRIFT_PX) {
          setState((prev) => (prev.data ? { ...prev, data: null } : prev));
        }
      }

      window.clearTimeout(stillTimerRef.current);
      stillTimerRef.current = window.setTimeout(() => {
        void fireHover(position);
      }, STILL_DURATION_MS);
    };

    const intervalId = window.setInterval(() => void poll(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
      window.clearTimeout(stillTimerRef.current);
    };
  }, []);

  return { ...state, cursor };
}
