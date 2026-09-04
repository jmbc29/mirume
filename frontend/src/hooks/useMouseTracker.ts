import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { HoverResponse } from "../types/hover";

const HOVER_ENDPOINT = "http://127.0.0.1:8123/hover";
const POLL_MS = 200;
const MIN_MOVE_PX = 10;
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
 * Tracks the cursor and calls the Mirume backend's /hover endpoint whenever
 * it settles somewhere new.
 *
 * The overlay window is click-through (`set_ignore_cursor_events(true)`), so the
 * webview never receives `mousemove` events while the cursor is over another
 * app. Instead we poll the `get_cursor_position` Rust command every
 * {@link POLL_MS}ms for the real global cursor position (top-left-origin logical
 * screen points), and fire a hover request only once the cursor has moved more
 * than {@link MIN_MOVE_PX} pixels from the last request.
 */
export function useMouseTracker(): MouseTrackerState & { cursor: CursorPosition } {
  const [state, setState] = useState<MouseTrackerState>({
    data: null,
    loading: false,
    error: null,
  });
  const [cursor, setCursor] = useState<CursorPosition>({ x: 0, y: 0 });

  const lastCalledRef = useRef<CursorPosition | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    const maybeFetchHover = async (position: CursorPosition) => {
      const last = lastCalledRef.current;
      if (last) {
        const distance = Math.hypot(position.x - last.x, position.y - last.y);
        if (distance <= MIN_MOVE_PX) {
          return;
        }
      }
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

      // If the cursor has moved well away from where the visible card was
      // triggered, hide it right away rather than waiting for the next request.
      const last = lastCalledRef.current;
      if (last) {
        const drift = Math.hypot(position.x - last.x, position.y - last.y);
        if (drift > MAX_DRIFT_PX) {
          setState((prev) => (prev.data ? { ...prev, data: null } : prev));
        }
      }

      void maybeFetchHover(position);
    };

    const intervalId = window.setInterval(() => void poll(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  return { ...state, cursor };
}
