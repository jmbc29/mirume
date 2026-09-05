import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { HoverResponse } from "../types/hover";

const HOVER_ENDPOINT = "http://127.0.0.1:8123/hover";
/** How often the real cursor position is sampled from the Rust side. */
const POLL_MS = 300;
/** Minimum distance between samples to count as the cursor having moved. */
const MIN_MOVE_PX = 20;
/** The cursor must sit still (no sample further than MIN_MOVE_PX away) this
 *  long before a /hover request is fired for it. */
const STILL_MS = 400;
/**
 * When a /hover response comes back empty, PaddleOCR may have simply missed
 * the line on this pass (see backend/ocr.py's own retry) — wait this long
 * and retry once at the same position before actually hiding the card.
 */
const HOVER_RETRY_MS = 300;
/**
 * Once the cursor drifts more than this far from the point that triggered the
 * last hover request, that reference point is stale — forget it so the next
 * still-pause fires a fresh /hover call. This does NOT clear the currently
 * displayed card; the card is only cleared when a /hover response actually
 * comes back empty (see fireHover), so a slight drift toward the card itself
 * (e.g. to click Save) doesn't make it disappear.
 */
const MAX_DRIFT_PX = 300;
/**
 * Once a card has been showing for less than this long, an empty /hover
 * response is ignored instead of clearing it. Without this, reaching for the
 * Save button re-triggers the debounced /hover cycle at the cursor's new
 * position (now over the card's own window, not the original text), which
 * comes back empty and would otherwise wipe the card out from under the
 * cursor before the click lands.
 */
const MIN_DISPLAY_MS = 2000;

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
 * The overlay window is click-through (`set_ignore_cursor_events(true)`), so
 * the webview never receives `mousemove` events while the cursor is over
 * another app — there is no browser event to hook here. Cursor position is
 * instead sampled from the `get_cursor_position` Rust command on a plain
 * `setInterval` every {@link POLL_MS}ms; that Rust-side poll is the only
 * source of position data this hook has.
 *
 * Firing /hover is debounced on top of that poll, not driven by it directly:
 * each sample that differs from the last *significant* one by more than
 * {@link MIN_MOVE_PX} resets a {@link STILL_MS}ms timer, and only when that
 * timer completes uninterrupted — meaning the cursor has genuinely stopped
 * somewhere — does a /hover request go out. A cursor sweeping across the
 * screen keeps resetting the timer and never triggers a request; only a
 * deliberate pause over one spot does.
 */
export function useMouseTracker(): MouseTrackerState & {
  cursor: CursorPosition;
  triggerPoint: CursorPosition;
  setPaused: (paused: boolean) => void;
} {
  const [state, setState] = useState<MouseTrackerState>({
    data: null,
    loading: false,
    error: null,
  });
  const [cursor, setCursor] = useState<CursorPosition>({ x: 0, y: 0 });
  // Cursor position at the moment the *currently displayed* data arrived.
  // Only updated when a new /hover response carries real content — this is
  // what the hover card locks its position to, so it doesn't chase the
  // live cursor once shown (see HoverCard).
  const [triggerPoint, setTriggerPoint] = useState<CursorPosition>({ x: 0, y: 0 });

  const lastCalledRef = useRef<CursorPosition | null>(null);
  const lastSampleRef = useRef<CursorPosition | null>(null);
  const stillTimerRef = useRef<number | undefined>(undefined);
  const requestIdRef = useRef(0);
  // When content last appeared — used to enforce MIN_DISPLAY_MS below.
  const cardShownAtRef = useRef<number | null>(null);
  // True while the cursor is physically over the card itself (see
  // HoverCard's onMouseEnter/onMouseLeave). Polling keeps sampling the
  // cursor while paused, but never fires or schedules a new /hover call —
  // otherwise moving onto the card to click Save would immediately queue a
  // fresh hover at that position (now over the card's own window, not the
  // original text), which comes back empty and clears the card.
  const pausedRef = useRef(false);
  const setPaused = (paused: boolean) => {
    pausedRef.current = paused;
    if (paused) {
      window.clearTimeout(stillTimerRef.current);
    }
  };

  useEffect(() => {
    let cancelled = false;

    const fireHover = async (position: CursorPosition, isRetry = false) => {
      if (!isRetry) {
        lastCalledRef.current = position;
      }

      // A retry re-queries the same logical request rather than starting a
      // new one, so a genuinely new hover elsewhere (which does bump this)
      // can invalidate a still-pending retry.
      const requestId = isRetry ? requestIdRef.current : ++requestIdRef.current;
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
        if (requestId !== requestIdRef.current || cancelled) {
          return;
        }
        const hasContent = json.tokens.length > 0 || json.translations.length > 0;
        if (!hasContent) {
          if (!isRetry) {
            // OCR can miss a line on a single pass — give it one more try
            // at the same spot before actually hiding the card.
            window.setTimeout(() => {
              if (!cancelled) {
                void fireHover(position, true);
              }
            }, HOVER_RETRY_MS);
            return;
          }
          const shownMs = cardShownAtRef.current ? Date.now() - cardShownAtRef.current : Infinity;
          if (shownMs < MIN_DISPLAY_MS) {
            // The card only just appeared — this empty response is almost
            // certainly from a /hover fired at the cursor's current spot
            // (e.g. now over the card itself), not evidence the original
            // text is gone. Keep showing the existing data.
            setState((prev) => ({ ...prev, loading: false, error: null }));
            return;
          }
          // Backend found no real screen text — this is the only case that
          // clears the card once shown; a drifting cursor alone never does.
          setState({ data: null, loading: false, error: null });
          return;
        }
        cardShownAtRef.current = Date.now();
        setTriggerPoint(position);
        setState({ data: json, loading: false, error: null });
      } catch (err) {
        if (requestId === requestIdRef.current) {
          const message = err instanceof Error ? err.message : String(err);
          setState((prev) => ({ ...prev, loading: false, error: message }));
        }
      }
    };

    // Poll cursor position every POLL_MS via the Rust side — a browser
    // mousemove listener would never fire on this click-through overlay.
    const intervalId = window.setInterval(() => {
      void (async () => {
        if (cancelled) return;

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

        if (pausedRef.current) {
          // Cursor is over the card itself — don't fire or schedule a new
          // /hover call while the user is interacting with it.
          return;
        }

        const lastSample = lastSampleRef.current;
        const moved =
          !lastSample ||
          Math.hypot(position.x - lastSample.x, position.y - lastSample.y) > MIN_MOVE_PX;
        if (!moved) {
          // Cursor is holding steady — leave any in-flight still-timer
          // running rather than restarting it on every sample.
          return;
        }
        lastSampleRef.current = position;

        // The cursor is actively moving again. Once it has drifted far enough
        // from the point that triggered the last /hover call, that reference
        // point no longer describes a nearby target — forget it so the next
        // still-pause fires a fresh call. This does NOT hide the card: the
        // card is only cleared when a /hover response itself comes back
        // empty, so drifting toward the card (e.g. to click Save) never
        // makes it disappear.
        const lastCalled = lastCalledRef.current;
        if (lastCalled) {
          const drift = Math.hypot(position.x - lastCalled.x, position.y - lastCalled.y);
          if (drift > MAX_DRIFT_PX) {
            lastCalledRef.current = null;
          }
        }

        window.clearTimeout(stillTimerRef.current);
        stillTimerRef.current = window.setTimeout(() => {
          if (!cancelled) {
            void fireHover(position);
          }
        }, STILL_MS);
      })();
    }, POLL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
      window.clearTimeout(stillTimerRef.current);
    };
  }, []);

  return { ...state, cursor, triggerPoint, setPaused };
}
