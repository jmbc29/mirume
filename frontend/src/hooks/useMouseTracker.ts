import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { hasRenderableContent } from "../lib/hoverContent";
import type { HoverResponse } from "../types/hover";

const HOVER_ENDPOINT = "http://127.0.0.1:8123/hover";
/** How often the real cursor position is sampled from the Rust side. */
const POLL_MS = 300;
/** Minimum distance between samples to count as the cursor having moved. */
const MIN_MOVE_PX = 20;
/** The cursor must sit still (no sample further than MIN_MOVE_PX away) this
 *  long before a /hover request is fired for it. */
const STILL_MS = 300;
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
 * Once a card has been showing for less than this long, a non-renderable
 * /hover response is ignored instead of clearing it. Without this, reaching
 * for the Save button re-triggers the debounced /hover cycle at the
 * cursor's new position (now over the card's own window, not the original
 * text), which comes back empty and would otherwise wipe the card out from
 * under the cursor before the click lands. A full 3 seconds — comfortably
 * longer than it takes to move the cursor from the trigger point to Save.
 */
const MIN_DISPLAY_MS = 3000;
/**
 * A shown card is re-checked at least this often even while the cursor sits
 * perfectly still. The context under it can change with no mouse movement at
 * all — switching browser tab by keyboard, scrolling the page, an app switch —
 * and a card left over from the previous context should not linger. The
 * re-check goes through the same /hover call, so a now-blocked surface
 * (claude.ai, another app) comes back empty and the card hides.
 */
const REVALIDATE_MS = 2000;
/**
 * Footprint of the card's own window — its {@link CARD_REGION_OFFSET} from the
 * trigger point plus its declared size in tauri.conf.json — padded by
 * {@link CARD_REGION_SLACK}. An empty /hover fired while the cursor is inside
 * this box is read as "the cursor is now over the card" (the user reaching for
 * Save) and is subject to the MIN_DISPLAY_MS grace; one fired outside it means
 * the context is genuinely gone and clears the card straight away.
 */
const CARD_REGION_OFFSET = 20;
const CARD_REGION_WIDTH = 380;
const CARD_REGION_HEIGHT = 500;
const CARD_REGION_SLACK = 40;

function cursorOverCard(pos: CursorPosition, cardTrigger: CursorPosition | null): boolean {
  if (!cardTrigger) return false;
  const left = cardTrigger.x + CARD_REGION_OFFSET - CARD_REGION_SLACK;
  const top = cardTrigger.y + CARD_REGION_OFFSET - CARD_REGION_SLACK;
  return (
    pos.x >= left &&
    pos.x <= left + CARD_REGION_WIDTH + 2 * CARD_REGION_SLACK &&
    pos.y >= top &&
    pos.y <= top + CARD_REGION_HEIGHT + 2 * CARD_REGION_SLACK
  );
}

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
  // Trigger point of the card currently on screen, or null when none is shown.
  // Lets an "empty because the cursor moved onto the card" response be told
  // apart from an "empty because the context is gone" one, and lets the poll
  // loop know a card is up so it can re-validate it while the cursor is still.
  const displayedTriggerRef = useRef<CursorPosition | null>(null);
  // When a (non-retry) /hover call last went out — rate-limits REVALIDATE_MS.
  const lastHoverAtRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    const fireHover = async (position: CursorPosition, isRetry = false) => {
      if (!isRetry) {
        lastCalledRef.current = position;
        lastHoverAtRef.current = Date.now();
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
        if (!hasRenderableContent(json)) {
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
          if (shownMs < MIN_DISPLAY_MS && cursorOverCard(position, displayedTriggerRef.current)) {
            // The card only just appeared and the cursor is now over it — this
            // non-renderable response is almost certainly from a /hover fired
            // at the cursor's current spot (e.g. reaching for Save), not
            // evidence the original text is gone. Keep showing the existing
            // data. An empty response from anywhere *else* (the cursor moved
            // off to blank space, the tab/app changed under a still cursor)
            // falls through and clears the card.
            setState((prev) => ({ ...prev, loading: false, error: null }));
            return;
          }
          // Backend found nothing worth showing and the cursor isn't on the
          // card: the context under it is genuinely gone (text scrolled away,
          // switched to claude.ai or another blocked surface, app switch).
          // A drifting cursor alone still never clears the card — only an
          // actual empty /hover response does (see the drift-handling block
          // below, which only ever resets a stale reference point).
          displayedTriggerRef.current = null;
          setState({ data: null, loading: false, error: null });
          return;
        }
        cardShownAtRef.current = Date.now();
        displayedTriggerRef.current = position;
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

        const lastSample = lastSampleRef.current;
        const moved =
          !lastSample ||
          Math.hypot(position.x - lastSample.x, position.y - lastSample.y) > MIN_MOVE_PX;
        if (!moved) {
          // Cursor is holding steady — leave any in-flight still-timer
          // running rather than restarting it on every sample. But if a card
          // is showing, re-fire /hover at the same spot every REVALIDATE_MS so
          // a context change that produced no mouse movement (keyboard tab
          // switch, page scroll, app switch) still tears a now-stale card down.
          if (
            displayedTriggerRef.current &&
            Date.now() - lastHoverAtRef.current > REVALIDATE_MS
          ) {
            void fireHover(position);
          }
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

  return { ...state, cursor, triggerPoint };
}
