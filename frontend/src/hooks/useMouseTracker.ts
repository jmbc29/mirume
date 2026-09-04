import { useEffect, useRef, useState } from "react";
import type { HoverResponse } from "../types/hover";

const HOVER_ENDPOINT = "http://127.0.0.1:8123/hover";
const DEBOUNCE_MS = 300;
const MIN_MOVE_PX = 10;

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
 * Mouse movement is debounced by {@link DEBOUNCE_MS}: a hover request only
 * fires once the cursor has been still for that long, and only if it moved
 * more than {@link MIN_MOVE_PX} pixels since the last request.
 */
export function useMouseTracker(): MouseTrackerState & { cursor: CursorPosition } {
  const [state, setState] = useState<MouseTrackerState>({
    data: null,
    loading: false,
    error: null,
  });
  const [cursor, setCursor] = useState<CursorPosition>({ x: 0, y: 0 });

  const lastCalledRef = useRef<CursorPosition | null>(null);
  const latestRef = useRef<CursorPosition>({ x: 0, y: 0 });
  const timeoutRef = useRef<number | undefined>(undefined);
  const requestIdRef = useRef(0);

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      const position = { x: event.screenX, y: event.screenY };
      latestRef.current = position;
      setCursor(position);

      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = window.setTimeout(() => {
        void maybeFetchHover(latestRef.current);
      }, DEBOUNCE_MS);
    };

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
        const data: HoverResponse = await response.json();
        if (requestId === requestIdRef.current) {
          setState({ data, loading: false, error: null });
        }
      } catch (err) {
        if (requestId === requestIdRef.current) {
          const message = err instanceof Error ? err.message : String(err);
          setState((prev) => ({ ...prev, loading: false, error: message }));
        }
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.clearTimeout(timeoutRef.current);
    };
  }, []);

  return { ...state, cursor };
}
