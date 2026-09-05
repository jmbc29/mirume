import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import HoverCard from "./components/HoverCard";
import type { HoverResponse } from "./types/hover";
import type { CursorPosition } from "./hooks/useMouseTracker";

interface HoverPayload {
  data: HoverResponse | null;
  triggerPoint: CursorPosition;
}

const EMPTY_PAYLOAD: HoverPayload = { data: null, triggerPoint: { x: 0, y: 0 } };

/**
 * Root of the hover card's own window (see `card.html`).
 *
 * The card used to render inline in the full-screen overlay window ("main").
 * It now lives in a separate, small, fixed-size window instead — see
 * `src-tauri/src/lib.rs`'s `show_card`/`hide_card` for why — so it needs its
 * own way to receive hover data from "main", which still owns the actual
 * cursor-polling/`/hover`-calling logic (`useMouseTracker`, in `App.tsx`).
 * Tauri's event system bridges the two: "main" `emit`s a `"hover-data"`
 * event on every change, and this listens for it.
 */
export default function CardApp() {
  const [payload, setPayload] = useState<HoverPayload>(EMPTY_PAYLOAD);

  useEffect(() => {
    const unlisten = listen<HoverPayload>("hover-data", (event) => {
      setPayload(event.payload);
    });
    return () => {
      void unlisten.then((fn) => fn());
    };
  }, []);

  return <HoverCard data={payload.data} triggerPoint={payload.triggerPoint} />;
}
