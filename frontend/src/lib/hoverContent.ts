import type { HoverResponse } from "../types/hover";

/**
 * Whether a /hover response is worth showing a card for.
 *
 * The single source of truth for this — shared by `useMouseTracker` (to
 * decide whether a response counts as real content vs. treated the same as
 * an empty one: retried once, then subject to the display grace period
 * before the card clears) and `HoverCard` (to decide whether to render).
 *
 * These two used to diverge: the hook only checked `tokens.length > 0`,
 * looser than this. A response with tokens but none of them real
 * content words (e.g. only particles/junk survived OCR) passed the hook's
 * loose check as "success" — bypassing MIN_DISPLAY_MS entirely, since that
 * only guards the hook's own *empty*-response path — and got handed to
 * HoverCard, which then independently and immediately hid it under this
 * stricter check. That mismatch was the actual cause of the card
 * "disappearing immediately even with no mouse movement": not a broken
 * grace period, but a grace period that was never being consulted at all
 * for that failure path.
 *
 * Hidden when:
 *   - there is no data, OR
 *   - there are no content-word tokens AND no translations, OR
 *   - (for the token path) no token carries a real JLPT level.
 */
export function hasRenderableContent(d: HoverResponse | null): boolean {
  if (!d) return false;
  const contentWords = d.tokens.filter((t) => t.is_content_word);
  if (contentWords.length === 0 && d.translations.length === 0) return false;
  if (d.translations.length === 0 && !d.tokens.some((t) => t.jlpt_level !== null)) {
    return false;
  }
  return true;
}
