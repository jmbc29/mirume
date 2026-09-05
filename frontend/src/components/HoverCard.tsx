import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { HoverResponse, TokenOut, TranslationOut } from "../types/hover";
import type { CursorPosition } from "../hooks/useMouseTracker";

const SAVE_WORD_ENDPOINT = "http://127.0.0.1:8123/save/word";
const SAVE_SENTENCE_ENDPOINT = "http://127.0.0.1:8123/save/sentence";
const AUTO_HIDE_MS = 4000;
const CURSOR_OFFSET = 20;
const CARD_MAX_WIDTH = 340;
const CARD_MAX_HEIGHT = 380;
const MAX_WORDS_SHOWN = 3;
const MEANING_MAX_CHARS = 50;

const JLPT_COLORS: Record<string, string> = {
  N5: "#22c55e",
  N4: "#14b8a6",
  N3: "#eab308",
  N2: "#f97316",
  N1: "#ef4444",
};
const UNKNOWN_COLOR = "#6b7280";

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function JlptBadge({ name }: { name: string }) {
  const color = JLPT_COLORS[name];
  if (!color) return null;
  return (
    <span
      style={{
        display: "inline-block",
        padding: "1px 6px",
        borderRadius: 6,
        fontSize: 10,
        fontWeight: 600,
        color: "#0a0a0a",
        backgroundColor: color,
        flexShrink: 0,
      }}
    >
      {name}
    </span>
  );
}

function SaveButton({ onClick, label }: { onClick: () => void; label: string }) {
  const [saved, setSaved] = useState(false);
  return (
    <button
      onClick={() => {
        onClick();
        setSaved(true);
      }}
      disabled={saved}
      style={{
        padding: "4px 12px",
        borderRadius: 6,
        border: "1px solid rgba(255,255,255,0.15)",
        backgroundColor: saved ? "rgba(34,197,94,0.2)" : "rgba(255,255,255,0.08)",
        color: "#e2e8f0",
        fontSize: 12,
        cursor: saved ? "default" : "pointer",
        flexShrink: 0,
      }}
    >
      {saved ? "Saved" : label}
    </button>
  );
}

/** Full detected sentence, each token colour-coded by its JLPT level. */
function SentenceDisplay({ tokens }: { tokens: TokenOut[] }) {
  return (
    <div style={{ fontSize: 16, lineHeight: 1.6 }}>
      {tokens.map((t, i) => (
        <span key={i} style={{ color: JLPT_COLORS[t.jlpt_name] ?? UNKNOWN_COLOR }}>
          {t.surface}
        </span>
      ))}
    </div>
  );
}

/** One compact row in the top-3 rarest-words list. */
function WordRow({ token, showBorder }: { token: TokenOut; showBorder: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: 8,
        padding: "6px 0",
        borderBottom: showBorder ? "1px solid rgba(255,255,255,0.08)" : "none",
      }}
    >
      <span style={{ fontSize: 20, fontWeight: 600, flexShrink: 0 }}>{token.surface}</span>
      <JlptBadge name={token.jlpt_name} />
      {token.reading && (
        <span style={{ fontSize: 12, color: "#94a3b8", flexShrink: 0 }}>{token.reading}</span>
      )}
      <span
        style={{
          fontSize: 12,
          color: "#cbd5e1",
          marginLeft: "auto",
          textAlign: "right",
          overflow: "hidden",
        }}
      >
        {token.meaning ? truncate(token.meaning, MEANING_MAX_CHARS) : ""}
      </span>
    </div>
  );
}

/** Top N content words, rarest (N1) first, easiest (N5) last, unknown last. */
function pickTopWords(tokens: TokenOut[]): TokenOut[] {
  const contentWords = tokens.filter((t) => t.is_content_word);
  return [...contentWords]
    .sort((a, b) => (a.jlpt_level ?? 6) - (b.jlpt_level ?? 6))
    .slice(0, MAX_WORDS_SHOWN);
}

function EnglishTranslationEntry({ translation }: { translation: TranslationOut }) {
  const save = () => {
    void fetch(SAVE_SENTENCE_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sentence: translation.translation,
        reading: translation.reading,
        translation: translation.source_text,
        jlpt_level: translation.jlpt_name,
      }),
    });
  };

  return (
    <div
      style={{
        paddingBottom: 10,
        marginBottom: 10,
        borderBottom: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <div style={{ fontSize: 12, opacity: 0.6 }}>{translation.source_text}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginTop: 4 }}>
        <span style={{ fontSize: 22, fontWeight: 600 }}>{translation.translation}</span>
        <JlptBadge name={translation.jlpt_name} />
      </div>
      {translation.reading && (
        <div style={{ fontSize: 13, opacity: 0.7, marginTop: 2 }}>{translation.reading}</div>
      )}
      {translation.grammar_patterns.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
          {translation.grammar_patterns.map((g) => (
            <span
              key={g.pattern}
              style={{
                fontSize: 11,
                padding: "2px 6px",
                borderRadius: 6,
                backgroundColor: "rgba(255,255,255,0.05)",
              }}
            >
              {g.pattern} · {g.jlpt_level}
            </span>
          ))}
        </div>
      )}
      {translation.alternatives.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>
          {translation.alternatives.map((alt, i) => (
            <div key={i} style={{ marginTop: 2 }}>
              {alt}
            </div>
          ))}
        </div>
      )}
      <div style={{ marginTop: 8 }}>
        <SaveButton onClick={save} label="Save sentence" />
      </div>
    </div>
  );
}

function clampToViewport(x: number, y: number): { left: number; top: number } {
  const maxLeft = Math.max(0, window.innerWidth - CARD_MAX_WIDTH - 8);
  const maxTop = Math.max(0, window.innerHeight - CARD_MAX_HEIGHT - 8);
  return { left: Math.min(x, maxLeft), top: Math.min(y, maxTop) };
}

interface HoverCardProps {
  data: HoverResponse | null;
  triggerPoint: CursorPosition;
}

/**
 * Whether the backend response is worth showing a card for.
 *
 * The card must stay hidden unless the cursor is actually over meaningful
 * Japanese text or a translation. It is hidden when:
 *   - there is no data, OR
 *   - there are no content-word tokens AND no translations, OR
 *   - (for the token path) no token carries a real JLPT level.
 */
function hasRenderableContent(d: HoverResponse | null): boolean {
  if (!d) return false;
  const contentWords = d.tokens.filter((t) => t.is_content_word);
  if (contentWords.length === 0 && d.translations.length === 0) return false;
  if (d.translations.length === 0 && !d.tokens.some((t) => t.jlpt_level !== null)) {
    return false;
  }
  return true;
}

export default function HoverCard({ data, triggerPoint }: HoverCardProps) {
  const [visible, setVisible] = useState(false);
  const [displayData, setDisplayData] = useState<HoverResponse | null>(null);
  // Screen position the card is pinned to. Set once when new data arrives and
  // never touched again while that data is showing, so the card doesn't
  // chase the live cursor and run away when the user reaches for Save.
  const [lockedPosition, setLockedPosition] = useState<CursorPosition | null>(null);
  const hideTimeoutRef = useRef<number | undefined>(undefined);

  const hasContent = hasRenderableContent(data);

  useEffect(() => {
    if (!hasContent) {
      setVisible(false);
      setDisplayData(null);
      window.clearTimeout(hideTimeoutRef.current);
      return;
    }
    setDisplayData(data);
    setLockedPosition(triggerPoint);
    setVisible(true);
    window.clearTimeout(hideTimeoutRef.current);
    hideTimeoutRef.current = window.setTimeout(() => setVisible(false), AUTO_HIDE_MS);
    return () => window.clearTimeout(hideTimeoutRef.current);
    // Deliberately keyed on `data` only — triggerPoint is captured at the
    // moment new data arrives, not tracked live.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // The frontend still pings the Rust side when the card shows/hides, but the
  // overlay is always kept click-through there — it must never block the app
  // underneath. The inner card div opts back into pointer events on its own.
  useEffect(() => {
    void invoke("set_click_through", { ignore: !visible }).catch(() => {
      // Not running inside Tauri (e.g. plain browser dev) — ignore.
    });
  }, [visible]);

  if (!displayData || !lockedPosition) return null;

  const { left, top } = clampToViewport(
    lockedPosition.x + CURSOR_OFFSET,
    lockedPosition.y + CURSOR_OFFSET
  );
  const topWords = pickTopWords(displayData.tokens);
  const totalContentWords = displayData.tokens.filter((t) => t.is_content_word).length;
  const hiddenCount = Math.max(0, totalContentWords - topWords.length);

  const saveAll = () => {
    for (const t of topWords) {
      void fetch(SAVE_WORD_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          word: t.surface,
          reading: t.reading,
          meaning: t.meaning,
          jlpt_level: t.jlpt_name,
          context_sentence: displayData.text,
        }),
      });
    }
  };

  return (
    <div style={{ position: "fixed", left, top, pointerEvents: "none" }}>
      <div
        style={{
          maxWidth: CARD_MAX_WIDTH,
          maxHeight: CARD_MAX_HEIGHT,
          overflow: "hidden",
          backgroundColor: "rgba(15, 15, 30, 0.92)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 14,
          padding: "14px 16px",
          boxShadow: "0 8px 40px rgba(0,0,0,0.5)",
          fontFamily: "system-ui, -apple-system, sans-serif",
          color: "#e2e8f0",
          opacity: visible ? 1 : 0,
          transition: "opacity 150ms ease",
          pointerEvents: "auto",
        }}
      >
        {displayData.tokens.length > 0 && (
          <>
            <SentenceDisplay tokens={displayData.tokens} />
            <div
              style={{
                height: 1,
                backgroundColor: "rgba(255,255,255,0.08)",
                margin: "10px 0",
              }}
            />
            <div>
              {topWords.map((t, i) => (
                <WordRow token={t} showBorder={i < topWords.length - 1} key={`word-${i}`} />
              ))}
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginTop: 10,
              }}
            >
              <span style={{ fontSize: 11, color: "#94a3b8" }}>
                {totalContentWords} word{totalContentWords === 1 ? "" : "s"} total
                {hiddenCount > 0 ? ` · ${hiddenCount} more` : ""}
              </span>
              <SaveButton onClick={saveAll} label="Save all" />
            </div>
          </>
        )}
        {displayData.translations.map((t, i) => (
          <EnglishTranslationEntry translation={t} key={`en-${i}`} />
        ))}
      </div>
    </div>
  );
}
