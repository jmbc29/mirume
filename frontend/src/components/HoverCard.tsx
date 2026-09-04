import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { HoverResponse, KanjiOut, TokenOut, TranslationOut } from "../types/hover";
import type { CursorPosition } from "../hooks/useMouseTracker";

const SAVE_WORD_ENDPOINT = "http://127.0.0.1:8123/save/word";
const SAVE_SENTENCE_ENDPOINT = "http://127.0.0.1:8123/save/sentence";
const AUTO_HIDE_MS = 4000;
const CURSOR_OFFSET = 20;
const CARD_MAX_WIDTH = 320;
const ESTIMATED_CARD_HEIGHT = 400;

const JLPT_COLORS: Record<string, string> = {
  N5: "#22c55e",
  N4: "#14b8a6",
  N3: "#eab308",
  N2: "#f97316",
  N1: "#ef4444",
};

function JlptBadge({ name }: { name: string }) {
  const color = JLPT_COLORS[name];
  if (!color) return null;
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: 6,
        fontSize: 11,
        fontWeight: 600,
        color: "#0a0a0a",
        backgroundColor: color,
      }}
    >
      {name}
    </span>
  );
}

function KanjiChip({ kanji }: { kanji: KanjiOut }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "4px 6px",
        borderRadius: 6,
        backgroundColor: "rgba(255,255,255,0.05)",
      }}
    >
      <span style={{ fontSize: 16 }}>{kanji.literal}</span>
      <JlptBadge name={kanji.jlpt_name} />
      <span style={{ fontSize: 11, opacity: 0.7 }}>{kanji.meaning}</span>
    </div>
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
        marginTop: 8,
        padding: "4px 12px",
        borderRadius: 6,
        border: "1px solid rgba(255,255,255,0.15)",
        backgroundColor: saved ? "rgba(34,197,94,0.2)" : "rgba(255,255,255,0.08)",
        color: "#e2e8f0",
        fontSize: 12,
        cursor: saved ? "default" : "pointer",
      }}
    >
      {saved ? "Saved" : label}
    </button>
  );
}

function JapaneseWordEntry({ token, kanji }: { token: TokenOut; kanji: KanjiOut[] }) {
  const relatedKanji = kanji.filter((k) => token.surface.includes(k.literal));

  const save = () => {
    void fetch(SAVE_WORD_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        word: token.surface,
        reading: token.reading,
        meaning: token.meaning,
        jlpt_level: token.jlpt_name,
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
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontSize: 26, fontWeight: 600 }}>{token.surface}</span>
        <JlptBadge name={token.jlpt_name} />
      </div>
      {token.reading && (
        <div style={{ fontSize: 13, opacity: 0.7, marginTop: 2 }}>{token.reading}</div>
      )}
      {token.meaning && <div style={{ fontSize: 13, marginTop: 6 }}>{token.meaning}</div>}
      {relatedKanji.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
          {relatedKanji.map((k) => (
            <KanjiChip kanji={k} key={k.literal} />
          ))}
        </div>
      )}
      <SaveButton onClick={save} label="Save word" />
    </div>
  );
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
        <span style={{ fontSize: 26, fontWeight: 600 }}>{translation.translation}</span>
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
      <SaveButton onClick={save} label="Save sentence" />
    </div>
  );
}

function clampToViewport(x: number, y: number): { left: number; top: number } {
  const maxLeft = Math.max(0, window.innerWidth - CARD_MAX_WIDTH - 8);
  const maxTop = Math.max(0, window.innerHeight - ESTIMATED_CARD_HEIGHT - 8);
  return { left: Math.min(x, maxLeft), top: Math.min(y, maxTop) };
}

interface HoverCardProps {
  data: HoverResponse | null;
  cursor: CursorPosition;
}

export default function HoverCard({ data, cursor }: HoverCardProps) {
  const [visible, setVisible] = useState(false);
  const [displayData, setDisplayData] = useState<HoverResponse | null>(null);
  const hideTimeoutRef = useRef<number | undefined>(undefined);

  const hasContent = !!data && (data.tokens.length > 0 || data.translations.length > 0);

  useEffect(() => {
    if (!hasContent) return;
    setDisplayData(data);
    setVisible(true);
    window.clearTimeout(hideTimeoutRef.current);
    hideTimeoutRef.current = window.setTimeout(() => setVisible(false), AUTO_HIDE_MS);
    return () => window.clearTimeout(hideTimeoutRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // Let clicks (e.g. the save button) through only while the card is showing;
  // otherwise the overlay stays click-through so it never blocks the app underneath.
  useEffect(() => {
    void invoke("set_click_through", { ignore: !visible }).catch(() => {
      // Not running inside Tauri (e.g. plain browser dev) — ignore.
    });
  }, [visible]);

  if (!displayData) return null;

  const { left, top } = clampToViewport(cursor.x + CURSOR_OFFSET, cursor.y + CURSOR_OFFSET);

  return (
    <div
      style={{
        position: "fixed",
        left,
        top,
        maxWidth: CARD_MAX_WIDTH,
        backgroundColor: "rgba(26,26,46,0.95)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 12,
        padding: 16,
        boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
        fontFamily: "system-ui, -apple-system, sans-serif",
        color: "#e2e8f0",
        opacity: visible ? 1 : 0,
        transition: "opacity 150ms ease",
        pointerEvents: visible ? "auto" : "none",
      }}
    >
      {displayData.tokens
        .filter((t) => t.is_content_word)
        .map((t, i) => (
          <JapaneseWordEntry token={t} kanji={displayData.kanji} key={`ja-${i}`} />
        ))}
      {displayData.translations.map((t, i) => (
        <EnglishTranslationEntry translation={t} key={`en-${i}`} />
      ))}
    </div>
  );
}
