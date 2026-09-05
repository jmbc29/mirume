import { useEffect, useMemo, useState } from "react";
import type {
  ExampleSentenceOut,
  FlashcardResponse,
  ReviewWordsResponse,
  SavedWordOut,
  StatsResponse,
} from "./types/review";

const API = "http://127.0.0.1:8123";

const JLPT_COLORS: Record<string, string> = {
  N5: "#22c55e",
  N4: "#14b8a6",
  N3: "#eab308",
  N2: "#f97316",
  N1: "#ef4444",
};
const LEVELS = ["N5", "N4", "N3", "N2", "N1"] as const;

function JlptBadge({ level }: { level: string }) {
  const color = JLPT_COLORS[level] ?? "#6b7280";
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
      {level}
    </span>
  );
}

function isDue(word: SavedWordOut): boolean {
  return new Date(word.due_date).getTime() <= Date.now();
}

// --------------------------------------------------------------------------- #
// Tab bar
// --------------------------------------------------------------------------- #

type Tab = "words" | "flashcards" | "stats";

function TabBar({ tab, onChange }: { tab: Tab; onChange: (tab: Tab) => void }) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "words", label: "Word List" },
    { id: "flashcards", label: "Flashcards" },
    { id: "stats", label: "Stats" },
  ];
  return (
    <div
      style={{
        display: "flex",
        gap: 4,
        padding: "12px 24px 0",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        flexShrink: 0,
      }}
    >
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          style={{
            padding: "10px 18px",
            borderRadius: "10px 10px 0 0",
            border: "none",
            borderBottom: tab === t.id ? "2px solid #60a5fa" : "2px solid transparent",
            background: tab === t.id ? "rgba(96,165,250,0.1)" : "transparent",
            color: tab === t.id ? "#e2e8f0" : "#94a3b8",
            fontSize: 14,
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

// --------------------------------------------------------------------------- #
// Tab 1 — Word List
// --------------------------------------------------------------------------- #

type LevelFilter = "All" | (typeof LEVELS)[number] | "Due today";

function WordListTab() {
  const [data, setData] = useState<ReviewWordsResponse | null>(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<LevelFilter>("All");
  const [expanded, setExpanded] = useState<number | null>(null);
  const [sentencesByWord, setSentencesByWord] = useState<Record<string, ExampleSentenceOut[]>>(
    {}
  );

  const load = () => {
    void fetch(`${API}/review/words`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null));
  };

  useEffect(load, []);

  const allWords = useMemo(() => {
    if (!data) return [];
    return Object.values(data.by_level).flat();
  }, [data]);

  // Per-level counts for the filter chips, so the word list itself shows the
  // JLPT breakdown (not just the Stats tab).
  const countFor = (opt: LevelFilter): number => {
    if (opt === "All") return allWords.length;
    if (opt === "Due today") return allWords.filter(isDue).length;
    return allWords.filter((w) => w.jlpt_level === opt).length;
  };

  const filteredWords = useMemo(() => {
    let words = allWords;
    if (filter === "Due today") {
      words = words.filter(isDue);
    } else if (filter !== "All") {
      words = words.filter((w) => w.jlpt_level === filter);
    }
    const query = search.trim().toLowerCase();
    if (query) {
      words = words.filter(
        (w) =>
          w.surface.toLowerCase().includes(query) ||
          (w.reading ?? "").toLowerCase().includes(query) ||
          (w.meaning ?? "").toLowerCase().includes(query)
      );
    }
    return words;
  }, [allWords, filter, search]);

  const toggleExpand = (word: SavedWordOut) => {
    if (expanded === word.id) {
      setExpanded(null);
      return;
    }
    setExpanded(word.id);
    if (!sentencesByWord[word.surface]) {
      void fetch(`${API}/word/${encodeURIComponent(word.surface)}/sentences`)
        .then((r) => r.json())
        .then((res: { sentences: ExampleSentenceOut[] }) =>
          setSentencesByWord((prev) => ({ ...prev, [word.surface]: res.sentences }))
        )
        .catch(() => setSentencesByWord((prev) => ({ ...prev, [word.surface]: [] })));
    }
  };

  const deleteWord = (id: number) => {
    void fetch(`${API}/review/words/${id}`, { method: "DELETE" }).then(() => load());
  };

  if (!data) {
    return <div style={{ opacity: 0.6 }}>Loading…</div>;
  }

  const filterOptions: LevelFilter[] = ["All", ...LEVELS, "Due today"];

  return (
    <div>
      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 22, fontWeight: 700 }}>{data.total} saved words</div>
        <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>
          {data.due_today} due for review today
        </div>
      </div>
      <input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search word, reading, or meaning…"
        style={{
          width: "100%",
          padding: "10px 14px",
          borderRadius: 10,
          border: "1px solid rgba(255,255,255,0.12)",
          background: "rgba(255,255,255,0.05)",
          color: "#e2e8f0",
          fontSize: 14,
          marginBottom: 12,
        }}
      />
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 16 }}>
        {filterOptions.map((opt) => (
          <button
            key={opt}
            onClick={() => setFilter(opt)}
            style={{
              padding: "5px 12px",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.12)",
              background: filter === opt ? "rgba(96,165,250,0.25)" : "transparent",
              color: filter === opt ? "#93c5fd" : "#94a3b8",
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {opt} <span style={{ opacity: 0.6 }}>{countFor(opt)}</span>
          </button>
        ))}
        <span style={{ marginLeft: "auto", fontSize: 12, color: "#64748b", alignSelf: "center" }}>
          {filteredWords.length} of {data.total} words
        </span>
      </div>

      <div
        style={{
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 12,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1.3fr 1fr 2fr 0.7fr 0.9fr 1.2fr 0.6fr",
            gap: 8,
            padding: "10px 14px",
            fontSize: 11,
            color: "#64748b",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: 0.4,
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <span>Word</span>
          <span>Reading</span>
          <span>Meaning</span>
          <span>JLPT</span>
          <span>Seen</span>
          <span>Next review</span>
          <span />
        </div>
        {filteredWords.length === 0 && (
          <div style={{ padding: 24, textAlign: "center", color: "#64748b", fontSize: 13 }}>
            No words match.
          </div>
        )}
        {filteredWords.map((word) => (
          <div key={word.id}>
            <div
              onClick={() => toggleExpand(word)}
              style={{
                display: "grid",
                gridTemplateColumns: "1.3fr 1fr 2fr 0.7fr 0.9fr 1.2fr 0.6fr",
                gap: 8,
                alignItems: "center",
                padding: "10px 14px",
                fontSize: 13,
                borderTop: "1px solid rgba(255,255,255,0.06)",
                cursor: "pointer",
                background: expanded === word.id ? "rgba(255,255,255,0.03)" : "transparent",
              }}
            >
              <span style={{ fontWeight: 600, fontSize: 15 }}>{word.surface}</span>
              <span style={{ color: "#94a3b8" }}>{word.reading ?? ""}</span>
              <span
                style={{
                  color: "#cbd5e1",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {word.meaning ?? ""}
              </span>
              <JlptBadge level={word.jlpt_level} />
              <span style={{ color: "#94a3b8" }}>{word.times_seen}</span>
              <span style={{ color: isDue(word) ? "#f97316" : "#94a3b8", fontSize: 12 }}>
                {isDue(word) ? "Due now" : new Date(word.due_date).toLocaleDateString()}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteWord(word.id);
                }}
                style={{
                  padding: "3px 8px",
                  borderRadius: 6,
                  border: "1px solid rgba(239,68,68,0.3)",
                  background: "rgba(239,68,68,0.1)",
                  color: "#f87171",
                  fontSize: 11,
                  cursor: "pointer",
                }}
              >
                Delete
              </button>
            </div>
            {expanded === word.id && (
              <div
                style={{
                  padding: "10px 14px 16px 14px",
                  borderTop: "1px solid rgba(255,255,255,0.06)",
                  background: "rgba(255,255,255,0.02)",
                }}
              >
                {word.context_sentence && (
                  <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8 }}>
                    From: {word.context_sentence}
                  </div>
                )}
                <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>
                  Example sentences
                </div>
                {(sentencesByWord[word.surface] ?? []).length === 0 && (
                  <div style={{ fontSize: 12, color: "#4b5563" }}>
                    {sentencesByWord[word.surface] ? "None found." : "Loading…"}
                  </div>
                )}
                {(sentencesByWord[word.surface] ?? []).map((s, i) => (
                  <div key={i} style={{ marginBottom: 8, fontSize: 13 }}>
                    <div>{s.japanese}</div>
                    <div style={{ color: "#94a3b8", fontSize: 12 }}>{s.english}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- #
// Tab 2 — Flashcards
// --------------------------------------------------------------------------- #

const GRADE_BUTTONS: { label: string; grade: number; color: string }[] = [
  { label: "Again", grade: 1, color: "#ef4444" },
  { label: "Hard", grade: 2, color: "#f97316" },
  { label: "Good", grade: 4, color: "#22c55e" },
  { label: "Easy", grade: 5, color: "#14b8a6" },
];

function FlashcardsTab() {
  const [card, setCard] = useState<FlashcardResponse | null>(null);
  const [flipped, setFlipped] = useState(false);
  const [reviewedCount, setReviewedCount] = useState(0);
  const [initialDue, setInitialDue] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  const loadCard = () => {
    setFlipped(false);
    void fetch(`${API}/review/flashcard`)
      .then((r) => r.json())
      .then((res: FlashcardResponse) => {
        setCard(res);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  useEffect(() => {
    void fetch(`${API}/review/words`)
      .then((r) => r.json())
      .then((res: ReviewWordsResponse) => setInitialDue(res.due_today));
    loadCard();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const grade = (value: number) => {
    if (!card?.word) return;
    void fetch(`${API}/review/grade`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word_id: card.word.id, grade: value }),
    }).then(() => {
      setReviewedCount((c) => c + 1);
      loadCard();
    });
  };

  if (loading) {
    return <div style={{ opacity: 0.6 }}>Loading…</div>;
  }

  const total = Math.max(initialDue ?? 0, reviewedCount + (card?.word ? 1 : 0));
  const progressPct = total > 0 ? Math.min(100, (reviewedCount / total) * 100) : 0;

  return (
    <div style={{ maxWidth: 560, margin: "0 auto" }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>
          {reviewedCount} of {total} due today
        </div>
        <div
          style={{
            height: 6,
            borderRadius: 3,
            background: "rgba(255,255,255,0.08)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${progressPct}%`,
              height: "100%",
              background: "#60a5fa",
              transition: "width 200ms ease",
            }}
          />
        </div>
      </div>

      {!card?.word ? (
        <div
          style={{
            textAlign: "center",
            padding: "80px 24px",
            fontSize: 22,
            color: "#94a3b8",
          }}
        >
          🎉 All done!
        </div>
      ) : (
        <>
          <div
            onClick={() => setFlipped((f) => !f)}
            style={{
              minHeight: 280,
              borderRadius: 16,
              border: "1px solid rgba(255,255,255,0.1)",
              background: "rgba(255,255,255,0.03)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              padding: 32,
              textAlign: "center",
              gap: 12,
            }}
          >
            {!flipped ? (
              <>
                <div style={{ fontSize: 48, fontWeight: 700 }}>{card.word.surface}</div>
                {card.word.reading && (
                  <div style={{ fontSize: 18, color: "#94a3b8" }}>{card.word.reading}</div>
                )}
                <JlptBadge level={card.word.jlpt_level} />
                <div style={{ fontSize: 11, color: "#4b5563", marginTop: 16 }}>
                  Click to flip
                </div>
              </>
            ) : (
              <>
                <div style={{ fontSize: 26, fontWeight: 600 }}>{card.word.meaning ?? "—"}</div>
                {card.sentences.length > 0 && (
                  <div style={{ marginTop: 16, fontSize: 14 }}>
                    <div>{card.sentences[0].japanese}</div>
                    <div style={{ color: "#94a3b8", fontSize: 13, marginTop: 4 }}>
                      {card.sentences[0].english}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          <div style={{ display: "flex", gap: 10, marginTop: 20, justifyContent: "center" }}>
            {GRADE_BUTTONS.map((g) => (
              <button
                key={g.label}
                onClick={() => grade(g.grade)}
                style={{
                  flex: 1,
                  padding: "12px 0",
                  borderRadius: 10,
                  border: `1px solid ${g.color}55`,
                  background: `${g.color}22`,
                  color: g.color,
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {g.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- #
// Tab 3 — Stats
// --------------------------------------------------------------------------- #

function StatsTab() {
  const [review, setReview] = useState<ReviewWordsResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);

  useEffect(() => {
    void fetch(`${API}/review/words`)
      .then((r) => r.json())
      .then(setReview);
    void fetch(`${API}/stats`)
      .then((r) => r.json())
      .then(setStats);
  }, []);

  if (!review || !stats) {
    return <div style={{ opacity: 0.6 }}>Loading…</div>;
  }

  const levelCounts = LEVELS.map((level) => ({
    level,
    count: review.by_level[level]?.length ?? 0,
  }));
  const maxCount = Math.max(1, ...levelCounts.map((l) => l.count));

  const statCard = (label: string, value: string | number) => (
    <div
      style={{
        flex: 1,
        padding: "18px 20px",
        borderRadius: 12,
        border: "1px solid rgba(255,255,255,0.08)",
        background: "rgba(255,255,255,0.03)",
      }}
    >
      <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700 }}>{value}</div>
    </div>
  );

  return (
    <div style={{ maxWidth: 640, margin: "0 auto" }}>
      <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
        {statCard("Total words saved", review.total)}
        {statCard("Due today", review.due_today)}
        {statCard("Review streak", `${review.streak}🔥`)}
      </div>

      <div
        style={{
          padding: 20,
          borderRadius: 12,
          border: "1px solid rgba(255,255,255,0.08)",
          marginBottom: 24,
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14 }}>Words by JLPT level</div>
        {levelCounts.map(({ level, count }) => (
          <div key={level} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <span style={{ width: 30, fontSize: 12, color: "#94a3b8" }}>{level}</span>
            <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", borderRadius: 4, height: 16 }}>
              <div
                style={{
                  width: `${(count / maxCount) * 100}%`,
                  height: "100%",
                  borderRadius: 4,
                  background: JLPT_COLORS[level],
                  minWidth: count > 0 ? 4 : 0,
                }}
              />
            </div>
            <span style={{ width: 28, fontSize: 12, color: "#94a3b8", textAlign: "right" }}>
              {count}
            </span>
          </div>
        ))}
      </div>

      <div
        style={{
          padding: 20,
          borderRadius: 12,
          border: "1px solid rgba(255,255,255,0.08)",
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14 }}>Most reviewed words</div>
        {stats.most_seen_words.length === 0 && (
          <div style={{ fontSize: 12, color: "#64748b" }}>Nothing saved yet.</div>
        )}
        {stats.most_seen_words.map((w) => (
          <div
            key={w.id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "6px 0",
              borderBottom: "1px solid rgba(255,255,255,0.05)",
              fontSize: 13,
            }}
          >
            <span style={{ fontWeight: 600 }}>{w.surface}</span>
            <JlptBadge level={w.jlpt_level} />
            <span style={{ color: "#94a3b8", marginLeft: "auto" }}>{w.times_seen}x</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- #
// App
// --------------------------------------------------------------------------- #

export default function ReviewApp() {
  const [tab, setTab] = useState<Tab>("words");

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <TabBar tab={tab} onChange={setTab} />
      <div style={{ flex: 1, overflow: "auto", padding: 24 }}>
        {tab === "words" && <WordListTab />}
        {tab === "flashcards" && <FlashcardsTab />}
        {tab === "stats" && <StatsTab />}
      </div>
    </div>
  );
}
