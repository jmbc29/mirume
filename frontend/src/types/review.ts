// Mirrors backend/main.py's review-window response models (Pydantic -> JSON).

export interface SavedWordOut {
  id: number;
  surface: string;
  lemma: string | null;
  reading: string | null;
  meaning: string | null;
  jlpt_level: string;
  context_sentence: string | null;
  times_seen: number;
  ease_factor: number;
  interval_days: number;
  repetitions: number;
  due_date: string;
  last_seen: string;
  created_at: string;
}

export interface ExampleSentenceOut {
  japanese: string;
  english: string;
}

export interface ReviewWordsResponse {
  total: number;
  by_level: Record<string, SavedWordOut[]>;
  due_today: number;
  streak: number;
}

export interface FlashcardResponse {
  word: SavedWordOut | null;
  sentences: ExampleSentenceOut[];
}

export interface StatsResponse {
  total_saved_words: number;
  total_saved_sentences: number;
  total_encounters: number;
  words_by_level: Record<string, number>;
  due_for_review: number;
  most_seen_words: SavedWordOut[];
}
