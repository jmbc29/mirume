// Mirrors backend/main.py's HoverResponse and friends (Pydantic -> JSON).

export interface TokenOut {
  surface: string;
  reading: string | null;
  lemma: string;
  part_of_speech: string | null;
  is_content_word: boolean;
  jlpt_level: number | null;
  jlpt_name: string;
  jlpt_estimated: boolean;
  meaning: string | null;
  dictionary_reading: string | null;
  jmdict_id: number | null;
}

export interface KanjiOut {
  literal: string;
  meaning: string;
  reading_on: string;
  reading_kun: string;
  jlpt_level: number | null;
  jlpt_name: string;
}

export interface GrammarPattern {
  pattern: string;
  jlpt_level: string;
}

export interface TranslationOut {
  source_text: string;
  translation: string;
  reading: string | null;
  jlpt_level: string;
  jlpt_name: string;
  grammar_patterns: GrammarPattern[];
  alternatives: string[];
}

export interface HoverResponse {
  text: string;
  source: string;
  language: "ja" | "en" | "mixed";
  request_point: { x: number; y: number };
  tokens: TokenOut[];
  kanji: KanjiOut[];
  translations: TranslationOut[];
}
