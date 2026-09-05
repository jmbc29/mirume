# Mirume

A macOS desktop overlay for learning Japanese. Hover over Japanese text anywhere
on screen and Mirume detects it, classifies every word by JLPT level (N5–N1),
and shows a hover card with reading, meaning, difficulty, a per-kanji breakdown,
and a plain-English translation of the sentence. Save words, sentences, and
grammar patterns to a local SQLite database and review them later with SM-2
spaced repetition. Everything stays on your machine (the optional DeepL /
Anthropic translation is the only network call, and only if you add a key).

## Project layout

```
mirume/
  backend/     FastAPI server + Japanese NLP + macOS text detection
  frontend/    Tauri + React transparent overlay (the hover card + review window)
  data/        Generated SQLite databases + raw JMdict/kanjidic2/Tatoeba source (gitignored)
  models/      fastText language-ID model, downloaded on first use (gitignored)
```

## Backend — getting started

Requires Python 3.11+ and macOS.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt

# One-time: download JMdict + kanjidic2 and build data/jmdict.db (~12 MB
# download, ~1 min). Re-run with `build --force` to refresh the sources.
python jlpt.py build

# One-time: download the Tatoeba slices and add example sentences to
# jmdict.db (~3 MB). Optional — without it the review flashcards and
# /word/{word}/sentences just return no examples.
python sentences.py build

# Port 8123 avoids a conflict with another local project on 8000.
# --reload-exclude keeps the reloader off the 30k-file venv/ (otherwise it
# pins a CPU core and a stray restart mid-request wedges /hover).
uvicorn main:app --reload --reload-exclude 'venv/*' --port 8123
```

The server listens on `http://127.0.0.1:8123`. Interactive API docs are at
`http://127.0.0.1:8123/docs`.

### macOS permissions

- **Accessibility** (required): `/hover` reads on-screen text via the
  Accessibility API. Grant it in **System Settings ▸ Privacy & Security ▸
  Accessibility** to the app running the server (your terminal, or Python),
  then restart it. Without it, `/hover` always returns the placeholder
  sentence (`source: "placeholder"`). Verify with `python accessibility.py`.
- **Screen Recording** (for the OCR fallback): Chrome/Electron web content is
  invisible to the AX tree, so `ocr.py` screenshots around the cursor and runs
  PaddleOCR. Grant Screen Recording to the same app and fully quit/reopen it.
  Without it, hovers over browser content that the AX tree misses return
  nothing. Verify with `python ocr.py <x> <y>`.

### Optional translation keys (`backend/.env`)

```
DEEPL_API_KEY=...       # English↔Japanese via DeepL (free tier: https://www.deepl.com/pro-api)
ANTHROPIC_API_KEY=...   # JA→EN sentence translation fallback (claude-sonnet-4-6) when DeepL is unset
```

With neither, the hover card still works — it just omits the English sentence
line, and English-text hovers are detected but not translated to Japanese.

### Endpoints

| Method   | Path                        | Description |
| -------- | --------------------------- | ----------- |
| `GET`    | `/health`                   | Liveness check; reports DB paths, dictionary-built and accessibility-permission status. |
| `POST`   | `/hover`                    | Accepts `{x, y}`. Reads on-screen text at that point (AX API, then screenshot OCR, then the focused element), detects Japanese / English / mixed, and returns per-word JLPT classifications + readings + meanings, a per-kanji breakdown, English→Japanese translations for any English runs, and `sentence_translation` (English gloss of the Japanese). Empty response when nothing is found or the context is blocked (dev tools, AI-chat sites). |
| `GET`    | `/word/{word}/sentences`    | Up to 3 Tatoeba example sentences for a word. |
| `POST`   | `/save/word` `/save/sentence` `/save/grammar` | Save an item to `mirume.db`. A repeat word save bumps its exposure count. |
| `POST`   | `/encounter`                | Log a passive hover exposure for progress stats. |
| `GET`    | `/review`                   | Saved words due for review today (SM-2). |
| `GET`    | `/review/words`             | Every saved word grouped by JLPT level, plus due-today count and streak (review window). |
| `GET`    | `/review/flashcard`         | Next due word + its example sentences. |
| `POST`   | `/review/grade`             | Grade a review (0–5) and reschedule per SM-2. |
| `DELETE` | `/review/words/{id}`        | Delete a saved word. |
| `GET`    | `/stats`                    | Aggregate progress statistics. |

## Frontend — getting started

Requires [Rust](https://www.rust-lang.org/tools/install) and Node 18+, plus the
backend running on port 8123.

```bash
cd frontend
npm install
npm run tauri dev        # also starts Vite; opens the transparent overlay
```

The overlay itself is invisible and click-through — it just polls the cursor
and shows the hover card (its own small window) when the cursor settles over
Japanese text. Reaching for **Save** / **Review** on the card works because
that window takes clicks (`acceptFirstMouse`); moving onto the card pins it so
it won't vanish mid-scroll.

Press **⌘⇧M** (or the card's **Review** button) to toggle the review window:
a word list grouped by JLPT level with per-word stats and delete, an SM-2
flashcard deck, and a stats tab.

Same macOS permissions as the backend apply to whichever process launches it.

### Windows

| Label    | File          | What it is |
| -------- | ------------- | ---------- |
| `main`   | `index.html`  | Full-screen transparent, click-through, above fullscreen spaces. Cursor polling + `/hover` calls only — no visible UI. |
| `card`   | `card.html`   | Small fixed-size hover card. Shown/positioned by Rust `show_card`; receives hover data from `main` over a Tauri event. |
| `review` | `review.html` | Normal opaque window, created lazily on first ⌘⇧M / Review click. |

## Backend modules

| File               | Responsibility |
| ------------------ | -------------- |
| `main.py`          | FastAPI app + routes; the `/hover` detection → routing → classification pipeline |
| `database.py`      | SQLAlchemy engines/sessions for `jmdict.db` and `mirume.db` |
| `models.py`        | `SavedWord`, `SavedSentence`, `SavedGrammar`, `WordEncounter`, `ReviewLog` |
| `tokeniser.py`     | fugashi + MeCab tokenisation (surface / reading / lemma / POS) |
| `jlpt.py`          | JMdict/kanjidic2 parser + `Entry`/`Kanji` tables + `lookup` / `classify_tokens` / `get_kanji_breakdown` / `get_example_sentences` |
| `language.py`      | fastText `lid.176` language identification (Japanese vs English vs mixed) |
| `accessibility.py` | macOS AX API text detection — `get_text_at_position` / `get_focused_text` + permission handling |
| `ocr.py`           | Screenshot + PaddleOCR fallback for content the AX tree can't see; frontmost-app / URL gating |
| `translator.py`    | English↔Japanese (DeepL) and JA→EN sentence gloss (DeepL → Claude fallback) |
| `sentences.py`     | Tatoeba example-sentence store in `jmdict.db` |
| `spaced_rep.py`    | SM-2 review scheduling |

## Data (`data/`, all gitignored)

| File | Source | Used for |
| ---- | ------ | -------- |
| `jmdict_e.xml` | [EDRDG JMdict](http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz) | word readings + meanings |
| `kanjidic2.xml` | [EDRDG kanjidic2](http://www.edrdg.org/kanjidic/kanjidic2.xml.gz) | per-kanji readings + meanings |
| `jlpt_vocab/n{1..5}.csv` | [elzup/jlpt-word-list](https://github.com/elzup/jlpt-word-list) (Tanos lists) | word-level JLPT tags |
| `kanji_data.json` | [davidluzgouveia/kanji-data](https://github.com/davidluzgouveia/kanji-data) | estimated *new* JLPT kanji levels |
| `tatoeba/*` | [Tatoeba per-language exports](https://downloads.tatoeba.org/exports/) | example sentences |
| `jmdict.db` | generated by `jlpt.py build` + `sentences.py build` | `Entry` / `Kanji` / sentence tables queried at runtime |
| `mirume.db` | created on first run | your saved words, sentences, grammar, review history |

JMdict has no official JLPT tags, so word levels come from the community-standard
(pre-2010) Tanos lists. Words on no list but made of known kanji get an estimated
level from their hardest kanji (`jlpt_estimated: true` in API responses).

## License

See [LICENSE](LICENSE).
