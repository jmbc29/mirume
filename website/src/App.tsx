import { useState } from 'react'

const REPO_URL = 'https://github.com/jmbc29/mirume'
const RELEASES_URL = 'https://github.com/jmbc29/mirume/releases/latest'
const ISSUES_URL = `${REPO_URL}/issues`

const jlptColor: Record<string, string> = {
  N5: '#22c55e',
  N4: '#14b8a6',
  N3: '#eab308',
  N2: '#f97316',
  N1: '#ef4444',
}

const words = [
  { word: '台風', reading: 'たいふう', level: 'N4', meaning: 'typhoon' },
  { word: '前線', reading: 'ぜんせん', level: 'N2', meaning: 'front (weather); front line' },
  { word: '警戒', reading: 'けいかい', level: 'N1', meaning: 'caution, vigilance, being on alert' },
]

const features = [
  {
    title: 'JLPT N1–N5 classification',
    body: 'Every word is tagged with its JLPT level and colour-coded, so you always know if it’s worth learning right now.',
  },
  {
    title: 'Furigana and meanings',
    body: 'Readings and English definitions appear instantly, pulled from JMdict — no dictionary app, no alt-tabbing.',
  },
  {
    title: 'Works anywhere',
    body: 'Chrome, Discord, Mail, PDFs, native apps — if it renders Japanese text on screen, Mirume can read it.',
  },
  {
    title: '233,000 example sentences',
    body: 'Real sentences from Tatoeba show you how a word is actually used, translated on the fly.',
  },
  {
    title: 'Spaced repetition review',
    body: 'Save anything you hover over and review it later with built-in flashcards.',
  },
  {
    title: '100% offline and private',
    body: 'Dictionary, OCR, and lookups all run locally. Nothing you hover over ever leaves your Mac.',
  },
  {
    title: 'Sentence translation',
    body: 'Full-sentence English translation for the line you’re reading, not just the word under your cursor.',
  },
  {
    title: 'Free and open source',
    body: 'MIT licensed. No account, no subscription, no telemetry.',
  },
]

const steps = [
  {
    n: '01',
    title: 'Install',
    body: 'Drag Mirume to Applications and grant Accessibility and Screen Recording permissions once.',
  },
  {
    n: '02',
    title: 'Hover',
    body: 'Move your cursor over any Japanese text on screen — a webpage, a chat, a PDF.',
  },
  {
    n: '03',
    title: 'Learn',
    body: 'Save words you don’t know and review them later with the built-in flashcard deck.',
  },
]

function JlptBadge({ level }: { level: string }) {
  return (
    <span className="jlpt-badge" style={{ backgroundColor: jlptColor[level] }}>
      {level}
    </span>
  )
}

function HoverCardDemo() {
  const [active, setActive] = useState(2)
  const current = words[active]

  return (
    <div className="demo-window">
      <div className="demo-titlebar">
        <div className="demo-dots">
          <span style={{ background: '#ff5f57' }} />
          <span style={{ background: '#febc2e' }} />
          <span style={{ background: '#28c840' }} />
        </div>
        <div className="demo-url">news.example.co.jp</div>
      </div>

      <div className="demo-body">
        <p className="demo-headline">気象庁、関東地方に大雨警戒を呼びかけ</p>
        <p className="demo-article">
          気象庁は6日、
          <span
            className={`demo-word ${active === 0 ? 'is-active' : ''}`}
            style={{ borderColor: jlptColor.N4 }}
            onMouseEnter={() => setActive(0)}
          >
            台風
          </span>
          の接近に伴い、
          <span
            className={`demo-word ${active === 1 ? 'is-active' : ''}`}
            style={{ borderColor: jlptColor.N2 }}
            onMouseEnter={() => setActive(1)}
          >
            前線
          </span>
          の影響で大雨が続くとして、関東地方に
          <span
            className={`demo-word ${active === 2 ? 'is-active' : ''}`}
            style={{ borderColor: jlptColor.N1 }}
            onMouseEnter={() => setActive(2)}
          >
            警戒
          </span>
          レベルの引き上げを発表した。
        </p>

        <div className={`hover-card hover-card-${active}`}>
          <div className="hover-card-top">
            <div className="hover-card-word">
              <span className="hover-card-kanji">{current.word}</span>
              <span className="hover-card-reading">{current.reading}</span>
            </div>
            <JlptBadge level={current.level} />
          </div>
          <p className="hover-card-meaning">{current.meaning}</p>
          <div className="hover-card-divider" />
          <p className="hover-card-sentence-en">
            &ldquo;The Meteorological Agency raised the alert level for the Kanto region due to
            heavy rain from an approaching typhoon front.&rdquo;
          </p>
        </div>
      </div>

      <p className="demo-hint">try hovering the underlined words</p>
    </div>
  )
}

export default function App() {
  return (
    <div className="page">
      <header className="nav">
        <div className="nav-inner">
          <div className="nav-brand">
            <span className="nav-logo">見</span>
            Mirume
          </div>
          <a className="nav-link" href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </div>
      </header>

      <main>
        <section className="hero">
          <h1>Learn Japanese from anything on your screen</h1>
          <p className="hero-sub">
            Hover over any Japanese text and instantly see readings, meanings, and JLPT levels.
            Works everywhere on your Mac.
          </p>
          <div className="hero-actions">
            <a className="btn btn-primary" href={RELEASES_URL} target="_blank" rel="noreferrer">
              Download for Mac
            </a>
            <a className="btn btn-secondary" href={REPO_URL} target="_blank" rel="noreferrer">
              View on GitHub
            </a>
          </div>
        </section>

        <section className="demo-section">
          <HoverCardDemo />
        </section>

        <section className="steps-section">
          <h2>How it works</h2>
          <div className="steps-grid">
            {steps.map((step) => (
              <div className="step" key={step.n}>
                <div className="step-n">{step.n}</div>
                <h3>{step.title}</h3>
                <p>{step.body}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="features-section">
          <h2>Everything you need to read faster</h2>
          <div className="features-grid">
            {features.map((f) => (
              <div className="feature" key={f.title}>
                <h3>{f.title}</h3>
                <p>{f.body}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="download-section">
          <h2>Mirume 1.0.0</h2>
          <p className="download-sub">Free for macOS Ventura 13+</p>
          <a className="btn btn-primary btn-large" href={RELEASES_URL} target="_blank" rel="noreferrer">
            Download for Mac
          </a>
          <p className="download-note">First launch: right-click the app and choose &ldquo;Open&rdquo;.</p>
        </section>
      </main>

      <footer className="footer">
        <p>
          Built for Japanese learners · MIT License ·{' '}
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>{' '}
          ·{' '}
          <a href="/privacy.html">Privacy</a> ·{' '}
          <a href={ISSUES_URL} target="_blank" rel="noreferrer">
            Support
          </a>
        </p>
      </footer>
    </div>
  )
}
