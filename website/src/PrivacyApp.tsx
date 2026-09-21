const REPO_URL = 'https://github.com/jmbc29/mirume'
const ISSUES_URL = `${REPO_URL}/issues`

export default function PrivacyApp() {
  return (
    <div className="page">
      <header className="nav">
        <div className="nav-inner">
          <a className="nav-brand" href="/" style={{ textDecoration: 'none' }}>
            <span className="nav-logo">見</span>
            Mirume
          </a>
          <a className="nav-link" href="/">
            Back to home
          </a>
        </div>
      </header>

      <main>
        <section className="legal">
          <h1>Privacy Policy</h1>
          <p className="legal-updated">Last updated September 2026</p>

          <p>
            Mirume is a hover-to-learn overlay that reads Japanese text on your screen. This
            page explains, plainly, what that involves.
          </p>

          <h2>Everything stays on your Mac</h2>
          <p>
            The dictionary, JLPT classification, OCR, spaced-repetition scheduling, and your
            saved words, sentences, and grammar patterns all live in local files on your Mac —
            a SQLite database in your Application Support folder. Mirume has no server. There is
            nothing to sync, and nothing we can see, because we never receive it.
          </p>

          <h2>What leaves your Mac (and only if you set it up)</h2>
          <p>
            Mirume can show a full-sentence English translation alongside the word-by-word
            breakdown. That single feature is the only thing that ever leaves your machine, and
            only when you've added an API key for it:
          </p>
          <ul className="legal-list">
            <li>
              <strong>DeepL</strong> — if you add a <code>DEEPL_API_KEY</code>, the sentence
              you're hovering is sent to DeepL's translation API to be translated.
            </li>
            <li>
              <strong>Anthropic (Claude)</strong> — if you add an <code>ANTHROPIC_API_KEY</code>{' '}
              instead (or DeepL is rate-limited), the sentence is sent to Anthropic's API for the
              same purpose.
            </li>
          </ul>
          <p>
            With no key configured, neither call happens — Mirume just omits the sentence
            translation and works entirely offline. Either way, only the specific sentence
            you're hovering is ever sent, and only to translate it. Nothing is sent in the
            background, on a schedule, or for any purpose other than fulfilling that one request.
          </p>

          <h2>No analytics, no tracking, no accounts</h2>
          <p>
            Mirume doesn't have an account system, doesn't phone home, and doesn't include any
            analytics or crash-reporting SDK. There's no telemetry to opt out of because none is
            collected. Error logs are written only to a file on your own Mac (
            <code>~/Library/Logs/Mirume/</code>) for your own debugging — they never leave your
            machine either.
          </p>

          <h2>Support</h2>
          <p>
            Found a bug, or have a question? The GitHub issue tracker is Mirume's only support
            channel — <a href={ISSUES_URL} target="_blank" rel="noreferrer">open an issue</a> and
            we'll take a look. Since it's a public tracker, don't include anything sensitive in a
            report.
          </p>
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
