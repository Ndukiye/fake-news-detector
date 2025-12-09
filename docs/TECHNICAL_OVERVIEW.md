# AuthentiScan — Technical Overview

## Purpose
AuthentiScan provides transparent, rule-based detection of fake news and phishing, with an optional lightweight ML layer. It analyzes a page’s URL and content, returns a 0–100 authenticity score, a verdict, an evidence list, and a per-test breakdown.

## Architecture
- Backend: Flask API (`app.py`) and analyzer engine (`analyzer.py`).
- Frontend: Chrome/Edge extension under `extension/` (Manifest V3) with popup UI.
- Data flow: Popup posts URL + options → backend fetches/analyses → returns score, verdict, evidence, tests → popup renders detailed UI.

## Backend API
- `GET /health` → `{ "status": "ok", "analyzer": "rule-based" }`
- `POST /analyze`
  - Input: `{ "articleLink": "<url>", "text": "<optional>", "options": { "phishing": true, "domain": true, "linguistic": true, "content": true, "fact": true, "ml": true } }`
  - Output: `{ "score": <0-100>, "verdict": "<string>", "evidence": [{ type, signal, impact }...], "tests": [{ name, status, impact, details }...] }`
- `POST /train` → Train ML from labeled examples `{ "data": [{ url, title, text, label: 1|0 }, ...] }`
- `GET /model` → Model availability `{ "available": true|false, "path": "<artifact>" }`

## Analyzer Engine (analyzer.py)
- URL/Phishing: Detects IP address domains, plain `http`, suspicious TLDs, deceptive patterns. Clean URL adds a bonus; each signal subtracts points.
- Domain Reputation: Whitelist boost for reputable domains; clean non-whitelist domains get a small bonus; suspicious TLD penalized.
- Linguistic Patterns: Flags sensational keywords, excessive caps/punctuation ratios, repeated long sentences; clean language adds a bonus.
- Title vs Content: Compares tokens; mismatch penalty only when both title and content are sufficiently long; skipped on reputable domains; alignment adds a bonus.
- Fact Verification: Extracts informative sentences; searches DuckDuckGo HTML; corroboration by reputable sources increases score; contradiction cues decrease; neutral adds a small bonus.
- ML (optional): If trained model exists, “Trained ML” adds continuous impact −15..+15 with `prob=...`; otherwise “Lightweight ML” uses interpretable feature signals.

## Scoring Model
- Baseline 50; add pass bonuses and subtract penalties; clamp to 0–100.
- Typical impacts:
  - URL/Phishing: clean +10; signals −10 each
  - Domain Reputation: whitelist +20; clean non-whitelist +5; suspicious TLD −20
  - Linguistic: clean +5; flags −5 each
  - Title vs Content: aligned +5; mismatch −5 (skipped for reputable domains)
  - Fact Verification: corroboration +10; neutral +3; contradiction −15
  - ML: probability mapped to −15..+15
- Verdict bands: 80–100 Authentic; 60–79 Likely Authentic; 40–59 Needs Review; 0–39 Likely Misleading.

## URL-only Fallback
- When content fetch fails or no text is extractable, backend runs URL-only: `phishing`, `domain`, `ml` enabled; disables text-dependent checks.
- Adds system evidence and a “Content Fetch” test with details.

## Extension
- Files: `extension/manifest.json`, `extension/popup.html`, `extension/popup.css`, `extension/popup.js`.
- UI: URL input + Analyze; checkboxes for available tests; circular score + progress bar; Evidence list; Test Breakdown with status badges and signed impacts; legend explains impacts add/subtract from the 50 baseline.
- Logic: Reads current tab URL; posts to `/analyze`; renders structured results; shows errors if backend unreachable.

## ML Training
- Lightweight trainer: No compiled deps; trains logistic-like weights over interpretable features; saves `ml_weights.json`; auto-loaded.
- Sklearn trainer (optional): TF‑IDF + LogisticRegression saving `model.pkl`/`vectorizer.pkl` (requires suitable wheels on Windows).
- Examples: `POST /train` with authentic (`label=1`) and suspicious (`label=0`) records; verify via `GET /model`.

## Deployment & Testing
- Start backend: `python app.py` (port 5000).
- Load extension: `chrome://extensions/` → Developer mode → Load unpacked → `extension/`.
- Analyze: open popup on any page; expand Evidence and Test Breakdown.
- Package: `dist/authentiscan.zip` is created for store submission.
- Troubleshooting: Backend unreachable → start server; fetch failures → URL-only fallback runs; ensure internet for fact-check.

## Configuration & Tuning
- Reputable domains: `REPUTABLE_DOMAINS` in `analyzer.py`.
- Impacts and thresholds: adjust in `calculate_authenticity_score` and `linguistic_analysis`.
- Options: enable/disable tests via `options` in requests or popup checkboxes.

## Security & Privacy
- No API keys required; analysis and ML are local.
- Only fetches content of analyzed pages; no history saved.
- External search via DuckDuckGo HTML; no paid AI services.
