# AuthentiScan Extension — Getting Started

## Prerequisites
- Chrome or Edge (latest)
- Backend running locally on `http://localhost:5000`

## Load the Extension (Developer Mode)
1. Open `chrome://extensions/` (or `edge://extensions/`).
2. Enable "Developer mode".
3. Click "Load unpacked" and select the `extension/` folder.
4. Pin the extension and click the shield icon to open the popup.

## Analyze a Page
1. Navigate to any article.
2. Open the popup and click "Analyze".
3. Expand "Evidence" and "Test Breakdown" to view rule results.

## Optional: Enable ML
- Tick "Lightweight ML" in the popup. If a trained model exists, it will be used; otherwise, the embedded lightweight model runs.
- Train from labeled examples via backend:
  - POST `http://localhost:5000/train` with `{"data":[{"url":"...","title":"...","text":"...","label":1|0}]}`.
  - Check `http://localhost:5000/model` for model availability.

## Packaging for Distribution
- A prebuilt ZIP is created under `dist/authentiscan.zip`.
- For Chrome Web Store/Edge Add‑ons submission:
  - Ensure `manifest.json` includes icons and host permissions.
  - Upload `authentiscan.zip`, add screenshots, description, and privacy notes.

## Backend
- Start: `python app.py`
- Health: `GET /health`
- Analyze: `POST /analyze` with `{"articleLink":"<url>", "options":{...}}`.
- URL‑only fallback: works when content fetch fails; shows a "Content Fetch" test with details.

## Troubleshooting
- Backend unreachable: start `python app.py` on port 5000.
- DNS/fetch failures: analysis runs with URL‑only fallback; check network access.
- Fact‑check: uses DuckDuckGo HTML; ensure internet connectivity.

## Notes
- All analysis is local; no paid AI APIs.
- Scores are explainable with explicit rule impacts.
