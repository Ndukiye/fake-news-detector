from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import os
import json
from analyzer import calculate_authenticity_score, train_ml_model, ml_model_status

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_article():
    data = request.get_json() or {}
    url = data.get('articleLink')
    provided_text = data.get('text')
    options = data.get('options') or {}

    if not url and not provided_text:
        return jsonify({'error': 'No article link or text provided'}), 400

    try:
        if provided_text:
            article_text = provided_text
            title = "(User-provided text)"
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Extract title
            title_tag = soup.find('title') or soup.find('h1')
            title = title_tag.get_text(strip=True) if title_tag else "(No title found)"

            # Extract main text
            main = soup.find('article') or soup.find('main')
            paragraphs = (main.find_all('p') if main else soup.find_all('p'))
            article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])

        article_text = (article_text or '').strip()
        if not article_text:
            # URL-only fallback: analyze using URL-based checks, skip text-dependent ones
            fallback_opts = {
                'phishing': bool(options.get('phishing', True)),
                'domain': bool(options.get('domain', True)),
                'linguistic': False,
                'content': False,
                'fact': False,
                'ml': bool(options.get('ml', True))
            }
            result = calculate_authenticity_score(url or "(url-only)", "(No content)", "", fallback_opts)
            result.setdefault('evidence', []).append({'type': 'system', 'signal': 'URL-only fallback (no content extracted)', 'impact': '+0'})
            result.setdefault('tests', []).append({'name': 'Content Fetch', 'status': 'neutral', 'impact': 0, 'details': ['No content extracted']})
            return jsonify(result), 200

        result = calculate_authenticity_score(url or "(text-only)", title, article_text, options)
        return jsonify(result), 200

    except requests.exceptions.RequestException as e:
        # URL-only fallback when network fetch fails
        url = (data or {}).get('articleLink')
        options = (data or {}).get('options') or {}
        if url:
            fallback_opts = {
                'phishing': bool(options.get('phishing', True)),
                'domain': bool(options.get('domain', True)),
                'linguistic': False,
                'content': False,
                'fact': False,
                'ml': bool(options.get('ml', True))
            }
            result = calculate_authenticity_score(url, "(Fetch failed)", "", fallback_opts)
            result.setdefault('evidence', []).append({'type': 'system', 'signal': 'URL-only fallback (fetch failed)', 'impact': '+0'})
            result.setdefault('tests', []).append({'name': 'Content Fetch', 'status': 'fail', 'impact': 0, 'details': [str(e)]})
            return jsonify(result), 200
        return jsonify({'error': f'Error fetching article: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'analyzer': 'rule-based'}), 200

@app.route('/', methods=['GET'])
def index():
    return (
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>AuthentiScan Backend</title>
            <style>
                body{font-family:Segoe UI,Tahoma,Arial,sans-serif;margin:0;padding:20px;background:#f6f8fa}
                .wrap{max-width:720px;margin:0 auto;background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:24px}
                h1{margin:0 0 12px;font-size:22px}
                .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
                input,textarea{width:100%;padding:10px;border:1px solid #d0d7de;border-radius:6px;font-size:14px}
                textarea{min-height:120px}
                button{padding:10px 14px;border:0;border-radius:6px;background:#2da44e;color:#fff;font-weight:600;cursor:pointer}
                .muted{color:#57606a;font-size:13px;margin:10px 0 18px}
                .result{white-space:pre-wrap;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;margin-top:12px}
                .row{display:flex;gap:10px;align-items:center}
                .badge{display:inline-block;padding:4px 8px;border-radius:999px;background:#eaeef2;color:#24292f;font-size:12px}
            </style>
        </head>
        <body>
            <div class="wrap">
                <h1>AuthentiScan Backend</h1>
                <div class="muted">Use this page to test the API endpoints. Health uses GET; Analyze uses POST with JSON.</div>
                <div class="row">
                    <span class="badge">Health</span>
                    <button id="btn-health">Check</button>
                </div>
                <div id="health" class="result" style="display:none"></div>
                <hr/>
                <div class="row" style="margin-bottom:8px">
                    <span class="badge">Analyze</span>
                    <button id="btn-analyze">Run</button>
                </div>
                <div class="grid">
                    <div>
                        <label>Article URL</label>
                        <input id="in-url" placeholder="https://example.com/article" />
                    </div>
                    <div>
                        <label>Enable ML</label>
                        <input id="in-ml" type="checkbox" checked />
                    </div>
                </div>
                <div style="margin-top:8px">
                    <label>Optional Text</label>
                    <textarea id="in-text" placeholder="Paste article text (optional)"></textarea>
                </div>
                <div class="result" id="analyze" style="display:none"></div>
            </div>
            <script>
                const healthBtn = document.getElementById('btn-health');
                const analyzeBtn = document.getElementById('btn-analyze');
                const healthBox = document.getElementById('health');
                const analyzeBox = document.getElementById('analyze');
                const inUrl = document.getElementById('in-url');
                const inText = document.getElementById('in-text');
                const inMl = document.getElementById('in-ml');
                healthBtn.onclick = async () => {
                    try {
                        const r = await fetch('/health');
                        const j = await r.json();
                        healthBox.textContent = JSON.stringify(j, null, 2);
                        healthBox.style.display = 'block';
                    } catch (e) {
                        healthBox.textContent = String(e);
                        healthBox.style.display = 'block';
                    }
                };
                analyzeBtn.onclick = async () => {
                    try {
                        const body = { articleLink: inUrl.value, text: inText.value, options: { phishing: true, domain: true, linguistic: true, content: true, fact: true, ml: inMl.checked } };
                        const r = await fetch('/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
                        const j = await r.json();
                        analyzeBox.textContent = JSON.stringify(j, null, 2);
                        analyzeBox.style.display = 'block';
                    } catch (e) {
                        analyzeBox.textContent = String(e);
                        analyzeBox.style.display = 'block';
                    }
                };
            </script>
        </body>
        </html>
        """
    )

@app.route('/model', methods=['GET'])
def model_status():
    return jsonify(ml_model_status()), 200

@app.route('/train', methods=['POST'])
def train():
    payload = request.get_json() or {}
    records = payload.get('data') or []
    result = train_ml_model(records)
    status = 200 if result.get('ok') else 400
    return jsonify(result), status

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
