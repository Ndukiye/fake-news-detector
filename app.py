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
