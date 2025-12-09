import re
import time
import os
import requests
import tldextract
from bs4 import BeautifulSoup
from typing import List, Dict, Any
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
except Exception:
    TfidfVectorizer = None
    LogisticRegression = None
    joblib = None
from datetime import datetime
from urllib.parse import urlparse
from collections import Counter

SENSATIONAL_WORDS = [
    "shocking", "unbelievable", "miracle", "secret", "leaked", "exclusive",
    "urgent", "breaking", "you won’t believe", "what happens next", "click here",
    "guaranteed", "100%", "never before seen", "mind-blowing", "devastating"
]
PHISH_KEYWORDS = [
    "login", "verify", "account", "suspend", "confirm", "update", "security",
    "alert", "urgent", "expires", "click here", "immediate action"
]
SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".top", ".click", ".download", ".work",
    ".date", ".party", ".racing", ".cricket", ".science", ".trade"
]

def extract_domain_info(url):
    """Return registered domain, subdomain, and TLD."""
    ext = tldextract.extract(url)
    return ext.registered_domain, ext.subdomain, ext.suffix

def domain_age_check(url):
    """Very light domain age proxy via WHOIS XML API free tier or similar."""
    domain, _, tld = extract_domain_info(url)
    if not domain:
        return {"score": 0, "note": "Could not extract domain"}
    # Placeholder: in production call a WHOIS API
    # Example: https://www.whoisxmlapi.com/whoisserver/WhoisService?apiKey=KEY&domainName=DOMAIN&outputFormat=JSON
    # Parse creationDate, compare to now.
    # For now we return neutral.
    return {"score": 0, "note": "Domain age check skipped (needs API)"}

def phishing_url_indicators(url):
    """Return list of phishing signals in the URL."""
    signals = []
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    # IP instead of hostname
    if re.match(r"^\d+\.\d+\.\d+\.\d+", netloc):
        signals.append("Uses IP address instead of domain")
    # Excessive dashes in domain
    if netloc.count("-") >= 4:
        signals.append("Many dashes in domain (possible typosquat)")
    # Suspicious TLD
    if any(netloc.endswith(tld) for tld in SUSPICIOUS_TLDS):
        signals.append("Suspicious TLD")
    # HTTPS missing
    if parsed.scheme != "https":
        signals.append("Not using HTTPS")
    # Long query strings
    if len(parsed.query) > 100:
        signals.append("Very long query string")
    # Suspicious keywords in path
    for kw in PHISH_KEYWORDS:
        if kw in path:
            signals.append(f"Path contains '{kw}'")
    return signals

def linguistic_analysis(text):
    flags = []
    lower = text.lower()
    sensational_count = sum(1 for w in SENSATIONAL_WORDS if w in lower)
    if sensational_count >= 3:
        flags.append(f"Contains {sensational_count} sensational words")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.2:
        flags.append("High ratio of uppercase letters")
    words = text.split()
    punct_count = sum(1 for c in text if c in "!?.")
    if punct_count > len(words) * 0.3:
        flags.append("Excessive punctuation")
    norm_sents = [re.sub(r"\s+", " ", s.strip().lower()) for s in re.split(r'[.!?]', text) if len(s.strip()) >= 30]
    counts = Counter(norm_sents)
    if any(c >= 3 for c in counts.values()):
        flags.append("Repeated sentences/phrases")
    return flags

def fact_check_snippets(text, max_queries=3):
    """
    Lightweight fact check using DuckDuckGo HTML results (no API key).
    - Extract up to max_queries informative sentences
    - Search each sentence
    - Score based on corroboration from reputable domains and snippet overlap
    Returns score in [-1, 1] and evidence list of notable matches
    """
    def extract_sentences(t):
        sents = [s.strip() for s in re.split(r'[.!?]', t) if len(s.strip()) >= 40]
        sents.sort(key=len, reverse=True)
        return sents[:max_queries]

    def search_duckduckgo(q, max_results=5):
        url = "https://duckduckgo.com/html/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
        }
        try:
            resp = requests.post(url, data={'q': q}, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            results = []
            for a in soup.select('a.result__a')[:max_results]:
                title = a.get_text(strip=True)
                href = a.get('href')
                # DuckDuckGo wraps URLs sometimes; attempt to extract real url
                real = href
                if real and real.startswith('/?'):  # fallback
                    real = f"https://duckduckgo.com{real}"
                snippet_el = a.find_parent('div').select_one('.result__snippet') if a.find_parent('div') else None
                snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ''
                results.append({'title': title, 'url': real, 'snippet': snippet})
            return results
        except Exception:
            return []

    def score_results(sentence, results):
        sent_tokens = set(re.findall(r'[a-zA-Z]{4,}', sentence.lower()))
        corroboration = 0
        notes = []
        for r in results:
            url = r.get('url') or ''
            reg_domain, _, _ = extract_domain_info(url)
            snip = (r.get('snippet') or '').lower()
            overlap = len(sent_tokens & set(re.findall(r'[a-zA-Z]{4,}', snip)))
            if reg_domain in REPUTABLE_DOMAINS and overlap >= 3:
                corroboration += 0.4
                notes.append(f"Corroborated by {reg_domain}: {r.get('title','')}")
            elif overlap >= 5:
                corroboration += 0.2
                notes.append(f"Similar phrasing found: {r.get('title','')}")
            # Simple contradiction heuristics
            if any(kw in snip for kw in ["false", "debunk", "misleading", "hoax", "not true"]):
                corroboration -= 0.4
                notes.append(f"Contradiction cue in {reg_domain}: {r.get('title','')}")
        return max(-1.0, min(1.0, corroboration)), notes

    sentences = extract_sentences(text)
    if not sentences:
        return {"score": 0.0, "evidence": ["No suitable sentences for fact-checking"]}

    total = 0.0
    evidence = []
    for s in sentences:
        results = search_duckduckgo(s)
        sc, notes = score_results(s, results)
        total += sc
        evidence.extend(notes[:3])

    avg = total / max(len(sentences), 1)
    return {"score": avg, "evidence": evidence if evidence else ["No corroboration or contradictions found"]}

REPUTABLE_DOMAINS = {
    "bbc.co.uk", "bbc.com", "reuters.com", "apnews.com", "nytimes.com",
    "theguardian.com", "washingtonpost.com", "wsj.com", "ft.com", "bloomberg.com"
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECT_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'ml_weights.json')
_ML_MODEL = None
_ML_VECT = None
_ML_WEIGHTS = None

def _load_ml_artifacts():
    global _ML_MODEL, _ML_VECT
    if joblib and os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        try:
            _ML_MODEL = joblib.load(MODEL_PATH)
            _ML_VECT = joblib.load(VECT_PATH)
        except Exception:
            _ML_MODEL = None
            _ML_VECT = None
    # Lightweight trained weights fallback
    if os.path.exists(WEIGHTS_PATH):
        try:
            import json
            with open(WEIGHTS_PATH, 'r', encoding='utf-8') as f:
                _ML_WEIGHTS = json.load(f)
        except Exception:
            _ML_WEIGHTS = None

_load_ml_artifacts()

def _sigmoid(x):
    return 1 / (1 + (2.718281828459045 ** (-x)))

def lightweight_ml_score(url, title, text):
    looks_url = False
    https = 0
    phish_count = 0
    reputable = 0
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    punct_ratio = (sum(1 for c in text if c in "!?.") / max(len(text.split()), 1))
    sensational_count = sum(1 for w in SENSATIONAL_WORDS if w in text.lower())
    title_overlap = 0.0

    try:
        parsed = urlparse(url)
        looks_url = parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        looks_url = False

    if looks_url:
        https = 1 if parsed.scheme == "https" else 0
        phish_count = len(phishing_url_indicators(url))
        reg_domain, _, _ = extract_domain_info(url)
        reputable = 1 if reg_domain in REPUTABLE_DOMAINS else 0

    if title:
        stop = {"the","a","an","and","or","of","to","in","on","for","with","by","at","from","as"}
        t_tokens = [w for w in re.findall(r"[a-zA-Z]+", title.lower()) if w not in stop]
        x_tokens = [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w not in stop]
        title_overlap = len(set(t_tokens) & set(x_tokens)) / max(len(set(t_tokens)), 1)

    w_https = 0.8
    w_reputable = 1.0
    w_caps = -1.2
    w_punct = -0.8
    w_sens = -0.6
    w_phish = -0.9
    w_title = 0.5
    bias = 0.0

    z = (
        w_https * https +
        w_reputable * reputable +
        w_caps * caps_ratio +
        w_punct * punct_ratio +
        w_sens * min(sensational_count, 6) / 6.0 +
        w_phish * min(phish_count, 5) / 5.0 +
        w_title * title_overlap +
        bias
    )
    prob = _sigmoid(z)
    return {
        "prob": prob,
        "features": {
            "https": https,
            "reputable": reputable,
            "caps_ratio": round(caps_ratio, 3),
            "punct_ratio": round(punct_ratio, 3),
            "sensational_count": sensational_count,
            "phish_count": phish_count,
            "title_overlap": round(title_overlap, 3)
        }
    }

def ml_model_status() -> Dict[str, Any]:
    return {
        "available": (_ML_MODEL is not None and _ML_VECT is not None and LogisticRegression is not None) or (_ML_WEIGHTS is not None),
        "path": MODEL_PATH if _ML_MODEL is not None else (WEIGHTS_PATH if _ML_WEIGHTS is not None else None)
    }

def train_ml_model(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # If sklearn is available, train TF-IDF + LR; otherwise train lightweight weights via SGD
    if TfidfVectorizer is None or LogisticRegression is None or joblib is None:
        # Lightweight training
        def features_for(r):
            url = r.get('url') or ''
            title = r.get('title') or ''
            text = r.get('text') or ''
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            punct_ratio = (sum(1 for c in text if c in "!?.") / max(len(text.split()), 1))
            sensational_count = sum(1 for w in SENSATIONAL_WORDS if w in text.lower())
            phish_count = len(phishing_url_indicators(url)) if url else 0
            reputable = 0
            https = 0
            try:
                parsed = urlparse(url)
                https = 1 if parsed.scheme == 'https' else 0
                reg_domain, _, _ = extract_domain_info(url)
                reputable = 1 if reg_domain in REPUTABLE_DOMAINS else 0
            except Exception:
                pass
            stop = {"the","a","an","and","or","of","to","in","on","for","with","by","at","from","as"}
            t_tokens = [w for w in re.findall(r"[a-zA-Z]+", title.lower()) if w not in stop]
            x_tokens = [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w not in stop]
            title_overlap = len(set(t_tokens) & set(x_tokens)) / max(len(set(t_tokens)), 1)
            return [https, reputable, caps_ratio, punct_ratio, min(sensational_count, 6)/6.0, min(phish_count,5)/5.0, title_overlap]

        X = []
        y = []
        for r in records:
            X.append(features_for(r))
            y.append(int(r.get('label', 0)))
        if len(X) < 4:
            return {"ok": False, "error": "Need at least 4 samples"}
        # SGD on logistic loss
        import random
        random.seed(0)
        w = [0.5, 0.8, -1.0, -0.8, -0.6, -0.9, 0.5]
        b = 0.0
        lr = 0.1
        for epoch in range(200):
            for i in range(len(X)):
                z = sum(w[j]*X[i][j] for j in range(len(w))) + b
                p = 1/(1+ (2.718281828459045 ** (-z)))
                grad = p - y[i]
                for j in range(len(w)):
                    w[j] -= lr * grad * X[i][j]
                b -= lr * grad
        # Save
        try:
            import json
            with open(WEIGHTS_PATH, 'w', encoding='utf-8') as f:
                json.dump({"w": w, "b": b}, f)
            global _ML_WEIGHTS
            _ML_WEIGHTS = {"w": w, "b": b}
            # Compute simple accuracy
            correct = 0
            for i in range(len(X)):
                z = sum(w[j]*X[i][j] for j in range(len(w))) + b
                p = 1/(1+ (2.718281828459045 ** (-z)))
                pred = 1 if p >= 0.5 else 0
                if pred == y[i]:
                    correct += 1
            acc = correct/len(X)
            return {"ok": True, "train_accuracy": acc, "samples": len(X), "mode": "lightweight"}
        except Exception as e:
            return {"ok": False, "error": f"Save error: {e}"}
    texts = []
    labels = []
    for r in records:
        url = r.get('url') or ''
        title = r.get('title') or ''
        text = r.get('text') or ''
        label = int(r.get('label', 0))
        texts.append(f"{title}\n{text}\n{url}")
        labels.append(label)
    if len(texts) < 4:
        return {"ok": False, "error": "Need at least 4 samples"}
    vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, labels)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)
    global _ML_MODEL, _ML_VECT
    _ML_MODEL, _ML_VECT = clf, vect
    acc = float(clf.score(X, labels))
    return {"ok": True, "train_accuracy": acc, "samples": len(texts)}

def predict_ml_probability(url: str, title: str, text: str) -> float:
    # Prefer trained sklearn model if available, else use lightweight weights
    if _ML_MODEL is not None and _ML_VECT is not None:
        s = f"{title}\n{text}\n{url}"
        X = _ML_VECT.transform([s])
        if hasattr(_ML_MODEL, 'predict_proba'):
            proba = _ML_MODEL.predict_proba(X)[0, 1]
        else:
            proba = float(_ML_MODEL.decision_function(X))
            proba = _sigmoid(proba)
        return float(proba)
    if _ML_WEIGHTS is not None:
        w = _ML_WEIGHTS.get('w', [])
        b = _ML_WEIGHTS.get('b', 0.0)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        punct_ratio = (sum(1 for c in text if c in "!?.") / max(len(text.split()), 1))
        sensational_count = sum(1 for w_ in SENSATIONAL_WORDS if w_ in text.lower())
        phish_count = len(phishing_url_indicators(url)) if url else 0
        reputable = 0
        https = 0
        try:
            parsed = urlparse(url)
            https = 1 if parsed.scheme == 'https' else 0
            reg_domain, _, _ = extract_domain_info(url)
            reputable = 1 if reg_domain in REPUTABLE_DOMAINS else 0
        except Exception:
            pass
        stop = {"the","a","an","and","or","of","to","in","on","for","with","by","at","from","as"}
        t_tokens = [w_ for w_ in re.findall(r"[a-zA-Z]+", title.lower()) if w_ not in stop]
        x_tokens = [w_ for w_ in re.findall(r"[a-zA-Z]+", text.lower()) if w_ not in stop]
        title_overlap = len(set(t_tokens) & set(x_tokens)) / max(len(set(t_tokens)), 1)
        feats = [https, reputable, caps_ratio, punct_ratio, min(sensational_count,6)/6.0, min(phish_count,5)/5.0, title_overlap]
        z = sum((w[i] if i < len(w) else 0.0)*feats[i] for i in range(len(feats))) + b
        return _sigmoid(z)
    return None
def calculate_authenticity_score(url, title, text, options=None):
    """
    Combine all signals into a 0-100 authenticity score.
    100 = very likely authentic, 0 = very likely fake/phish.
    """
    evidence = []
    tests = []
    score = 50

    # Helper: check if the provided url looks valid
    def _looks_like_url(u: str) -> bool:
        try:
            parsed = urlparse(u)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    opts = {"phishing": True, "domain": True, "linguistic": True, "content": True, "fact": True, "ml": False}
    if isinstance(options, dict):
        for k in list(opts.keys()):
            if k in options:
                opts[k] = bool(options[k])

    phish_signals = []
    if opts["phishing"]:
        if isinstance(url, str) and _looks_like_url(url):
            phish_signals = phishing_url_indicators(url)
            for sig in phish_signals:
                score -= 10
                evidence.append({"type": "phishing", "signal": sig, "impact": "-10"})
        if phish_signals:
            tests.append({
                "name": "URL/Phishing",
                "status": "fail",
                "impact": -10 * len(phish_signals),
                "details": phish_signals
            })
        else:
            pass_bonus = 10 if isinstance(url, str) and _looks_like_url(url) else 0
            score += pass_bonus
            tests.append({
                "name": "URL/Phishing",
                "status": "pass",
                "impact": pass_bonus,
                "details": []
            })
            if pass_bonus:
                evidence.append({"type": "phishing", "signal": "No phishing signals", "impact": f"+{pass_bonus}"})
    else:
        tests.append({"name": "URL/Phishing", "status": "disabled", "impact": 0, "details": []})

    if opts["domain"]:
        domain_info = domain_age_check(url) if (isinstance(url, str) and _looks_like_url(url)) else {"score": 0, "note": "Domain checks skipped (no URL)"}
        domain_details = []
        domain_impact = 0
        if "suspicious TLD" in [s for s in phish_signals]:
            score -= 20
            domain_impact -= 20
            domain_details.append("Suspicious TLD")
            evidence.append({"type": "domain", "signal": "Suspicious TLD", "impact": "-20"})
        if isinstance(url, str) and _looks_like_url(url):
            reg_domain, _, _ = extract_domain_info(url)
            if reg_domain in REPUTABLE_DOMAINS:
                score += 20
                domain_impact += 20
                domain_details.append("Reputable domain (whitelist)")
                evidence.append({"type": "domain", "signal": "Reputable domain (whitelist)", "impact": "+20"})
            else:
                if domain_impact == 0:
                    score += 5
                    domain_impact += 5
                    domain_details.append("Domain checks passed")
                    evidence.append({"type": "domain", "signal": "Domain checks passed", "impact": "+5"})
        if domain_info["score"] < 0:
            score += domain_info["score"]
            domain_impact += domain_info["score"]
            domain_details.append(domain_info["note"])
            evidence.append({"type": "domain", "signal": domain_info["note"], "impact": str(domain_info["score"])})
        tests.append({
            "name": "Domain Reputation",
            "status": "pass" if domain_impact > 0 else ("fail" if domain_impact < 0 else "neutral"),
            "impact": domain_impact,
            "details": domain_details
        })
    else:
        tests.append({"name": "Domain Reputation", "status": "disabled", "impact": 0, "details": []})

    if opts["linguistic"]:
        linguistics = linguistic_analysis(text)
        if linguistics:
            for flag in linguistics:
                score -= 5
                evidence.append({"type": "linguistic", "signal": flag, "impact": "-5"})
            tests.append({
                "name": "Linguistic Patterns",
                "status": "fail",
                "impact": -5 * len(linguistics),
                "details": linguistics
            })
        else:
            score += 5
            tests.append({
                "name": "Linguistic Patterns",
                "status": "pass",
                "impact": 5,
                "details": []
            })
            evidence.append({"type": "linguistic", "signal": "No linguistic red flags", "impact": "+5"})
    else:
        tests.append({"name": "Linguistic Patterns", "status": "disabled", "impact": 0, "details": []})

    if opts["content"] and title:
        stop = {"the","a","an","and","or","of","to","in","on","for","with","by","at","from","as"}
        title_tokens = [w for w in re.findall(r"[a-zA-Z]+", title.lower()) if w not in stop]
        text_tokens = [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w not in stop]
        title_set = set(title_tokens)
        text_set = set(text_tokens)
        overlap = len(title_set & text_set) / max(len(title_set), 1)
        penalty = 0
        if len(title_tokens) >= 5 and len(text_tokens) >= 100 and overlap < 0.2:
            penalty = -5
            if isinstance(url, str) and _looks_like_url(url):
                reg_domain, _, _ = extract_domain_info(url)
                if reg_domain in REPUTABLE_DOMAINS:
                    penalty = 0
        if penalty:
            score += penalty
            evidence.append({"type": "content", "signal": "Title/content mismatch", "impact": str(penalty)})
        else:
            score += 5
        tests.append({
            "name": "Title vs Content",
            "status": "fail" if penalty < 0 else "pass",
            "impact": penalty if penalty < 0 else 5,
            "details": ["Mismatch"] if penalty < 0 else []
        })
    elif not opts["content"]:
        tests.append({"name": "Title vs Content", "status": "disabled", "impact": 0, "details": []})

    if opts["fact"]:
        fact_result = fact_check_snippets(text)
        fact_impact = 0
        if fact_result["score"] < -0.3:
            score -= 15
            fact_impact -= 15
            evidence.append({"type": "fact", "signal": "Possible contradictions found", "impact": "-15"})
        elif fact_result["score"] > 0.3:
            score += 10
            fact_impact += 10
            evidence.append({"type": "fact", "signal": "Corroborated by search", "impact": "+10"})
        else:
            score += 3
            fact_impact += 3
        tests.append({
            "name": "Fact Verification",
            "status": "pass" if fact_impact > 0 else ("fail" if fact_impact < 0 else "neutral"),
            "impact": fact_impact,
            "details": fact_result.get("evidence", [])
        })
    else:
        tests.append({"name": "Fact Verification", "status": "disabled", "impact": 0, "details": []})

    # 6. Lightweight ML (optional)
    if opts.get("ml"):
        prob = predict_ml_probability(url, title, text)
        if prob is None:
            ml = lightweight_ml_score(url, title, text)
            impact = round((ml["prob"] - 0.5) * 30)
            score += impact
            tests.append({
                "name": "Lightweight ML",
                "status": "pass" if impact > 0 else ("fail" if impact < 0 else "neutral"),
                "impact": impact,
                "details": [f"prob={ml['prob']:.2f}"]
            })
        else:
            impact = round((prob - 0.5) * 30)
            score += impact
            tests.append({
                "name": "Trained ML",
                "status": "pass" if impact > 0 else ("fail" if impact < 0 else "neutral"),
                "impact": impact,
                "details": [f"prob={prob:.2f}"]
            })

    # Clamp
    score = max(0, min(100, score))
    verdict = (
        "Authentic" if score >= 80 else
        "Likely Authentic" if score >= 60 else
        "Needs Review" if score >= 40 else
        "Likely Misleading" if score >= 20 else
        "Fake/Phish"
    )
    return {"score": score, "verdict": verdict, "evidence": evidence, "tests": tests}
