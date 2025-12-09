document.addEventListener('DOMContentLoaded', () => {
  let apiBase = 'https://authentiscan-601g.onrender.com';
  if (window.chrome && chrome.storage && chrome.storage.sync) {
    chrome.storage.sync.get({ backendUrl: apiBase }, (items) => {
      apiBase = items.backendUrl || apiBase;
    });
  }
  const analyzeBtn = document.getElementById('analyze-btn');
  const loading = document.getElementById('loading');
  const scoreSection = document.getElementById('score-section');
  const scoreBadge = document.getElementById('score-badge');
  const verdictText = document.getElementById('verdict-text');
  const evidenceSection = document.getElementById('evidence-section');
  const evidenceList = document.getElementById('evidence-list');
  const errorMessage = document.getElementById('error-message');
  const manualInput = document.getElementById('manual-input');
  const manualUrl = document.getElementById('manual-url');
  const testsSection = document.getElementById('tests-section');
  const testsList = document.getElementById('tests-list');
  const testsLegend = document.getElementById('tests-legend');
  const chkPhish = document.getElementById('chk-phish');
  const chkFact = document.getElementById('chk-fact');
  const chkDomain = document.getElementById('chk-domain');
  const chkLinguistic = document.getElementById('chk-linguistic');
  const chkMl = document.getElementById('chk-ml');

  if (window.chrome && chrome.tabs && chrome.tabs.query) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      manualUrl.value = tab.url;
      analyzeBtn.onclick = () => analyze(manualUrl.value.trim());
    });
  } else {
    // Fallback for non-extension environment (testing)
    analyzeBtn.onclick = () => analyze(manualUrl.value.trim());
  }

  function analyze(url) {
    show(loading);
    hide(scoreSection);
    hide(evidenceSection);
    hide(errorMessage);
    hide(testsSection);
    hide(testsLegend);

    const options = {
      phishing: chkPhish.checked,
      fact: chkFact.checked,
      domain: chkDomain.checked,
      linguistic: chkLinguistic.checked,
      content: true,
      ml: chkMl.checked
    };
    fetch(`${apiBase}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ articleLink: url, options })
    })
      .then(r => r.json())
      .then(data => {
        hide(loading);
        if (typeof data.score === 'number') {
          renderResult(data);
        } else {
          showError(data.error || 'Unknown error');
        }
      })
      .catch(err => {
        hide(loading);
        showError(`Backend unreachable. Set backend URL in extension storage (backendUrl) or ensure server at ${apiBase}`);
      });
  }

  function renderResult(data) {
    const score = data.score;
    const verdict = data.verdict;
    let cls = 'badge-neutral';
    if (score >= 80) cls = 'badge-good';
    else if (score >= 60) cls = 'badge-ok';
    else if (score >= 40) cls = 'badge-caution';
    else cls = 'badge-bad';

    scoreBadge.className = `score-circle ${cls}`;
    scoreBadge.textContent = `${score}`;
    verdictText.textContent = `${verdict}`;
    const fill = document.getElementById('bar-fill');
    fill.style.width = `${score}%`;

    if (data.evidence && data.evidence.length) {
      evidenceList.innerHTML = '';
      data.evidence.forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${item.type.toUpperCase()}:</strong> ${item.signal} <em>(${item.impact})</em>`;
        evidenceList.appendChild(li);
      });
      show(evidenceSection);
    }
    if (data.tests && data.tests.length) {
      testsList.innerHTML = '';
      data.tests.forEach(t => {
        const li = document.createElement('li');
        const statusClass = t.status === 'pass' ? 'status-pass' : (t.status === 'fail' ? 'status-fail' : 'status-neutral');
        const details = t.details && t.details.length ? `<ul class="test-details">${t.details.map(d => `<li>${d}</li>`).join('')}</ul>` : '';
        const impactSigned = t.impact > 0 ? `+${t.impact}` : `${t.impact}`;
        li.innerHTML = `<span class="test-name">${t.name}</span> <span class="${statusClass}">${t.status}</span> <span class="test-impact">${impactSigned}</span>${details}`;
        testsList.appendChild(li);
      });
      show(testsSection);
      show(testsLegend);
    }
    show(scoreSection);
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    show(errorMessage);
  }

  function show(el) {
    el.classList.remove('hidden');
  }
  function hide(el) {
    el.classList.add('hidden');
  }
});
