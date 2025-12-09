document.getElementById('detectForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const articleLink = document.getElementById('articleLink').value;
    const resultsDiv = document.getElementById('results');
    const resultContent = document.getElementById('resultContent');
    const scoreBadge = document.getElementById('scoreBadge');
    const evidenceList = document.getElementById('evidenceList');
    const evidenceDetails = document.getElementById('evidenceDetails');

    // Show loading state
    resultContent.innerHTML = 'Analyzing the news article... Please wait.';
    scoreBadge.innerHTML = '';
    evidenceList.innerHTML = '';
    evidenceDetails.open = false;
    resultsDiv.classList.remove('hidden');

    try {
        const response = await fetch('http://localhost:5000/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ articleLink: articleLink })
        });

        const data = await response.json();

        if (response.ok) {
            // Score badge
            const score = data.score;
            const verdict = data.verdict;
            let badgeClass = 'badge-neutral';
            if (score >= 80) badgeClass = 'badge-good';
            else if (score >= 60) badgeClass = 'badge-ok';
            else if (score >= 40) badgeClass = 'badge-caution';
            else badgeClass = 'badge-bad';
            scoreBadge.innerHTML = `<span class="${badgeClass}">${score}/100 – ${verdict}</span>`;

            // Main result
            resultContent.innerHTML = `
                <p><strong>Article Link:</strong> <a href="${articleLink}" target="_blank">${articleLink}</a></p>
                <p><strong>Verdict:</strong> ${verdict}</p>
            `;

            // Evidence list
            if (data.evidence && data.evidence.length) {
                evidenceDetails.style.display = 'block';
                data.evidence.forEach(item => {
                    const li = document.createElement('li');
                    li.innerHTML = `<strong>${item.type.toUpperCase()}:</strong> ${item.signal} <em>(${item.impact})</em>`;
                    evidenceList.appendChild(li);
                });
            } else {
                evidenceDetails.style.display = 'none';
            }
        } else {
            resultContent.innerHTML = `<p><strong>Error:</strong> ${data.error || 'An unknown error occurred.'}</p>`;
            scoreBadge.innerHTML = '';
            evidenceDetails.style.display = 'none';
        }
    } catch (error) {
        resultContent.innerHTML = `<p><strong>Network Error:</strong> Could not connect to the backend server. Please ensure the backend is running. (${error.message})</p>`;
        scoreBadge.innerHTML = '';
        evidenceDetails.style.display = 'none';
        console.error('Error:', error);
    }
});