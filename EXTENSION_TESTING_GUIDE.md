# Browser Extension Testing Guide

## 🛡️ Fake News Detector Extension Testing

### Prerequisites
- Chrome or Edge browser
- Flask backend running on http://localhost:5000
- Extension files in the `extension/` directory

### Step 1: Load the Extension

1. **Open Chrome Extensions Page**
   - Navigate to `chrome://extensions/`
   - Enable "Developer mode" toggle (top right)

2. **Load Extension**
   - Click "Load unpacked" button
   - Select the `extension/` directory
   - Extension should appear in your extensions list

3. **Verify Installation**
   - Look for the 🛡️ shield icon in your browser toolbar
   - If not visible, click the puzzle piece icon and pin the extension

### Step 2: Test Extension Functionality

#### Test 1: Good Article Analysis
1. Navigate to a reputable news site (e.g., reuters.com, bbc.com)
2. Click the extension icon
3. Verify:
   - Popup appears with current URL detected
   - "Analyze Article" button is enabled
   - Click to analyze and wait for results
   - Should show high authenticity score (70+)

#### Test 2: Suspicious Content Detection
1. Use the test page at: http://localhost:8080/extension_test.html
2. Click "Test Suspicious Article" button
3. Verify the extension detects:
   - Sensational language patterns
   - Excessive punctuation
   - Clickbait indicators
   - Lower authenticity score (30-50)

#### Test 3: Phishing URL Detection
1. Use the test page and click "Test Phishing URL"
2. Verify the extension detects:
   - IP address URLs
   - HTTP (non-HTTPS) usage
   - Urgency language patterns
   - Very low authenticity score (0-30)

### Step 3: Verify Evidence Display

Each analysis should show:
- **Score Badge**: Color-coded (green/yellow/orange/red)
- **Verdict**: Clear classification (Reliable/Likely Reliable/Likely Misleading/Misleading)
- **Evidence Breakdown**: Specific reasons with impact scores
- **Rule Categories**: phishing/linguistic/content/domain

### Step 4: Test Error Handling

1. **Backend Offline**: Stop Flask server and test - should show error message
2. **Network Issues**: Disconnect internet and test - should handle gracefully
3. **Invalid URLs**: Test with malformed URLs - should not crash

### Expected Results Summary

| Test Case | Expected Score | Key Indicators |
|-----------|---------------|----------------|
| Good Article | 70-100 | HTTPS, neutral tone, credible domain |
| Suspicious Article | 30-50 | Sensational language, excessive caps/punctuation |
| Phishing URL | 0-30 | IP address, HTTP, urgency language |

### Troubleshooting

**Extension Not Loading:**
- Verify all files are in `extension/` directory
- Check manifest.json syntax
- Ensure Chrome Developer mode is enabled

**Backend Connection Errors:**
- Confirm Flask is running on localhost:5000
- Check CORS headers in app.py
- Verify extension has proper permissions

**No Analysis Results:**
- Check browser console for JavaScript errors
- Verify content extraction is working
- Test backend directly with curl commands

### Performance Notes

- Analysis typically takes 2-5 seconds
- Extension popup auto-sizes to content
- Evidence list is scrollable for long results
- Score colors update immediately upon analysis

## 🎉 Success Criteria

✅ Extension loads without errors  
✅ Popup displays current page URL  
✅ Analysis returns consistent scores  
✅ Evidence breakdown shows specific reasons  
✅ Color coding matches score ranges  
✅ Error handling works properly  
✅ Both web app and extension work identically  

The extension is ready for use when all these criteria are met!