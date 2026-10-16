/* ===============================
   BACKEND CONFIG
=============================== */

// The API is served from the same address as the site: by Vercel in
// production and by Flask in local development. To use another backend, set
// window.HTEP_API_BASE before this script loads.
function getApiBase() {
    return window.HTEP_API_BASE || '';
}

// The free hosting plan accepts request bodies up to 4.5 MB
const MAX_FILE_BYTES = 4 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = ['pdf', 'png', 'jpg', 'jpeg'];

document.addEventListener('DOMContentLoaded', () => {

    /* ===============================
       THEME TOGGLE (SUN / MOON SVG)
    =============================== */

    const root = document.documentElement;
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    // Theme is applied in <head> before paint; fall back to light
    let savedTheme = root.getAttribute('data-theme');
    if (savedTheme !== 'dark' && savedTheme !== 'light') savedTheme = 'light';
    root.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const newTheme = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            root.setAttribute('data-theme', newTheme);
            try { localStorage.setItem('theme', newTheme); } catch (_) {}
            updateThemeIcon(newTheme);
        });
    }

    function updateThemeIcon(theme) {
        if (themeIcon) {
            themeIcon.src = theme === 'dark' ? 'assets/sun.svg' : 'assets/moon.svg';
        }
        if (themeToggle) {
            themeToggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
        }
    }

    /* ===============================
       INDEX PAGE LOGIC (UPLOAD)
    =============================== */

    const pdfInput = document.getElementById('pdfInput');
    const extractBtn = document.getElementById('extractBtn');
    const uploadBox = document.getElementById('uploadBox');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const status = document.getElementById('statusMessage');

    if (!pdfInput || !extractBtn) return;

    const extractBtnLabel = extractBtn.innerHTML;
    let selectedFile = null;

    // Wake the backend while the user picks a file (Cloud Run cold starts take ~1-2 min)
    const warmup = warmUpBackend();

    pdfInput.addEventListener('change', () => {
        if (pdfInput.files.length > 0) selectFile(pdfInput.files[0]);
    });

    if (uploadBox) {
        // Whole drop zone opens the picker (covers the "Browse files" button too)
        uploadBox.addEventListener('click', (e) => {
            if (e.target === pdfInput || extractBtn.classList.contains('is-busy')) return;
            pdfInput.click();
        });

        ['dragenter', 'dragover'].forEach((evt) => {
            uploadBox.addEventListener(evt, (e) => {
                e.preventDefault();
                uploadBox.classList.add('is-dragover');
            });
        });

        uploadBox.addEventListener('dragleave', (e) => {
            if (!uploadBox.contains(e.relatedTarget)) uploadBox.classList.remove('is-dragover');
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('is-dragover');
            if (extractBtn.classList.contains('is-busy')) return;
            const files = e.dataTransfer && e.dataTransfer.files;
            if (files && files.length > 0) selectFile(files[0]);
        });

        // A file dropped outside the zone shouldn't navigate away from the page
        ['dragover', 'drop'].forEach((evt) => {
            window.addEventListener(evt, (e) => e.preventDefault());
        });
    }

    function selectFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();

        if (!ACCEPTED_EXTENSIONS.includes(ext)) {
            resetSelection();
            setStatus('Unsupported file type. Please upload a PDF, JPEG or PNG.', true);
            return;
        }
        if (file.size > MAX_FILE_BYTES) {
            resetSelection();
            setStatus(`That file is ${formatBytes(file.size)}. Please upload a file under 4 MB.`, true);
            return;
        }

        selectedFile = file;
        setStatus('');
        renderFileChip(file, ext);
        if (uploadBox) uploadBox.classList.add('has-file');
        extractBtn.disabled = false;
    }

    function resetSelection() {
        selectedFile = null;
        pdfInput.value = '';
        if (fileNameDisplay) fileNameDisplay.textContent = '';
        if (uploadBox) uploadBox.classList.remove('has-file');
        extractBtn.disabled = true;
    }

    function renderFileChip(file, ext) {
        if (!fileNameDisplay) return;
        fileNameDisplay.textContent = '';

        const icon = document.createElement('i');
        icon.className = 'fas ' + (ext === 'pdf' ? 'fa-file-pdf' : 'fa-file-image');
        icon.setAttribute('aria-hidden', 'true');

        const name = document.createElement('span');
        name.className = 'file-name';
        name.textContent = file.name;

        const size = document.createElement('span');
        size.className = 'file-size';
        size.textContent = formatBytes(file.size);

        fileNameDisplay.append(icon, name, size);
    }

    function setStatus(message, isError = false) {
        if (!status) return;
        status.textContent = message;
        status.classList.toggle('is-error', isError && !!message);
    }

    extractBtn.addEventListener('click', async () => {

        const file = selectedFile;
        if (!file) return;

        extractBtn.disabled = true;
        extractBtn.classList.add('is-busy');
        extractBtn.innerHTML = '<span class="btn-spinner" aria-hidden="true"></span> Extracting…';
        setStatus('');

        const formData = new FormData();
        formData.append('file', file);

        let timeoutId, ticker;

        // Show elapsed time while processing; `phase` says what is happening now
        let elapsed = 0;
        let phase = 'Waiting for the extraction engine to start…';
        const tick = () => {
            const hint = elapsed >= 20 ? ' Multi-page PDFs and handwritten prescriptions take longer.' : '';
            setStatus(`${phase} ${elapsed}s.${hint}`);
        };

        try {
            ticker = setInterval(() => { elapsed++; tick(); }, 1000);

            // Preferred: PaddleOCR's models on the user's GPU (public/ocr.js); only
            // the recognized text goes to the server. Otherwise the server reads it.
            let data = await readInBrowser(file, (msg) => { phase = msg; tick(); });

            if (!data) {
                // Don't race the warm-up: both requests would load the ML engines at once
                phase = 'Waiting for the extraction engine to start…';
                await warmup;

                // Abort just after the host's 300 s request limit so the server's limit governs
                const controller = new AbortController();
                timeoutId = setTimeout(() => controller.abort(), 310 * 1000);
                phase = 'Reading your prescription…';
                tick();

                const response = await fetch(getApiBase() + '/api/digitize', {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });
                if (!response.ok) throw new Error(await serverError(response));
                data = await response.json();
                data.ocr_location = 'server';
            }
            clearTimeout(timeoutId);
            clearInterval(ticker);

            // Store corrected text (falls back to ocr_text if not available)
            const displayText = data.corrected_text || data.ocr_text || data.text || '';
            localStorage.setItem('extractedText', displayText);
            localStorage.setItem('fileName', file.name);

            // Store full response for output page (drugs, diseases, corrections)
            localStorage.setItem('htepResponse', JSON.stringify(data));

            window.location.href = 'output.html';

        } catch (error) {
            clearTimeout(timeoutId);
            clearInterval(ticker);
            console.error(error);
            let errorMsg = "Error processing file.";
            if (error.name === 'AbortError') {
                errorMsg = "Request timed out — the server took too long. Try a smaller image or try again later.";
            } else if (error instanceof TypeError) {
                errorMsg = "The extraction server didn't respond — it may be restarting. Please try again in a minute.";
            } else if (error.message) {
                errorMsg = "Error: " + error.message;
            }
            setStatus(errorMsg, true);
            extractBtn.disabled = false;
            extractBtn.classList.remove('is-busy');
            extractBtn.innerHTML = extractBtnLabel;
        }
    });

});

// OCR in the browser, then build the record on the server from the text.
// Resolves to the record, or null if this browser can't (no WebGPU) or the
// in-browser OCR failed, in which case the caller uses the server's OCR.
async function readInBrowser(file, onStatus) {
    let ocr;
    try {
        ocr = await import('./ocr.js');
        if (!(await ocr.isSupported())) return null;
    } catch (err) {
        console.warn('In-browser OCR unavailable:', err);
        return null;
    }

    let read;
    try {
        read = await ocr.readDocument(file, onStatus);
    } catch (err) {
        // Page limits are the user's to fix; anything else falls back to the server
        if (/limit is/.test(err.message || '')) throw err;
        console.warn('In-browser OCR failed, using the server instead:', err);
        return null;
    }

    onStatus('Extracting the prescription details…');
    const response = await fetch(getApiBase() + '/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file: file.name, pages: read.pages }),
    });
    if (!response.ok) throw new Error(await serverError(response));
    const data = await response.json();
    // The on-device vision model reads this preview if the OCR looks unreliable
    data.page_image = data.vlm && data.vlm.recommended ? read.preview : null;
    data.ocr_location = 'browser';
    return data;
}

async function serverError(response) {
    let message = "Server error (" + response.status + ")";
    try {
        const body = await response.json();
        if (body.error) message = body.error;
    } catch (_) {}
    return message;
}

/* ===============================
   BACKEND WARM-UP / STATUS PILL
=============================== */

async function warmUpBackend() {
    const pill = document.getElementById('engineStatus');
    const label = document.getElementById('engineStatusText');

    const setState = (state, text) => {
        if (pill) pill.dataset.state = state;
        if (label) label.textContent = text;
    };

    setState('warming', 'Waking up the extraction engine…');

    const controller = new AbortController();
    // A cold start on a fresh instance (image pull + model load) can take ~3 min
    const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000);

    try {
        const res = await fetch(getApiBase() + '/api/status', { signal: controller.signal });
        if (!res.ok) throw new Error('Status ' + res.status);
        setState('ready', 'Extraction engine ready');
    } catch (err) {
        console.warn('Backend warm-up failed:', err);
        setState('offline', 'Engine unreachable — extraction may be slow or fail');
    } finally {
        clearTimeout(timeoutId);
    }
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
