/* ===============================
   BACKEND CONFIG
=============================== */

// Production backend: local machine via Cloudflare quick tunnel (GCP billing is off;
// Cloud Run was https://htep-151629903010.us-central1.run.app). To point the frontend
// elsewhere, set window.HTEP_API_BASE before this script loads.
const HTEP_BACKEND_URL = 'https://consist-pipes-neighbor-fingers.trycloudflare.com';

function getApiBase() {
    if (window.HTEP_API_BASE) return window.HTEP_API_BASE;

    const host = window.location.hostname;
    if (host === 'localhost' || host === '127.0.0.1') return 'http://127.0.0.1:5000';

    // Frontend served by the backend itself (Cloud Run also serves /web)
    if (host.endsWith('.run.app')) return window.location.origin;

    return HTEP_BACKEND_URL;
}

const MAX_FILE_BYTES = 5 * 1024 * 1024;
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
            setStatus(`That file is ${formatBytes(file.size)}. Please upload a file under 5 MB.`, true);
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

        // Show elapsed time while processing
        let elapsed = 0;
        const tick = (prefix) => {
            const hint = elapsed >= 20 ? ' Multi-page PDFs and handwritten prescriptions take longer.' : '';
            setStatus(`${prefix} ${elapsed}s.${hint}`);
        };

        try {
            // Don't race the warm-up: both requests would load the ML engines at once
            ticker = setInterval(() => { elapsed++; tick('Waiting for the extraction engine to start…'); }, 1000);
            await warmup;
            clearInterval(ticker);

            // Abort just after Cloud Run's 600s request timeout so the server's limit governs
            const controller = new AbortController();
            timeoutId = setTimeout(() => controller.abort(), 10 * 60 * 1000);

            tick('Reading your prescription…');
            ticker = setInterval(() => { elapsed++; tick('Reading your prescription…'); }, 1000);

            const response = await fetch(getApiBase() + '/jobs', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });

            if (!response.ok) throw new Error(await serverError(response));

            const { job_id } = await response.json();
            const data = await pollJob(job_id, controller.signal);
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

async function serverError(response) {
    let message = "Server error (" + response.status + ")";
    try {
        const body = await response.json();
        if (body.error) message = body.error;
    } catch (_) {}
    return message;
}

// A multi-page PDF can take minutes, longer than proxies such as Cloudflare
// tunnels (100s) keep a request open, so the server processes the file in the
// background and we poll for the result.
async function pollJob(jobId, signal) {
    let failures = 0;
    while (true) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        if (signal.aborted) throw new DOMException('Aborted', 'AbortError');

        let res;
        try {
            res = await fetch(getApiBase() + '/jobs/' + jobId, { signal });
        } catch (err) {
            // Ride out brief network blips; give up after ~10s of failures
            if (err.name === 'AbortError' || ++failures > 5) throw err;
            continue;
        }
        if (res.status >= 500 && ++failures <= 5) continue;
        if (!res.ok) throw new Error(await serverError(res));
        failures = 0;

        const job = await res.json();
        if (job.status === 'done') return job.result;
        if (job.status === 'error') throw new Error(job.error || 'Processing failed');
    }
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
        const res = await fetch(getApiBase() + '/status', { signal: controller.signal });
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
