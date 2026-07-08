document.addEventListener('DOMContentLoaded', () => {

    /* ===============================
       THEME TOGGLE (SUN / MOON SVG)
    =============================== */

    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    // Load saved theme or default to light
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    if (themeIcon) {
        themeIcon.src = savedTheme === "dark"
            ? "assets/sun.svg"
            : "assets/moon.svg";
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === "dark" ? "light" : "dark";

            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            if (themeIcon) {
                themeIcon.src = newTheme === "dark"
                    ? "assets/sun.svg"
                    : "assets/moon.svg";
            }
        });
    }

    /* ===============================
       INDEX PAGE LOGIC (UPLOAD)
    =============================== */

    const pdfInput = document.getElementById('pdfInput');
    const extractBtn = document.getElementById('extractBtn');
    const uploadTriggerBtn = document.getElementById('uploadTriggerBtn');

    if (pdfInput) {
        pdfInput.addEventListener('change', () => {
            const fileNameDisplay = document.getElementById('fileNameDisplay');
            if (pdfInput.files.length > 0) {
                fileNameDisplay.innerText = "Selected: " + pdfInput.files[0].name;
                if (extractBtn) extractBtn.style.display = "inline-block";
            }
        });
    }

    if (uploadTriggerBtn && pdfInput) {
        uploadTriggerBtn.addEventListener('click', () => {
            pdfInput.click();
        });
    }

    if (extractBtn) {
        extractBtn.addEventListener('click', async () => {

            const file = pdfInput.files[0];
            const loader = document.getElementById('loader');
            const status = document.getElementById('statusMessage');

            if (!file) return;

            extractBtn.disabled = true;
            extractBtn.innerText = "Processing...";
            if (loader) loader.style.display = "block";
            if (status) status.innerText = "";

            const formData = new FormData();
            formData.append('file', file);

            try {
                // Dynamic API URL: uses localhost for dev, deployed backend URL for production
                const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
                    ? 'http://127.0.0.1:5000'
                    : (window.HTEP_API_BASE || 'https://healthcare-text-extraction-platform.onrender.com');

                const response = await fetch(API_BASE + '/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error("Server error");

                const data = await response.json();

                // Store corrected text (falls back to ocr_text if not available)
                const displayText = data.corrected_text || data.ocr_text || data.text || '';
                localStorage.setItem('extractedText', displayText);
                localStorage.setItem('fileName', file.name);

                // Store full response for output page (drugs, diseases, corrections)
                localStorage.setItem('htepResponse', JSON.stringify(data));

                window.location.href = 'output.html';

            } catch (error) {
                console.error(error);
                if (status) status.innerText = "Error processing file.";
                extractBtn.disabled = false;
                extractBtn.innerText = "Start Extraction";
                if (loader) loader.style.display = "none";
            }
        });
    }

});
