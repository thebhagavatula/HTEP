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
                const response = await fetch('http://127.0.0.1:5000/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error("Server error");

                const data = await response.json();

                localStorage.setItem('extractedText', data.text);
                localStorage.setItem('fileName', file.name);

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

    /* ===============================
       OUTPUT PAGE LOGIC
    =============================== */

    const displayText = document.getElementById('displayText');

    if (displayText) {

        const text = localStorage.getItem('extractedText');
        const fileName = localStorage.getItem('fileName');
        const fileSub = document.getElementById('fileNameSub');

        if (text) {
            displayText.innerText = text;
            if (fileSub) fileSub.innerText = "Source: " + (fileName || "Unknown File");
        } else {
            displayText.innerText = "No text found.";
        }

        const copyBtn = document.getElementById('copyBtn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(displayText.innerText).then(() => {
                    alert("Copied!");
                });
            });
        }
    }

});
