document.addEventListener('DOMContentLoaded', () => {
    // --- Logic for Index Page (Upload) ---
    const pdfInput = document.getElementById('pdfInput');
    const extractBtn = document.getElementById('extractBtn');
    const uploadTriggerBtn = document.getElementById('uploadTriggerBtn'); // New ID for the trigger button

    // 1. Handle File Selection
    if (pdfInput) {
        pdfInput.addEventListener('change', () => {
            const fileNameDisplay = document.getElementById('fileNameDisplay');
            if (pdfInput.files.length > 0) {
                fileNameDisplay.innerText = "Selected: " + pdfInput.files[0].name;
                if (extractBtn) extractBtn.style.display = "inline-block";
            }
        });
    }

    // 2. Trigger File Input from Custom Button
    if (uploadTriggerBtn && pdfInput) {
        uploadTriggerBtn.addEventListener('click', () => {
            pdfInput.click();
        });
    }

    // 3. Handle PDF Upload & Extraction
    if (extractBtn) {
        extractBtn.addEventListener('click', async () => {
            const file = pdfInput.files[0];
            const loader = document.getElementById('loader');
            const status = document.getElementById('statusMessage');

            if (!file) return;

            // UI Updates
            extractBtn.disabled = true;
            extractBtn.innerText = "Processing...";
            if (loader) loader.style.display = "block";
            if (status) status.innerText = "";

            const formData = new FormData();
            formData.append('file', file);

            try {
                // Ensure backend is running on port 5000
                const response = await fetch('http://127.0.0.1:5000/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error("Server error");

                const data = await response.json();

                // Save result and redirect
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

    // --- Logic for Output Page (Result) ---
    const displayText = document.getElementById('displayText');
    
    if (displayText) {
        // Load data on page load
        const text = localStorage.getItem('extractedText');
        const fileName = localStorage.getItem('fileName');
        const fileSub = document.getElementById('fileNameSub');

        if (text) {
            displayText.innerText = text;
            if (fileSub) fileSub.innerText = "Source: " + (fileName || "Unknown File");
        } else {
            displayText.innerText = "No text found.";
        }

        // Handle Copy Button
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