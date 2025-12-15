# 🏥 HTEP – Medical Document AI Pipeline

HTEP (Healthcare Text Extraction Pipeline) is an end-to-end **medical document processing system** that combines **OCR, Intelligent Character Recognition (ICR), document segmentation, and classification** to extract structured information from scanned medical documents.

This project is designed as a **modular, extensible pipeline**, with a strong focus on **handwritten / scanned characters** and **medical-domain documents**.

---

## ✨ Key Features

### 🔠 Intelligent Character Recognition (ICR)
- Custom-trained CNN model (28×28 grayscale)
- Recognizes **A–Z and 0–9**
- Robust preprocessing for scanned & handwritten characters
- Supports:
  - Single character testing
  - Word-level recognition via character segmentation

### 📄 OCR for Medical Documents
- Uses **Tesseract OCR** for text extraction
- Supports **PDFs and images**
- Image preprocessing for better OCR accuracy

### ✂️ Document Segmentation
- Rule-based segmentation of medical documents into:
  - Patient Info
  - Diagnosis
  - Medications
  - Lab Results
  - Treatment Plan
  - History, Examination, etc.

### 🧠 Medical Document Classification
- Classifies documents into:
  - Prescription
  - Lab Report
  - Discharge Summary
  - Consultation Notes
  - Radiology Reports
  - Progress Notes
- Rule-based confidence scoring
- Urgency detection (routine / high / urgent)

---

## 🗂️ Project Structure
htep-ai/
│
├── src/
│ ├── icr/
│ │ ├── inference.py # ICR model inference
│ │ ├── train_model.py # CNN / ML training pipeline
│ │ ├── dataset_preparation.py
│ │ └── preprocessing.py # Character preprocessing logic
│ │
│ ├── ocr/
│ │ └── extractor.py # OCR using Tesseract
│ │
│ ├── segmentation/
│ │ └── segmenter.py # Medical document segmentation
│ │
│ ├── classification/
│ │ └── classifier.py # Medical document classifier
│
├── scripts/
│ └── test_scanned_icr.py # Single-character ICR testing
│
├── test_word_icr.py # Word-level ICR testing
│
├── data/
│ ├── icr_training/
│ │ ├── train/
│ │ ├── test/
│ │ └── scanned/
│ │ └── words/
│ │
│ ├── processed/
│ │ └── word_chars/ # Debug character outputs
│
├── models/
│ └── icr_model.* # Trained ICR model + metadata
│
├── main.py # Full medical document pipeline
├── config.py
└── README.md


---

## 🧪 ICR Workflow (How Character Recognition Works)

1. **Input Image** (scanned or handwritten)
2. **Preprocessing**
   - Grayscale conversion
   - Auto polarity correction
   - Thresholding (Otsu / adaptive)
   - Morphological cleanup
   - Forced resize to **28×28**
3. **CNN Prediction**
   - Trained on block letters
   - Outputs character + confidence

---

## 🔤 Word Recognition (Current Approach)

Since the ICR model is trained **only on single characters**, word recognition is done by:

1. Segmenting a word image into individual characters
2. Preprocessing each character independently
3. Predicting characters one-by-one
4. Combining predictions into a word

> ⚠️ This works best for **clearly separated block letters** (not cursive).


