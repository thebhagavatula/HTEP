/* ===============================
   IN-BROWSER OCR (PaddleOCR models)
=============================== */

// Runs PaddleOCR's PP-OCRv6 medium detection and recognition models in the
// user's browser with ONNX Runtime Web on WebGPU. These are the models
// PaddleOCR itself uses; on one server CPU they need ~80 s per image, on the
// user's GPU a few seconds, and hosting stays $0. The document never leaves
// the device: only the recognized text is sent to the server.
//
// Each step mirrors RapidOCR (Python) so results match the server-side
// engine: DetPreProcess, DBPostProcess, get_rotate_crop_image,
// TextRecognizer.resize_norm_img and CTCLabelDecode. OpenCV.js provides the
// same image operations as cv2.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';

const OPENCV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js';
const PDFJS_URL = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@6.3.289/build/pdf.min.mjs';
const PDFJS_WORKER_URL = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@6.3.289/build/pdf.worker.min.mjs';

const MODEL_DIR = 'models/ocr/';
const DET_MODEL = MODEL_DIR + 'PP-OCRv6_det_medium.onnx';
const REC_MODEL = MODEL_DIR + 'PP-OCRv6_rec_medium.onnx';
const REC_KEYS = MODEL_DIR + 'PP-OCRv6_rec_medium.keys.json';
export const DOWNLOAD_MB = 150;   // both models + OpenCV.js, once per browser

// Settings shared with the server (src/ocr/extractor.py and RapidOCR's config)
const PDF_DPI = 300;
const MAX_SIDE = 2000;             // Global.max_side_len
const MIN_SIDE = 30;               // Global.min_side_len
const DET_LIMIT_SIDE = 1600;       // as PaddleOCR in the audit build: limit 1600, type "max"
const DET_THRESH = 0.3;
const DET_BOX_THRESH = 0.5;
const DET_UNCLIP_RATIO = 1.6;
const DET_MAX_CANDIDATES = 1000;
const DET_MIN_SIZE = 3;
const BOX_SORT_Y_THRESHOLD = 10;
const REC_HEIGHT = 48;
const REC_BASE_WIDTH = 320;
const REC_BATCH = 6;
const TEXT_SCORE = 0.5;
export const MAX_PDF_PAGES = 10;

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

export async function isSupported() {
    if (!navigator.gpu) return false;
    try {
        return !!(await navigator.gpu.requestAdapter());
    } catch (_) {
        return false;
    }
}

/* ---------- Loading ---------- */

let engine = null;

// Resolves to { cv }. The OpenCV.js module object has its own then(), so a
// promise must never be resolved with it directly: the promise would adopt it,
// call then() again and loop forever.
function loadOpenCV() {
    if (window.cv && window.cv.Mat) return Promise.resolve({ cv: window.cv });
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = OPENCV_URL;
        script.async = true;
        script.onerror = () => reject(new Error('Could not load OpenCV.js'));
        script.onload = () => {
            const cv = window.cv;
            if (cv instanceof Promise) cv.then(m => { window.cv = m; resolve({ cv: m }); }, reject);   // 5.x builds
            else if (cv.Mat) resolve({ cv });
            else cv.onRuntimeInitialized = () => resolve({ cv });
        };
        document.head.appendChild(script);
    });
}

async function fetchWithProgress(url, onBytes) {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Could not download ' + url);
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        onBytes(received);
    }
    const buffer = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
    return buffer;
}

function loadEngine(onStatus) {
    if (engine) return engine;
    engine = (async () => {
        const progress = { det: 0, rec: 0 };
        const report = () => onStatus(`Downloading the text reader: ${Math.round((progress.det + progress.rec) / 2 ** 20)} of ~140 MB`);
        const [{ cv }, detBytes, recBytes, keys] = await Promise.all([
            loadOpenCV(),
            fetchWithProgress(DET_MODEL, b => { progress.det = b; report(); }),
            fetchWithProgress(REC_MODEL, b => { progress.rec = b; report(); }),
            fetch(REC_KEYS).then(r => r.json()),
        ]);
        onStatus('Starting the text reader…');
        // WebGPU only: on the CPU these models take minutes and freeze the page, so
        // a device without a working GPU uses the server's OCR instead (app.js)
        const options = { executionProviders: ['webgpu'] };
        const det = await ort.InferenceSession.create(detBytes, options);
        const rec = await ort.InferenceSession.create(recBytes, options);
        // CTCLabelDecode: "blank" first, a space last
        const characters = ['blank', ...keys, ' '];
        return { cv, det, rec, characters };
    })();
    engine.catch(() => { engine = null; });
    return engine;
}

/* ---------- Public API ---------- */

/**
 * OCR an uploaded File (PDF, PNG or JPEG).
 * Resolves to { pages: [{ text, confidence }], preview } where preview is a
 * JPEG data URL of the first page (for the on-device vision model).
 */
export async function readDocument(file, onStatus) {
    const eng = await loadEngine(onStatus);
    const canvases = file.type === 'application/pdf' || /\.pdf$/i.test(file.name)
        ? await renderPdf(file, onStatus)
        : [await imageToCanvas(file)];

    const pages = [];
    for (let i = 0; i < canvases.length; i++) {
        onStatus(canvases.length > 1 ? `Reading page ${i + 1} of ${canvases.length}…` : 'Reading your prescription…');
        pages.push(await ocrCanvas(eng, canvases[i]));
    }
    return { pages, preview: previewDataUrl(canvases[0]) };
}

/* ---------- Input ---------- */

async function imageToCanvas(file) {
    const bitmap = await createImageBitmap(file);   // applies EXIF orientation, like cv2.imread
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    canvas.getContext('2d').drawImage(bitmap, 0, 0);
    bitmap.close();
    return canvas;
}

async function renderPdf(file, onStatus) {
    const pdfjs = await import(PDFJS_URL);
    pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_URL;
    const pdf = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
    if (pdf.numPages > MAX_PDF_PAGES) {
        throw new Error(`This PDF has ${pdf.numPages} pages; the limit is ${MAX_PDF_PAGES}. Upload the prescription pages only.`);
    }
    const canvases = [];
    for (let n = 1; n <= pdf.numPages; n++) {
        onStatus(`Opening page ${n} of ${pdf.numPages}…`);
        const page = await pdf.getPage(n);
        const viewport = page.getViewport({ scale: PDF_DPI / 72 });
        const canvas = document.createElement('canvas');
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        await page.render({ canvasContext: ctx, viewport }).promise;
        canvases.push(canvas);
    }
    return canvases;
}

function previewDataUrl(canvas, maxSide = 1280) {
    const scale = Math.min(1, maxSide / Math.max(canvas.width, canvas.height));
    const out = document.createElement('canvas');
    out.width = Math.round(canvas.width * scale);
    out.height = Math.round(canvas.height * scale);
    out.getContext('2d').drawImage(canvas, 0, 0, out.width, out.height);
    return out.toDataURL('image/jpeg', 0.85);
}

/* ---------- OCR pipeline ---------- */

async function ocrCanvas(eng, canvas) {
    const { cv } = eng;
    // The models were trained on BGR images (cv2), so convert RGBA -> BGR
    const rgba = cv.imread(canvas);
    const img = new cv.Mat();
    cv.cvtColor(rgba, img, cv.COLOR_RGBA2BGR);
    rgba.delete();
    return ocrMat(eng, img);
}

// OCR a BGR cv.Mat (deleted when done). Exported for the parity test, which
// compares this pipeline with RapidOCR's Python output on the same images.
export async function ocrMat(eng, img) {
    const { cv } = eng;
    try {
        img = resizeWithinBounds(cv, img);
        const boxes = await detect(eng, img);
        if (!boxes.length) return { text: '', confidence: 0 };

        const crops = boxes.map(box => rotateCrop(cv, img, box));
        const results = await recognize(eng, crops);
        crops.forEach(c => c.delete());

        const kept = results.filter(r => r.score >= TEXT_SCORE && r.text);
        const confidence = kept.length ? kept.reduce((s, r) => s + r.score, 0) / kept.length : 0;
        return { text: kept.map(r => r.text).join('\n').trim(), confidence: Math.round(confidence * 1000) / 1000 };
    } finally {
        img.delete();
    }
}

// resize_image_within_bounds: cap the longest side at 2000, sides multiples of 32
function resizeWithinBounds(cv, img) {
    const fit = (ratio) => {
        const h = Math.round(Math.trunc(img.rows * ratio) / 32) * 32;
        const w = Math.round(Math.trunc(img.cols * ratio) / 32) * 32;
        const out = new cv.Mat();
        cv.resize(img, out, new cv.Size(w, h));
        img.delete();
        img = out;
    };
    if (Math.max(img.rows, img.cols) > MAX_SIDE) fit(MAX_SIDE / Math.max(img.rows, img.cols));
    if (Math.min(img.rows, img.cols) < MIN_SIDE) fit(MIN_SIDE / Math.min(img.rows, img.cols));
    return img;
}

async function detect(eng, img) {
    const { cv, det } = eng;
    const srcH = img.rows, srcW = img.cols;

    // DetPreProcess: limit the longest side, multiples of 32, normalize with mean/std 0.5
    const ratio = Math.max(srcH, srcW) > DET_LIMIT_SIDE ? DET_LIMIT_SIDE / Math.max(srcH, srcW) : 1;
    const h = Math.round(Math.trunc(srcH * ratio) / 32) * 32;
    const w = Math.round(Math.trunc(srcW * ratio) / 32) * 32;
    const resized = new cv.Mat();
    cv.resize(img, resized, new cv.Size(w, h));
    const input = toCHW(resized.data, h, w, 0.5, 0.5);
    resized.delete();

    const output = await det.run({ x: new ort.Tensor('float32', input, [1, 3, h, w]) });
    const pred = Object.values(output)[0];
    const [mapH, mapW] = [pred.dims[2], pred.dims[3]];
    const prob = await pred.getData();

    // DBPostProcess
    const prob32 = cv.matFromArray(mapH, mapW, cv.CV_32F, prob);
    const bitmap = new cv.Mat(mapH, mapW, cv.CV_8U);
    for (let i = 0; i < prob.length; i++) bitmap.data[i] = prob[i] > DET_THRESH ? 255 : 0;
    const kernel = cv.Mat.ones(2, 2, cv.CV_8U);
    cv.dilate(bitmap, bitmap, kernel);
    kernel.delete();

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(bitmap, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    bitmap.delete();
    hierarchy.delete();

    const boxes = [];
    const count = Math.min(contours.size(), DET_MAX_CANDIDATES);
    for (let i = 0; i < count; i++) {
        const contour = contours.get(i);
        const [points, sside] = miniBox(cv, cv.minAreaRect(contour));
        contour.delete();
        if (sside < DET_MIN_SIZE) continue;
        if (boxScoreFast(cv, prob32, points) < DET_BOX_THRESH) continue;

        const [box, sside2] = miniBox(cv, unclipRect(cv, points));
        if (sside2 < DET_MIN_SIZE + 2) continue;
        boxes.push(box.map(([x, y]) => [
            Math.min(Math.max(Math.round(x / mapW * srcW), 0), srcW),
            Math.min(Math.max(Math.round(y / mapH * srcH), 0), srcH),
        ]));
    }
    contours.delete();
    prob32.delete();

    // filter_det_res + sorted_boxes
    const kept = boxes.map(b => clipBox(orderClockwise(b), srcH, srcW))
        .filter(b => Math.trunc(dist(b[0], b[1])) > 3 && Math.trunc(dist(b[0], b[3])) > 3);
    return sortBoxes(kept);
}

// get_mini_boxes: the rectangle's corners ordered top-left, top-right, bottom-right, bottom-left
function miniBox(cv, rect) {
    const pts = cv.RotatedRect.points(rect).map(p => [p.x, p.y]).sort((a, b) => a[0] - b[0]);
    const [i1, i4] = pts[1][1] > pts[0][1] ? [0, 1] : [1, 0];
    const [i2, i3] = pts[3][1] > pts[2][1] ? [2, 3] : [3, 2];
    return [[pts[i1], pts[i2], pts[i3], pts[i4]], Math.min(rect.size.width, rect.size.height)];
}

// box_score_fast: mean probability inside the box
function boxScoreFast(cv, prob, box) {
    const h = prob.rows, w = prob.cols;
    const xs = box.map(p => p[0]), ys = box.map(p => p[1]);
    const xmin = clamp(Math.floor(Math.min(...xs)), 0, w - 1), xmax = clamp(Math.ceil(Math.max(...xs)), 0, w - 1);
    const ymin = clamp(Math.floor(Math.min(...ys)), 0, h - 1), ymax = clamp(Math.ceil(Math.max(...ys)), 0, h - 1);
    const mask = cv.Mat.zeros(ymax - ymin + 1, xmax - xmin + 1, cv.CV_8U);
    const poly = cv.matFromArray(4, 1, cv.CV_32SC2, box.flatMap(([x, y]) => [Math.trunc(x - xmin), Math.trunc(y - ymin)]));
    const polys = new cv.MatVector();
    polys.push_back(poly);
    cv.fillPoly(mask, polys, new cv.Scalar(1));
    const roi = prob.roi(new cv.Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
    const score = cv.mean(roi, mask)[0];
    roi.delete(); mask.delete(); poly.delete(); polys.delete();
    return score;
}

// unclip: grow the box by area * ratio / perimeter on every side (pyclipper's round
// offset of a rectangle has this exact minimum-area rectangle)
function unclipRect(cv, box) {
    const w = dist(box[0], box[1]), h = dist(box[1], box[2]);
    const d = (w * h) * DET_UNCLIP_RATIO / (2 * (w + h));
    const cx = box.reduce((s, p) => s + p[0], 0) / 4, cy = box.reduce((s, p) => s + p[1], 0) / 4;
    const angle = Math.atan2(box[1][1] - box[0][1], box[1][0] - box[0][0]) * 180 / Math.PI;
    return new cv.RotatedRect(new cv.Point(cx, cy), new cv.Size(w + 2 * d, h + 2 * d), angle);
}

function orderClockwise(pts) {
    const xs = [...pts].sort((a, b) => a[0] - b[0]);
    const [tl, bl] = xs.slice(0, 2).sort((a, b) => a[1] - b[1]);
    const [tr, br] = xs.slice(2).sort((a, b) => a[1] - b[1]);
    return [tl, tr, br, bl];
}

function clipBox(box, h, w) {
    return box.map(([x, y]) => [Math.trunc(Math.min(Math.max(x, 0), w - 1)), Math.trunc(Math.min(Math.max(y, 0), h - 1))]);
}

// sorted_boxes: top to bottom; boxes within 10 px vertically form a line, read left to right
function sortBoxes(boxes) {
    const byY = boxes.map((b, i) => ({ b, i })).sort((p, q) => p.b[0][1] - q.b[0][1] || p.i - q.i).map(p => p.b);
    let line = 0;
    const withLine = byY.map((b, k) => {
        if (k > 0 && b[0][1] - byY[k - 1][0][1] >= BOX_SORT_Y_THRESHOLD) line++;
        return { b, line };
    });
    return withLine.sort((p, q) => p.line - q.line || p.b[0][0] - q.b[0][0]).map(p => p.b);
}

// get_rotate_crop_image: straighten the box with a perspective warp
function rotateCrop(cv, img, box) {
    const width = Math.trunc(Math.max(dist(box[0], box[1]), dist(box[2], box[3])));
    const height = Math.trunc(Math.max(dist(box[0], box[3]), dist(box[1], box[2])));
    const src = cv.matFromArray(4, 1, cv.CV_32FC2, box.flat());
    const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, width, 0, width, height, 0, height]);
    const M = cv.getPerspectiveTransform(src, dst);
    const crop = new cv.Mat();
    cv.warpPerspective(img, crop, M, new cv.Size(width, height), cv.INTER_CUBIC, cv.BORDER_REPLICATE);
    src.delete(); dst.delete(); M.delete();
    if (crop.rows / crop.cols >= 1.5) {
        cv.rotate(crop, crop, cv.ROTATE_90_COUNTERCLOCKWISE);   // np.rot90
    }
    return crop;
}

async function recognize(eng, crops) {
    const { cv, rec, characters } = eng;
    const results = new Array(crops.length);
    // Sort by aspect ratio so each batch pads to a similar width
    const order = crops.map((c, i) => [c.cols / c.rows, i]).sort((a, b) => a[0] - b[0]).map(p => p[1]);

    for (let start = 0; start < order.length; start += REC_BATCH) {
        const batch = order.slice(start, start + REC_BATCH);
        let maxRatio = REC_BASE_WIDTH / REC_HEIGHT;
        batch.forEach(i => { maxRatio = Math.max(maxRatio, crops[i].cols / crops[i].rows); });
        const width = Math.trunc(REC_HEIGHT * maxRatio);

        const input = new Float32Array(batch.length * 3 * REC_HEIGHT * width);
        batch.forEach((i, n) => {
            const crop = crops[i];
            const resizedW = Math.min(width, Math.ceil(REC_HEIGHT * crop.cols / crop.rows));
            const resized = new cv.Mat();
            cv.resize(crop, resized, new cv.Size(resizedW, REC_HEIGHT));
            // (x / 255 - 0.5) / 0.5, CHW, zero padding on the right
            const plane = REC_HEIGHT * width;
            const base = n * 3 * plane;
            const d = resized.data;
            for (let y = 0; y < REC_HEIGHT; y++) {
                for (let x = 0; x < resizedW; x++) {
                    const p = (y * resizedW + x) * 3;
                    for (let c = 0; c < 3; c++) input[base + c * plane + y * width + x] = (d[p + c] / 255 - 0.5) / 0.5;
                }
            }
            resized.delete();
        });

        const output = await rec.run({ x: new ort.Tensor('float32', input, [batch.length, 3, REC_HEIGHT, width]) });
        const pred = Object.values(output)[0];
        const [, steps, classes] = pred.dims;
        const probs = await pred.getData();

        // CTCLabelDecode: greedy, merge repeats, drop "blank"
        batch.forEach((i, n) => {
            let text = '', total = 0, kept = 0, previous = -1;
            for (let t = 0; t < steps; t++) {
                const off = (n * steps + t) * classes;
                let best = 0, bestP = probs[off];
                for (let k = 1; k < classes; k++) if (probs[off + k] > bestP) { bestP = probs[off + k]; best = k; }
                if (best !== 0 && best !== previous) { text += characters[best]; total += bestP; kept++; }
                previous = best;
            }
            results[i] = { text, score: kept ? total / kept : 0 };
        });
    }
    return results;
}

/* ---------- Helpers ---------- */

function toCHW(bgr, h, w, mean, std) {
    const out = new Float32Array(3 * h * w);
    const plane = h * w;
    for (let i = 0; i < plane; i++) {
        out[i] = (bgr[i * 3] / 255 - mean) / std;
        out[plane + i] = (bgr[i * 3 + 1] / 255 - mean) / std;
        out[2 * plane + i] = (bgr[i * 3 + 2] / 255 - mean) / std;
    }
    return out;
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
