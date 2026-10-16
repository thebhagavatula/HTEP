/* ===============================
   ON-DEVICE VISION MODEL
=============================== */

// Reads a prescription image with a vision-language model running in the
// user's browser on WebGPU, so the image never leaves the device and there is
// no GPU server to pay for. The model (~2.5 GB) downloads on first use and is
// cached by the browser. Its answer is merged on the server by /api/merge,
// which checks every value against the OCR text (src/pipeline/grounding.py).

// 3.8.1: 4.3.0 fails on Qwen2-VL with WebGPU ("Shape mismatch attempting to re-use buffer")
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const MODEL_ID = 'onnx-community/Qwen2-VL-2B-Instruct';
const DTYPE = { embed_tokens: 'q4f16', vision_encoder: 'fp16', decoder_model_merged: 'q4f16' };
export const DOWNLOAD_GB = 2.5;

// Same instructions as the server-side VLM (src/vlm/client.py)
const PROMPT =
    'You read medical prescriptions. Copy every value exactly as it is printed or handwritten, ' +
    'keeping the original spelling and units. Use null for a field that is not visible. Never guess, ' +
    'never correct, and never add medicines or diagnoses that are not written in the image. ' +
    'Return only JSON with keys: patient_name, age, gender, date, doctor_name, hospital, ' +
    'diagnosis (list of strings), medicines (list of {name, dosage, frequency, duration}), ' +
    'instructions (list of strings).';

// Qwen2-VL spends one token per 28x28 patch; a capped side keeps it fast
const MAX_IMAGE_SIDE = 896;

export async function isSupported() {
    if (!navigator.gpu) return false;
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter && adapter.features.has('shader-f16');
    } catch (_) {
        return false;
    }
}

let loading = null;

function loadModel(onStatus) {
    if (loading) return loading;
    loading = (async () => {
        const { Qwen2VLForConditionalGeneration, AutoProcessor } = await import(TRANSFORMERS_URL);

        // Sum the progress of every model file into one figure
        const files = {};
        const progress_callback = (p) => {
            if (p.status !== 'progress' || !p.total) return;
            files[p.file] = { loaded: p.loaded, total: p.total };
            const loaded = Object.values(files).reduce((s, f) => s + f.loaded, 0);
            onStatus(`Downloading the AI model: ${(loaded / 2 ** 30).toFixed(2)} of ~${DOWNLOAD_GB} GB`);
        };

        const processor = await AutoProcessor.from_pretrained(MODEL_ID);
        const model = await Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, {
            dtype: DTYPE,
            device: 'webgpu',
            progress_callback,
        });
        return { processor, model };
    })();
    loading.catch(() => { loading = null; });   // allow a retry after a failure
    return loading;
}

/**
 * Read the prescription in `imageDataUrl`. Resolves to the model's fields as
 * an object, or throws with a message suitable for the user.
 */
export async function readPrescription(imageDataUrl, onStatus) {
    const { RawImage } = await import(TRANSFORMERS_URL);
    const { processor, model } = await loadModel(onStatus);

    onStatus('Reading the prescription on your device…');
    let image = await RawImage.fromURL(imageDataUrl);
    const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(image.width, image.height));
    image = await image.resize(
        Math.max(28, Math.round(image.width * scale / 28) * 28),
        Math.max(28, Math.round(image.height * scale / 28) * 28),
    );

    const conversation = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: PROMPT }] }];
    const text = processor.apply_chat_template(conversation, { add_generation_prompt: true });
    const inputs = await processor(text, image);
    const ids = await model.generate({ ...inputs, max_new_tokens: 512, do_sample: false });
    const answer = processor.batch_decode(
        ids.slice(null, [inputs.input_ids.dims.at(-1), null]),
        { skip_special_tokens: true },
    )[0];

    const data = parseJson(answer);
    if (!data) throw new Error('The AI model did not return a readable answer. Please try again.');
    return data;
}

// The model may wrap its JSON in a code fence or add text around it
function parseJson(answer) {
    const start = answer.indexOf('{');
    const end = answer.lastIndexOf('}');
    if (start < 0 || end <= start) return null;
    try {
        const data = JSON.parse(answer.slice(start, end + 1));
        return data && typeof data === 'object' && !Array.isArray(data) ? data : null;
    } catch (_) {
        return null;
    }
}
