const modelStatus = document.getElementById("modelStatus");
const generateButton = document.getElementById("generateButton");
const downloadButton = document.getElementById("downloadButton");
const message = document.getElementById("message");
const result = document.getElementById("result");
const noteCount = document.getElementById("noteCount");

let session = null;
let latestMidi = null;

const MIN_PITCH = 48;
const VOCAB_SIZE = 37;
const SEED = [0, 2, 4, 7, 9, 7, 4, 2];

async function loadModel() {
    try {
        ort.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        session = await ort.InferenceSession.create(
            "models/melody_lstm.onnx",
            { executionProviders: ["wasm"] }
        );

        modelStatus.textContent = "AI model ready";
        modelStatus.style.color = "#86efac";
        message.textContent = "Your AI music model is ready.";
        generateButton.disabled = false;
    } catch (error) {
        console.error(error);
        modelStatus.textContent = "Model loading failed";
        message.textContent =
            "Could not load the AI model. Check your connection and model file.";
    }
}

function softmaxSample(logits) {
    const temperature = 0.9;
    const scaled = logits.map(value => value / temperature);
    const maximum = Math.max(...scaled);
    const probabilities = scaled.map(value =>
        Math.exp(value - maximum)
    );

    const total = probabilities.reduce((sum, value) => sum + value, 0);
    let random = Math.random() * total;

    for (let i = 0; i < probabilities.length; i++) {
        random -= probabilities[i];
        if (random <= 0) return i;
    }

    return probabilities.length - 1;
}

async function generateNotes(length) {
    const notes = [...SEED];

    while (notes.length < length) {
        const input = new ort.Tensor(
            "int64",
            BigInt64Array.from(notes.slice(-8), value => BigInt(value)),
            [1, 8]
        );

        const output = await session.run({ notes: input });
        const logits = output.logits.data;
        const finalStep = logits.slice(-VOCAB_SIZE);
        const nextNote = softmaxSample(Array.from(finalStep));

        notes.push(nextNote);
    }

    return notes.map(index => MIN_PITCH + index);
}

function writeVariableLength(value) {
    let buffer = value & 0x7f;

    while ((value >>= 7)) {
        buffer <<= 8;
        buffer |= ((value & 0x7f) | 0x80);
    }

    const bytes = [];

    while (true) {
        bytes.push(buffer & 0xff);
        if (buffer & 0x80) buffer >>= 8;
        else break;
    }

    return bytes;
}

function makeMidi(pitches) {
    const events = [];
    const ticksPerNote = 480;
    const noteDuration = 360;

    pitches.forEach((pitch, index) => {
        const start = index * ticksPerNote;

        events.push({
            tick: start,
            priority: 1,
            bytes: [0x90, pitch, 90]
        });

        events.push({
            tick: start + noteDuration,
            priority: 0,
            bytes: [0x80, pitch, 0]
        });
    });

    events.sort((a, b) =>
        a.tick - b.tick || a.priority - b.priority
    );

    const track = [];
    let previousTick = 0;

    for (const event of events) {
        const delta = event.tick - previousTick;
        track.push(...writeVariableLength(delta), ...event.bytes);
        previousTick = event.tick;
    }

    track.push(0x00, 0xff, 0x2f, 0x00);

    const header = [
        0x4d, 0x54, 0x68, 0x64,
        0x00, 0x00, 0x00, 0x06,
        0x00, 0x00,
        0x00, 0x01,
        0x01, 0xe0
    ];

    const trackHeader = [
        0x4d, 0x54, 0x72, 0x6b,
        (track.length >>> 24) & 0xff,
        (track.length >>> 16) & 0xff,
        (track.length >>> 8) & 0xff,
        track.length & 0xff
    ];

    return new Blob(
        [new Uint8Array([...header, ...trackHeader, ...track])],
        { type: "audio/midi" }
    );
}

generateButton.addEventListener("click", async () => {
    try {
        generateButton.disabled = true;
        result.classList.add("hidden");
        message.textContent = "Your AI is composing a melody...";

        const length = Number(
            document.getElementById("length").value
        );

        const pitches = await generateNotes(length);

        latestMidi = makeMidi(pitches);

        noteCount.textContent =
            `${pitches.length} notes generated successfully.`;

        result.classList.remove("hidden");
        message.textContent = "Your melody has been generated!";
    } catch (error) {
        console.error(error);
        message.textContent =
            "Melody generation failed. Please check the browser console.";
    } finally {
        generateButton.disabled = false;
    }
});

downloadButton.addEventListener("click", () => {
    if (!latestMidi) return;

    const url = URL.createObjectURL(latestMidi);
    const link = document.createElement("a");

    link.href = url;
    link.download = "MelodyForge_AI_Melody.mid";
    link.click();

    URL.revokeObjectURL(url);
});

loadModel();
