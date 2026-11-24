import * as ort from "onnxruntime-web";
import "./style.css";

const statusEl = document.querySelector("#modelStatus");
const generateButton = document.querySelector("#generate");
const message = document.querySelector("#message");
const lengthInput = document.querySelector("#length");
const temperatureInput = document.querySelector("#temperature");
const result = document.querySelector("#result");

let session;
let latestMidi;

const LOW = 48;
const VOCAB = 37;

lengthInput.addEventListener("input", () => {
  document.querySelector("#lengthValue").textContent = `${lengthInput.value} notes`;
});
temperatureInput.addEventListener("input", () => {
  document.querySelector("#temperatureValue").textContent = Number(temperatureInput.value).toFixed(1);
});

async function loadModel() {
  try {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("/models/melody_lstm.onnx", {
      executionProviders: ["wasm"]
    });
    statusEl.textContent = "AI MODEL READY";
    statusEl.classList.add("ready");
    message.textContent = "Your trained LSTM is ready to create music.";
    generateButton.disabled = false;
  } catch (error) {
    statusEl.textContent = "MODEL ERROR";
    message.textContent = `Could not load model: ${error.message}`;
    console.error(error);
  }
}

function sample(logits, temperature) {
  const scaled = Array.from(logits, value => value / temperature);
  const max = Math.max(...scaled);
  const weights = scaled.map(value => Math.exp(value - max));
  const total = weights.reduce((a, b) => a + b, 0);
  let target = Math.random() * total;

  for (let i = 0; i < weights.length; i++) {
    target -= weights[i];
    if (target <= 0) return i;
  }
  return weights.length - 1;
}

async function generateNotes(length, temperature) {
  const notes = [0, 2, 4, 7, 9, 7, 4, 2];

  for (let i = 0; i < length; i++) {
    const context = notes.slice(-8);
    const input = new ort.Tensor("int64", BigInt64Array.from(context, BigInt), [1, 8]);
    const outputs = await session.run({ notes: input });
    const logits = outputs.logits.data;
    const offset = (8 - 1) * VOCAB;
    const next = sample(logits.slice(offset, offset + VOCAB), temperature);
    notes.push(next);
  }
  return notes;
}

function writeMidi(notes) {
  const events = [];
  const ticksPerBeat = 480;
  const ticksPerNote = 144;

  events.push(0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20);

  notes.forEach((note, index) => {
    const pitch = note + LOW;
    const start = index * ticksPerNote;
    const end = start + 110;

    events.push({ tick: start, bytes: [0x90, pitch, 95] });
    events.push({ tick: end, bytes: [0x80, pitch, 0] });
  });

  events.sort((a, b) => a.tick - b.tick);
  const track = [...events.slice(0, 0)];
  let previousTick = 0;

  for (const event of events) {
    const delta = event.tick - previousTick;
    track.push(...variableLength(delta), ...event.bytes);
    previousTick = event.tick;
  }

  track.push(0x00, 0xFF, 0x2F, 0x00);

  const header = [
    0x4D,0x54,0x68,0x64, 0,0,0,6, 0,0, 0,1,
    (ticksPerBeat >> 8) & 255, ticksPerBeat & 255
  ];
  const lengthBytes = [
    (track.length >>> 24) & 255,
    (track.length >>> 16) & 255,
    (track.length >>> 8) & 255,
    track.length & 255
  ];
  return new Uint8Array([...header, 0x4D,0x54,0x72,0x6B, ...lengthBytes, ...track]);
}

function variableLength(value) {
  let buffer = value & 0x7F;
  const bytes = [];
  while ((value >>= 7)) {
    buffer <<= 8;
    buffer |= ((value & 0x7F) | 0x80);
  }
  while (true) {
    bytes.push(buffer & 255);
    if (buffer & 0x80) buffer >>= 8;
    else break;
  }
  return bytes;
}

function drawNotes(notes) {
  const canvas = document.querySelector("#visualizer");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const min = Math.min(...notes);
  const max = Math.max(...notes);
  const range = Math.max(1, max - min);
  const width = canvas.width / notes.length;

  notes.forEach((note, index) => {
    const height = 25 + ((note - min) / range) * 120;
    const x = index * width + 1;
    const y = canvas.height - height - 12;
    ctx.fillStyle = `hsl(${250 + (note * 3) % 80}, 85%, 68%)`;
    ctx.fillRect(x, y, Math.max(2, width - 2), height);
  });
}

generateButton.addEventListener("click", async () => {
  generateButton.disabled = true;
  result.hidden = true;
  message.textContent = "Generating notes with your trained LSTM...";
  try {
    const notes = await generateNotes(
      Number(lengthInput.value),
      Number(temperatureInput.value)
    );
    latestMidi = writeMidi(notes);
    drawNotes(notes);
    result.hidden = false;
    message.textContent = "Melody generated successfully.";
  } catch (error) {
    message.textContent = `Generation failed: ${error.message}`;
    console.error(error);
  } finally {
    generateButton.disabled = false;
  }
});

document.querySelector("#download").addEventListener("click", () => {
  if (!latestMidi) return;
  const blob = new Blob([latestMidi], { type: "audio/midi" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "melodyforge_generated.mid";
  link.click();
  URL.revokeObjectURL(url);
});

loadModel();
