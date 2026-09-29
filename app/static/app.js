const ui = {
  talk: document.querySelector("#talk"), status: document.querySelector("#status"),
  dot: document.querySelector("#status-dot"), timer: document.querySelector("#timer"),
  lamp: document.querySelector("#lamp"), result: document.querySelector("#result"),
  transcript: document.querySelector("#transcript"), response: document.querySelector("#response"),
  inspection: document.querySelector("#inspection"), error: document.querySelector("#error"),
  replay: document.querySelector("#replay"), supplied: document.querySelector("#supplied-text"),
  sendReplay: document.querySelector("#send-replay")
};

let socket, stream, context, source, processor, turnId, timerHandle, startedAt, activeStatus = "idle";
const schemaVersion = "1.0";

function connect() {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.onmessage = ({data}) => handleEvent(JSON.parse(data));
  socket.onclose = () => setStatus("Disconnected. Reload to reconnect.", "error");
}

function sendEvent(event, extra = {}) {
  socket.send(JSON.stringify({schema_version: schemaVersion, event, turn_id: turnId, ...extra}));
}

function setStatus(text, pose = "idle") {
  activeStatus = pose;
  ui.status.textContent = text;
  window.lampController.play(pose);
  ui.dot.style.background = pose === "listening" ? "#f2aa52" : "#6d7068";
}

function handleEvent(message) {
  if (message.turn_id !== turnId && message.turn_id !== "unknown") return;
  if (message.event === "status") {
    setStatus(message.status === "processing" ? "Finding the signal…" : message.status, message.status);
    if (message.status === "cancelled") turnId = null;
  } else if (message.event === "error") {
    ui.error.textContent = message.message;
    setStatus("Please try again", "error");
  } else if (message.event === "result") {
    ui.result.hidden = false;
    ui.transcript.textContent = `“${message.transcript}”`;
    ui.response.textContent = message.response_text;
    ui.inspection.replaceChildren(...Object.entries(message.probabilities).map(([label, value]) => {
      const row = document.createElement("div"); row.className = "probability";
      const name = document.createElement("span"); name.textContent = label;
      const bar = document.createElement("span"); bar.className = "bar";
      const fill = document.createElement("span"); fill.style.width = `${value * 100}%`; bar.append(fill);
      const number = document.createElement("span"); number.textContent = `${Math.round(value * 100)}%`;
      row.append(name, bar, number); return row;
    }));
    setStatus(message.uncertain ? "A tentative reading" : `Signal: ${message.emotion}`, message.gesture_id);
    turnId = null;
  }
}

function downsample(input, fromRate) {
  if (fromRate === 16000) return input;
  const ratio = fromRate / 16000, output = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < output.length; i++) output[i] = input[Math.floor(i * ratio)];
  return output;
}

function pcm16(samples) {
  const result = new Int16Array(samples.length);
  samples.forEach((value, i) => result[i] = Math.max(-1, Math.min(1, value)) * (value < 0 ? 32768 : 32767));
  return result.buffer;
}

async function startRecording(event) {
  event.preventDefault(); if (processor || socket.readyState !== WebSocket.OPEN) return;
  ui.error.textContent = "";
  try { stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true}}); }
  catch (error) { ui.error.textContent = "Microphone permission is required to record."; return; }
  turnId = crypto.randomUUID(); sendEvent("start_turn");
  context = new AudioContext(); source = context.createMediaStreamSource(stream);
  processor = context.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = ({inputBuffer}) => socket.send(pcm16(downsample(inputBuffer.getChannelData(0), context.sampleRate)));
  source.connect(processor); processor.connect(context.destination);
  ui.talk.classList.add("active"); ui.talk.textContent = "Release to finish";
  startedAt = Date.now(); timerHandle = setInterval(updateTimer, 100);
  setTimeout(() => { if (processor) stopRecording(); }, 10000);
}

function updateTimer() { const seconds = (Date.now() - startedAt) / 1000; ui.timer.textContent = `00:${seconds.toFixed(1).padStart(4, "0")}`; }

async function stopRecording() {
  if (!processor) return;
  processor.disconnect(); source.disconnect(); stream.getTracks().forEach(track => track.stop()); await context.close();
  processor = null; clearInterval(timerHandle); ui.talk.classList.remove("active"); ui.talk.textContent = "Hold to talk";
  sendEvent("end_turn");
}

async function cancelTurn() {
  if (!turnId || activeStatus === "idle") return;
  if (processor) {
    processor.disconnect(); source.disconnect(); stream.getTracks().forEach(track => track.stop()); await context.close();
    processor = null; clearInterval(timerHandle); ui.talk.classList.remove("active");
    ui.talk.textContent = "Hold to talk";
  }
  sendEvent("cancel_turn"); setStatus("Cancelled", "idle");
}

async function replay() {
  const file = ui.replay.files[0]; if (!file) { ui.error.textContent = "Choose a mono 16 kHz PCM WAV first."; return; }
  const buffer = await file.arrayBuffer(); const view = new DataView(buffer);
  if (view.getUint16(22, true) !== 1 || view.getUint32(24, true) !== 16000 || view.getUint16(34, true) !== 16) {
    ui.error.textContent = "Replay requires mono 16 kHz PCM16 WAV."; return;
  }
  let offset = 12, dataOffset = -1, dataLength = 0;
  while (offset + 8 <= buffer.byteLength) {
    const id = String.fromCharCode(...new Uint8Array(buffer, offset, 4)); const length = view.getUint32(offset + 4, true);
    if (id === "data") { dataOffset = offset + 8; dataLength = length; break; } offset += 8 + length + (length % 2);
  }
  if (dataOffset < 0) { ui.error.textContent = "The WAV has no data chunk."; return; }
  turnId = crypto.randomUUID(); sendEvent("start_turn", {transcript: ui.supplied.value || null});
  const end = Math.min(dataOffset + dataLength, buffer.byteLength);
  for (let cursor = dataOffset; cursor < end; cursor += 3200) {
    socket.send(buffer.slice(cursor, Math.min(cursor + 3200, end)));
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  sendEvent("end_turn");
}

ui.talk.addEventListener("pointerdown", startRecording);
ui.talk.addEventListener("pointerup", stopRecording);
ui.talk.addEventListener("pointercancel", stopRecording);
ui.sendReplay.addEventListener("click", replay);
document.addEventListener("keydown", event => { if (event.key === "Escape") cancelTurn(); });
connect();
