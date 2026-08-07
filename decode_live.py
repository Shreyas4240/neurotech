"""
decode_live.py

Step 3–4 (online): pull EEG from the LSL "EEG" stream, run the trained
CSP+LDA pipeline on a sliding window, and broadcast predictions over a
WebSocket so the 3D brain frontend can close the loop.

Also pulls the optional "Markers" stream so rolling accuracy can be shown
when replaying a labeled dataset.

Message schema (JSON, every UPDATE_SECONDS):
  {
    "type": "prediction",
    "prediction": "left" | "right" | "rest",
    "confidence": 0.0-1.0,
    "probs": {"left": ..., "right": ..., "rest": ...},
    "band_power": {"C3": ..., "C4": ...},
    "marker": "T0" | "T1" | "T2" | null,
    "rolling_accuracy": 0.0-1.0 | null,
    "n_scored": int,
    "timestamp": float
  }

Usage:
  # Terminal A:  python lsl.py
  # Terminal B:  python decode_live.py
  # Terminal C:  serve brain_replica and open /bci

  # Wire the frontend without a trained model / LSL:
  python decode_live.py --mock
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path

# Unbuffered logs when launched without a TTY (Cursor/CI terminals)
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from joblib import load
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt
import websockets
from websockets.asyncio.server import serve

# --- Config ---------------------------------------------------------------

WINDOW_SECONDS = 2.0
UPDATE_SECONDS = 0.25
LOW_HZ, HIGH_HZ = 8.0, 30.0
WS_HOST = "0.0.0.0"
WS_PORT = 8765
MODEL_PATH = Path("models/csp_lda.joblib")

# Marker → class (PhysioNet EEGBCI left/right fist imagery runs)
MARKER_TO_CLASS = {"T0": "rest", "T1": "left", "T2": "right"}
MOTOR_CHANNELS = ["C3", "C4"]


# --- Signal helpers -------------------------------------------------------

def butter_bandpass(sfreq: float, low: float, high: float, order: int = 4):
    nyq = sfreq / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def bandpass(window: np.ndarray, b, a) -> np.ndarray:
    """window: (n_samples, n_channels) → same shape, filtered."""
    return filtfilt(b, a, window, axis=0)


def band_power(filtered: np.ndarray) -> np.ndarray:
    return np.mean(filtered ** 2, axis=0)


# --- LSL ------------------------------------------------------------------

def connect_lsl(timeout: float = 15.0):
    print("Looking for LSL 'EEG' stream...")
    eeg_streams = resolve_byprop("name", "EEG", timeout=timeout)
    if not eeg_streams:
        raise RuntimeError("No 'EEG' stream found. Start lsl.py first.")
    eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=10)

    print("Looking for LSL 'Markers' stream...")
    marker_streams = resolve_byprop("name", "Markers", timeout=2.0)
    marker_inlet = StreamInlet(marker_streams[0]) if marker_streams else None
    if marker_inlet is None:
        print("  (no Markers stream — rolling accuracy disabled)")

    info = eeg_inlet.info()
    sfreq = float(info.nominal_srate())
    n_ch = info.channel_count()

    ch = info.desc().child("channels").child("channel")
    names = []
    while ch.name() == "channel":
        names.append(ch.child_value("label"))
        ch = ch.next_sibling()
    if len(names) != n_ch:
        names = [f"ch{i}" for i in range(n_ch)]

    print(f"Connected: {n_ch} ch @ {sfreq} Hz  names[:6]={names[:6]}")
    return eeg_inlet, marker_inlet, sfreq, names


# --- Decoder state --------------------------------------------------------

class LiveDecoder:
    def __init__(self, model_path: Path):
        bundle = load(model_path)
        self.pipeline = bundle["pipeline"]
        self.meta = bundle["meta"]
        self.int_to_class = {int(k): v for k, v in self.meta["int_to_class"].items()}
        self.class_to_int = self.meta["class_to_int"]
        self.model_ch_names = list(self.meta["ch_names"])
        self.channel_map = None  # filled once we see the stream
        self.history = deque(maxlen=80)  # for rolling accuracy
        self.n_correct = 0
        self.n_scored = 0

    def bind_channels(self, stream_names: list[str]):
        missing = [c for c in self.model_ch_names if c not in stream_names]
        if missing:
            raise RuntimeError(
                f"Stream is missing {len(missing)} channels the model needs "
                f"(e.g. {missing[:5]}). Retrain on matching channel set."
            )
        self.channel_map = [stream_names.index(c) for c in self.model_ch_names]
        print(f"Channel map locked ({len(self.channel_map)} EEG channels).")

    def predict(self, window: np.ndarray) -> dict:
        """
        window: (n_samples, n_stream_channels) in volts (as from LSL/MNE).
        Returns prediction dict with probs + confidence.
        3-class model: left / right / rest (rest learned from T0 epochs).
        """
        x = window[:, self.channel_map].T[np.newaxis, :, :]
        proba = self.pipeline.predict_proba(x)[0]
        pred_i = int(np.argmax(proba))
        pred = self.int_to_class[pred_i]
        probs = {self.int_to_class[i]: float(proba[i]) for i in range(len(proba))}
        # Ensure all three keys exist even if an older 2-class model is loaded
        for name in ("left", "right", "rest"):
            probs.setdefault(name, 0.0)
        return {
            "prediction": pred,
            "confidence": float(proba[pred_i]),
            "probs": probs,
        }

    def score_against_marker(self, prediction: str, marker: str | None):
        if marker is None:
            return None
        truth = MARKER_TO_CLASS.get(marker)
        if truth is None:
            return None
        self.n_scored += 1
        if prediction == truth:
            self.n_correct += 1
        self.history.append(1 if prediction == truth else 0)
        return self.n_correct / self.n_scored

    @property
    def rolling_accuracy(self) -> float | None:
        if not self.history:
            return None
        return float(np.mean(self.history))


# --- WebSocket hub --------------------------------------------------------

class Hub:
    def __init__(self):
        self.clients: set = set()

    async def register(self, ws):
        self.clients.add(ws)
        print(f"Client connected ({len(self.clients)} total)")

    async def unregister(self, ws):
        self.clients.discard(ws)
        print(f"Client disconnected ({len(self.clients)} total)")

    async def broadcast(self, payload: dict):
        if not self.clients:
            return
        msg = json.dumps(payload)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)


# --- Mock mode ------------------------------------------------------------

async def mock_loop(hub: Hub):
    """Emit plausible fake predictions so the frontend can be wired first."""
    classes = ["left", "right", "rest"]
    i = 0
    print("MOCK mode — broadcasting synthetic predictions")
    while True:
        pred = classes[i % 3]
        # Hold each class for ~2.5s (10 updates)
        hold = (i // 10) % 3
        pred = classes[hold]
        conf = 0.55 + 0.4 * abs(np.sin(i / 7))
        other = (1.0 - conf) / 2.0
        probs = {c: (conf if c == pred else other) for c in classes}
        payload = {
            "type": "prediction",
            "prediction": pred,
            "confidence": float(conf),
            "probs": probs,
            "band_power": {
                "C3": float(8 + 4 * (pred == "right")),
                "C4": float(8 + 4 * (pred == "left")),
            },
            "marker": None,
            "rolling_accuracy": None,
            "n_scored": 0,
            "timestamp": time.time(),
            "mock": True,
        }
        await hub.broadcast(payload)
        print(f"[mock] {pred:5s}  conf={conf:.2f}")
        i += 1
        await asyncio.sleep(UPDATE_SECONDS)


# --- Live loop ------------------------------------------------------------

async def live_loop(hub: Hub, decoder: LiveDecoder):
    eeg_inlet, marker_inlet, sfreq, names = await asyncio.to_thread(connect_lsl)
    decoder.bind_channels(names)

    window_size = int(WINDOW_SECONDS * sfreq)
    b, a = butter_bandpass(sfreq, LOW_HZ, HIGH_HZ)

    motor_idx = {ch: names.index(ch) for ch in MOTOR_CHANNELS if ch in names}

    buffer: list = []
    current_marker = None
    last_emit = 0.0

    print(f"Decoding every {UPDATE_SECONDS}s on {WINDOW_SECONDS}s windows "
          f"({window_size} samples)...\n")

    while True:
        chunk, _ = await asyncio.to_thread(
            eeg_inlet.pull_chunk, 0.05, 64
        )
        if chunk:
            buffer.extend(chunk)
            if len(buffer) > window_size:
                buffer = buffer[-window_size:]

        if marker_inlet is not None:
            sample, _ = await asyncio.to_thread(marker_inlet.pull_sample, 0.0)
            if sample:
                current_marker = sample[0]

        now = time.time()
        if now - last_emit < UPDATE_SECONDS or len(buffer) < window_size:
            await asyncio.sleep(0.01)
            continue

        last_emit = now
        window = np.asarray(buffer, dtype=np.float64)  # volts

        # Match training: bandpass 8–30 Hz before CSP (training filtered the Raw)
        filtered_v = bandpass(window, b, a)

        # Band-power readout for HUD (µV²)
        power = band_power(filtered_v * 1e6)
        band = {ch: float(power[idx]) for ch, idx in motor_idx.items()}

        result = await asyncio.to_thread(decoder.predict, filtered_v)
        # Score against all three PhysioNet markers (T0/T1/T2)
        rolling = decoder.score_against_marker(result["prediction"], current_marker)

        payload = {
            "type": "prediction",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probs": result["probs"],
            "band_power": band,
            "marker": current_marker,
            "rolling_accuracy": rolling if rolling is not None else decoder.rolling_accuracy,
            "n_scored": decoder.n_scored,
            "timestamp": now,
            "mock": False,
        }
        await hub.broadcast(payload)

        acc_str = (
            f"acc={payload['rolling_accuracy']:.2f}"
            if payload["rolling_accuracy"] is not None
            else "acc=—"
        )
        print(
            f"pred={result['prediction']:5s}  conf={result['confidence']:.2f}  "
            f"marker={current_marker or '?':3s}  {acc_str}  "
            f"C3={band.get('C3', 0):.1f} C4={band.get('C4', 0):.1f}"
        )


# --- Server ---------------------------------------------------------------

async def ws_handler(websocket, hub: Hub):
    await hub.register(websocket)
    try:
        # Keep the connection open; we only push server → client.
        async for _ in websocket:
            pass
    finally:
        await hub.unregister(websocket)


async def main_async(args):
    hub = Hub()

    async def handler(ws):
        await ws_handler(ws, hub)

    print(f"WebSocket listening on ws://{WS_HOST}:{WS_PORT}")
    async with serve(handler, WS_HOST, WS_PORT):
        if args.mock:
            await mock_loop(hub)
        else:
            if not args.model.exists():
                raise SystemExit(
                    f"Model not found at {args.model}. "
                    f"Run: python train_decoder.py"
                )
            decoder = LiveDecoder(args.model)
            print(
                f"Loaded model (CV acc={decoder.meta['cv_accuracy_mean']:.1%} "
                f"on subjects {decoder.meta['subjects']})"
            )
            await live_loop(hub, decoder)


def main():
    parser = argparse.ArgumentParser(description="Live MI decoder + WebSocket")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Broadcast synthetic predictions (no LSL / no model needed)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
