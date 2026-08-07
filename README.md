# Closed-Loop Motor Imagery BCI

Live EEG (PhysioNet replayed over LSL, or a real headset) → sliding-window
bandpass → CSP+LDA decode → WebSocket → Three.js brain mesh + cursor control.

```
EEG source (lsl.py)
        │ LSL
        ▼
decode_live.py  ──WebSocket :8765──▶  /bci  (brain mesh + cursor)
   CSP + LDA
```

## Quick start

```bash
# from digi_bci/
source .venv/bin/activate
pip install -r requirements.txt

# 1. Train the decoder once (PhysioNet subject 7 — strong MI performer)
python train_decoder.py
# → 3-class CSP+LDA (left / right / rest from T0) ≈ 89% CV on subject 7

# 2. Three terminals:
python lsl.py                  # replay subject 7 MI as live LSL (loops)
python decode_live.py          # real-time CSP+LDA → ws://localhost:8765
cd brain_replica && python main.py   # serves http://localhost:8080/bci
```

Open **http://localhost:8080/bci**. You should see:

- Motor cortex regions lighting up with each prediction (contralateral)
- A cursor sliding left / right / center with imagined fist class
- Live confidence, class probabilities, C3/C4 band power, and rolling
  accuracy vs. T0/T1/T2 markers (typically ~80–90% on the subject-7 replay)

### Frontend-only wiring check (no model / no LSL)

```bash
python decode_live.py --mock
# then open /bci — synthetic left→right→rest cycle
```

## Pipeline pieces

| File | Role |
|------|------|
| `lsl.py` | PhysioNet EEGBCI replay → LSL `EEG` + `Markers` |
| `bandpower.py` | Step-2 monitor: rolling 8–30 Hz power at C3/C4 |
| `train_decoder.py` | Offline CSP (6) + LDA on subjects 1–5, runs 4/8/12 |
| `decode_live.py` | Sliding window decode + WebSocket broadcast |
| `brain_replica/static/bci.html` | Closed-loop 3D scene |
| `models/csp_lda.joblib` | Saved pipeline + channel metadata |

## Hardware swap

Anything that publishes an LSL stream named `EEG` with the same 64-channel
10–20 labels works — OpenBCI, Muse (via a bridge), or BrainFlow→LSL.
Retrain if the channel set differs: `python train_decoder.py --subjects …`.

## Classes

| Prediction | PhysioNet marker | Cursor | Cortex glow |
|------------|------------------|--------|-------------|
| `left` | T1 | ← | Right frontal + parietal |
| `right` | T2 | → | Left frontal + parietal |
| `rest` | T0 | center | dim |
