"""
eeg_bandpower_monitor.py

Step 2 of the BCI pipeline: real-time filtering.

Connects to the "EEG" and "Markers" LSL streams produced by
eeg_lsl_streamer.py, keeps a rolling window of the most recent EEG samples,
bandpass-filters that window to the mu/beta range (8-30 Hz - the band where
motor imagery shows up), and prints live power at C3 and C4 (the standard
left/right motor cortex electrodes) alongside whatever marker is currently
active.

This is the piece that turns "raw numbers streaming in" into "a signal we
can actually reason about." The CSP/LDA decoder (step 3) will replace the
simple C3/C4 print-out with a real left/right hand classifier, but the
windowing + filtering logic here stays the same.

Install first:
    pip install pylsl scipy numpy

Run this WHILE eeg_lsl_streamer.py is running in another terminal:
    python eeg_bandpower_monitor.py
"""

import numpy as np
from scipy.signal import butter, filtfilt
from pylsl import StreamInlet, resolve_byprop

# --- Config -----------------------------------------------------------

WINDOW_SECONDS = 1.0     # how much history we filter/analyze at once
UPDATE_SECONDS = 0.25    # how often we recompute and print
LOW_HZ, HIGH_HZ = 8.0, 30.0   # mu + beta band, where motor imagery lives

# The channels we care about for left/right hand motor imagery.
# C3 = left motor cortex (controls right hand), C4 = right motor cortex
# (controls left hand) - this crossover is normal neuroanatomy, not a bug.
CHANNELS_OF_INTEREST = ["C3", "C4"]


def connect_to_streams():
    print("Looking for 'EEG' stream...")
    eeg_streams = resolve_byprop("name", "EEG", timeout=10)
    if not eeg_streams:
        raise RuntimeError(
            "No 'EEG' stream found. Is eeg_lsl_streamer.py running?")
    eeg_inlet = StreamInlet(eeg_streams[0])

    print("Looking for 'Markers' stream...")
    marker_streams = resolve_byprop("name", "Markers", timeout=10)
    marker_inlet = StreamInlet(marker_streams[0]) if marker_streams else None
    if marker_inlet is None:
        print("  (no marker stream found - continuing without labels)")

    return eeg_inlet, marker_inlet


def get_channel_indices(inlet):
    """Read channel names out of the stream's metadata so we know which
    columns of incoming data correspond to C3 and C4."""
    info = inlet.info()
    ch = info.desc().child("channels").child("channel")
    names = []
    while ch.name() == "channel":
        names.append(ch.child_value("label"))
        ch = ch.next_sibling()

    indices = {}
    for target in CHANNELS_OF_INTEREST:
        if target in names:
            indices[target] = names.index(target)
        else:
            print(f"  warning: channel '{target}' not found in stream")
    return indices


def bandpass_filter(window, sfreq, low, high):
    """Zero-phase Butterworth bandpass on a (samples, channels) window."""
    nyq = sfreq / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, window, axis=0)


def band_power(filtered_window):
    """Mean squared amplitude per channel - a simple proxy for band power."""
    return np.mean(filtered_window ** 2, axis=0)


def main():
    eeg_inlet, marker_inlet = connect_to_streams()
    sfreq = eeg_inlet.info().nominal_srate()
    window_size = int(WINDOW_SECONDS * sfreq)

    ch_indices = get_channel_indices(eeg_inlet)
    if not ch_indices:
        raise RuntimeError("None of the target channels were found - check "
                            "CHANNELS_OF_INTEREST against your stream.")

    print(f"Connected. Sample rate: {sfreq} Hz. "
          f"Window: {WINDOW_SECONDS}s ({window_size} samples).\n")

    buffer = []  # rolling list of samples, each sample is a list of floats
    current_marker = "?"
    last_print = 0.0

    import time
    start = time.time()

    while True:
        # Pull whatever new EEG samples have arrived
        chunk, _ = eeg_inlet.pull_chunk(timeout=0.1, max_samples=32)
        if chunk:
            buffer.extend(chunk)
            # Keep only the most recent window_size samples
            if len(buffer) > window_size:
                buffer = buffer[-window_size:]

        # Check for a new marker (non-blocking)
        if marker_inlet is not None:
            sample, _ = marker_inlet.pull_sample(timeout=0.0)
            if sample:
                current_marker = sample[0]

        now = time.time()
        if now - last_print >= UPDATE_SECONDS and len(buffer) == window_size:
            last_print = now
            # MNE stores EEG in volts (~1e-5 scale) - convert to microvolts
            # (the standard unit for reading EEG) so the numbers are
            # human-readable instead of rounding to 0.00
            window = np.array(buffer) * 1e6  # shape: (window_size, n_channels)

            filtered = bandpass_filter(window, sfreq, LOW_HZ, HIGH_HZ)
            power = band_power(filtered)

            parts = []
            for name, idx in ch_indices.items():
                parts.append(f"{name}: {power[idx]:8.2f}")

            elapsed = now - start
            print(f"[{elapsed:6.1f}s] marker={current_marker:>3}  "
                  + "  ".join(parts))


if __name__ == "__main__":
    main()