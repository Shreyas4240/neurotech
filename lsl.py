"""
eeg_lsl_streamer.py

Step 1 of the BCI pipeline: EEG source.

Loads the PhysioNet EEG Motor Movement/Imagery dataset (via MNE) and replays
it in real time over two LSL streams:

  1. "EEG"     - the actual channel data, paced to match the original
                 sampling rate, so downstream code sees it exactly like a
                 live headset feed.
  2. "Markers" - event labels (rest / left fist / right fist / both fists /
                 both feet) pushed at the moment each event occurs, so your
                 decoder training script can build labeled epochs later.

Install first:
    pip install mne pylsl numpy

Run:
    python eeg_lsl_streamer.py

Then anything that connects to LSL (your preprocessing/decoder script, or
even just LSL's own "lsl_viewer") will see it as a live stream.
"""

import time
import sys
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from pylsl import StreamInfo, StreamOutlet, local_clock

sys.stdout.reconfigure(line_buffering=True)

# --- Config ---------------------------------------------------------------

SUBJECT = 7  # keep in sync with train_decoder.DEFAULT_SUBJECTS
# Runs 4, 8, 12 = left/right fist motor imagery
# Runs 6, 10, 14 = both fists/both feet motor imagery
# (see MNE eegbci docs for the full run/task mapping)
RUNS = [4, 8, 12]

CHUNK_SIZE = 32  # samples pushed per loop iteration


def load_data():
    """Download (if needed) and load the PhysioNet motor imagery runs."""
    raw_fnames = eegbci.load_data(SUBJECT, RUNS)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

    # Standardize channel names (PhysioNet uses a slightly nonstandard
    # naming scheme with trailing dots)
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore")

    return raw


def build_outlets(raw):
    """Create the LSL EEG stream and the LSL marker stream."""
    ch_names = raw.info["ch_names"]
    sfreq = raw.info["sfreq"]

    eeg_info = StreamInfo(
        name="EEG",
        type="EEG",
        channel_count=len(ch_names),
        nominal_srate=sfreq,
        channel_format="float32",
        source_id="eegbci-replay-eeg",
    )
    chns = eeg_info.desc().append_child("channels")
    for ch in ch_names:
        chns.append_child("channel").append_child_value("label", ch)

    marker_info = StreamInfo(
        name="Markers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,  # irregular rate: pushed only when an event happens
        channel_format="string",
        source_id="eegbci-replay-markers",
    )

    eeg_outlet = StreamOutlet(eeg_info)
    marker_outlet = StreamOutlet(marker_info)
    return eeg_outlet, marker_outlet


def get_events(raw):
    """Extract (sample_index, label) pairs from the raw annotations."""
    events, event_id = mne.events_from_annotations(raw)
    id_to_label = {v: k for k, v in event_id.items()}
    return [(int(samp), id_to_label[eid]) for samp, _, eid in events]


def stream(raw, eeg_outlet, marker_outlet):
    """Replay the recording in real time over both LSL outlets."""
    data = raw.get_data().T  # shape: (n_samples, n_channels)
    sfreq = raw.info["sfreq"]
    events = get_events(raw)
    event_idx = 0
    n_samples = data.shape[0]

    print(f"Streaming {n_samples} samples at {sfreq} Hz "
          f"({n_samples / sfreq:.1f}s of recording)...")

    start_time = local_clock()
    sample_i = 0

    while sample_i < n_samples:
        chunk = data[sample_i:sample_i + CHUNK_SIZE]
        eeg_outlet.push_chunk(chunk.tolist())

        # Fire any markers whose sample index falls within this chunk
        while (event_idx < len(events)
               and events[event_idx][0] < sample_i + CHUNK_SIZE):
            _, label = events[event_idx]
            marker_outlet.push_sample([label])
            print(f"  [{sample_i / sfreq:6.1f}s] marker: {label}")
            event_idx += 1

        sample_i += CHUNK_SIZE

        # Pace the loop to match real time instead of dumping all data at once
        target_time = start_time + (sample_i / sfreq)
        sleep_time = target_time - local_clock()
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("Done streaming. Restart the script to loop again.")


def main():
    print("Loading PhysioNet motor imagery data (downloads on first run)...")
    raw = load_data()
    print(f"Loaded {len(raw.info['ch_names'])} channels @ {raw.info['sfreq']} Hz")

    eeg_outlet, marker_outlet = build_outlets(raw)
    print("LSL streams 'EEG' and 'Markers' are now live. "
          "Waiting 2s for consumers to discover them...")
    time.sleep(2)

    # Loop forever so a portfolio demo can run unattended
    while True:
        stream(raw, eeg_outlet, marker_outlet)
        print("Looping recording from the start...\n")
        time.sleep(1)


if __name__ == "__main__":
    main()