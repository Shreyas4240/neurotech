"""
train_decoder.py

Step 3 (offline): train a CSP + LDA motor-imagery classifier on PhysioNet
EEGBCI left/right fist imagery runs, then save the fitted pipeline so
decode_live.py can load it for real-time inference.

Classes:
  left  - imagined left fist  (T1)
  right - imagined right fist (T2)
  rest  - inter-trial baseline (T0) — trained as a real class, not a gate

Usage:
  python train_decoder.py
  python train_decoder.py --subjects 1 2 3 --out models/csp_lda.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
from joblib import dump
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# Motor-imagery runs: left/right fist imagery
DEFAULT_RUNS = [4, 8, 12]
# Subject 7 is a strong MI performer on PhysioNet (~98% CSP+LDA CV).
# Keep lsl.py SUBJECT in sync so the live replay matches the trained model.
DEFAULT_SUBJECTS = [7]

# Epoch window relative to cue onset (seconds)
TMIN, TMAX = 0.5, 3.5
LOW_HZ, HIGH_HZ = 8.0, 30.0
N_CSP_COMPONENTS = 6

# Genuine 3-class: rest is learned from T0 baseline epochs, not a
# confidence threshold at inference time.
LABEL_MAP = {"T0": "rest", "T1": "left", "T2": "right"}
CLASS_TO_INT = {"left": 0, "right": 1, "rest": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}


def load_subject(subject: int, runs: list[int]) -> mne.io.BaseRaw:
    fnames = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(f, preload=True, verbose="ERROR") for f in fnames]
    raw = concatenate_raws(raws, verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore")
    # Keep EEG only
    raw.pick(picks="eeg")
    return raw


def make_epochs(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass, epoch around cues, return (X, y) with y in {0,1,2}."""
    raw_f = raw.copy().filter(LOW_HZ, HIGH_HZ, fir_design="firwin", verbose="ERROR")

    events, event_id = mne.events_from_annotations(raw_f, verbose="ERROR")
    # Keep PhysioNet codes in event_id; rename keys to our class labels.
    # MNE requires event_id values to match events[:, 2], not our class ints.
    selected = {}
    code_to_class = {}
    for name, code in event_id.items():
        label = LABEL_MAP.get(str(name))
        if label is not None:
            selected[label] = int(code)
            code_to_class[int(code)] = CLASS_TO_INT[label]

    if not selected:
        raise RuntimeError(f"No usable events found. event_id={event_id}")

    epochs = mne.Epochs(
        raw_f,
        events,
        event_id=selected,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    X = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)
    y = np.array([code_to_class[int(c)] for c in epochs.events[:, -1]])
    return X, y


def build_pipeline() -> Pipeline:
    csp = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return Pipeline([("csp", csp), ("lda", lda)])


def train(subjects: list[int], runs: list[int]) -> tuple[Pipeline, dict]:
    Xs, ys = [], []
    ch_names = None
    sfreq = None

    for subj in subjects:
        print(f"Loading subject {subj:03d} runs {runs}...")
        raw = load_subject(subj, runs)
        if ch_names is None:
            ch_names = list(raw.ch_names)
            sfreq = float(raw.info["sfreq"])
        X, y = make_epochs(raw)
        print(f"  epochs: {X.shape[0]}  channels: {X.shape[1]}  samples: {X.shape[2]}")
        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    print(f"\nTotal epochs: {X.shape[0]}  class counts: "
          + ", ".join(f"{INT_TO_CLASS[c]}={int((y == c).sum())}" for c in sorted(INT_TO_CLASS)))

    clf = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)
    train_acc = float(clf.score(X, y))
    print(f"Train accuracy (full fit): {train_acc:.3f}")

    meta = {
        "ch_names": ch_names,
        "sfreq": sfreq,
        "tmin": TMIN,
        "tmax": TMAX,
        "low_hz": LOW_HZ,
        "high_hz": HIGH_HZ,
        "n_csp": N_CSP_COMPONENTS,
        "class_to_int": CLASS_TO_INT,
        "int_to_class": INT_TO_CLASS,
        "subjects": subjects,
        "runs": runs,
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "train_accuracy": train_acc,
        "window_seconds": TMAX - TMIN,
    }
    return clf, meta


def main():
    parser = argparse.ArgumentParser(description="Train CSP+LDA MI decoder")
    parser.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--runs", type=int, nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=Path("models/csp_lda.joblib"))
    args = parser.parse_args()

    clf, meta = train(args.subjects, args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump({"pipeline": clf, "meta": meta}, args.out)
    print(f"\nSaved model → {args.out.resolve()}")
    print(f"CV accuracy to report: {meta['cv_accuracy_mean']:.1%} "
          f"± {meta['cv_accuracy_std']:.1%}")


if __name__ == "__main__":
    main()
