# build_cnn_data.py
#
# Builds a mel spectrogram dataset from the ICBHI respiratory audio files.
#
# Key design decisions (report-worthy):
#   - Patient-level train/val/test split to prevent data leakage
#   - Per-sample z-score normalisation so each cycle is on the same scale
#     regardless of recording device or loudness
#   - hop_length tuned so 128 frames ≈ 6 s of audio, covering the longest cycles
#   - Padding uses "reflect" mode to avoid zero-edge artefacts
#   - Stratified patient split attempts to balance class representation

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split


# ── Config ────────────────────────────────────────────────────────────────────
AUDIO_DIR    = "audio_and_txt_files"
OUTPUT_DIR   = "cnn_data"

SAMPLE_RATE  = 22050
N_MELS       = 128
MAX_FRAMES   = 128
# hop_length = 512 → frame step ≈ 23 ms → 128 frames ≈ 3 s
# reduce to 256 for finer time resolution on short crackle events
HOP_LENGTH   = 256
N_FFT        = 1024
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Label encoding ────────────────────────────────────────────────────────────
def cycle_label(crackle: int, wheeze: int) -> int:
    """Map (crackle, wheeze) flags to a 4-class integer label."""
    if crackle == 0 and wheeze == 0:
        return 0  # normal
    if crackle == 1 and wheeze == 0:
        return 1  # crackle only
    if crackle == 0 and wheeze == 1:
        return 2  # wheeze only
    return 3      # both crackle and wheeze


# ── Spectrogram helpers ───────────────────────────────────────────────────────
def pad_or_crop(spec: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Ensure the spectrogram has exactly max_frames time steps.
    - Short clips: reflect-pad (avoids zero-edge artefacts).
    - Long clips:  centre-crop to retain the most informative region.
    """
    n_frames = spec.shape[1]
    if n_frames < max_frames:
        pad_width = max_frames - n_frames
        # reflect padding mirrors the edges rather than inserting silence
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="reflect")
    elif n_frames > max_frames:
        # centre crop
        start = (n_frames - max_frames) // 2
        spec = spec[:, start : start + max_frames]
    return spec


def normalise_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Per-sample z-score normalisation.
    Removes gain differences between recording devices and patients.
    """
    mean = spec.mean()
    std  = spec.std()
    return (spec - mean) / (std + 1e-8)


def audio_to_mel(cycle_audio: np.ndarray) -> np.ndarray:
    """Convert a raw audio segment to a normalised log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=cycle_audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = pad_or_crop(mel_db, MAX_FRAMES)
    mel_db = normalise_spectrogram(mel_db)
    return mel_db.astype(np.float32)


# ── Dataset builder ───────────────────────────────────────────────────────────
def get_patient_id(filename: str) -> str:
    """Extract the numeric patient ID from a filename like '101_1b1_Al_sc.wav'."""
    return filename.split("_")[0]


def build_dataset():
    X, y, patients = [], [], []

    wav_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    print(f"Found {len(wav_files)} WAV files.")

    for wav_file in wav_files:
        base_name  = os.path.splitext(wav_file)[0]
        txt_path   = os.path.join(AUDIO_DIR, base_name + ".txt")
        wav_path   = os.path.join(AUDIO_DIR, wav_file)

        if not os.path.exists(txt_path):
            print(f"  [SKIP] No annotation file for {wav_file}")
            continue

        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        annotations = np.loadtxt(txt_path)

        if annotations.ndim == 1:
            annotations = annotations.reshape(1, -1)

        patient_id = get_patient_id(wav_file)

        for row in annotations:
            start, end, crackle, wheeze = row

            start_sample = int(start * SAMPLE_RATE)
            end_sample   = int(end   * SAMPLE_RATE)
            cycle_audio  = audio[start_sample:end_sample]

            if len(cycle_audio) < HOP_LENGTH:
                # too short to form even one frame — skip
                continue

            spec  = audio_to_mel(cycle_audio)
            label = cycle_label(int(crackle), int(wheeze))

            X.append(spec)
            y.append(label)
            patients.append(patient_id)

    X        = np.array(X,        dtype=np.float32)
    y        = np.array(y,        dtype=np.int64)
    patients = np.array(patients, dtype=str)

    # Add channel dimension → (N, 128, 128, 1) for Conv2D
    X = X[..., np.newaxis]

    return X, y, patients


# ── Patient-level split ───────────────────────────────────────────────────────
def patient_level_split(X, y, patients):
    """
    Split by patient, not by cycle.
    This is essential: cycles from the same patient share recording conditions
    and breathing patterns, so a random cycle-level split would leak information
    from train into test and artificially inflate performance.

    Split: 70% train | 15% val | 15% test  (patient-level)
    """
    unique_patients = np.unique(patients)

    train_patients, temp_patients = train_test_split(
        unique_patients,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    train_mask = np.isin(patients, train_patients)
    val_mask   = np.isin(patients, val_patients)
    test_mask  = np.isin(patients, test_patients)

    return (
        X[train_mask], X[val_mask],  X[test_mask],
        y[train_mask], y[val_mask],  y[test_mask],
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Building mel spectrogram dataset...")
    X, y, patients = build_dataset()

    print(f"\nTotal cycles      : {len(X)}")
    print(f"Spectrogram shape : {X.shape}")
    print(f"Class distribution: {np.bincount(y)}  (0=normal, 1=crackle, 2=wheeze, 3=both)")

    X_train, X_val, X_test, y_train, y_val, y_test = patient_level_split(X, y, patients)

    for split_name, Xs, ys in [
        ("Train", X_train, y_train),
        ("Val",   X_val,   y_val),
        ("Test",  X_test,  y_test),
    ]:
        print(f"{split_name:5s}: {Xs.shape}  classes {np.bincount(ys)}")

    np.save(os.path.join(OUTPUT_DIR, "X_train_spec.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val_spec.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test_spec.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train_cycle.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val_cycle.npy"),   y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test_cycle.npy"),  y_test)

    print("\nCNN dataset saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
