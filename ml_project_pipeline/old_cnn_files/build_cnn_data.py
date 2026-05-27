# build_cnn_data.py

import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split


AUDIO_DIR = "audio_and_txt_files"
OUTPUT_DIR = "cnn_data"

SAMPLE_RATE = 22050
N_MELS = 128
MAX_FRAMES = 128
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def cycle_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return 0  # normal
    if crackle == 1 and wheeze == 0:
        return 1  # crackle
    if crackle == 0 and wheeze == 1:
        return 2  # wheeze
    return 3      # both


def pad_or_crop(spec, max_frames):
    if spec.shape[1] < max_frames:
        pad_width = max_frames - spec.shape[1]
        return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    return spec[:, :max_frames]


def audio_to_mel(cycle_audio):
    mel = librosa.feature.melspectrogram(
        y=cycle_audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = pad_or_crop(mel_db, MAX_FRAMES)

    return mel_db


def get_patient_id(filename):
    return filename.split("_")[0]


def build_dataset():
    X, y, patients = [], [], []

    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]
        txt_file = base_name + ".txt"

        wav_path = os.path.join(AUDIO_DIR, wav_file)
        txt_path = os.path.join(AUDIO_DIR, txt_file)

        if not os.path.exists(txt_path):
            continue

        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        annotations = np.loadtxt(txt_path)

        if annotations.ndim == 1:
            annotations = annotations.reshape(1, -1)

        patient_id = get_patient_id(wav_file)

        for row in annotations:
            start, end, crackle, wheeze = row

            start_sample = int(start * SAMPLE_RATE)
            end_sample = int(end * SAMPLE_RATE)

            cycle_audio = audio[start_sample:end_sample]

            if len(cycle_audio) == 0:
                continue

            spec = audio_to_mel(cycle_audio)
            label = cycle_label(int(crackle), int(wheeze))

            X.append(spec)
            y.append(label)
            patients.append(patient_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    patients = np.array(patients)

    X = X[..., np.newaxis]

    return X, y, patients


def patient_level_split(X, y, patients):
    unique_patients = np.unique(patients)

    train_patients, temp_patients = train_test_split(
        unique_patients,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    train_mask = np.isin(patients, train_patients)
    val_mask = np.isin(patients, val_patients)
    test_mask = np.isin(patients, test_patients)

    return (
        X[train_mask], X[val_mask], X[test_mask],
        y[train_mask], y[val_mask], y[test_mask]
    )


def main():
    print("Building mel spectrogram dataset...")

    X, y, patients = build_dataset()

    print(f"Total cycles: {len(X)}")
    print(f"Spectrogram shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    X_train, X_val, X_test, y_train, y_val, y_test = patient_level_split(
        X, y, patients
    )

    np.save(os.path.join(OUTPUT_DIR, "X_train_spec.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val_spec.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test_spec.npy"), X_test)

    np.save(os.path.join(OUTPUT_DIR, "y_train_cycle.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val_cycle.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test_cycle.npy"), y_test)

    print("CNN dataset saved successfully.")
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")


if __name__ == "__main__":
    main()