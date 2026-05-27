#@title 3. Feature Extraction (cached to disk after first run)

import numpy as np
import librosa
import time
from collections import OrderedDict
from tqdm.auto import tqdm

CACHE = "data/features/raw_features.npz"

SR = 22050
N_MFCC = 40
N_MELS = 128

def extract_features(y, sr=SR):
    """~558 features: MFCC + delta + delta2 + mel + chroma + tonnetz + HPSS + spectral + ZCR/RMS."""
    feats = OrderedDict()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for name, mat in [("mfcc", mfcc), ("delta_mfcc", delta), ("delta2_mfcc", delta2)]:
        for i in range(mat.shape[0]):
            feats[f"{name}_mean_{i}"] = mat[i].mean()
            feats[f"{name}_std_{i}"] = mat[i].std()

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    for i in range(mel_db.shape[0]):
        feats[f"mel_mean_{i}"] = mel_db[i].mean()
        feats[f"mel_std_{i}"] = mel_db[i].std()

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        feats[f"chroma_mean_{i}"] = chroma[i].mean()
        feats[f"chroma_std_{i}"] = chroma[i].std()

    sc = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(sc.shape[0]):
        feats[f"spectral_contrast_mean_{i}"] = sc[i].mean()
        feats[f"spectral_contrast_std_{i}"] = sc[i].std()

    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for i in range(tonnetz.shape[0]):
            feats[f"tonnetz_mean_{i}"] = tonnetz[i].mean()
            feats[f"tonnetz_std_{i}"] = tonnetz[i].std()
    except Exception:
        for i in range(6):
            feats[f"tonnetz_mean_{i}"] = 0.0
            feats[f"tonnetz_std_{i}"] = 0.0

    try:
        y_h, y_p = librosa.effects.hpss(y)
        feats["harmonic_energy"] = float(np.mean(y_h**2))
        feats["percussive_energy"] = float(np.mean(y_p**2))
    except Exception:
        feats["harmonic_energy"] = 0.0
        feats["percussive_energy"] = 0.0

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["spectral_centroid_mean"]  = cent.mean(); feats["spectral_centroid_std"]  = cent.std()
    feats["spectral_bandwidth_mean"] = bw.mean();   feats["spectral_bandwidth_std"] = bw.std()
    feats["spectral_rolloff_mean"]   = roll.mean(); feats["spectral_rolloff_std"]   = roll.std()

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    feats["zcr_mean"] = zcr.mean(); feats["zcr_std"] = zcr.std()
    feats["rms_mean"] = rms.mean(); feats["rms_std"] = rms.std()

    return feats


if os.path.exists(CACHE):
    print(f"Loading cached features from {CACHE}")
    z = np.load(CACHE, allow_pickle=True)
    X_all = z["X"]
    y_cycle_all = z["y_cycle"]
    y_disease_all = z["y_disease"]
    pids_all = z["pids"]
    feat_names = list(z["feat_names"])
    disease_to_id = {k: int(v) for k, v in z["disease_to_id"].item().items()}
    print(f"  X: {X_all.shape}, features: {len(feat_names)}")
else:
    print("Building cycle list from annotations...")
    disease_classes = sorted(diag["disease"].unique())
    disease_to_id = {d: i for i, d in enumerate(disease_classes)}
    patient_to_disease = dict(zip(diag["patient_id"], diag["disease"]))

    all_cycles = []
    wav_files = sorted([f for f in os.listdir(ROOT) if f.endswith(".wav")])
    for wav in wav_files:
        base = wav.replace(".wav", "")
        txt = f"{ROOT}/{base}.txt"
        if not os.path.exists(txt):
            continue
        pid = int(base.split("_")[0])
        if pid not in patient_to_disease:
            continue
        try:
            ann = pd.read_csv(txt, sep="\t", header=None,
                              names=["start","end","crackle","wheeze"])
        except Exception:
            continue
        for _, r in ann.iterrows():
            all_cycles.append({
                "patient_id": pid, "recording": base,
                "start": float(r["start"]), "end": float(r["end"]),
                "crackle": int(r["crackle"]), "wheeze": int(r["wheeze"]),
            })

    cycle_df = pd.DataFrame(all_cycles)
    print(f"Total cycles: {len(cycle_df)} from {cycle_df['patient_id'].nunique()} patients")

    def cycle_label(row):
        c, w = row["crackle"], row["wheeze"]
        if c and w: return 3
        if c: return 1
        if w: return 2
        return 0

    cycle_df["cycle_label"] = cycle_df.apply(cycle_label, axis=1)
    cycle_df["disease_label"] = cycle_df["patient_id"].map(
        lambda p: disease_to_id[patient_to_disease[p]])

    print("\nExtracting features (~20 min on Colab CPU)...")
    X_rows, y_cycle, y_disease, patient_ids, feat_names = [], [], [], [], None
    t0 = time.time()

    grouped = cycle_df.groupby("recording")
    for rec_name, rec_cycles in tqdm(grouped, total=len(grouped)):
        wav_path = f"{ROOT}/{rec_name}.wav"
        try:
            y_full, sr = librosa.load(wav_path, sr=SR)
        except Exception:
            continue
        for _, row in rec_cycles.iterrows():
            s = int(row["start"] * SR)
            e = int(row["end"] * SR)
            if e - s < SR * 0.2:
                continue
            try:
                feats = extract_features(y_full[s:e], sr=sr)
            except Exception:
                continue
            if feat_names is None:
                feat_names = list(feats.keys())
            X_rows.append(list(feats.values()))
            y_cycle.append(row["cycle_label"])
            y_disease.append(row["disease_label"])
            patient_ids.append(row["patient_id"])

    X_all = np.array(X_rows, dtype=np.float32)
    y_cycle_all = np.array(y_cycle, dtype=np.int32)
    y_disease_all = np.array(y_disease, dtype=np.int64)
    pids_all = np.array(patient_ids, dtype=np.int32)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"X shape: {X_all.shape}")
    print(f"NaN: {np.isnan(X_all).sum()}, Inf: {np.isinf(X_all).sum()}")

    # Cache
    np.savez_compressed(CACHE,
        X=X_all, y_cycle=y_cycle_all, y_disease=y_disease_all,
        pids=pids_all, feat_names=np.array(feat_names),
        disease_to_id=np.array(disease_to_id, dtype=object))
    print(f"Cached to {CACHE} ({os.path.getsize(CACHE)//1024//1024} MB)")

disease_map = {str(v): k for k, v in disease_to_id.items()}
cycle_map = {0:"Normal", 1:"Crackle", 2:"Wheeze", 3:"Both"}