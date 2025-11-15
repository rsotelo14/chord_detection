# build_frames_dataset.py
import numpy as np
import librosa
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# ---- Config ----
SR = 11025                   # paper
HOP = 512                    # ~46.4 ms @ 11025 Hz
BINS_PER_OCT = 36
N_BINS = 180                 # 5 octavas * 36
FMIN = 110.0                 # ~A2
SPLICE_CTX = 1               # t-1, t, t+1  -> 3 frames
MIN_FRAME_DUR = HOP / SR     # ~0.046 s

# Beatles paths (ajusta raíz)
BASE = Path("The Beatles Annotations")
AUDIO_ROOT = BASE / "audio" / "The Beatles"
LAB_ROOT   = BASE / "chordlab" / "The Beatles"
OUT_FRAMES = Path("analysis_out_frames")

NOTES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
ENH = {"C#":"Db","D#":"Eb","F#":"Gb","G#":"Ab","A#":"Bb"}

def norm_label(lab: str) -> str:
    if lab.upper() == "N": return "N"
    parts = lab.split(":")
    root = parts[0]
    rest = parts[1] if len(parts) > 1 else "maj"
    root = ENH.get(root, root).title()
    qual = "min" if "min" in rest else "maj"
    return f"{root}:{qual}" if root in NOTES else "N"

def load_chords_lab(p):
    rows=[]
    with open(p,"r") as f:
        for line in f:
            xs=line.strip().split()
            if len(xs)>=3:
                rows.append((float(xs[0]), float(xs[1]), xs[2]))
    rows.sort(key=lambda x:x[0])
    return rows

def label_for_time(t, intervals):
    # devuelve label del acorde que cubre t, o "N"
    # intervals: [(t0,t1,lab), ...]
    # asume no solapados
    # búsqueda lineal (rápida para este tamaño)
    for t0,t1,lab in intervals:
        if t>=t0 and t<t1:
            return norm_label(lab)
    return "N"

def compute_cqt_frames(y, sr=SR, hop=HOP):
    C = librosa.cqt(y=y, sr=sr, hop_length=hop,
                    fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCT)
    # magnitud log
    X = np.log1p(np.abs(C)).astype(np.float32)   # (n_bins, T)
    times = librosa.frames_to_time(np.arange(X.shape[1]), sr=sr, hop_length=hop)
    return X, times

def make_superframes(X_2d, ctx=SPLICE_CTX):
    # X_2d: (feat, T) -> (T, (2*ctx+1)*feat)
    feat, T = X_2d.shape
    pad = np.pad(X_2d, ((0,0),(ctx,ctx)), mode="edge")
    out = []
    for t in range(T):
        sl = pad[:, t: t+2*ctx+1]   # (feat, 2*ctx+1)
        out.append(sl.T.reshape(-1))  # ( (2*ctx+1)*feat ,)
    return np.stack(out, axis=0)  # (T, F')

def build_frame_dataset(csv_out="frames_dataset.csv",
                        pca_out="pca.joblib", scaler_out="scaler.joblib",
                        npz_out="frames_dataset.npz"):
    rows = []
    X_all = []
    y_all = []
    g_all = []
    t_all = []
    n_audio = 0

    # -------- 1) Recolectar TODOS los frames (para fit PCA/Scaler SOLO en train después)
    # Guardamos por canción para luego splittear por grupo.
    per_song = []  # list of dicts with song-level matrices

    for audio_file in AUDIO_ROOT.rglob("*.mp3"):
        rel = audio_file.relative_to(AUDIO_ROOT)    # album/track.mp3
        lab_file = LAB_ROOT / rel.with_suffix(".lab")
        if not lab_file.exists():
            print("Lab faltante:", lab_file)
            continue

        try:
            y, sr = librosa.load(audio_file, sr=SR, mono=True)
            X_cqt, times = compute_cqt_frames(y, sr=sr, hop=HOP)  # (180, T)
            chord_intervals = load_chords_lab(lab_file)

            # Label por frame (por centro de frame)
            labels = [label_for_time(t, chord_intervals) for t in times]

            # descartar frames "N" (opcional; podés dejar N si querés 25 clases completas)
            keep = [i for i,l in enumerate(labels) if l != "N"]
            if not keep:
                continue

            X_kept = X_cqt[:, keep]                 # (180, Tk)
            labels_kept = [labels[i] for i in keep]
            times_kept  = [times[i]  for i in keep]

            per_song.append({
                "rel": str(rel.with_suffix("")),
                "X180": X_kept,                     # (180, Tk)
                "labels": labels_kept,              # len Tk
                "times": times_kept
            })
            n_audio += 1
            print(f"OK ({n_audio}): {rel.name} -> {len(keep)} frames")

        except Exception as e:
            print("ERROR en", audio_file, ":", e)

    if not per_song:
        print("No hay datos.")
        return

    # -------- 2) Split por canción (train/test) para fit PCA/Scaler SOLO en train
    # Aquí, para simplificar, 75/25 por canción (igual que tu MLP baseline)
    rng = np.random.default_rng(42)
    idxs = np.arange(len(per_song))
    rng.shuffle(idxs)
    k = int(0.75 * len(per_song))
    train_idx, test_idx = idxs[:k], idxs[k:]

    # Concatenar TRAIN para fit PCA y Scaler
    X_train_concat = []
    for i in train_idx:
        X_train_concat.append(per_song[i]["X180"].T)  # (Tk,180)
    X_train_concat = np.concatenate(X_train_concat, axis=0)  # (Ntr,180)

    # PCA (retené var > 98% p.ej.)
    pca = PCA(n_components=0.98, svd_solver="full", whiten=False, random_state=0)
    X_train_pca = pca.fit_transform(X_train_concat)   # (Ntr, Dp)
    # Scaler sobre componentes PCA
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_pca_norm = scaler.fit_transform(X_train_pca)

    joblib.dump(pca, pca_out)
    joblib.dump(scaler, scaler_out)
    print(f"PCA dims: {X_train_pca.shape[1]}  (guardado en {pca_out})")

    # -------- 3) Transformar TODAS las canciones: PCA + Z-score + Splicing
    for i, item in enumerate(per_song):
        rel = item["rel"]
        X180 = item["X180"]            # (180, T)
        labs = item["labels"]
        times = item["times"]
        # Proyección PCA por frame
        Xp = pca.transform(X180.T)     # (T, Dp)
        Xn = scaler.transform(Xp)      # (T, Dp)
        # Splicing
        Xsf = make_superframes(Xn.T, ctx=SPLICE_CTX)  # (T, (2c+1)*Dp)

        # Guardar filas
        for t_idx, (feat, lab, tt) in enumerate(zip(Xsf, labs, times)):
            rows.append({
                "album_track": rel,
                "t": tt,
                "label": lab
            })
            X_all.append(feat)
            y_all.append(lab)
            g_all.append(rel)
            t_all.append(tt)

    df = pd.DataFrame(rows)
    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=object)
    g_all = np.array(g_all, dtype=object)
    t_all = np.array(t_all, dtype=np.float32)

    df.to_csv(csv_out, index=False)
    np.savez(npz_out, X=X_all, y=y_all, groups=g_all, times=t_all)
    print(f"\n✅ frames CSV: {csv_out} ({len(df)} filas)")
    print(f"✅ frames NPZ:  {npz_out}  X.shape={X_all.shape}")
    print("   (input_dim = (2*ctx+1)*PCA_dims)")
    
    # -------- 4) Class balance (frames) -> CSV + Plot
    try:
        OUT_FRAMES.mkdir(parents=True, exist_ok=True)
        counts = pd.Series(y_all).value_counts().sort_values(ascending=False)
        counts_path_csv = OUT_FRAMES / "class_balance_counts_frames.csv"
        counts.to_frame("count").to_csv(counts_path_csv)

        plt.figure(figsize=(12, 6))
        counts.plot(kind="bar", color="#4C78A8")
        plt.title("Frame-level class counts")
        plt.ylabel("Num frames")
        plt.xlabel("Chord class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        counts_path_png = OUT_FRAMES / "class_balance_counts_frames.png"
        plt.savefig(counts_path_png, dpi=150)
        plt.close()
        print(f"✅ class balance guardado: {counts_path_csv}")
        print(f"✅ class balance plot:     {counts_path_png}")
    except Exception as e:
        print("⚠️ No se pudo generar el class balance:", e)
    
if __name__ == "__main__":
    build_frame_dataset()
