# inference_frames.py
"""
Script para hacer inferencia sobre archivos de audio y guardar predicciones en formato .lab
"""

import numpy as np
import librosa
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model

# --- Config consistente con build_frames_dataset.py ---
SR = 11025
HOP = 512
BINS_PER_OCT = 36
N_BINS = 180
FMIN = 110.0
CTX = 1  # splicing t-1,t,t+1

NOTES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
ENH = {"C#":"Db","D#":"Eb","F#":"Gb","G#":"Ab","A#":"Bb"}

def norm_label(lab: str) -> str:
    if lab.upper() == "N": return "N"
    parts = lab.split(":")
    root = ENH.get(parts[0], parts[0]).title()
    qual = "min" if (len(parts)>1 and "min" in parts[1]) else "maj"
    return f"{root}:{qual}" if root in NOTES else "N"

def cqt_frames(y, sr=SR, hop=HOP):
    """Calcula CQT en frames."""
    C = librosa.cqt(y=y, sr=sr, hop_length=hop, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCT)
    # Usar log1p como en build_frames_dataset.py para consistencia
    C_mag = np.abs(C)
    X = np.log1p(C_mag).astype(np.float32)   # (bins, T)
    
    # Verificar valores finitos (evitar NaN/inf que causan las warnings)
    X = np.nan_to_num(X, nan=0.0, posinf=20.0, neginf=0.0)
    
    times = librosa.frames_to_time(np.arange(X.shape[1]), sr=sr, hop_length=hop)
    return X, times

def splice(Xpca_norm_2d, ctx=CTX):
    """Concatena frames adyacentes: (feat, T) -> (T, (2ctx+1)*feat)"""
    feat, T = Xpca_norm_2d.shape
    pad = np.pad(Xpca_norm_2d, ((0,0),(ctx,ctx)), mode="edge")
    out=[]
    for t in range(T):
        sl = pad[:, t:t+2*ctx+1]  # (feat, 2ctx+1)
        out.append(sl.T.reshape(-1))
    return np.stack(out, axis=0)

def viterbi_log(pi_log, A_log, B_log):
    """Viterbi en espacio logarítmico."""
    T, K = B_log.shape
    dp = np.full((T, K), -np.inf, dtype=np.float32)
    back = np.zeros((T, K), dtype=np.int32)

    dp[0] = pi_log + B_log[0]
    for t in range(1, T):
        prev = dp[t-1][:,None] + A_log  # (K,K)
        back[t] = np.argmax(prev, axis=0)
        dp[t] = np.max(prev, axis=0) + B_log[t]
    
    # backtrace
    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(dp[-1])
    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]
    return path

def build_hmm(num_classes, stay_prob=0.995):
    """Construye HMM con alta probabilidad de permanecer en el mismo estado."""
    K = num_classes
    A = np.full((K,K), (1.0 - stay_prob)/(K-1), dtype=np.float64)
    np.fill_diagonal(A, stay_prob)
    pi = np.full(K, 1.0/K, dtype=np.float64)
    return np.log(pi), np.log(A)

def merge_consecutive_labels(times, labels):
    """Agrupa frames consecutivos con el mismo label. Devuelve [(t0, t1, label), ...]"""
    if len(times) == 0:
        return []
    intervals = []
    t0 = times[0]
    label0 = labels[0]
    for i in range(1, len(times)):
        if labels[i] != label0:
            t1 = times[i-1] + (HOP / SR)
            intervals.append((t0, t1, label0))
            t0 = times[i]
            label0 = labels[i]
    # último segmento
    t1 = times[-1] + (HOP / SR)
    intervals.append((t0, t1, label0))
    return intervals

def infer_on_audio(audio_path, pca_path, scaler_path, model_path, labels_txt, use_hmm=True):
    """
    Realiza inferencia sobre un archivo de audio.
    
    Returns:
        times: (T,) tiempos de cada frame
        labels: (T,) labels predichas
        intervals: [(t0, t1, label), ...] intervalos fusionados
    """
    # 1) Cargar audio
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    # 2) Features por frame (CQT log)
    X180, times = cqt_frames(y)

    # 3) PCA + Z-score
    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)
    Xp = pca.transform(X180.T)         # (T, Dp)
    Xn = scaler.transform(Xp)          # (T, Dp)

    # 4) Splicing (t-1,t,t+1)
    Xsf = splice(Xn.T, ctx=CTX)        # (T, F')

    # 5) Predicciones MLP
    model = load_model(model_path)
    P = model.predict(Xsf, verbose=0)  # (T, K) posteriors
    classes = np.loadtxt(labels_txt, dtype=str)

    # 6) HMM para suavizar (opcional)
    if use_hmm:
        pi_log, A_log = build_hmm(num_classes=P.shape[1], stay_prob=0.995)
        eps = 1e-8
        B_log = np.log(np.clip(P, eps, 1.0))
        z = viterbi_log(pi_log, A_log, B_log)
        labels_est = classes[z]
    else:
        labels_est = classes[np.argmax(P, axis=1)]

    # 7) Fusionar intervalos consecutivos
    intervals = merge_consecutive_labels(times, labels_est)
    
    return times, labels_est, intervals

def save_lab_file(intervals, output_path):
    """Guarda intervalos en formato .lab."""
    with open(output_path, "w") as f:
        for t0, t1, lab in intervals:
            f.write(f"{t0:.6f} {t1:.6f} {lab}\n")

if __name__ == "__main__":
    import sys
    
    # Rutas de archivos del modelo
    PCA = Path("pca.joblib")
    SCAL = Path("scaler.joblib")
    MODEL = Path("analysis_out_frames/dnn_bottleneck.h5")
    MAP = Path("analysis_out_frames/label_mapping.txt")
    
    if len(sys.argv) > 1:
        # Modo comando: inferir en audio proporcionado
        audio_path = Path(sys.argv[1])
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        print(f"Inferencia en: {audio_path}")
        times, labels, intervals = infer_on_audio(audio_path, PCA, SCAL, MODEL, MAP, use_hmm=True)
        
        output_file = output_dir / f"{audio_path.stem}_predicted.lab"
        save_lab_file(intervals, output_file)
        
        print(f"✅ Archivo .lab guardado en: {output_file}")
    else:
        # Ejemplo de uso por defecto
        print("Uso: python3 inference_frames.py <ruta_al_audio>")
        print("\nEjemplo:")
        print("  python3 inference_frames.py test_audios/cancion.mp3")

