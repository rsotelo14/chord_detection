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

def compute_beat_intervals(y, sr, frame_times, frame_labels, classes, group_size=1):
    """Crea intervalos a nivel beat usando majority voting de labels por frame.

    Si no se detectan beats suficientes, devuelve None para indicar fallback.
    """
    # Detectar beats en frames de la misma rejilla de hop para consistencia
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    if beat_frames is None or len(beat_frames) < 2:
        return None

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    audio_end = len(y) / sr

    # Asegurar cubrir [0, audio_end]
    bt = beat_times
    if bt[0] > 0.0:
        bt = np.concatenate([[0.0], bt])
    if bt[-1] < audio_end:
        bt = np.concatenate([bt, [audio_end]])

    # Mapear labels a índices para votar
    class_to_idx = {c: i for i, c in enumerate(classes)}
    label_indices = np.array([class_to_idx.get(l, -1) for l in frame_labels], dtype=np.int32)

    # Asegurar tamaño de grupo válido
    if group_size is None or group_size < 1:
        group_size = 1

    intervals = []
    prev_idx = None
    # Iterar en grupos de beats consecutivos
    for i in range(0, len(bt) - 1, group_size):
        t0 = bt[i]
        t1 = bt[min(i + group_size, len(bt) - 1)]
        mask = (frame_times >= t0) & (frame_times < t1)
        inds = np.where(mask)[0]
        if inds.size == 0:
            # Si no hay frames dentro del beat, extender el anterior si existe
            if intervals:
                # Extender último segmento hasta t1
                last_t0, _, last_lab = intervals[-1]
                intervals[-1] = (last_t0, t1, last_lab)
            else:
                # Crear silencio/None; lo omitimos
                pass
            continue
        vote_indices = label_indices[inds]
        vote_indices = vote_indices[vote_indices >= 0]
        if vote_indices.size == 0:
            # Ningún índice válido; omitir
            continue
        counts = np.bincount(vote_indices, minlength=len(classes))
        winner_idx = int(np.argmax(counts))
        winner_lab = classes[winner_idx]

        if intervals and winner_idx == prev_idx:
            # Unir con el anterior
            last_t0, _, last_lab = intervals[-1]
            intervals[-1] = (last_t0, t1, last_lab)
        else:
            intervals.append((t0, t1, winner_lab))
            prev_idx = winner_idx

    return intervals

def infer_on_audio(audio_path, pca_path, scaler_path, model_path, labels_txt, use_hmm=True, beat_sync=False, beat_group=1):
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

    # 7) Generar intervalos
    if beat_sync:
        beat_intervals = compute_beat_intervals(y, SR, times, labels_est, classes, group_size=beat_group)
        if beat_intervals is not None and len(beat_intervals) > 0:
            intervals = beat_intervals
        else:
            intervals = merge_consecutive_labels(times, labels_est)
    else:
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
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('audio_path', type=str)
        parser.add_argument('--smooth', action='store_true', help='Usar HMM para suavizar predicciones')
        parser.add_argument('--beat-sync', action='store_true', help='Alinear a beats y votar mayoritario por beat/grupo')
        parser.add_argument('--beat-group', type=int, default=2, help='Cantidad de beats por voto (default: 4)')
        args = parser.parse_args()

        audio_path = Path(args.audio_path)
        use_hmm = args.smooth if args.smooth else True  # HMM activado por defecto
        use_beat_sync = bool(args.beat_sync)
        beat_group = int(args.beat_group)

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        print(f"Inferencia en: {audio_path}")
        times, labels, intervals = infer_on_audio(
            audio_path, PCA, SCAL, MODEL, MAP,
            use_hmm=use_hmm,
            beat_sync=use_beat_sync,
            beat_group=beat_group
        )

        output_file = output_dir / f"{audio_path.stem}_predicted.lab"
        save_lab_file(intervals, output_file)

        print(f"✅ Archivo .lab guardado en: {output_file}")
    else:
        # Ejemplo de uso por defecto
        print("Uso: python3 inference_frames.py <ruta_al_audio>")
        print("\nEjemplo:")
        print("  python3 inference_frames.py test_audios/cancion.mp3")

