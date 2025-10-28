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

def load_lab(path):
    rows=[]
    with open(path,"r") as f:
        for line in f:
            p=line.strip().split()
            if len(p)>=3:
                rows.append((float(p[0]), float(p[1]), norm_label(p[2])))
    rows.sort(key=lambda x:x[0])
    return rows  # [(t0,t1,label_norm),...]

def label_at_time(t, intervals):
    # devuelve label norm para instante t
    for t0,t1,lab in intervals:
        if t>=t0 and t<t1:
            return lab
    return "N"

def cqt_frames(y, sr=SR, hop=HOP):
    C = librosa.cqt(y=y, sr=sr, hop_length=hop, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCT)
    # Usar log1p como en build_frames_dataset.py para consistencia
    # con el entrenamiento del PCA
    C_mag = np.abs(C)
    X = np.log1p(C_mag).astype(np.float32)   # (bins, T)
    
    # Verificar valores finitos (evitar NaN/inf que causan las warnings)
    X = np.nan_to_num(X, nan=0.0, posinf=20.0, neginf=0.0)
    
    times = librosa.frames_to_time(np.arange(X.shape[1]), sr=sr, hop_length=hop)
    return X, times

def splice(Xpca_norm_2d, ctx=CTX):
    # Xpca_norm_2d: (feat, T) -> (T, (2ctx+1)*feat)
    feat, T = Xpca_norm_2d.shape
    pad = np.pad(Xpca_norm_2d, ((0,0),(ctx,ctx)), mode="edge")
    out=[]
    for t in range(T):
        sl = pad[:, t:t+2*ctx+1]  # (feat, 2ctx+1)
        out.append(sl.T.reshape(-1))
    return np.stack(out, axis=0)

# --- HMM / Viterbi simple (log-domain), con sesgo fuerte a permanencia ---
def viterbi_log(pi_log, A_log, B_log):
    # pi_log: (K,), A_log: (K,K), B_log: (T,K) -> devuelve estados (T,)
    T, K = B_log.shape
    dp = np.full((T, K), -np.inf, dtype=np.float32)
    back = np.zeros((T, K), dtype=np.int32)

    dp[0] = pi_log + B_log[0]
    for t in range(1, T):
        # para cada estado j, max_i dp[t-1,i] + A[i,j]
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
    # Matriz de transición con fuerte prob. de permanencia
    K = num_classes
    A = np.full((K,K), (1.0 - stay_prob)/(K-1), dtype=np.float64)
    np.fill_diagonal(A, stay_prob)
    pi = np.full(K, 1.0/K, dtype=np.float64)
    return np.log(pi), np.log(A)

# --- Merge intervalos consecutivos iguales ---
def merge_consecutive_labels(times, labels):
    """
    Agrupa frames consecutivos con el mismo label
    Devuelve: [(t0, t1, label), ...]
    """
    if len(times) == 0:
        return []
    intervals = []
    t0 = times[0]
    label0 = labels[0]
    for i in range(1, len(times)):
        if labels[i] != label0:
            # fin del segmento actual
            t1 = times[i-1] + (HOP / SR)  # aprox fin del frame anterior
            intervals.append((t0, t1, label0))
            t0 = times[i]
            label0 = labels[i]
    # último segmento
    t1 = times[-1] + (HOP / SR)
    intervals.append((t0, t1, label0))
    return intervals

# --- WCSR (time-weighted accuracy) ---
def wcsr_from_framewise(ref_intervals, ref_times, est_labels, exclude_N=True):
    """
    ref_intervals: [(t0,t1,label_norm),...]
    ref_times: (T,) centro de cada frame
    est_labels: (T,) labels predichas para cada frame
    """
    # duración por frame ~ HOP/SR (constante)
    dt = HOP / SR
    t_ok = 0.0
    t_tot = 0.0
    for ti, lab_hat in zip(ref_times, est_labels):
        lab_ref = label_at_time(ti, ref_intervals)
        if exclude_N and (lab_ref == "N"):
            continue
        t_tot += dt
        if lab_hat == lab_ref:
            t_ok += dt
    return (t_ok / t_tot) if t_tot > 0 else 0.0

# --- INFERENCIA + WCSR PARA UNA CANCIÓN ---
def infer_and_wcsr(audio_path, lab_path, pca_path, scaler_path, model_path, labels_txt, use_hmm=True):
    # 1) cargar audio y .lab
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    ref = load_lab(lab_path)

    # 2) features por frame (CQT log)
    X180, times = cqt_frames(y)

    # 3) PCA + Z-score (aplicados a CADA frame; PCA entrenado en TRAIN)
    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)
    Xp = pca.transform(X180.T)         # (T, Dp)
    Xn = scaler.transform(Xp)          # (T, Dp)

    # 4) splicing (t-1,t,t+1) SOBRE los componentes normalizados
    Xsf = splice(Xn.T, ctx=CTX)        # (T, F')

    # 5) predicciones MLP
    model = load_model(model_path)
    P = model.predict(Xsf, verbose=0)  # (T, K) posteriors
    classes = np.loadtxt(labels_txt, dtype=str)  # orden del softmax

    # 6) (opcional) HMM para suavizar
    if use_hmm:
        pi_log, A_log = build_hmm(num_classes=P.shape[1], stay_prob=0.995)
        # B_log = log likelihood por estado (usamos posterior como proxy)
        eps = 1e-8
        B_log = np.log(np.clip(P, eps, 1.0))
        z = viterbi_log(pi_log, A_log, B_log)        # índices
        labels_est = classes[z]
    else:
        labels_est = classes[np.argmax(P, axis=1)]

    # 7) WCSR
    score = wcsr_from_framewise(ref, times, labels_est, exclude_N=True)
    return score, times, labels_est

# Mapeo de canciones en wcsr_test a archivos .lab
SONG_MAPPINGS = {
    "For you blue.mp3": "12_-_Let_It_Be/11_-_For_You_Blue.lab",
    "Love Me Do.mp3": "01_-_Please_Please_Me/08_-_Love_Me_Do.lab",
    "Misery.mp3": "01_-_Please_Please_Me/02_-_Misery.lab",
    "Please Please Me.mp3": "01_-_Please_Please_Me/07_-_Please_Please_Me.lab",
}

def evaluate_all_songs(wcsr_test_dir="wcsr_test", output_dir="outputs"):
    """Evalúa todas las canciones en wcsr_test y guarda predicciones."""
    wcsr_path = Path(wcsr_test_dir)
    chordlab_path = Path("The Beatles Annotations/chordlab/The Beatles")
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Archivos del modelo
    PCA = Path("pca.joblib")
    SCAL = Path("scaler.joblib")
    MODEL = Path("analysis_out_frames/dnn_bottleneck.h5")
    MAP = Path("analysis_out_frames/label_mapping.txt")
    
    results = []
    total_wcsr = 0.0
    n_songs = 0
    
    # Procesar cada canción
    audio_files = sorted(wcsr_path.glob("*.mp3"))
    print(f"\n🎵 Evaluando {len(audio_files)} canciones en wcsr_test...\n")
    
    for audio_file in audio_files:
        audio_name = audio_file.name
        print(f"▶ {audio_name}")
        
        if audio_name not in SONG_MAPPINGS:
            print(f"  ⚠️  Sin mapeo para: {audio_name}")
            continue
        
        lab_relative = SONG_MAPPINGS[audio_name]
        lab_path = chordlab_path / lab_relative
        
        if not lab_path.exists():
            print(f"  ❌ .lab no existe: {lab_path}")
            continue
        
        try:
            # Inferencia
            w, t, L = infer_and_wcsr(audio_file, lab_path, PCA, SCAL, MODEL, MAP, use_hmm=True)
            total_wcsr += w
            n_songs += 1
            
            results.append({
                'song': audio_name,
                'wcsr': w
            })
            
            print(f"  WCSR: {w:.4f} ({w*100:.2f}%)")
            
            # Guardar archivo .lab
            intervals = merge_consecutive_labels(t, L)
            output_file = out_dir / f"{audio_file.stem}_predicted.lab"
            
            with open(output_file, "w") as f:
                for t0, t1, lab in intervals:
                    f.write(f"{t0:.6f} {t1:.6f} {lab}\n")
            
            print(f"  ✅ Guardado: {output_file}\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Métricas globales
    if n_songs > 0:
        global_wcsr = total_wcsr / n_songs
        print(f"\n{'='*60}")
        print(f"📊 RESULTADOS GLOBALES")
        print(f"{'='*60}")
        print(f"\nCanciones evaluadas: {n_songs}")
        print(f"WCSR Promedio: {global_wcsr:.4f} ({global_wcsr*100:.2f}%)")
        print(f"\nResultados por canción:")
        for i, r in enumerate(sorted(results, key=lambda x: x['wcsr'], reverse=True), 1):
            print(f"  {i}. {r['song']:30s}  WCSR={r['wcsr']:.4f} ({r['wcsr']*100:.2f}%)")
        print(f"\n{'='*60}\n")
    
    return results

# NOTA: Este archivo ha sido reemplazado por:
# - inference_frames.py: para hacer inferencia y guardar .lab
# - evaluate_wcsr_frames.py: para evaluar métricas WCSR
#
# Para mantener compatibilidad hacia atrás, este archivo redirige
# a las nuevas funciones.

if __name__ == "__main__":
    import sys
    
    print("⚠️  Este archivo ha sido reemplazado.")
    print("\nUsa:")
    print("  python3 inference_frames.py <audio.mp3>     # Inferir")
    print("  python3 evaluate_wcsr_frames.py               # Evaluar métricas")
    print()
