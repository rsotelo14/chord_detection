import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from pathlib import Path
from typing import List, Tuple, Union, Optional
import pickle

import numpy as np
import pandas as pd
import librosa

# ---- Config / rutas por defecto ----
ANALYSIS_OUT = Path("analysis_out")
OUTPUTS_DIR = Path("outputs")

# Por defecto usamos regresión logística (cambiar aquí para usar MLP)
MODEL_TYPE = "mlp"  # "logreg" o "mlp"

if MODEL_TYPE == "logreg":
    DEFAULT_MODEL = ANALYSIS_OUT / "baseline_logreg_model.pkl"
    DEFAULT_LABELS = ANALYSIS_OUT / "logreg_label_mapping.txt"
    DEFAULT_SCALER_STATS = None  # El pipeline de sklearn ya incluye el scaler
else:  # mlp
    DEFAULT_MODEL = ANALYSIS_OUT / "baseline_mlp_model.h5"
    DEFAULT_LABELS = ANALYSIS_OUT / "mlp_label_mapping.txt"
    DEFAULT_SCALER_STATS = ANALYSIS_OUT / "mlp_scaler_stats.npz"

SR = 22050
HOP = 512

NOTE_ORDER = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]

# HMM por defecto
DEFAULT_HMM = ANALYSIS_OUT / "hmm_model.npz"
EPSILON = 1e-10


class ChordHMM:
    """
    Hidden Markov Model para transiciones de acordes.
    Usa Viterbi para suavizar predicciones del MLP.
    """
    
    def __init__(self, class_names, transition_matrix, initial_probs):
        self.class_names = list(class_names)
        self.n_states = len(class_names)
        self.transition_matrix = transition_matrix
        self.initial_probs = initial_probs
        self.chord_to_idx = {chord: i for i, chord in enumerate(class_names)}
    
    def viterbi(self, emission_probs, transition_weight=1.0):
        """
        Algoritmo de Viterbi para encontrar la secuencia más probable.
        
        Args:
            emission_probs: matriz (T, N) de probabilidades del MLP
            transition_weight: peso de las transiciones (0.0=solo MLP, 1.0=balanceado, >1.0=favorece transiciones)
                              Valores recomendados: 0.1-0.5 para dar más peso al audio
        
        Returns:
            best_path: array de índices de acordes
        """
        T = len(emission_probs)
        
        # Log-espacio para estabilidad
        log_emission = np.log(emission_probs + EPSILON)
        log_transition = np.log(self.transition_matrix + EPSILON)
        log_initial = np.log(self.initial_probs + EPSILON)
        
        # Aplicar peso a las transiciones
        log_transition_weighted = log_transition * transition_weight
        
        # DP
        viterbi = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)
        
        # Inicialización
        viterbi[0] = log_initial + log_emission[0]
        
        # Recursión
        for t in range(1, T):
            for s in range(self.n_states):
                trans_probs = viterbi[t-1] + log_transition_weighted[:, s]
                backpointer[t, s] = np.argmax(trans_probs)
                viterbi[t, s] = trans_probs[backpointer[t, s]] + log_emission[t, s]
        
        # Backtrack
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(viterbi[-1])
        
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]
        
        return best_path


def load_hmm(hmm_path: Path) -> Optional[ChordHMM]:
    """Carga el modelo HMM pre-entrenado desde archivo npz."""
    if not hmm_path.exists():
        return None
    
    # Cargar matrices desde npz
    data = np.load(hmm_path, allow_pickle=True)
    class_names = data['class_names']
    transition_matrix = data['transition_matrix']
    initial_probs = data['initial_probs']
    
    # Crear objeto HMM con las matrices cargadas
    hmm = ChordHMM(class_names, transition_matrix, initial_probs)
    
    return hmm


def load_model_auto(model_path: Path) -> Union[object, object]:
    """
    Carga un modelo automáticamente detectando si es sklearn (.pkl) o keras (.h5)
    """
    if model_path.suffix == ".pkl":
        import joblib
        return joblib.load(model_path), "sklearn"
    elif model_path.suffix == ".h5":
        from tensorflow.keras.models import load_model as keras_load_model
        return keras_load_model(model_path), "keras"
    else:
        raise ValueError(f"Formato de modelo no soportado: {model_path.suffix}")


def load_scaler_stats(scaler_npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(scaler_npz_path)
    return data["mean"], data["scale"]


def standardize_features(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # Evitar división por cero
    safe_scale = np.where(scale == 0, 1.0, scale)
    return (X - mean) / safe_scale


def compute_chroma(y: np.ndarray, sr: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
    y_harm = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)
    times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)
    return chroma, times


def segment_beats(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    # Construir intervalos [t_i, t_{i+1}) y uno final hasta el fin del audio
    if len(beat_times) == 0:
        return np.array([0.0]), np.array([librosa.get_duration(y=y, sr=sr)])
    starts = beat_times
    ends = np.concatenate([beat_times[1:], [librosa.get_duration(y=y, sr=sr)]])
    return starts, ends


def aggregate_chroma_over_interval(chroma: np.ndarray, times: np.ndarray, t0: float, t1: float) -> np.ndarray:
    mask = (times >= t0) & (times < t1)
    if not np.any(mask):
        # Si no hay frames en el intervalo, devolver vector nulo
        return np.zeros((chroma.shape[0],), dtype=np.float32)
    vec = np.median(chroma[:, mask], axis=1)
    s = np.sum(vec)
    if s > 1e-8:
        vec = vec / s
    return vec.astype(np.float32)


def infer_on_audio(
    audio_path: Path,
    model_path: Path = DEFAULT_MODEL,
    labels_path: Path = DEFAULT_LABELS,
    scaler_stats_path: Path = DEFAULT_SCALER_STATS,
    sr: int = SR,
    hop_length: int = HOP,
    beats_per_segment: int = 4,
    use_hmm: bool = False,
    hmm_path: Path = DEFAULT_HMM,
    transition_weight: float = 0.3,
) -> pd.DataFrame:
    # Cargar recursos
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Archivo de etiquetas no encontrado: {labels_path}")

    # Cargar modelo (auto-detecta tipo)
    model, model_type = load_model_auto(model_path)
    class_names = np.loadtxt(labels_path, dtype=str)
    
    # Cargar HMM si está habilitado
    hmm = None
    if use_hmm:
        hmm = load_hmm(hmm_path)
        if hmm is None:
            print(f"⚠️  HMM no encontrado en {hmm_path}, continuando sin HMM")
            use_hmm = False
        else:
            print(f"✨ Usando HMM (peso transiciones={transition_weight})")
    
    # Solo cargar scaler para modelos Keras/MLP
    if model_type == "keras":
        if scaler_stats_path is None or not scaler_stats_path.exists():
            raise FileNotFoundError(f"Scaler stats no encontrado: {scaler_stats_path}")
        mean, scale = load_scaler_stats(scaler_stats_path)

    # Audio -> cromas
    y, _sr = librosa.load(audio_path, sr=sr, mono=True)
    chroma, times = compute_chroma(y, sr=sr, hop_length=hop_length)

    # Beats -> intervalos
    starts, ends = segment_beats(y, sr=sr)

    # Agregar cromas cada N beats (por defecto 4)
    feats = []
    rows = []
    n_beats = len(starts)
    if n_beats == 0:
        return pd.DataFrame(columns=["t_start","t_end","label_pred","p_pred"])  # vacío

    i = 0
    while i < n_beats:
        j = min(i + beats_per_segment - 1, n_beats - 1)
        t0 = float(starts[i])
        t1 = float(ends[j])
        vec = aggregate_chroma_over_interval(chroma, times, t0, t1)
        rows.append({"t_start": t0, "t_end": t1, **{f"chroma_{n}": float(vec[k]) for k, n in enumerate(NOTE_ORDER)}})
        feats.append(vec)
        i += beats_per_segment

    if len(feats) == 0:
        return pd.DataFrame(columns=["t_start","t_end","label_pred","p_pred"])  # vacío

    X = np.stack(feats, axis=0)

    # Inferencia según tipo de modelo
    if model_type == "sklearn":
        # El pipeline de sklearn ya incluye el scaler
        prob = model.predict_proba(X)
        if use_hmm:
            # Aplicar Viterbi con peso ajustable
            best_path = hmm.viterbi(prob, transition_weight=transition_weight)
            y_pred = class_names[best_path]
        else:
            y_pred = model.predict(X)
        p_pred = np.max(prob, axis=1)
    else:  # keras
        # Estandarizar con medias y std guardados
        X_std = standardize_features(X, mean=mean, scale=scale)
        prob = model.predict(X_std, verbose=0)
        
        if use_hmm:
            # Aplicar Viterbi con peso ajustable
            best_path = hmm.viterbi(prob, transition_weight=transition_weight)
            y_pred = class_names[best_path]
        else:
            y_pred_idx = np.argmax(prob, axis=1)
            y_pred = class_names[y_pred_idx]
        
        p_pred = np.max(prob, axis=1)

    df_out = pd.DataFrame(rows)
    df_out["label_pred"] = y_pred
    df_out["p_pred"] = p_pred
    return df_out


def merge_consecutive_same_label(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    df_sorted = df.sort_values("t_start")
    merged = []
    cur_start = float(df_sorted.iloc[0]["t_start"]) 
    cur_end = float(df_sorted.iloc[0]["t_end"]) 
    cur_label = str(df_sorted.iloc[0]["label_pred"]) 
    for _, row in df_sorted.iloc[1:].iterrows():
        label = str(row["label_pred"]) 
        if label == cur_label:
            cur_end = float(row["t_end"]) 
        else:
            merged.append((cur_start, cur_end, cur_label))
            cur_start = float(row["t_start"]) 
            cur_end = float(row["t_end"]) 
            cur_label = label
    merged.append((cur_start, cur_end, cur_label))
    return merged


def save_outputs(df: pd.DataFrame, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_lab = out_prefix.with_suffix(".lab")
    merged = merge_consecutive_same_label(df[["t_start","t_end","label_pred"]])
    with open(out_lab, "w") as f:
        for t0, t1, lab in merged:
            f.write(f"{t0:.6f}\t{t1:.6f}\t{lab}\n")


def main():
    parser = argparse.ArgumentParser(description="Inferencia de acordes agrupando beats usando MLP guardado")
    parser.add_argument("audio", type=str, help="Ruta al archivo de audio (.mp3/.wav)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Ruta al modelo .h5")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS), help="Ruta a mlp_label_mapping.txt")
    parser.add_argument("--scaler", type=str, default=str(DEFAULT_SCALER_STATS), help="Ruta a mlp_scaler_stats.npz")
    parser.add_argument("--sr", type=int, default=SR, help="Sample rate para carga de audio")
    parser.add_argument("--hop", type=int, default=HOP, help="Hop length para cromas")
    parser.add_argument("--beats-per-segment", type=int, default=4, help="Número de beats por segmento (p.ej., 4)")
    parser.add_argument("--use-hmm", action="store_true", help="Usar HMM para suavizar predicciones (mejora coherencia)")
    parser.add_argument("--hmm", type=str, default=str(DEFAULT_HMM), help="Ruta al modelo HMM .pkl")
    parser.add_argument("--transition-weight", type=float, default=0.3, help="Peso de transiciones HMM (0.0=solo audio, 1.0=balanceado, default=0.3)")
    parser.add_argument("--out", type=str, default=None, help="Prefijo de salida (sin extensión). Por defecto usa outputs/<audio_stem>")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model)
    labels_path = Path(args.labels)
    scaler_path = Path(args.scaler)
    hmm_path = Path(args.hmm)

    df_pred = infer_on_audio(
        audio_path=audio_path,
        model_path=model_path,
        labels_path=labels_path,
        scaler_stats_path=scaler_path,
        sr=args.sr,
        hop_length=args.hop,
        beats_per_segment=args.beats_per_segment,
        use_hmm=args.use_hmm,
        hmm_path=hmm_path,
        transition_weight=args.transition_weight,
    )

    if args.out is None:
        out_prefix = OUTPUTS_DIR / audio_path.stem
    else:
        out_prefix = Path(args.out)

    save_outputs(df_pred, out_prefix=out_prefix)
    print(f"✅ Guardado: {out_prefix.with_suffix('.lab')}")


if __name__ == "__main__":
    main()


