# hmm_postprocessing.py
"""
HMM Post-Processing para detección de acordes.

El HMM aprende probabilidades de transición entre acordes del dataset de entrenamiento
y usa el algoritmo de Viterbi para suavizar las predicciones del MLP, haciéndolas
más coherentes musicalmente.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score


CSV = Path("dataset_chords_merged.csv")
OUT = Path("analysis_out")
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Pequeño valor para evitar log(0)
EPSILON = 1e-10


class ChordHMM:
    """
    Hidden Markov Model para transiciones de acordes.
    
    - Estados ocultos: acordes reales
    - Observaciones: probabilidades del MLP
    - Aprende matriz de transición de secuencias reales
    - Usa Viterbi para decodificar secuencia más probable
    """
    
    def __init__(self, class_names, smoothing=0.01):
        """
        Args:
            class_names: lista de nombres de acordes (en orden del encoder)
            smoothing: factor de suavizado Laplace para transiciones no vistas
        """
        self.class_names = list(class_names)
        self.n_states = len(class_names)
        self.smoothing = smoothing
        
        # Matrices del HMM (se llenan en fit)
        self.transition_matrix = None  # P(chord_t | chord_t-1)
        self.initial_probs = None      # P(chord_0)
        
        # Mapeo nombre -> índice
        self.chord_to_idx = {chord: i for i, chord in enumerate(class_names)}
        
    def fit(self, df_train):
        """
        Aprende probabilidades de transición y probabilidades iniciales
        desde secuencias de acordes en el dataset de entrenamiento.
        
        Args:
            df_train: DataFrame con columnas ['album_track', 't_start', 'label']
                     ordenado por album_track y t_start
        """
        print("📊 Aprendiendo matriz de transición del HMM...")
        
        # Inicializar contadores
        transition_counts = np.zeros((self.n_states, self.n_states))
        initial_counts = np.zeros(self.n_states)
        
        # Agrupar por canción y contar transiciones
        for track_name, track_df in df_train.groupby('album_track'):
            # Ordenar por tiempo para garantizar secuencia correcta
            track_df = track_df.sort_values('t_start')
            labels = track_df['label'].values
            
            if len(labels) == 0:
                continue
                
            # Acorde inicial de la canción
            first_idx = self.chord_to_idx[labels[0]]
            initial_counts[first_idx] += 1
            
            # Contar transiciones consecutivas
            for i in range(len(labels) - 1):
                from_chord = labels[i]
                to_chord = labels[i + 1]
                from_idx = self.chord_to_idx[from_chord]
                to_idx = self.chord_to_idx[to_chord]
                transition_counts[from_idx, to_idx] += 1
        
        # Normalizar a probabilidades con suavizado Laplace
        # P(to | from) = (count(from→to) + α) / (sum(count(from→*)) + α*N)
        transition_probs = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            row_sum = transition_counts[i].sum()
            for j in range(self.n_states):
                transition_probs[i, j] = (transition_counts[i, j] + self.smoothing) / \
                                        (row_sum + self.smoothing * self.n_states)
        
        # Probabilidades iniciales
        total_initial = initial_counts.sum()
        initial_probs = (initial_counts + self.smoothing) / \
                       (total_initial + self.smoothing * self.n_states)
        
        self.transition_matrix = transition_probs
        self.initial_probs = initial_probs
        
        # Reporte de transiciones más comunes
        print("\n🎵 Top 10 transiciones de acordes más frecuentes:")
        top_transitions = []
        for i in range(self.n_states):
            for j in range(self.n_states):
                if transition_counts[i, j] > 0:
                    top_transitions.append((
                        self.class_names[i],
                        self.class_names[j],
                        transition_counts[i, j],
                        transition_probs[i, j]
                    ))
        top_transitions.sort(key=lambda x: x[2], reverse=True)
        
        for from_chord, to_chord, count, prob in top_transitions[:10]:
            print(f"  {from_chord:8s} → {to_chord:8s}  |  {int(count):4d} veces  |  P={prob:.3f}")
        
        return self
    
    def viterbi(self, emission_probs):
        """
        Algoritmo de Viterbi para encontrar la secuencia más probable de acordes.
        
        Args:
            emission_probs: matriz (T, N) donde T es longitud de secuencia,
                           N es número de acordes. emission_probs[t, i] es
                           P(observación_t | acorde_i) según el MLP.
        
        Returns:
            best_path: array de índices de acordes (longitud T)
        """
        T = len(emission_probs)  # Longitud de la secuencia
        
        # Matrices en log-espacio para estabilidad numérica
        log_emission = np.log(emission_probs + EPSILON)
        log_transition = np.log(self.transition_matrix + EPSILON)
        log_initial = np.log(self.initial_probs + EPSILON)
        
        # DP: viterbi[t, s] = máxima log-prob de llegar al estado s en tiempo t
        viterbi = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)
        
        # Inicialización (t=0)
        viterbi[0] = log_initial + log_emission[0]
        
        # Recursión
        for t in range(1, T):
            for s in range(self.n_states):
                # Mejor estado previo que lleva a s
                trans_probs = viterbi[t-1] + log_transition[:, s]
                backpointer[t, s] = np.argmax(trans_probs)
                viterbi[t, s] = trans_probs[backpointer[t, s]] + log_emission[t, s]
        
        # Backtrack: encontrar mejor camino
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(viterbi[-1])
        
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]
        
        return best_path
    
    def predict_sequence(self, emission_probs):
        """
        Predice secuencia de acordes usando Viterbi.
        
        Args:
            emission_probs: probabilidades del MLP (T, N)
        
        Returns:
            chord_names: lista de nombres de acordes predichos
        """
        best_path = self.viterbi(emission_probs)
        return [self.class_names[idx] for idx in best_path]
    
    def save_transition_matrix(self, path):
        """Guarda la matriz de transición como imagen."""
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(self.transition_matrix, cmap='YlOrRd', vmin=0, vmax=0.3)
        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Probabilidad de transición', rotation=270, labelpad=20)
        
        ax.set_xticks(np.arange(self.n_states))
        ax.set_yticks(np.arange(self.n_states))
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Acorde siguiente")
        ax.set_ylabel("Acorde actual")
        ax.set_title("Matriz de Transición HMM\nP(acorde_siguiente | acorde_actual)")
        
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"💾 Matriz de transición guardada en: {path}")


def predict_with_hmm_by_track(df_test, mlp_probs, hmm, le):
    """
    Aplica Viterbi por cada canción (track) en el set de test.
    
    Args:
        df_test: DataFrame de test con 'album_track'
        mlp_probs: probabilidades del MLP para cada fila de df_test
        hmm: modelo ChordHMM entrenado
        le: LabelEncoder
    
    Returns:
        y_pred_hmm: predicciones suavizadas por HMM (índices)
    """
    y_pred_hmm = np.zeros(len(df_test), dtype=int)
    offset = 0
    
    for track_name, track_df in df_test.groupby('album_track'):
        track_df = track_df.sort_values('t_start')
        track_indices = track_df.index.values
        track_len = len(track_indices)
        
        # Probabilidades del MLP para esta canción
        track_probs = mlp_probs[offset:offset+track_len]
        
        # Viterbi sobre la secuencia
        best_path = hmm.viterbi(track_probs)
        
        # Guardar predicciones
        y_pred_hmm[offset:offset+track_len] = best_path
        offset += track_len
    
    return y_pred_hmm


if __name__ == "__main__":
    print("=" * 60)
    print("🎼 HMM Post-Processing para Detección de Acordes")
    print("=" * 60)
    
    # ----- Cargar datos -----
    print("\n📁 Cargando dataset...")
    df = pd.read_csv(CSV)
    
    # Asegurar que está ordenado (importante para secuencias)
    df = df.sort_values(['album_track', 't_start']).reset_index(drop=True)
    
    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    class_names = le.classes_
    groups = df['album_track'].values
    
    print(f"   Total de segmentos: {len(df)}")
    print(f"   Número de acordes: {len(class_names)}")
    print(f"   Número de canciones: {df['album_track'].nunique()}")
    
    # ----- Split train/test por canción -----
    print(f"\n✂️  Split train/test ({int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    y_test = y[test_idx]
    
    print(f"   Train: {len(df_train)} segmentos, {df_train['album_track'].nunique()} canciones")
    print(f"   Test:  {len(df_test)} segmentos, {df_test['album_track'].nunique()} canciones")
    
    # ----- Entrenar HMM -----
    hmm = ChordHMM(class_names, smoothing=0.01)
    hmm.fit(df_train)
    
    # Guardar visualización de matriz de transición
    hmm.save_transition_matrix(OUT / "hmm_transition_matrix.png")
    
    # Guardar modelo HMM para usar en inferencia
    import pickle
    with open(OUT / "hmm_model.pkl", "wb") as f:
        pickle.dump(hmm, f)
    print(f"💾 Modelo HMM guardado en: {OUT / 'hmm_model.pkl'}")
    
    # ----- Cargar predicciones del MLP baseline -----
    print("\n🤖 Cargando predicciones del MLP baseline...")
    mlp_predictions_path = OUT / "baseline_mlp_predictions.csv"
    
    if not mlp_predictions_path.exists():
        print(f"❌ ERROR: No se encontraron predicciones del MLP en {mlp_predictions_path}")
        print("   Ejecuta primero: python baseline_mlp.py")
        exit(1)
    
    mlp_preds_df = pd.read_csv(mlp_predictions_path)
    
    # Cargar modelo y obtener probabilidades (necesitamos las probs, no solo la clase predicha)
    print("   Cargando modelo MLP para obtener probabilidades...")
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import StandardScaler
    
    model = load_model(OUT / "baseline_mlp_model.h5")
    scaler_stats = np.load(OUT / "mlp_scaler_stats.npz")
    mean, scale = scaler_stats["mean"], scaler_stats["scale"]
    
    # Extraer features y estandarizar
    NOTE_COLS = [f"chroma_{n}" for n in ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]]
    X_test = df_test[NOTE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    X_test_scaled = scaler.transform(X_test)
    
    # Obtener probabilidades del MLP
    mlp_probs = model.predict(X_test_scaled, verbose=0)
    y_pred_mlp = np.argmax(mlp_probs, axis=1)
    
    # ----- Aplicar Viterbi por canción -----
    print("\n🔮 Aplicando algoritmo de Viterbi por canción...")
    df_test_sorted = df_test.sort_values(['album_track', 't_start']).reset_index(drop=True)
    mlp_probs_sorted = mlp_probs  # Ya están en orden porque df_test está ordenado
    
    y_pred_hmm_indices = []
    for track_name, track_df in df_test_sorted.groupby('album_track'):
        track_indices = track_df.index.values
        track_probs = mlp_probs_sorted[track_indices]
        
        # Aplicar Viterbi
        best_path = hmm.viterbi(track_probs)
        y_pred_hmm_indices.extend(best_path)
    
    y_pred_hmm = np.array(y_pred_hmm_indices)
    
    # Reordenar para que coincida con el índice original de test
    y_test_sorted = df_test_sorted['label'].values
    y_test_sorted_encoded = le.transform(y_test_sorted)
    
    # ----- Evaluación -----
    print("\n" + "=" * 60)
    print("📊 RESULTADOS - COMPARACIÓN MLP vs MLP+HMM")
    print("=" * 60)
    
    # Métricas MLP solo
    acc_mlp = accuracy_score(y_test_sorted_encoded, np.argmax(mlp_probs_sorted, axis=1))
    f1_mlp = f1_score(y_test_sorted_encoded, np.argmax(mlp_probs_sorted, axis=1), average='macro')
    bacc_mlp = balanced_accuracy_score(y_test_sorted_encoded, np.argmax(mlp_probs_sorted, axis=1))
    
    # Métricas MLP + HMM
    acc_hmm = accuracy_score(y_test_sorted_encoded, y_pred_hmm)
    f1_hmm = f1_score(y_test_sorted_encoded, y_pred_hmm, average='macro')
    bacc_hmm = balanced_accuracy_score(y_test_sorted_encoded, y_pred_hmm)
    
    print("\n🤖 MLP Baseline:")
    print(f"   Accuracy:          {acc_mlp:.4f}")
    print(f"   Macro F1:          {f1_mlp:.4f}")
    print(f"   Balanced Accuracy: {bacc_mlp:.4f}")
    
    print("\n✨ MLP + HMM (Viterbi):")
    print(f"   Accuracy:          {acc_hmm:.4f}  ({'+' if acc_hmm > acc_mlp else ''}{acc_hmm - acc_mlp:.4f})")
    print(f"   Macro F1:          {f1_hmm:.4f}  ({'+' if f1_hmm > f1_mlp else ''}{f1_hmm - f1_mlp:.4f})")
    print(f"   Balanced Accuracy: {bacc_hmm:.4f}  ({'+' if bacc_hmm > bacc_mlp else ''}{bacc_hmm - bacc_mlp:.4f})")
    
    # Classification report
    report_hmm = classification_report(
        le.inverse_transform(y_test_sorted_encoded),
        le.inverse_transform(y_pred_hmm),
        digits=3
    )
    
    report_text = f"""HMM Post-Processing Results
================================

MLP Baseline:
  Accuracy:          {acc_mlp:.4f}
  Macro F1:          {f1_mlp:.4f}
  Balanced Accuracy: {bacc_mlp:.4f}

MLP + HMM:
  Accuracy:          {acc_hmm:.4f}  (Δ {acc_hmm - acc_mlp:+.4f})
  Macro F1:          {f1_hmm:.4f}  (Δ {f1_hmm - f1_mlp:+.4f})
  Balanced Accuracy: {bacc_hmm:.4f}  (Δ {bacc_hmm - bacc_mlp:+.4f})

Classification Report (MLP + HMM):
{report_hmm}
"""
    
    (OUT / "hmm_results.txt").write_text(report_text)
    print(f"\n💾 Reporte guardado en: {OUT / 'hmm_results.txt'}")
    
    # Matriz de confusión
    cm = confusion_matrix(
        le.inverse_transform(y_test_sorted_encoded),
        le.inverse_transform(y_pred_hmm),
        labels=class_names
    )
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Proporción', rotation=270, labelpad=15)
    
    thresh = cm_normalized.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            if count > 0:
                text = f'{pct:.2f}\n({count})'
                ax.text(j, i, text,
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=6)
    
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Matriz de confusión — MLP + HMM (normalizada por fila)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "hmm_confusion.png", dpi=150)
    plt.close()
    
    print(f"💾 Matriz de confusión guardada en: {OUT / 'hmm_confusion.png'}")
    
    # Guardar predicciones
    pred_df = df_test_sorted.copy()
    pred_df['label_true'] = y_test_sorted
    pred_df['label_mlp'] = le.inverse_transform(np.argmax(mlp_probs_sorted, axis=1))
    pred_df['label_hmm'] = le.inverse_transform(y_pred_hmm)
    pred_df['p_mlp'] = np.max(mlp_probs_sorted, axis=1)
    
    pred_df.to_csv(OUT / "hmm_predictions.csv", index=False)
    print(f"💾 Predicciones guardadas en: {OUT / 'hmm_predictions.csv'}")
    
    print("\n✅ ¡Listo! El HMM ha mejorado la coherencia de las predicciones.")

