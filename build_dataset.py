import numpy as np
import librosa
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ---- Config ----
AUDIO_PATH = Path("The Beatles Annotations/audio/The Beatles/12_-_Let_It_Be/06_-_Let_It_Be.mp3")
CHORDS_LAB = Path("The Beatles Annotations/chordlab/The Beatles/12_-_Let_It_Be/06_-_Let_It_Be.lab")
SR = 22050
HOP = 512

NOTES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
ENH = {"C#":"Db","D#":"Eb","F#":"Gb","G#":"Ab","A#":"Bb"}

# ---- Normalización de etiquetas ----
def norm_label(lab: str) -> str:
    """
    Normaliza etiquetas del .lab a 25 clases:
      - 'Root:maj' o 'Root:min'
      - 'N' para no-chord
    """
    if lab.upper() == "N":
        return "N"

    parts = lab.split(":")
    root = parts[0]
    rest = parts[1] if len(parts) > 1 else "maj"

    # convertir enarmónicos a bemoles
    root = ENH.get(root, root).title()

    # decidir calidad
    if "min" in rest:
        qual = "min"
    else:
        qual = "maj"

    if root not in NOTES:
        return "N"
    return f"{root}:{qual}"

# ---- Lectura de anotaciones ----
def load_chords_lab(path_lab):
    rows = []
    with open(path_lab, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start, end, lab = float(parts[0]), float(parts[1]), parts[2]
                rows.append((start, end, lab))
    rows.sort(key=lambda x: x[0])
    return rows  # [(t0, t1, label), ...]

# ---- Audio + features (croma "sano") ----
def compute_chroma_dense(y, sr=SR, hop_length=HOP):
    y_harm = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)  # normalización por columna
    times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)
    return chroma, times  # (12, T), (T,)

# ---- Agregación por intervalo de acorde ----
def aggregate_chroma_per_chord(chroma, times_frame, chord_intervals, min_dur=0.25):
    X, y, intervals = [], [], []
    for (t0, t1, lab) in chord_intervals:
        if (t1 - t0) < min_dur:
            continue
        mask = (times_frame >= t0) & (times_frame < t1)
        if not np.any(mask):
            continue
        vec = np.median(chroma[:, mask], axis=1)
        s = np.sum(vec)
        if s > 1e-8:
            vec = vec / s
        X.append(vec)
        y.append(norm_label(lab))     # etiquetas ya normalizadas
        intervals.append((t0, t1))    # timestamps del acorde
    X = np.stack(X) if X else np.zeros((0,12))
    return X, y, intervals

# --------- GENERADOR DE DATASET (.csv) ----------
def build_dataset_csv(csv_out="dataset_chords.csv"):
    base = Path("The Beatles Annotations")
    labs_root = base / "chordlab" / "The Beatles"
    audios_root = base / "audio" / "The Beatles"

    rows = []
    n_ok, n_skip = 0, 0

    for audio_file in audios_root.rglob("*.mp3"):
        # ej: "12_-_Let_It_Be/06_-_Let_It_Be.mp3"
        rel = audio_file.relative_to(audios_root)
        lab_file = labs_root / rel.with_suffix(".lab")

        if not lab_file.exists():
            print("Lab faltante para", audio_file)
            n_skip += 1
            continue

        try:
            # cargar audio + lab
            y, sr = librosa.load(audio_file, sr=SR, mono=True)
            chord_intervals = load_chords_lab(lab_file)

            # features densas
            chroma, times_frame = compute_chroma_dense(y, sr=sr, hop_length=HOP)

            # agregación por acorde
            X_chords, y_chords, chord_times = aggregate_chroma_per_chord(
                chroma, times_frame, chord_intervals, min_dur=0.25
            )

            # armar filas para el CSV
            for vec, lab, (t0, t1) in zip(X_chords, y_chords, chord_times):
                row = {
                    "album_track": str(rel.with_suffix("")),  # ej: 12_-_Let_It_Be/06_-_Let_It_Be
                    "t_start": t0,
                    "t_end": t1,
                    "label": lab,
                }
                # agregar 12 columnas de cromas
                for i, note in enumerate(NOTES):
                    row[f"chroma_{note}"] = float(vec[i])
                rows.append(row)

            n_ok += 1
            print(f"OK ({n_ok}):", audio_file.name, "->", len(y_chords), "segmentos")

        except Exception as e:
            print("ERROR en", audio_file, ":", e)
            n_skip += 1

    # guardar CSV
    if rows:
        df = pd.DataFrame(rows)
        
        # Filtrar clase "N" (sin acorde)
        n_before = len(df)
        df = df[df['label'] != 'N'].copy()
        n_after = len(df)
        n_filtered = n_before - n_after
        
        df.to_csv(csv_out, index=False)
        print(f"\n✅ Dataset guardado en: {csv_out}  ({len(df)} filas, {df['label'].nunique()} clases)")
        if n_filtered > 0:
            print(f"   🗑️  Filtrados {n_filtered} segmentos con label 'N' (sin acorde)")
    else:
        print("\n⚠️ No se generaron filas. Revisá que existan pares audio/lab consistentes.")

# --------- RUN DEMO (una canción) + DATASET ----------
if __name__ == "__main__":
    # --- Demo con una canción (tu bloque original) ---
    y, sr = librosa.load(AUDIO_PATH, sr=SR, mono=True)
    chord_intervals = load_chords_lab(CHORDS_LAB)
    chroma, times_frame = compute_chroma_dense(y, sr=sr, hop_length=HOP)
    X_chords, y_chords, chord_times = aggregate_chroma_per_chord(chroma, times_frame, chord_intervals, min_dur=0.25)

    print("X_chords shape:", X_chords.shape)
    print("y_chords length:", len(y_chords))
    print("Ejemplo etiquetas normalizadas:", y_chords[:10])

    idx = 40 if len(y_chords) > 40 else 0
    vec = X_chords[idx]; lab = y_chords[idx]; t0, t1 = chord_times[idx]
    print("Top3 notas:", [NOTES[k] for k in np.argsort(vec)[-3:][::-1]])
    plt.figure(figsize=(8,4))
    plt.bar(NOTES, vec, color="steelblue")
    plt.title(f"Croma del acorde #{idx+1} [{t0:.2f}-{t1:.2f}s] — Etiqueta: {lab}")
    plt.xlabel("Notas"); plt.ylabel("Energía normalizada"); plt.tight_layout(); plt.show()

    # --- Generar CSV para todos los pares audio/lab disponibles ---
    build_dataset_csv(csv_out="dataset_chords.csv")
