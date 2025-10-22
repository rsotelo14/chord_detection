import numpy as np
import librosa
import pandas as pd
import json
from pathlib import Path

# ---- Config ----
SR = 22050
HOP = 512

NOTES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
ENH = {"C#":"Db","D#":"Eb","F#":"Gb","G#":"Ab","A#":"Bb"}

# ---- Normalización de etiquetas (igual que build_dataset.py) ----
def norm_label(lab: str) -> str:
    """
    Normaliza etiquetas a 25 clases:
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


# ---- Lectura de anotaciones JAMS ----
def load_chords_jams(path_jams):
    """
    Carga las anotaciones de acordes desde un archivo .jams
    Usa la primera anotación de acordes (instructed/simple)
    Retorna: [(t0, t1, label), ...]
    """
    with open(path_jams, 'r') as f:
        data = json.load(f)
    
    # Buscar anotaciones de acordes
    chord_annotations = [a for a in data['annotations'] if a['namespace'] == 'chord']
    
    if not chord_annotations:
        raise ValueError(f"No se encontraron anotaciones de acordes en {path_jams}")
    
    # Usar la primera anotación (instructed/simple)
    chord_ann = chord_annotations[0]
    
    rows = []
    for segment in chord_ann['data']:
        t0 = segment['time']
        duration = segment['duration']
        t1 = t0 + duration
        label = segment['value']
        rows.append((t0, t1, label))
    
    # Ordenar por tiempo de inicio
    rows.sort(key=lambda x: x[0])
    return rows


# ---- Audio + features (croma "sano") ----
def compute_chroma_dense(y, sr=SR, hop_length=HOP):
    """
    Calcula cromas usando componente armónica
    """
    y_harm = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)  # normalización por columna
    times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)
    return chroma, times  # (12, T), (T,)


# ---- Agregación por intervalo de acorde ----
def aggregate_chroma_per_chord(chroma, times_frame, chord_intervals, min_dur=0.25):
    """
    Agrega cromas por cada intervalo de acorde
    """
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


# --------- GENERADOR DE DATASET GUITARSET (.csv) ----------
def build_guitarset_dataset_csv(csv_out="dataset_chords_guitarset.csv", filter_styles=None):
    """
    Construye el dataset de GuitarSet procesando solo los archivos 'comp'
    
    Args:
        csv_out: nombre del archivo CSV de salida
        filter_styles: lista de estilos a incluir (ej: ['Rock', 'SS'])
                      Si es None, incluye todos los estilos
    """
    base = Path(".")
    annotation_dir = base / "annotation"
    audio_dir = base / "audio_mono-mic"
    
    # Buscar solo archivos JAMS de tipo "comp"
    jams_files = sorted(annotation_dir.glob("*_comp.jams"))
    
    # Filtrar por estilos si se especifica
    if filter_styles:
        jams_files_filtered = []
        for jams_file in jams_files:
            # Verificar si el nombre contiene alguno de los estilos
            if any(style in jams_file.name for style in filter_styles):
                jams_files_filtered.append(jams_file)
        jams_files = jams_files_filtered
        print(f"🎯 Filtrado por estilos: {', '.join(filter_styles)}")
        print(f"   Archivos que coinciden: {len(jams_files)}\n")
    
    rows = []
    n_ok, n_skip = 0, 0
    
    print(f"🎸 Procesando GuitarSet ({len(jams_files)} archivos 'comp')...\n")
    
    for jams_file in jams_files:
        # Archivo de audio correspondiente
        # Ej: "00_BN1-129-Eb_comp.jams" -> "00_BN1-129-Eb_comp_mic.wav"
        audio_file = audio_dir / f"{jams_file.stem}_mic.wav"
        
        if not audio_file.exists():
            print(f"⚠️  Audio faltante para {jams_file.name}")
            n_skip += 1
            continue
        
        try:
            # Cargar anotaciones de acordes
            chord_intervals = load_chords_jams(jams_file)
            
            # Cargar audio
            y, sr = librosa.load(audio_file, sr=SR, mono=True)
            
            # Features densas (cromas)
            chroma, times_frame = compute_chroma_dense(y, sr=sr, hop_length=HOP)
            
            # Agregación por acorde
            X_chords, y_chords, chord_times = aggregate_chroma_per_chord(
                chroma, times_frame, chord_intervals, min_dur=0.25
            )
            
            # Armar filas para el CSV
            for vec, lab, (t0, t1) in zip(X_chords, y_chords, chord_times):
                row = {
                    "album_track": jams_file.stem,  # Ej: "00_BN1-129-Eb_comp"
                    "t_start": t0,
                    "t_end": t1,
                    "label": lab,
                }
                # Agregar 12 columnas de cromas
                for i, note in enumerate(NOTES):
                    row[f"chroma_{note}"] = float(vec[i])
                rows.append(row)
            
            n_ok += 1
            print(f"✅ ({n_ok:3}/{len(jams_files)}) {jams_file.name:40} -> {len(y_chords):3} segmentos")
        
        except Exception as e:
            print(f"❌ ERROR en {jams_file.name}: {e}")
            n_skip += 1
    
    # Guardar CSV
    if rows:
        df = pd.DataFrame(rows)
        
        # Filtrar clase "N" (sin acorde)
        n_before = len(df)
        df = df[df['label'] != 'N'].copy()
        n_after = len(df)
        n_filtered = n_before - n_after
        
        df.to_csv(csv_out, index=False)
        print(f"\n✅ Dataset GuitarSet guardado en: {csv_out}")
        print(f"   📊 Total de filas: {len(df)}")
        if n_filtered > 0:
            print(f"   🗑️  Filtrados {n_filtered} segmentos con label 'N' (sin acorde)")
        print(f"   📊 Clases únicas: {df['label'].nunique()}")
        print(f"   📊 Distribución de clases:")
        class_counts = df['label'].value_counts().sort_index()
        for label, count in class_counts.items():
            print(f"      {label:10} : {count:4} segmentos")
    else:
        print("\n⚠️ No se generaron filas. Revisá que existan pares audio/jams consistentes.")
    
    print(f"\n📈 Resumen:")
    print(f"   ✅ Procesados correctamente: {n_ok}")
    print(f"   ❌ Con errores/saltados: {n_skip}")


if __name__ == "__main__":
    # Filtrar solo Rock y Singer-Songwriter (acordes más simples)
    build_guitarset_dataset_csv(
        csv_out="dataset_chords_guitarset.csv",
        filter_styles=['Rock', 'SS']  # Rock y Singer-Songwriter
    )

