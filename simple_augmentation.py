# simple_augmentation.py
"""
Data Augmentation conservador usando solo pitch shifting.
Solo aumenta clases minoritarias (<80 samples) usando transposición.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

CSV_INPUT = Path("dataset_chords_merged.csv")
CSV_OUTPUT = Path("dataset_chords_augmented_simple.csv")
OUT = Path("analysis_out")

NOTE_ORDER = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
NOTE_COLS = [f"chroma_{n}" for n in NOTE_ORDER]

THRESHOLD = 80  # Solo aumentar clases con menos de esto
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def get_note_index(note):
    """Retorna el índice de una nota en NOTE_ORDER."""
    return NOTE_ORDER.index(note)


def parse_chord_label(label):
    """
    Parsea una etiqueta de acorde.
    
    Ejemplos:
        'G:min' → ('G', 'min')
        'Db:maj' → ('Db', 'maj')
    
    Returns:
        (root, quality) tuple
    """
    parts = label.split(':')
    if len(parts) != 2:
        raise ValueError(f"Formato de acorde inválido: {label}")
    return parts[0], parts[1]


def transpose_chord_label(label, semitones):
    """
    Transpone una etiqueta de acorde.
    
    Args:
        label: ej. 'G:min'
        semitones: +2 para transponer 2 semitonos arriba
    
    Returns:
        nuevo label: ej. 'A:min'
    """
    root, quality = parse_chord_label(label)
    root_idx = get_note_index(root)
    new_root_idx = (root_idx + semitones) % 12
    new_root = NOTE_ORDER[new_root_idx]
    return f"{new_root}:{quality}"


def pitch_shift_chroma(chroma, semitones):
    """
    Transpone un vector de chroma rotándolo circularmente.
    
    IMPORTANTE: np.roll con valor positivo mueve hacia la DERECHA (índices mayores).
    Si queremos transponer +2 semitonos (ej: G→A), el valor en la posición de G
    debe moverse a la posición de A, que está 2 posiciones más adelante.
    
    Args:
        chroma: array (12,) con valores de chroma
        semitones: número de semitonos a transponer
    
    Returns:
        chroma transpuesto
    
    Verificación:
        - Orden: ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
        - G está en índice 7, A en índice 9
        - Para G→A necesitamos +2 semitonos
        - np.roll(chroma, 2) mueve cada valor 2 posiciones a la derecha
        - El valor que estaba en índice 7 (G) va al índice 9 (A) ✓
    """
    return np.roll(chroma, semitones)


def verify_pitch_shift():
    """Verifica que el pitch shifting funcione correctamente."""
    print("\n🔍 Verificando pitch shifting...")
    
    # Crear un acorde de prueba: G con solo la nota G activa
    test_chroma = np.zeros(12)
    g_idx = NOTE_ORDER.index('G')  # índice 7
    test_chroma[g_idx] = 1.0
    
    print(f"   Acorde de prueba: G (solo nota G activa)")
    print(f"   Índice de G: {g_idx}")
    print(f"   Vector original: {test_chroma}")
    
    # Transponer +2 semitonos (G → A)
    transposed = pitch_shift_chroma(test_chroma, 2)
    a_idx = NOTE_ORDER.index('A')  # índice 9
    
    print(f"\n   Transposición: +2 semitonos (G → A)")
    print(f"   Índice de A: {a_idx}")
    print(f"   Vector transpuesto: {transposed}")
    print(f"   ¿Valor activo en A? {transposed[a_idx] == 1.0}")
    
    if transposed[a_idx] == 1.0:
        print("   ✅ Pitch shifting funciona correctamente!\n")
        return True
    else:
        print("   ❌ ERROR: Pitch shifting NO funciona correctamente!\n")
        return False


def augment_minority_class(df_class, label, target_samples):
    """
    Aumenta una clase minoritaria usando pitch shifting.
    
    Estrategia:
    - Genera versiones transpuestas (+/-1, +/-2, +/-3 semitonos)
    - Alterna transposiciones para distribuir uniformemente
    """
    if len(df_class) >= target_samples:
        return df_class
    
    samples_needed = target_samples - len(df_class)
    augmented_rows = []
    
    # Transposiciones a usar (evitar 0 y extremos muy grandes)
    transpositions = [-3, -2, -1, 1, 2, 3]  # Evitar 0 (sería duplicado exacto)
    
    print(f"  {label:10s}: {len(df_class):3d} → {target_samples:3d} (añadiendo {samples_needed} samples)")
    
    for i in range(samples_needed):
        # Seleccionar ejemplo base aleatorio
        base_idx = np.random.randint(0, len(df_class))
        base_row = df_class.iloc[base_idx]
        
        # Seleccionar transposición
        semitones = transpositions[i % len(transpositions)]
        
        # Transponer chroma
        base_chroma = base_row[NOTE_COLS].values
        new_chroma = pitch_shift_chroma(base_chroma, semitones)
        
        # Normalizar (por si acaso)
        s = np.sum(new_chroma)
        if s > 1e-8:
            new_chroma = new_chroma / s
        
        # Crear nueva fila (mantiene misma label, solo cambia chroma)
        new_row = base_row.copy()
        for j, note in enumerate(NOTE_ORDER):
            new_row[f'chroma_{note}'] = float(new_chroma[j])
        
        augmented_rows.append(new_row)
    
    # Combinar originales con aumentados
    if augmented_rows:
        augmented_df = pd.concat([df_class, pd.DataFrame(augmented_rows)], ignore_index=True)
    else:
        augmented_df = df_class
    
    return augmented_df


def augment_dataset(df, threshold=80, target_min=100):
    """
    Aumenta solo las clases minoritarias usando pitch shifting.
    
    Args:
        df: DataFrame original
        threshold: solo aumentar clases con menos de estos samples
        target_min: llevar cada clase minoritaria al menos a este número
    """
    print("=" * 60)
    print("🎵 Data Augmentation Simple - Pitch Shifting")
    print("=" * 60)
    
    # Verificar que pitch shifting funcione
    if not verify_pitch_shift():
        raise RuntimeError("Pitch shifting no funciona correctamente!")
    
    # Contar clases
    class_counts = df['label'].value_counts().sort_index()
    
    print(f"Dataset original:")
    print(f"  Total samples: {len(df)}")
    print(f"  Clases: {len(class_counts)}")
    print(f"  Min: {class_counts.min()}, Max: {class_counts.max()}")
    print(f"  Media: {class_counts.mean():.1f}, Mediana: {class_counts.median():.1f}")
    
    print(f"\n🎯 Estrategia:")
    print(f"  - Solo aumentar clases con <{threshold} samples")
    print(f"  - Target mínimo: {target_min} samples por clase")
    print(f"  - Técnica: Pitch shifting (transposición)")
    
    # Identificar clases minoritarias
    minority_classes = class_counts[class_counts < threshold]
    print(f"\n📊 Clases minoritarias (<{threshold}): {len(minority_classes)}")
    
    # Augmentar
    augmented_dfs = []
    total_added = 0
    
    print(f"\n🔄 Aumentando clases minoritarias...")
    
    for label in sorted(class_counts.index):
        df_class = df[df['label'] == label].copy()
        original_count = len(df_class)
        
        if original_count < threshold:
            # Calcular target (al menos target_min, pero no más de threshold)
            target = min(max(target_min, original_count), threshold)
            df_aug = augment_minority_class(df_class, label, target)
            added = len(df_aug) - original_count
            total_added += added
            augmented_dfs.append(df_aug)
        else:
            augmented_dfs.append(df_class)
            print(f"  {label:10s}: {original_count:3d} (sin cambios)")
    
    # Combinar todo
    df_augmented = pd.concat(augmented_dfs, ignore_index=True)
    
    print(f"\n✅ Augmentation completado:")
    print(f"   Clases aumentadas: {len(minority_classes)}")
    print(f"   Samples añadidos: {total_added}")
    print(f"   Total original: {len(df)}")
    print(f"   Total final: {len(df_augmented)}")
    print(f"   Incremento: +{(len(df_augmented)/len(df) - 1)*100:.1f}%")
    
    return df_augmented


def visualize_comparison(df_original, df_augmented, output_path, threshold):
    """Visualiza el antes/después del augmentation."""
    counts_orig = df_original['label'].value_counts().sort_index()
    counts_aug = df_augmented['label'].value_counts().sort_index()
    
    labels = counts_orig.index
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Colores: rojo si era minoritaria, azul si no
    colors_orig = ['#FF6B6B' if c < threshold else '#95E1D3' for c in counts_orig]
    colors_aug = ['#4ECDC4' if c < threshold else '#95E1D3' for c in counts_orig]
    
    bars1 = ax.bar(x - width/2, counts_orig, width, label='Original', alpha=0.8, color=colors_orig)
    bars2 = ax.bar(x + width/2, counts_aug, width, label='Aumentado', alpha=0.8, color=colors_aug)
    
    # Línea de threshold
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Clase de Acorde', fontsize=12)
    ax.set_ylabel('Cantidad de Segmentos', fontsize=12)
    ax.set_title('Data Augmentation Simple - Pitch Shifting (solo clases minoritarias)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Anotar incrementos
    for i, (orig, aug) in enumerate(zip(counts_orig, counts_aug)):
        if aug > orig:
            increase = aug - orig
            ax.text(i + width/2, aug, f'+{increase}', 
                   ha='center', va='bottom', fontsize=7, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n📊 Gráfico guardado en: {output_path}")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n🎵 Data Augmentation Simple - Pitch Shifting")
    
    # Cargar dataset
    df = pd.read_csv(CSV_INPUT)
    print(f"\n📁 Dataset cargado: {CSV_INPUT}")
    
    # Aplicar augmentation conservador
    df_augmented = augment_dataset(
        df,
        threshold=THRESHOLD,      # Solo aumentar si tiene <80 samples
        target_min=80            # Llevar cada clase minoritaria a ~80
    )
    
    # Guardar
    df_augmented.to_csv(CSV_OUTPUT, index=False)
    print(f"\n💾 Dataset aumentado guardado en: {CSV_OUTPUT}")
    
    # Visualizar
    visualize_comparison(df, df_augmented, 
                        OUT / "data_augmentation_simple.png", 
                        THRESHOLD)
    
    # Estadísticas finales
    print("\n" + "=" * 60)
    print("📈 Estadísticas Finales")
    print("=" * 60)
    
    counts_final = df_augmented['label'].value_counts().sort_index()
    print(f"\nDistribución final:")
    print(f"  Total samples: {len(df_augmented)}")
    print(f"  Min: {counts_final.min()}, Max: {counts_final.max()}")
    print(f"  Media: {counts_final.mean():.1f}")
    print(f"  Mediana: {counts_final.median():.1f}")
    print(f"  Desv. estándar: {counts_final.std():.1f}")
    
    print("\n✅ ¡Listo para reentrenar!")
    print(f"   Usa: python baseline_mlp.py (modificando CSV a '{CSV_OUTPUT}')")



