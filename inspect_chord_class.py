import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuración
NOTES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
NOTE_COLS = [f"chroma_{n}" for n in NOTES]

# ===== CONFIGURACIÓN: Cambia estos valores =====
DATASET = "guitarset"  # "beatles" o "guitarset"
CHORD_LABEL = "A:maj"  # Acorde a inspeccionar
N_EXAMPLES = 6         # Número de ejemplos a mostrar
RANDOM = True          # True para aleatorio, False para primeros N
# ===============================================

def plot_multiple_chromas(df, label, n_examples=6, random=False):
    """
    Plotea múltiples ejemplos del mismo acorde
    """
    # Filtrar por etiqueta
    df_filtered = df[df['label'] == label].reset_index(drop=True)
    
    if len(df_filtered) == 0:
        print(f"⚠️  No se encontraron ejemplos con label='{label}'")
        print(f"   Labels disponibles: {sorted(df['label'].unique())}")
        return
    
    print(f"Total de ejemplos con '{label}': {len(df_filtered)}")
    
    # Seleccionar ejemplos
    n_to_show = min(n_examples, len(df_filtered))
    
    if random:
        indices = np.random.choice(len(df_filtered), size=n_to_show, replace=False)
    else:
        indices = range(n_to_show)
    
    # Configurar subplot grid
    n_cols = 3
    n_rows = int(np.ceil(n_to_show / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]
        
        row = df_filtered.iloc[idx]
        chroma = row[NOTE_COLS].values
        track = row['album_track']
        t_start = row['t_start']
        t_end = row['t_end']
        
        # Colorear la nota fundamental
        root_note = label.split(':')[0]
        colors = ['green' if note == root_note else 'coral' for note in NOTES]
        
        bars = ax.bar(NOTES, chroma, color=colors, alpha=0.7, edgecolor='darkred', linewidth=1)
        
        # Top 3 notas
        top3_indices = np.argsort(chroma)[-3:][::-1]
        top3_notes = [NOTES[j] for j in top3_indices]
        
        ax.set_ylim(0, max(chroma) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlabel('Notas', fontsize=9)
        ax.set_ylabel('Energía', fontsize=9)
        
        title = f'Ejemplo #{i+1} (fila {idx})\n{track}\n'
        title += f'[{t_start:.2f}s - {t_end:.2f}s]\n'
        title += f'Top3: {", ".join(top3_notes)}'
        ax.set_title(title, fontsize=9, fontweight='bold')
        
        # Añadir valores en barras destacadas
        for bar, val, note in zip(bars, chroma, NOTES):
            if val > 0.15:  # Solo valores significativos
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=7)
    
    # Ocultar subplots vacíos
    for i in range(n_to_show, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes[row_idx, col_idx].axis('off')
    
    fig.suptitle(f'Múltiples Ejemplos de "{label}" - {DATASET.upper()}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Mostrando {n_to_show} ejemplos de '{label}'")


def show_statistics(df, label):
    """
    Muestra estadísticas de los vectores cromáticos para una clase
    """
    df_filtered = df[df['label'] == label]
    
    if len(df_filtered) == 0:
        return
    
    # Calcular promedio y desviación estándar
    chroma_matrix = df_filtered[NOTE_COLS].values
    chroma_mean = chroma_matrix.mean(axis=0)
    chroma_std = chroma_matrix.std(axis=0)
    
    print(f"\n📊 Estadísticas de '{label}' ({len(df_filtered)} ejemplos):")
    print(f"{'Nota':<6} {'Promedio':>10} {'Desv.Est':>10}")
    print("-" * 30)
    for note, mean, std in zip(NOTES, chroma_mean, chroma_std):
        print(f"{note:<6} {mean:>10.4f} {std:>10.4f}")
    
    # Top notas en promedio
    top3_indices = np.argsort(chroma_mean)[-3:][::-1]
    top3_notes = [NOTES[i] for i in top3_indices]
    top3_values = [chroma_mean[i] for i in top3_indices]
    
    print(f"\nTop 3 notas promedio:")
    for note, val in zip(top3_notes, top3_values):
        print(f"   {note}: {val:.4f}")


def main():
    # Cargar dataset
    if DATASET == "beatles":
        df = pd.read_csv("dataset_chords.csv")
    elif DATASET == "guitarset":
        df = pd.read_csv("dataset_chords_guitarset.csv")
    else:
        print(f"❌ Dataset '{DATASET}' no reconocido")
        return
    
    print("="*70)
    print(f"🔍 INSPECTOR DE CLASE DE ACORDE - {DATASET.upper()}")
    print("="*70)
    print(f"Acorde: {CHORD_LABEL}")
    print()
    
    # Mostrar estadísticas
    show_statistics(df, CHORD_LABEL)
    print()
    
    # Plotear ejemplos
    plot_multiple_chromas(df, CHORD_LABEL, n_examples=N_EXAMPLES, random=RANDOM)


if __name__ == "__main__":
    main()

