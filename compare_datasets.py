import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración de visualización
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Directorios
OUT = Path("analysis_out")
OUT.mkdir(exist_ok=True)

# Nombres de las notas para las gráficas de cromas
NOTES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
NOTE_COLS = [f"chroma_{n}" for n in NOTES]

def load_datasets():
    """Carga ambos datasets"""
    df_beatles = pd.read_csv("dataset_chords.csv")
    df_guitarset = pd.read_csv("dataset_chords_guitarset.csv")
    return df_beatles, df_guitarset


def plot_class_balance(df_beatles, df_guitarset):
    """
    Grafica el balanceo de clases de ambos datasets
    """
    # Contar clases en ambos datasets
    beatles_counts = df_beatles['label'].value_counts().sort_index()
    guitarset_counts = df_guitarset['label'].value_counts().sort_index()
    
    # Obtener todas las clases únicas
    all_labels = sorted(set(beatles_counts.index) | set(guitarset_counts.index))
    
    # Rellenar con 0 las clases faltantes
    beatles_values = [beatles_counts.get(label, 0) for label in all_labels]
    guitarset_values = [guitarset_counts.get(label, 0) for label in all_labels]
    
    # Crear gráfica de barras agrupadas
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(all_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, beatles_values, width, label='Beatles', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, guitarset_values, width, label='GuitarSet', color='coral', alpha=0.8)
    
    ax.set_xlabel('Clases de Acordes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cantidad de Segmentos', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Balanceo de Clases: Beatles vs GuitarSet', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUT / "datasets_class_balance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfica de balanceo de clases guardada")
    
    # Estadísticas
    print("\n📊 Estadísticas de Balanceo:")
    print(f"   Beatles    - Total clases: {len(beatles_counts)}, Promedio: {beatles_counts.mean():.1f}, "
          f"Min: {beatles_counts.min()}, Max: {beatles_counts.max()}")
    print(f"   GuitarSet  - Total clases: {len(guitarset_counts)}, Promedio: {guitarset_counts.mean():.1f}, "
          f"Min: {guitarset_counts.min()}, Max: {guitarset_counts.max()}")


def plot_example_chroma(df_beatles, df_guitarset):
    """
    Grafica ejemplos de vectores cromáticos de ambos datasets
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Ejemplo de Beatles
    idx_beatles = np.random.randint(0, len(df_beatles))
    row_beatles = df_beatles.iloc[idx_beatles]
    chroma_beatles = row_beatles[NOTE_COLS].values
    label_beatles = row_beatles['label']
    track_beatles = row_beatles['album_track']
    t_start_beatles = row_beatles['t_start']
    t_end_beatles = row_beatles['t_end']
    
    axes[0].bar(NOTES, chroma_beatles, color='steelblue', alpha=0.7, edgecolor='navy')
    axes[0].set_title(f'Beatles - Acorde: {label_beatles}\n{track_beatles}\n[{t_start_beatles:.2f}s - {t_end_beatles:.2f}s]', 
                      fontweight='bold', fontsize=11)
    axes[0].set_xlabel('Notas', fontweight='bold')
    axes[0].set_ylabel('Energía Normalizada', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim(0, max(chroma_beatles) * 1.1)
    
    # Ejemplo de GuitarSet
    idx_guitarset = np.random.randint(0, len(df_guitarset))
    row_guitarset = df_guitarset.iloc[idx_guitarset]
    chroma_guitarset = row_guitarset[NOTE_COLS].values
    label_guitarset = row_guitarset['label']
    track_guitarset = row_guitarset['album_track']
    t_start_guitarset = row_guitarset['t_start']
    t_end_guitarset = row_guitarset['t_end']
    
    axes[1].bar(NOTES, chroma_guitarset, color='coral', alpha=0.7, edgecolor='darkred')
    axes[1].set_title(f'GuitarSet - Acorde: {label_guitarset}\n{track_guitarset}\n[{t_start_guitarset:.2f}s - {t_end_guitarset:.2f}s]', 
                      fontweight='bold', fontsize=11)
    axes[1].set_xlabel('Notas', fontweight='bold')
    axes[1].set_ylabel('Energía Normalizada', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_ylim(0, max(chroma_guitarset) * 1.1)
    
    plt.suptitle('Ejemplos de Vectores Cromáticos', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "datasets_chroma_examples.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfica de ejemplos cromáticos guardada")


def plot_dataset_comparison(df_beatles, df_guitarset):
    """
    Compara los tamaños de los datasets con diferentes métricas
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Número total de segmentos
    datasets = ['Beatles', 'GuitarSet']
    total_segments = [len(df_beatles), len(df_guitarset)]
    colors = ['steelblue', 'coral']
    
    bars = axes[0, 0].bar(datasets, total_segments, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Número de Segmentos', fontweight='bold')
    axes[0, 0].set_title('Total de Segmentos de Acordes', fontweight='bold', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars, total_segments)):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., val,
                       f'{val}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Número de tracks/archivos únicos
    n_tracks_beatles = df_beatles['album_track'].nunique()
    n_tracks_guitarset = df_guitarset['album_track'].nunique()
    n_tracks = [n_tracks_beatles, n_tracks_guitarset]
    
    bars = axes[0, 1].bar(datasets, n_tracks, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Número de Tracks', fontweight='bold')
    axes[0, 1].set_title('Total de Canciones/Tracks Únicos', fontweight='bold', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars, n_tracks)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., val,
                       f'{val}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 3. Número de clases únicas
    n_classes_beatles = df_beatles['label'].nunique()
    n_classes_guitarset = df_guitarset['label'].nunique()
    n_classes = [n_classes_beatles, n_classes_guitarset]
    
    bars = axes[1, 0].bar(datasets, n_classes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Número de Clases', fontweight='bold')
    axes[1, 0].set_title('Total de Clases de Acordes Únicas', fontweight='bold', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars, n_classes)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., val,
                       f'{val}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 4. Duración promedio de segmentos
    duration_beatles = (df_beatles['t_end'] - df_beatles['t_start']).mean()
    duration_guitarset = (df_guitarset['t_end'] - df_guitarset['t_start']).mean()
    durations = [duration_beatles, duration_guitarset]
    
    bars = axes[1, 1].bar(datasets, durations, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Duración (segundos)', fontweight='bold')
    axes[1, 1].set_title('Duración Promedio de Segmentos', fontweight='bold', fontsize=12)
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars, durations)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.2f}s',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('Comparación de Tamaños de Datasets', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUT / "datasets_size_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfica de comparación de tamaños guardada")
    
    # Tabla comparativa en texto
    print("\n📊 Comparación Detallada:")
    print("="*70)
    print(f"{'Métrica':<40} {'Beatles':>12} {'GuitarSet':>12}")
    print("="*70)
    print(f"{'Total de segmentos':<40} {len(df_beatles):>12} {len(df_guitarset):>12}")
    print(f"{'Tracks únicos':<40} {n_tracks_beatles:>12} {n_tracks_guitarset:>12}")
    print(f"{'Clases únicas':<40} {n_classes_beatles:>12} {n_classes_guitarset:>12}")
    print(f"{'Duración promedio segmento (s)':<40} {duration_beatles:>12.2f} {duration_guitarset:>12.2f}")
    print(f"{'Segmentos por track (promedio)':<40} {len(df_beatles)/n_tracks_beatles:>12.1f} {len(df_guitarset)/n_tracks_guitarset:>12.1f}")
    print("="*70)


def main():
    print("="*70)
    print("🎸 ANÁLISIS COMPARATIVO: Beatles vs GuitarSet")
    print("="*70)
    print()
    
    # Cargar datasets
    print("📂 Cargando datasets...")
    df_beatles, df_guitarset = load_datasets()
    print(f"   ✅ Beatles: {len(df_beatles)} segmentos")
    print(f"   ✅ GuitarSet: {len(df_guitarset)} segmentos")
    print()
    
    # Generar gráficas
    print("📊 Generando visualizaciones...")
    print()
    
    plot_class_balance(df_beatles, df_guitarset)
    print()
    
    plot_example_chroma(df_beatles, df_guitarset)
    print()
    
    plot_dataset_comparison(df_beatles, df_guitarset)
    print()
    
    print("="*70)
    print(f"✅ Análisis completo guardado en: {OUT.resolve()}")
    print("="*70)


if __name__ == "__main__":
    main()


