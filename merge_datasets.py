import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración
OUT = Path("analysis_out")
OUT.mkdir(exist_ok=True)

BEATLES_CSV = "dataset_chords.csv"
GUITARSET_CSV = "dataset_chords_guitarset.csv"
MERGED_CSV = "dataset_chords_merged.csv"


def merge_datasets():
    """
    Fusiona los datasets de Beatles y GuitarSet
    """
    print("="*70)
    print("🔗 FUSIÓN DE DATASETS: Beatles + GuitarSet")
    print("="*70)
    print()
    
    # 1. Cargar datasets
    print("📂 Cargando datasets...")
    df_beatles = pd.read_csv(BEATLES_CSV)
    df_guitarset = pd.read_csv(GUITARSET_CSV)
    
    print(f"   ✅ Beatles:    {len(df_beatles):5} segmentos, {df_beatles['label'].nunique():2} clases")
    print(f"   ✅ GuitarSet:  {len(df_guitarset):5} segmentos, {df_guitarset['label'].nunique():2} clases")
    print()
    
    # 2. Agregar columna de fuente para poder identificar el origen
    df_beatles['source'] = 'beatles'
    df_guitarset['source'] = 'guitarset'
    
    # 3. Fusionar
    print("🔗 Fusionando datasets...")
    df_merged = pd.concat([df_beatles, df_guitarset], ignore_index=True)
    
    # Filtrar clase "N" (sin acorde) por seguridad
    n_before = len(df_merged)
    df_merged = df_merged[df_merged['label'] != 'N'].copy()
    n_after = len(df_merged)
    n_filtered = n_before - n_after
    
    print(f"   ✅ Dataset fusionado: {len(df_merged)} segmentos")
    if n_filtered > 0:
        print(f"   🗑️  Filtrados {n_filtered} segmentos con label 'N' (sin acorde)")
    print(f"   ✅ Clases únicas: {df_merged['label'].nunique()}")
    print()
    
    # 4. Verificar estructura
    print("🔍 Verificando estructura...")
    expected_cols = ['album_track', 't_start', 't_end', 'label'] + \
                    [f"chroma_{n}" for n in ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]] + \
                    ['source']
    
    if set(df_merged.columns) == set(expected_cols):
        print("   ✅ Estructura correcta")
    else:
        print("   ⚠️  Advertencia: estructura de columnas inesperada")
    
    # 5. Estadísticas de fusión
    print()
    print("📊 Estadísticas del dataset fusionado:")
    print(f"   Total de segmentos:     {len(df_merged)}")
    print(f"   Segmentos de Beatles:   {len(df_merged[df_merged['source']=='beatles'])} ({len(df_merged[df_merged['source']=='beatles'])/len(df_merged)*100:.1f}%)")
    print(f"   Segmentos de GuitarSet: {len(df_merged[df_merged['source']=='guitarset'])} ({len(df_merged[df_merged['source']=='guitarset'])/len(df_merged)*100:.1f}%)")
    print(f"   Clases únicas:          {df_merged['label'].nunique()}")
    print(f"   Tracks únicos:          {df_merged['album_track'].nunique()}")
    
    # 6. Guardar CSV
    print()
    print(f"💾 Guardando dataset fusionado en: {MERGED_CSV}")
    df_merged.to_csv(MERGED_CSV, index=False)
    print(f"   ✅ Guardado exitosamente")
    
    return df_merged


def plot_class_balance(df_merged):
    """
    Genera gráfico de balance de clases del dataset fusionado
    """
    print()
    print("📊 Generando visualización de balance de clases...")
    
    # Contar clases
    class_counts = df_merged['label'].value_counts().sort_index()
    
    # Contar por fuente
    df_beatles = df_merged[df_merged['source'] == 'beatles']
    df_guitarset = df_merged[df_merged['source'] == 'guitarset']
    
    beatles_counts = df_beatles['label'].value_counts()
    guitarset_counts = df_guitarset['label'].value_counts()
    
    # Obtener todas las clases
    all_labels = sorted(class_counts.index)
    
    # Crear arrays para gráfico apilado
    beatles_values = [beatles_counts.get(label, 0) for label in all_labels]
    guitarset_values = [guitarset_counts.get(label, 0) for label in all_labels]
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(all_labels))
    width = 0.6
    
    # Barras apiladas
    bars1 = ax.bar(x, beatles_values, width, label='Beatles', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, guitarset_values, width, bottom=beatles_values, 
                   label='GuitarSet', color='coral', alpha=0.8)
    
    ax.set_xlabel('Clases de Acordes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cantidad de Segmentos', fontsize=12, fontweight='bold')
    ax.set_title('Balance de Clases - Dataset Fusionado (Beatles + GuitarSet)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Añadir totales en las barras
    for i, label in enumerate(all_labels):
        total = beatles_values[i] + guitarset_values[i]
        if total > 0:
            ax.text(i, total + 5, f'{total}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar
    output_file = OUT / "class_balance_counts_merged.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Guardado en: {output_file}")
    plt.close()
    
    # Guardar CSV con conteos
    counts_df = pd.DataFrame({
        'label': all_labels,
        'beatles': beatles_values,
        'guitarset': guitarset_values,
        'total': [b + g for b, g in zip(beatles_values, guitarset_values)]
    })
    
    counts_csv = OUT / "class_balance_counts_merged.csv"
    counts_df.to_csv(counts_csv, index=False)
    print(f"   ✅ Tabla guardada en: {counts_csv}")
    
    # Imprimir tabla
    print()
    print("📊 Distribución de clases (Top 15):")
    print("="*70)
    print(f"{'Clase':<12} {'Beatles':>10} {'GuitarSet':>12} {'Total':>10}")
    print("="*70)
    
    # Ordenar por total descendente
    counts_df_sorted = counts_df.sort_values('total', ascending=False)
    for _, row in counts_df_sorted.head(15).iterrows():
        print(f"{row['label']:<12} {int(row['beatles']):>10} {int(row['guitarset']):>12} {int(row['total']):>10}")
    
    print("="*70)
    
    # Estadísticas generales
    print()
    print("📈 Estadísticas de balance:")
    print(f"   Clase más frecuente:  {counts_df_sorted.iloc[0]['label']} ({int(counts_df_sorted.iloc[0]['total'])} segmentos)")
    print(f"   Clase menos frecuente: {counts_df_sorted.iloc[-1]['label']} ({int(counts_df_sorted.iloc[-1]['total'])} segmentos)")
    print(f"   Promedio por clase:    {counts_df['total'].mean():.1f} segmentos")
    print(f"   Desviación estándar:   {counts_df['total'].std():.1f}")


def main():
    # Fusionar datasets
    df_merged = merge_datasets()
    
    # Generar visualización
    plot_class_balance(df_merged)
    
    print()
    print("="*70)
    print("✅ FUSIÓN COMPLETA")
    print("="*70)
    print(f"   Dataset fusionado: {MERGED_CSV}")
    print(f"   Visualizaciones: {OUT.resolve()}")
    print("="*70)


if __name__ == "__main__":
    main()


