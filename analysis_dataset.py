# analysis_dataset.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("dataset_chords.csv")
OUT_DIR = Path("analysis_out")
OUT_DIR.mkdir(exist_ok=True)

# ========== Carga ==========
df = pd.read_csv(CSV)

# Duración de cada segmento
df["dur"] = df["t_end"] - df["t_start"]

# Extra: columnas útiles
df["album"] = df["album_track"].apply(lambda s: s.split("/")[0])
df["track"] = df["album_track"].apply(lambda s: s.split("/")[-1])

# ========== 1) Balance de clases ==========
counts = df["label"].value_counts().rename_axis("label").reset_index(name="count")
counts["prop"] = counts["count"] / counts["count"].sum()
counts.to_csv(OUT_DIR / "class_balance_counts.csv", index=False)
print("\n--- Balance de clases (por segmentos) ---")
print(counts)

# Plot barras de clases
plt.figure(figsize=(10,4))
plt.bar(counts["label"], counts["count"])
plt.xticks(rotation=45, ha="right")
plt.title("Frecuencia de clases (segmentos)")
plt.ylabel("cantidad")
plt.tight_layout()
plt.savefig(OUT_DIR / "class_balance_counts.png", dpi=150)
plt.close()

# ========== 2) Duración total y media por clase ==========
dur_stats = df.groupby("label")["dur"].agg(total_dur="sum", mean_dur="mean", median_dur="median").reset_index()
dur_stats = dur_stats.sort_values("total_dur", ascending=False)
dur_stats.to_csv(OUT_DIR / "class_duration_stats.csv", index=False)
print("\n--- Duración por clase (s) ---")
print(dur_stats.head(15))

# Plot duración total por clase
plt.figure(figsize=(10,4))
plt.bar(dur_stats["label"], dur_stats["total_dur"])
plt.xticks(rotation=45, ha="right")
plt.title("Duración total por clase (s)")
plt.ylabel("segundos")
plt.tight_layout()
plt.savefig(OUT_DIR / "class_total_duration.png", dpi=150)
plt.close()

# ========== 3) Distribución de longitudes de segmento ==========
plt.figure(figsize=(6,4))
plt.hist(df["dur"], bins=40)
plt.title("Distribución de duración de segmentos")
plt.xlabel("duración (s)")
plt.ylabel("frecuencia")
plt.tight_layout()
plt.savefig(OUT_DIR / "segment_duration_hist.png", dpi=150)
plt.close()

# ========== 4) Cobertura por tema (cuántos segmentos y tiempo total) ==========
per_track = df.groupby("album_track").agg(
    n_segments=("label", "size"),
    dur_total=("dur", "sum")
).reset_index().sort_values("dur_total", ascending=False)
per_track.to_csv(OUT_DIR / "per_track_coverage.csv", index=False)
print("\n--- Cobertura por tema (top 10 por duración total) ---")
print(per_track.head(10))

# ========== 5) Matriz tema x clase (para ver qué aparece en cada canción) ==========
pivot = pd.crosstab(df["album_track"], df["label"])
pivot.to_csv(OUT_DIR / "track_by_class_matrix.csv")
print("\nMatriz tema x clase guardada en track_by_class_matrix.csv")

# ========== 6) Chequeos rápidos ==========
print("\n--- Chequeos rápidos ---")
print("Filas totales:", len(df))
print("Clases presentes:", df['label'].nunique(), "→", sorted(df['label'].unique()))
print("Porcentaje 'N':", (df['label'].eq('N').mean()*100).round(2), "%")
print("Duración total (min):", (df['dur'].sum()/60).round(2))

print(f"\nListo. Archivos guardados en: {OUT_DIR.resolve()}")
