# baseline_logreg.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt

CSV = Path("dataset_chords.csv")
OUT = Path("analysis_out")
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25  # porcentaje de canciones para test

# ========== Carga ==========
df = pd.read_csv(CSV)

# Features (12 cromas)
NOTE_COLS = [f"chroma_{n}" for n in ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]]
X = df[NOTE_COLS].values.astype(np.float32)
y = df["label"].values
groups = df["album_track"].values  # split por canción

# ========== Split por canción ==========
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ========== Modelo (pipeline) ==========
# Aunque las cromas ya están normalizadas por suma, la RL mejora con estandarización
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="saga",            # robusto con multinomial y regularización
        class_weight="balanced",
        n_jobs=-1
    ))
])

pipe.fit(X_train, y_train)

# ========== Evaluación ==========
y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
bacc = balanced_accuracy_score(y_test, y_pred)

print("\n--- Métricas (test, split por canción) ---")
print(f"Accuracy:            {acc:.3f}")
print(f"Macro F1:            {f1m:.3f}")
print(f"Balanced Accuracy:   {bacc:.3f}")

# Reporte por clase
report = classification_report(y_test, y_pred, digits=3)
print("\n--- Classification report ---\n", report)

# Guardar reporte
(OUT / "baseline_logreg_report.txt").write_text(
    f"Accuracy: {acc:.4f}\nMacro F1: {f1m:.4f}\nBalanced Acc: {bacc:.4f}\n\n{report}"
)

# ========== Matriz de confusión ==========
labels_sorted = sorted(np.unique(np.concatenate([y_test, y_pred])))
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(len(labels_sorted)),
    yticks=np.arange(len(labels_sorted)),
    xticklabels=labels_sorted,
    yticklabels=labels_sorted,
    ylabel="True label",
    xlabel="Predicted label",
    title="Matriz de confusión — Regresión Logística (test)"
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT / "baseline_logreg_confusion.png", dpi=150)
plt.close()

# ========== Dump de predicciones (opcional) ==========
pred_df = pd.DataFrame({
    "album_track": df.loc[test_idx, "album_track"].values,
    "t_start": df.loc[test_idx, "t_start"].values,
    "t_end": df.loc[test_idx, "t_end"].values,
    "label_true": y_test,
    "label_pred": y_pred
})
pred_df.to_csv(OUT / "baseline_logreg_predictions.csv", index=False)

print(f"\n✅ Resultados guardados en: {OUT.resolve()}")
