# baseline_mlp.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

CSV = Path("dataset_chords_merged.csv")
OUT = Path("analysis_out")
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
BATCH = 64
EPOCHS = 100
HIDDEN = 256
HIDDEN2 = 128
HIDDEN3 = 64
DROPOUT = 0.40

NOTE_COLS = [f"chroma_{n}" for n in ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]]

def load_data():
    df = pd.read_csv(CSV)
    X = df[NOTE_COLS].values.astype(np.float32)
    y = df["label"].values
    groups = df["album_track"].values
    return X, y, groups, df

def train_test_split_groups(X, y, groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx)

def build_mlp(input_dim, num_classes):
    l2 = tf.keras.regularizers.l2(1e-4)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        layers.Dense(64, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

if __name__ == "__main__":
    # ----- datos -----
    X, y_str, groups, df = load_data()

    # Label encoder en labels (persistimos el mapping)
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = le.classes_
    num_classes = len(class_names)
    np.savetxt(OUT / "mlp_label_mapping.txt", class_names, fmt="%s")

    # Split por canción (test externo)
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split_groups(X, y, groups)

    # Split de validación por canción dentro del train
    groups_tr = groups[tr_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_sub_idx, val_idx = next(gss_val.split(X_tr, y_tr, groups=groups_tr))

    X_tr_sub, y_tr_sub = X_tr[tr_sub_idx], y_tr[tr_sub_idx]
    X_val,    y_val    = X_tr[val_idx],    y_tr[val_idx]

    # Escalado (fit SOLO en train-sub)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_sub = scaler.fit_transform(X_tr_sub)
    X_val    = scaler.transform(X_val)
    X_te     = scaler.transform(X_te)
    # Guardar scaler para reproducibilidad (opcional: medias y stds)
    np.savez(OUT / "mlp_scaler_stats.npz", mean=scaler.mean_, scale=scaler.scale_)

    # Class weights balanceados para clases presentes en train-sub
    classes_present = np.unique(y_tr_sub)
    cw_values = compute_class_weight(class_weight="balanced", classes=classes_present, y=y_tr_sub)
    class_weight = {int(c): float(w) for c, w in zip(classes_present, cw_values)}

    # ----- modelo -----
    tf.random.set_seed(RANDOM_STATE)
    model = build_mlp(input_dim=X_tr_sub.shape[1], num_classes=num_classes)

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        X_tr_sub, y_tr_sub,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
        callbacks=cb,
        class_weight=class_weight
    )

    # ----- evaluación -----
    y_pred_prob = model.predict(X_te, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_te, y_pred)

    print("\n--- Métricas MLP (test, split por canción) ---")
    print(f"Accuracy:          {acc:.3f}")
    print(f"Macro F1:          {f1m:.3f}")
    print(f"Balanced Accuracy: {bacc:.3f}")

    # Reporte por clase (con nombres originales)
    report = classification_report(le.inverse_transform(y_te), le.inverse_transform(y_pred), digits=3)
    (OUT / "baseline_mlp_report.txt").write_text(
        f"Accuracy: {acc:.4f}\nMacro F1: {f1m:.4f}\nBalanced Acc: {bacc:.4f}\n\n{report}"
    )
    print("\n--- Classification report ---\n", report)

    # Matriz de confusión (normalizada por fila = por clase verdadera)
    labels_sorted = list(class_names)  # mantener orden del encoder
    cm = confusion_matrix(le.inverse_transform(y_te), le.inverse_transform(y_pred), labels=labels_sorted)
    
    # Normalizar por fila (cada fila suma 100%)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    # Manejar división por cero (clases sin ejemplos)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Proporción', rotation=270, labelpad=15)
    
    # Añadir texto en cada celda mostrando porcentaje y conteo (solo si no es 0)
    thresh = cm_normalized.max() / 2.
    for i in range(len(labels_sorted)):
        for j in range(len(labels_sorted)):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            # Mostrar porcentaje y conteo solo si no es 0
            if count > 0:
                text = f'{pct:.2f}\n({count})'
                ax.text(j, i, text,
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=6)
    
    ax.set(
        xticks=np.arange(len(labels_sorted)),
        yticks=np.arange(len(labels_sorted)),
        xticklabels=labels_sorted,
        yticklabels=labels_sorted,
        ylabel="True label",
        xlabel="Predicted label",
        title="Matriz de confusión — MLP (test, normalizada por fila)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "baseline_mlp_confusion.png", dpi=150)
    plt.close()

    # Curvas de entrenamiento (loss)
    plt.figure(figsize=(7,4))
    plt.plot(history.history.get("loss", []), label="train")
    plt.plot(history.history.get("val_loss", []), label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP loss por época")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "baseline_mlp_loss.png", dpi=150)
    plt.close()

    # Guardar predicciones
    pred_df = pd.DataFrame({
        "album_track": df.loc[te_idx, "album_track"].values,
        "t_start": df.loc[te_idx, "t_start"].values,
        "t_end": df.loc[te_idx, "t_end"].values,
        "label_true": le.inverse_transform(y_te),
        "label_pred": le.inverse_transform(y_pred),
        "p_pred": np.max(y_pred_prob, axis=1),
    })
    pred_df.to_csv(OUT / "baseline_mlp_predictions.csv", index=False)

    # Guardar el modelo (opcional)
    model.save(OUT / "baseline_mlp_model.h5")

    print(f"\n✅ Resultados guardados en: {OUT.resolve()}")
