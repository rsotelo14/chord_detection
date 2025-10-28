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

CSV = Path("dataset_chords.csv")  # Volver al dataset original
OUT = Path("analysis_out")
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
BATCH = 64
EPOCHS = 50
HIDDEN = 128
HIDDEN2 = 256
HIDDEN3 = 64
DROPOUT = 0.40

NOTE_COLS = None  # se detectan dinámicamente desde el CSV (todas las columnas 'chroma_*')

def load_data():
    df = pd.read_csv(CSV)
    # seleccionar todas las cols que empiezan con 'chroma_'
    feat_cols = [c for c in df.columns if c.startswith("chroma_")]
    # ordenar numéricamente si son chroma_0..chroma_35
    try:
        feat_cols = sorted(feat_cols, key=lambda c: int(c.split("_")[1]))
    except Exception:
        feat_cols = sorted(feat_cols)
    X = df[feat_cols].values.astype(np.float32)
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

def build_bottleneck_mlp(input_dim, num_classes):
    """Arquitectura bottleneck: comprime en el medio y expande"""
    l2 = tf.keras.regularizers.l2(1e-4)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Sección 1: Compresión hacia el bottleneck
        layers.Dense(128, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        layers.Dense(64, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        # BOTTLENECK: Punto más pequeño
        layers.Dense(32, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        # Sección 2: Expansión desde el bottleneck
        layers.Dense(64, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        layers.Dense(128, kernel_regularizer=l2),
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

    # ----- modelos -----
    tf.random.set_seed(RANDOM_STATE)
    
    # Modelo original
    model_original = build_mlp(input_dim=X_tr_sub.shape[1], num_classes=num_classes)
    
    # Modelo bottleneck
    model_bottleneck = build_bottleneck_mlp(input_dim=X_tr_sub.shape[1], num_classes=num_classes)

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    ]

    print("\n🔄 Entrenando modelo ORIGINAL...")
    history_original = model_original.fit(
        X_tr_sub, y_tr_sub,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
        callbacks=cb,
        class_weight=class_weight
    )

    print("\n🔄 Entrenando modelo BOTTLENECK...")
    tf.random.set_seed(RANDOM_STATE)  # Reset seed para comparación justa
    history_bottleneck = model_bottleneck.fit(
        X_tr_sub, y_tr_sub,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
        callbacks=cb,
        class_weight=class_weight
    )

    # ----- evaluación -----
    print("\n📊 EVALUANDO MODELOS...")
    
    # Evaluar modelo original
    y_pred_prob_orig = model_original.predict(X_te, verbose=0)
    y_pred_orig = np.argmax(y_pred_prob_orig, axis=1)
    
    acc_orig = accuracy_score(y_te, y_pred_orig)
    f1m_orig = f1_score(y_te, y_pred_orig, average="macro")
    bacc_orig = balanced_accuracy_score(y_te, y_pred_orig)
    
    # Evaluar modelo bottleneck
    y_pred_prob_bott = model_bottleneck.predict(X_te, verbose=0)
    y_pred_bott = np.argmax(y_pred_prob_bott, axis=1)
    
    acc_bott = accuracy_score(y_te, y_pred_bott)
    f1m_bott = f1_score(y_te, y_pred_bott, average="macro")
    bacc_bott = balanced_accuracy_score(y_te, y_pred_bott)

    print("\n" + "="*60)
    print("📊 COMPARACIÓN DE MODELOS")
    print("="*60)
    print(f"{'Métrica':<20} {'Original':<12} {'Bottleneck':<12} {'Diferencia':<12}")
    print("-"*60)
    print(f"{'Accuracy':<20} {acc_orig:.3f}        {acc_bott:.3f}        {acc_bott-acc_orig:+.3f}")
    print(f"{'Macro F1':<20} {f1m_orig:.3f}        {f1m_bott:.3f}        {f1m_bott-f1m_orig:+.3f}")
    print(f"{'Balanced Acc':<20} {bacc_orig:.3f}        {bacc_bott:.3f}        {bacc_bott-bacc_orig:+.3f}")
    print("="*60)
    
    # Determinar mejor modelo
    if acc_bott > acc_orig:
        print("🏆 BOTTLENECK es MEJOR!")
        best_model = model_bottleneck
        best_pred = y_pred_bott
        best_pred_prob = y_pred_prob_bott
        best_history = history_bottleneck
        model_name = "bottleneck"
    else:
        print("🏆 ORIGINAL es MEJOR!")
        best_model = model_original
        best_pred = y_pred_orig
        best_pred_prob = y_pred_prob_orig
        best_history = history_original
        model_name = "original"

    # Reporte por clase del mejor modelo
    report = classification_report(le.inverse_transform(y_te), le.inverse_transform(best_pred), digits=3)
    (OUT / f"best_model_report_{model_name}.txt").write_text(
        f"Accuracy: {acc_bott if model_name=='bottleneck' else acc_orig:.4f}\nMacro F1: {f1m_bott if model_name=='bottleneck' else f1m_orig:.4f}\nBalanced Acc: {bacc_bott if model_name=='bottleneck' else bacc_orig:.4f}\n\n{report}"
    )
    print(f"\n--- Classification report ({model_name.upper()}) ---\n", report)

    # Matriz de confusión (normalizada por fila = por clase verdadera)
    labels_sorted = list(class_names)  # mantener orden del encoder
    cm = confusion_matrix(le.inverse_transform(y_te), le.inverse_transform(best_pred), labels=labels_sorted)
    
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
        title=f"Matriz de confusión — {model_name.upper()} MLP (test, normalizada por fila)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "baseline_mlp_confusion.png", dpi=150)
    plt.close()

    # Curvas de entrenamiento (loss)
    plt.figure(figsize=(7,4))
    plt.plot(best_history.history.get("loss", []), label="train")
    plt.plot(best_history.history.get("val_loss", []), label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} MLP loss por época")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "baseline_mlp_loss.png", dpi=150)
    plt.close()

    # Guardar predicciones del mejor modelo
    pred_df = pd.DataFrame({
        "album_track": df.loc[te_idx, "album_track"].values,
        "t_start": df.loc[te_idx, "t_start"].values,
        "t_end": df.loc[te_idx, "t_end"].values,
        "label_true": le.inverse_transform(y_te),
        "label_pred": le.inverse_transform(best_pred),
        "p_pred": np.max(best_pred_prob, axis=1),
    })
    pred_df.to_csv(OUT / "baseline_mlp_predictions.csv", index=False)

    # Guardar el mejor modelo
    best_model.save(OUT / "baseline_mlp_model.h5")

    print(f"\n✅ Resultados guardados en: {OUT.resolve()}")
