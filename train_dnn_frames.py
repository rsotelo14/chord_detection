# train_dnn_frames.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import matplotlib
matplotlib.use('Agg')  # use non-interactive backend
import matplotlib.pyplot as plt

IN_NPZ = Path("frames_dataset.npz")
IN_CSV = Path("frames_dataset.csv")
OUT = Path("analysis_out_frames"); OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
BATCH = 128
EPOCHS = 50

def load_data():
    z = np.load(IN_NPZ, allow_pickle=True)
    X = z["X"].astype(np.float32)        # (N, F)
    y = z["y"]                            # str labels
    groups = z["groups"]                  # album_track
    df = pd.read_csv(IN_CSV)
    return X, y, groups, df

def split_groups(X,y,groups,test_size=0.25,seed=RANDOM_STATE):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    return tr_idx, te_idx

def build_mlp(input_dim, num_classes):
    l2 = tf.keras.regularizers.l2(1e-4)
    m = models.Sequential([
        layers.Input((input_dim,)),
        layers.Dense(1024, kernel_regularizer=l2),
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.3),
        layers.Dense(1024, kernel_regularizer=l2),
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    m.compile(optimizer=optimizers.Adam(1e-4),  # LR más bajo para más estabilidad
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

def build_bottleneck(input_dim, num_classes):
    l2 = tf.keras.regularizers.l2(1e-4)
    m = models.Sequential([
        layers.Input((input_dim,)),
        layers.Dense(1024, kernel_regularizer=l2),
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.3),
        layers.Dense(512, kernel_regularizer=l2),
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.3),
        layers.Dense(256, kernel_regularizer=l2),   # bottleneck
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.3),
        layers.Dense(512, kernel_regularizer=l2),
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.3),
        layers.Dense(1024, kernel_regularizer=l2),
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    m.compile(optimizer=optimizers.Adam(1e-4),  # LR más bajo
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

def plot_history_loss(histories, labels, out_path):
    plt.figure(figsize=(8, 5))
    for h, label in zip(histories, labels):
        l = h.history.get("loss", [])
        v = h.history.get("val_loss", [])
        epochs = range(1, len(l) + 1)
        plt.plot(epochs, l, label=f"{label} Train loss")
        plt.plot(epochs, v, '--', label=f"{label} Val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    X, y_str, groups, dfmeta = load_data()

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = le.classes_
    np.savetxt(OUT/"label_mapping.txt", classes, fmt="%s")

    tr_idx, te_idx = split_groups(X, y, groups)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    groups_tr = groups[tr_idx]

    # val split por canción dentro del train
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_sub, val_idx = next(gss_val.split(X_tr, y_tr, groups=groups_tr))
    X_tr_sub, y_tr_sub = X_tr[tr_sub], y_tr[tr_sub]
    X_val,    y_val    = X_tr[val_idx], y_tr[val_idx]

    tf.random.set_seed(RANDOM_STATE)
    input_dim = X_tr_sub.shape[1]
    num_classes = len(classes)

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, min_delta=5e-4),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    # ---- Class weights (balanced) ----
    # Calculados sobre el split de entrenamiento efectivo (X_tr_sub, y_tr_sub)
    class_weight_vals = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_tr_sub
    )
    class_weight = {int(i): float(w) for i, w in enumerate(class_weight_vals)}
    # Guardar para referencia
    with open(OUT/"class_weights.txt", "w") as f:
        for i, w in class_weight.items():
            f.write(f"{i}\t{classes[i]}\t{w:.6f}\n")
    print("\n⚖️  Class weights (balanced) aplicados. Ejemplos:")
    for i in range(min(5, num_classes)):
        print(f"  idx={i:2d} clase={classes[i]:8s} w={class_weight[i]:.3f}")

    print("\n🔄 Entrenando MLP común…")
    m_common = build_mlp(input_dim, num_classes)
    h_common = m_common.fit(
        X_tr_sub, y_tr_sub,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cb,
        verbose=1,
        class_weight=class_weight,
    )

    print("\n🔄 Entrenando MLP bottleneck…")
    tf.random.set_seed(RANDOM_STATE)
    m_bott = build_bottleneck(input_dim, num_classes)
    h_bott = m_bott.fit(
        X_tr_sub, y_tr_sub,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cb,
        verbose=1,
        class_weight=class_weight,
    )

    # Plot and save the training/validation loss curves
    plot_path = OUT / "train_val_loss.png"
    plot_history_loss([h_common, h_bott], ["Common", "Bottleneck"], plot_path)
    print(f"📈 Plot guardado en {plot_path}")

    # Eval
    def eval_model(m, tag):
        p = m.predict(X_te, verbose=0)
        yhat = np.argmax(p, axis=1)
        acc  = accuracy_score(y_te, yhat)
        f1m  = f1_score(y_te, yhat, average="macro")
        bacc = balanced_accuracy_score(y_te, yhat)
        print(f"\n[{tag}] acc={acc:.3f}  f1_macro={f1m:.3f}  bacc={bacc:.3f}")
        rep = classification_report(le.inverse_transform(y_te), le.inverse_transform(yhat), digits=3)
        (OUT/f"report_{tag}.txt").write_text(rep)
        return acc, f1m, bacc, yhat, p

    acc1,f11,bacc1,yh1,p1 = eval_model(m_common, "common")
    acc2,f12,bacc2,yh2,p2 = eval_model(m_bott, "bottleneck")

    # Guardar ambos modelos
    m_common.save(OUT/"dnn_common.h5")
    m_bott.save(OUT/"dnn_bottleneck.h5")
    print(f"\n💾 Modelos guardados: {OUT}/dnn_common.h5 y {OUT}/dnn_bottleneck.h5")

    # Informe simple de cuál rindió mejor en test
    if acc2 > acc1:
        print("\n🏆 En test, bottleneck > common por accuracy.")
    elif acc2 < acc1:
        print("\n🏆 En test, common > bottleneck por accuracy.")
    else:
        print("\n⚖️  En test, ambos empatan en accuracy.")
