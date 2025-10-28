# train_dnn_frames.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

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
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, min_delta=1e-3),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    print("\n🔄 Entrenando MLP común…")
    m_common = build_mlp(input_dim, num_classes)
    h_common = m_common.fit(X_tr_sub, y_tr_sub, validation_data=(X_val,y_val),
                            epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=1)

    print("\n🔄 Entrenando MLP bottleneck…")
    tf.random.set_seed(RANDOM_STATE)
    m_bott = build_bottleneck(input_dim, num_classes)
    h_bott = m_bott.fit(X_tr_sub, y_tr_sub, validation_data=(X_val,y_val),
                        epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=1)

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

    if acc2>acc1:
        best=m_bott; yhat=yh2; prob=p2; tag="bottleneck"
    else:
        best=m_common; yhat=yh1; prob=p1; tag="common"

    best.save(OUT/f"dnn_{tag}.h5")
    print(f"\n🏆 Mejor: {tag}  → guardado en {OUT}/dnn_{tag}.h5")
