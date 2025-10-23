# improved_mlp.py
"""
MLP mejorado con varias arquitecturas para probar.
Experimenta con diferentes configuraciones para mejorar la detección de acordes.
"""

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
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers

CSV = Path("dataset_chords_merged.csv")
OUT = Path("analysis_out")
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
BATCH = 64
EPOCHS = 150
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


# ==================== ARQUITECTURAS ====================

def build_baseline_mlp(input_dim, num_classes):
    """Arquitectura actual (baseline)"""
    l2 = regularizers.l2(1e-4)
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
    ], name="baseline")
    return model


def build_deeper_mlp(input_dim, num_classes):
    """MLP más profundo: 3 capas ocultas con gradual reduction"""
    l2 = regularizers.l2(1e-4)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(256, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        
        layers.Dense(128, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        
        layers.Dense(64, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation="softmax"),
    ], name="deeper")
    return model


def build_wide_mlp(input_dim, num_classes):
    """MLP más ancho: capas con más neuronas"""
    l2 = regularizers.l2(1e-4)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(512, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.4),
        
        layers.Dense(256, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation="softmax"),
    ], name="wide")
    return model


def build_residual_mlp(input_dim, num_classes):
    """MLP con residual connections (skip connections)"""
    l2 = regularizers.l2(1e-4)
    
    inputs = layers.Input(shape=(input_dim,))
    
    # Primera expansión
    x = layers.Dense(128, kernel_regularizer=l2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    # Bloque residual 1
    x1 = layers.Dense(128, kernel_regularizer=l2)(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x = layers.Add()([x, x1])  # Skip connection
    x = layers.Dropout(0.3)(x)
    
    # Bloque residual 2
    x2 = layers.Dense(128, kernel_regularizer=l2)(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x = layers.Add()([x, x2])  # Skip connection
    x = layers.Dropout(0.2)(x)
    
    # Capa final
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="residual")
    return model


def build_pyramid_mlp(input_dim, num_classes):
    """MLP piramidal: expande y luego contrae"""
    l2 = regularizers.l2(1e-4)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Expansión
        layers.Dense(64, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),
        
        layers.Dense(128, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        
        layers.Dense(256, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.4),
        
        # Contracción
        layers.Dense(128, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        
        layers.Dense(64, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation="softmax"),
    ], name="pyramid")
    return model


# ==================== ENTRENAMIENTO ====================

def train_model(model, X_tr, y_tr, X_val, y_val, class_weight, model_name):
    """Entrena un modelo y retorna history"""
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=15, 
            min_delta=1e-3, 
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=7, 
            min_lr=1e-6, 
            verbose=0
        ),
    ]
    
    print(f"\n{'='*60}")
    print(f"🏋️  Entrenando {model_name}...")
    print(f"{'='*60}")
    print(f"Parámetros: {model.count_params():,}")
    
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=0,
        callbacks=cb,
        class_weight=class_weight
    )
    
    return history


def evaluate_model(model, X_te, y_te, le, model_name):
    """Evalúa un modelo y retorna métricas"""
    y_pred_prob = model.predict(X_te, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_te, y_pred)
    
    print(f"\n📊 {model_name} - Resultados:")
    print(f"   Accuracy:          {acc:.4f}")
    print(f"   Macro F1:          {f1m:.4f}")
    print(f"   Balanced Accuracy: {bacc:.4f}")
    
    return {
        'model_name': model_name,
        'accuracy': acc,
        'macro_f1': f1m,
        'balanced_acc': bacc,
        'predictions': y_pred,
        'probabilities': y_pred_prob
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Experimentando con arquitecturas MLP mejoradas")
    print("=" * 60)
    
    # ----- Datos -----
    X, y_str, groups, df = load_data()
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = le.classes_
    num_classes = len(class_names)
    
    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {num_classes}")
    
    # Split
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split_groups(X, y, groups)
    
    # Validación
    groups_tr = groups[tr_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_sub_idx, val_idx = next(gss_val.split(X_tr, y_tr, groups=groups_tr))
    
    X_tr_sub, y_tr_sub = X_tr[tr_sub_idx], y_tr[tr_sub_idx]
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]
    
    # Escalado
    scaler = StandardScaler()
    X_tr_sub = scaler.fit_transform(X_tr_sub)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)
    
    # Class weights
    classes_present = np.unique(y_tr_sub)
    cw_values = compute_class_weight(class_weight="balanced", classes=classes_present, y=y_tr_sub)
    class_weight = {int(c): float(w) for c, w in zip(classes_present, cw_values)}
    
    # ----- Arquitecturas a probar -----
    architectures = {
        'baseline': build_baseline_mlp,
        'deeper': build_deeper_mlp,
        'wide': build_wide_mlp,
        'residual': build_residual_mlp,
        'pyramid': build_pyramid_mlp,
    }
    
    # ----- Entrenar y evaluar -----
    tf.random.set_seed(RANDOM_STATE)
    results = []
    
    for arch_name, build_fn in architectures.items():
        model = build_fn(input_dim=X_tr_sub.shape[1], num_classes=num_classes)
        history = train_model(model, X_tr_sub, y_tr_sub, X_val, y_val, class_weight, arch_name)
        result = evaluate_model(model, X_te, y_te, le, arch_name)
        result['history'] = history
        results.append(result)
        
        # Guardar mejor modelo hasta ahora
        if len(results) == 1 or result['accuracy'] > max(r['accuracy'] for r in results[:-1]):
            print(f"   ⭐ ¡Nuevo mejor modelo!")
            model.save(OUT / f"improved_mlp_{arch_name}.h5")
    
    # ----- Comparación final -----
    print("\n" + "=" * 60)
    print("📊 COMPARACIÓN FINAL")
    print("=" * 60)
    
    # Tabla de resultados
    results_df = pd.DataFrame([{
        'Arquitectura': r['model_name'],
        'Accuracy': f"{r['accuracy']:.4f}",
        'Macro F1': f"{r['macro_f1']:.4f}",
        'Balanced Acc': f"{r['balanced_acc']:.4f}",
    } for r in results])
    
    print("\n" + results_df.to_string(index=False))
    
    # Encontrar el mejor
    best_result = max(results, key=lambda r: r['accuracy'])
    print(f"\n🏆 Mejor modelo: {best_result['model_name']}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   Mejora sobre baseline: +{(best_result['accuracy'] - results[0]['accuracy'])*100:.2f}%")
    
    # Guardar reporte del mejor modelo
    report = classification_report(
        le.inverse_transform(y_te),
        le.inverse_transform(best_result['predictions']),
        digits=3
    )
    
    report_text = f"""Improved MLP Results - Best Architecture: {best_result['model_name']}
{'='*60}

Accuracy:          {best_result['accuracy']:.4f}
Macro F1:          {best_result['macro_f1']:.4f}
Balanced Accuracy: {best_result['balanced_acc']:.4f}

Comparison with baseline:
  Accuracy improvement: +{(best_result['accuracy'] - results[0]['accuracy'])*100:.2f}%
  F1 improvement:       +{(best_result['macro_f1'] - results[0]['macro_f1'])*100:.2f}%

Classification Report:
{report}
"""
    
    (OUT / "improved_mlp_report.txt").write_text(report_text)
    
    # Gráfico comparativo
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['accuracy', 'macro_f1', 'balanced_acc']
    titles = ['Accuracy', 'Macro F1', 'Balanced Accuracy']
    
    for ax, metric, title in zip(axes, metrics, titles):
        values = [r[metric] for r in results]
        names = [r['model_name'] for r in results]
        colors = ['#FF6B6B' if i == 0 else '#4ECDC4' if v == max(values) else '#95E1D3' 
                  for i, v in enumerate(values)]
        
        bars = ax.bar(names, values, color=colors)
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Anotar valores
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Comparación de Arquitecturas MLP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / "improved_mlp_comparison.png", dpi=150)
    plt.close()
    
    print(f"\n💾 Resultados guardados en: {OUT.resolve()}")
    print(f"   - improved_mlp_{best_result['model_name']}.h5 (mejor modelo)")
    print(f"   - improved_mlp_report.txt")
    print(f"   - improved_mlp_comparison.png")
    
    print("\n✅ Experimento completado!")



