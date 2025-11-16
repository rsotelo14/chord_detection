# evaluate_dnn_frames.py
"""
Script para evaluar los modelos DNN entrenados con train_dnn_frames.py
Calcula accuracy, macro F1 y balanced accuracy en train, validation y test sets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
import tensorflow as tf

# Configuración (debe coincidir con train_dnn_frames.py)
IN_NPZ = Path("frames_dataset.npz")
IN_CSV = Path("frames_dataset.csv")
OUT = Path("analysis_out_frames")
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42

def load_data():
    """Carga los datos del dataset de frames."""
    z = np.load(IN_NPZ, allow_pickle=True)
    X = z["X"].astype(np.float32)        # (N, F)
    y = z["y"]                            # str labels
    groups = z["groups"]                  # album_track
    df = pd.read_csv(IN_CSV)
    return X, y, groups, df

def split_groups(X, y, groups, test_size=0.25, seed=RANDOM_STATE):
    """Divide los datos en train y test usando GroupShuffleSplit."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    return tr_idx, te_idx

def evaluate_model(model, X, y_true, label_encoder, split_name, model_name):
    """
    Evalúa un modelo y retorna métricas.
    
    Args:
        model: modelo de Keras cargado
        X: features (N, F)
        y_true: labels verdaderos (índices numéricos)
        label_encoder: LabelEncoder para convertir índices a strings
        split_name: nombre del split ('train', 'val', 'test')
        model_name: nombre del modelo ('common' o 'bottleneck')
    
    Returns:
        dict con métricas: accuracy, f1_macro, balanced_accuracy
    """
    # Predicciones
    P = model.predict(X, verbose=0)  # (N, K)
    y_pred = np.argmax(P, axis=1)    # (N,)
    
    # Métricas
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    # Classification report
    y_true_str = label_encoder.inverse_transform(y_true)
    y_pred_str = label_encoder.inverse_transform(y_pred)
    rep = classification_report(y_true_str, y_pred_str, digits=3, zero_division=0)
    
    return {
        'accuracy': acc,
        'f1_macro': f1m,
        'balanced_accuracy': bacc,
        'report': rep,
        'y_true': y_true,
        'y_pred': y_pred
    }

def main():
    print("=" * 70)
    print("Evaluación de modelos DNN (frames)")
    print("=" * 70)
    
    # 1) Cargar datos
    print("\n📂 Cargando datos...")
    X, y_str, groups, dfmeta = load_data()
    print(f"   Total de frames: {len(X)}")
    print(f"   Dimensiones de features: {X.shape[1]}")
    
    # 2) Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = le.classes_
    num_classes = len(classes)
    print(f"   Número de clases: {num_classes}")
    
    # Verificar que el mapeo de labels coincide
    label_mapping_path = OUT / "label_mapping.txt"
    if label_mapping_path.exists():
        saved_classes = np.loadtxt(label_mapping_path, dtype=str)
        if not np.array_equal(classes, saved_classes):
            print(f"   ⚠️  Advertencia: las clases no coinciden con las guardadas")
        else:
            print(f"   ✅ Mapeo de labels verificado")
    
    # 3) Split train/test (mismo que en train_dnn_frames.py)
    print("\n🔀 Dividiendo datos en train/test...")
    tr_idx, te_idx = split_groups(X, y, groups, test_size=0.25, seed=RANDOM_STATE)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    groups_tr = groups[tr_idx]
    
    print(f"   Train: {len(X_tr)} frames")
    print(f"   Test:  {len(X_te)} frames")
    
    # 4) Split validation dentro de train (mismo que en train_dnn_frames.py)
    print("\n🔀 Dividiendo train en train_sub/validation...")
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_sub, val_idx = next(gss_val.split(X_tr, y_tr, groups=groups_tr))
    X_tr_sub, y_tr_sub = X_tr[tr_sub], y_tr[tr_sub]
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]
    
    print(f"   Train sub: {len(X_tr_sub)} frames")
    print(f"   Validation: {len(X_val)} frames")
    
    # 5) Cargar modelos
    model_common_path = OUT / "dnn_common.h5"
    model_bottleneck_path = OUT / "dnn_bottleneck.h5"
    
    models_to_eval = {}
    if model_common_path.exists():
        print(f"\n📦 Cargando modelo: {model_common_path.name}")
        models_to_eval['common'] = tf.keras.models.load_model(model_common_path)
    else:
        print(f"\n⚠️  No se encontró: {model_common_path}")
    
    if model_bottleneck_path.exists():
        print(f"📦 Cargando modelo: {model_bottleneck_path.name}")
        models_to_eval['bottleneck'] = tf.keras.models.load_model(model_bottleneck_path)
    else:
        print(f"⚠️  No se encontró: {model_bottleneck_path}")
    
    if not models_to_eval:
        print("\n❌ No se encontraron modelos para evaluar.")
        return
    
    # 6) Evaluar cada modelo en cada split
    print("\n" + "=" * 70)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 70)
    
    all_results = []
    
    for model_name, model in models_to_eval.items():
        print(f"\n{'='*70}")
        print(f"Modelo: {model_name.upper()}")
        print(f"{'='*70}")
        
        # Evaluar en train_sub
        print(f"\n📊 Evaluando en TRAIN_SUB...")
        results_train = evaluate_model(model, X_tr_sub, y_tr_sub, le, 'train', model_name)
        print(f"   Accuracy:           {results_train['accuracy']:.4f}")
        print(f"   Macro F1:           {results_train['f1_macro']:.4f}")
        print(f"   Balanced Accuracy:  {results_train['balanced_accuracy']:.4f}")
        
        # Evaluar en validation
        print(f"\n📊 Evaluando en VALIDATION...")
        results_val = evaluate_model(model, X_val, y_val, le, 'val', model_name)
        print(f"   Accuracy:           {results_val['accuracy']:.4f}")
        print(f"   Macro F1:           {results_val['f1_macro']:.4f}")
        print(f"   Balanced Accuracy:  {results_val['balanced_accuracy']:.4f}")
        
        # Evaluar en test
        print(f"\n📊 Evaluando en TEST...")
        results_test = evaluate_model(model, X_te, y_te, le, 'test', model_name)
        print(f"   Accuracy:           {results_test['accuracy']:.4f}")
        print(f"   Macro F1:           {results_test['f1_macro']:.4f}")
        print(f"   Balanced Accuracy:  {results_test['balanced_accuracy']:.4f}")
        
        # Guardar resultados
        all_results.append({
            'model': model_name,
            'split': 'train',
            'accuracy': results_train['accuracy'],
            'f1_macro': results_train['f1_macro'],
            'balanced_accuracy': results_train['balanced_accuracy'],
            'n_samples': len(X_tr_sub)
        })
        all_results.append({
            'model': model_name,
            'split': 'validation',
            'accuracy': results_val['accuracy'],
            'f1_macro': results_val['f1_macro'],
            'balanced_accuracy': results_val['balanced_accuracy'],
            'n_samples': len(X_val)
        })
        all_results.append({
            'model': model_name,
            'split': 'test',
            'accuracy': results_test['accuracy'],
            'f1_macro': results_test['f1_macro'],
            'balanced_accuracy': results_test['balanced_accuracy'],
            'n_samples': len(X_te)
        })
        
        # Guardar classification reports
        report_train_path = OUT / f"report_{model_name}_train.txt"
        report_val_path = OUT / f"report_{model_name}_validation.txt"
        report_test_path = OUT / f"report_{model_name}_test.txt"
        
        report_train_path.write_text(f"Classification Report - {model_name} - TRAIN\n" + 
                                     "="*70 + "\n\n" + results_train['report'])
        report_val_path.write_text(f"Classification Report - {model_name} - VALIDATION\n" + 
                                   "="*70 + "\n\n" + results_val['report'])
        report_test_path.write_text(f"Classification Report - {model_name} - TEST\n" + 
                                    "="*70 + "\n\n" + results_test['report'])
        
        print(f"\n💾 Reports guardados:")
        print(f"   - {report_train_path.name}")
        print(f"   - {report_val_path.name}")
        print(f"   - {report_test_path.name}")
    
    # 7) Guardar resumen en CSV
    df_results = pd.DataFrame(all_results)
    csv_path = OUT / "evaluation_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Resumen de métricas guardado en: {csv_path}")
    
    # 8) Mostrar tabla resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE MÉTRICAS")
    print("=" * 70)
    print("\n" + df_results.to_string(index=False))
    
    print("\n✅ Evaluación completada!")

if __name__ == "__main__":
    main()

