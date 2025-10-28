#!/usr/bin/env python3
"""
Script para registrar automáticamente resultados de experimentos
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import json

class ExperimentLogger:
    def __init__(self, log_file="experiment_log.json"):
        self.log_file = Path(log_file)
        self.experiments = self.load_experiments()
    
    def load_experiments(self):
        """Carga experimentos existentes"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_experiments(self):
        """Guarda experimentos a archivo JSON"""
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def log_experiment(self, 
                      experiment_name,
                      accuracy=None,
                      macro_f1=None,
                      balanced_acc=None,
                      wcsr_independent=None,
                      wcsr_by_song=None,
                      notes="",
                      config=None):
        """Registra un nuevo experimento"""
        
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "metrics": {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "balanced_acc": balanced_acc,
                "wcsr_independent": wcsr_independent
            },
            "wcsr_by_song": wcsr_by_song or {},
            "notes": notes,
            "config": config or {}
        }
        
        self.experiments.append(experiment)
        self.save_experiments()
        
        print(f"✅ Experimento '{experiment_name}' registrado")
        return experiment
    
    def get_best_wcsr(self):
        """Obtiene el mejor WCSR independiente"""
        best_wcsr = 0
        best_exp = None
        
        for exp in self.experiments:
            wcsr = exp["metrics"].get("wcsr_independent")
            if wcsr and wcsr > best_wcsr:
                best_wcsr = wcsr
                best_exp = exp
        
        return best_wcsr, best_exp
    
    def print_summary(self):
        """Imprime resumen de todos los experimentos"""
        print("\n" + "="*80)
        print("📊 RESUMEN DE EXPERIMENTOS")
        print("="*80)
        
        for i, exp in enumerate(self.experiments, 1):
            metrics = exp["metrics"]
            print(f"\n{i}. {exp['experiment_name']}")
            print(f"   📅 {exp['timestamp']}")
            
            if metrics["accuracy"]:
                print(f"   🎯 Accuracy: {metrics['accuracy']:.3f}")
            if metrics["macro_f1"]:
                print(f"   📈 Macro F1: {metrics['macro_f1']:.3f}")
            if metrics["balanced_acc"]:
                print(f"   ⚖️  Balanced Acc: {metrics['balanced_acc']:.3f}")
            if metrics["wcsr_independent"]:
                print(f"   🎵 WCSR Independiente: {metrics['wcsr_independent']:.3f}")
            
            if exp["notes"]:
                print(f"   📝 Notas: {exp['notes']}")
        
        # Mejor resultado
        best_wcsr, best_exp = self.get_best_wcsr()
        if best_exp:
            print(f"\n🏆 MEJOR RESULTADO:")
            print(f"   Experimento: {best_exp['experiment_name']}")
            print(f"   WCSR: {best_wcsr:.3f}")
    
    def export_to_csv(self, output_file="experiment_results.csv"):
        """Exporta resultados a CSV"""
        data = []
        for exp in self.experiments:
            row = {
                "experiment_name": exp["experiment_name"],
                "timestamp": exp["timestamp"],
                "accuracy": exp["metrics"].get("accuracy"),
                "macro_f1": exp["metrics"].get("macro_f1"),
                "balanced_acc": exp["metrics"].get("balanced_acc"),
                "wcsr_independent": exp["metrics"].get("wcsr_independent"),
                "notes": exp["notes"]
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"✅ Resultados exportados a {output_file}")


def main():
    """Función principal para usar desde línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Log de experimentos de chord detection")
    parser.add_argument("--name", required=True, help="Nombre del experimento")
    parser.add_argument("--accuracy", type=float, help="Accuracy")
    parser.add_argument("--macro-f1", type=float, help="Macro F1")
    parser.add_argument("--balanced-acc", type=float, help="Balanced Accuracy")
    parser.add_argument("--wcsr", type=float, help="WCSR Independiente")
    parser.add_argument("--notes", help="Notas del experimento")
    
    args = parser.parse_args()
    
    logger = ExperimentLogger()
    logger.log_experiment(
        experiment_name=args.name,
        accuracy=args.accuracy,
        macro_f1=args.macro_f1,
        balanced_acc=args.balanced_acc,
        wcsr_independent=args.wcsr,
        notes=args.notes
    )
    logger.print_summary()


if __name__ == "__main__":
    main()

