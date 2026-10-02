import os
import random
import numpy as np
import pandas as pd

from trainer import ClassificationTrainer
from models.vit_core import ViTModel
from models.microvit_core import MicroViTModel
from models.mobilenet_core import MobileNetModel
from models.resnet_core import ResNetModel

from utils.data import get_in_distribution_dataset, get_out_distribution_dataset
from utils.evaluation import evaluate_and_report
from configs.config import BaseConfig, MicroViTConfig, MobileNetConfig, ResNetConfig
from utils.loader import load_pretrain_encoder, load_student_model

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def freeze_backbone(model: nn.Module, head_name: str = "classifier") -> None:
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(f"{head_name}.")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  -> Layer Enabled for Linear Probing: {name}")

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuración del Benchmark
    data_fractions = [1.0, 0.5, 0.25, 0.10, 0.05]
    seeds = [42, 183, 320, 543, 999]
    noise_percentage = 0.0
    base_save_dir = "./MicroViTS3_Distilled_Benchmark_Results/"

    # Ruta del Checkpoint Maestro o Estudiante a evaluar
    PRETRAINED_CHECKPOINT_PATH = "./MicroViTS3_Student/checkpoints/best_student.pth"
    MODEL_TYPE_TO_LOAD = "microvit" # Teacher (it is a vit), vit, microvit, mobilenet or resnet

    # Hiperparámetros Base
    EPOCHS = 30
    BASE_LR = 1e-3
    MIN_LR = 1e-5
    BETA_1 = 0.9
    BETA_2 = 0.999
    WEIGHT_DECAY = 0.01
    
    BATCH_SIZE = 192
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 3
    HEAD_NAME = "classifier"

    results = []

    for fraction in data_fractions:
        print(f"\n{'='*50}\nEvaluando Escasez de Datos: {fraction*100}%\n{'='*50}")

        frac_metrics = {"idd_acc": [], "idd_f1": [], "ood_acc": [], "ood_f1": []}

        for seed in seeds:
            print(f"\n--- Ejecución con Semilla: {seed} ---")
            set_seed(seed)

            # 1. Preparar Datasets
            train_loader, val_loader, test_loader, master_label_map = get_in_distribution_dataset(
                    h5_path = "./In_Distribution_Dataset/IDD.h5",
                    csv_path = "./In_Distribution_Dataset/IDD_labels.csv",
                    train_fraction = fraction,
                    batch_size = BATCH_SIZE,
                    seed = seed,
                    num_workers = NUM_WORKERS,
                    prefetch_factor = PREFETCH_FACTOR,
                    noise_percentage = noise_percentage,
                    noise_seed = seed
            )

            ood_loader = get_out_distribution_dataset(
                    h5_path = "./Out_Distribution_Dataset/ODD.h5",
                    csv_path = "./Out_Distribution_Dataset/ODD_labels.csv",
                    label_map = master_label_map,
                    batch_size = BATCH_SIZE,
                    num_workers = NUM_WORKERS,
                    prefetch_factor = PREFETCH_FACTOR,
                    noise_percentage = noise_percentage,
                    noise_seed = seed
            )

            num_classes = len(master_label_map)

            # 2. Inicializar Modelo Maestro o Destilado
            print(f"Cargando Pesos Pre-Entrenados desde {PRETRAINED_CHECKPOINT_PATH}...")
            if MODEL_TYPE_TO_LOAD == "Teacher":
                model = load_pretrain_encoder(
                        checkpoint_path = PRETRAINED_CHECKPOINT_PATH,
                        device = device,
                        num_classes = num_classes
                )
            else:
                model = load_student_model(
                        checkpoint_path = PRETRAINED_CHECKPOINT_PATH,
                        model_type = MODEL_TYPE_TO_LOAD,
                        device = device,
                        num_classes = num_classes
                )

            # Congelar Pesos
            freeze_backbone(model)

            model = torch.compile(model)

            # 3. Optimizador y LRScheduler
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr = BASE_LR, betas = (BETA_1, BETA_2), weight_decay = WEIGHT_DECAY
            )

            scheduler = CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = MIN_LR)

            # 4. Configurar Directorios de Guardado Dinámicos
            run_dir = os.path.join(base_save_dir, f"Frac_{fraction}", f"Seed_{seed}")

            # 5. Entrenar
            trainer = ClassificationTrainer(
                    model = model,
                    train_dataloader = train_loader,
                    val_dataloader = val_loader,
                    optimizer = optimizer,
                    lr_scheduler = scheduler,
                    device = device,
                    epochs = EPOCHS,
                    save_dir = run_dir,
                    freeze_backbone = True,
                    head_name = HEAD_NAME
            )
            trainer.train()

            # 6. Cargar el mejor modelo para evaluación
            checkpoint_path = os.path.join(run_dir, "checkpoints", "best_model.pth")
            best_ckpt = torch.load(checkpoint_path, map_location = device, weights_only = False)

            raw_state_dict = best_ckpt["model_state_dict"]

            clean_state_dict = {}
            for key, value in raw_state_dict.items():
                clean_key = key.replace("_orig_mod.", "")
                clean_state_dict[clean_key] = value

            unwrapped_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            unwrapped_model.load_state_dict(clean_state_dict, strict = True)

            # 7. Evaluación y Reporte
            print("Evaluando en IDD Test...")
            idd_acc, idd_f1 = evaluate_and_report(unwrapped_model, test_loader, master_label_map, device, run_dir, "IDD")

            print("Evaluando en ODD Test...")
            odd_acc, odd_f1 = evaluate_and_report(unwrapped_model, ood_loader, master_label_map, device, run_dir, "ODD")

            frac_metrics["idd_acc"].append(idd_acc)
            frac_metrics["idd_f1"].append(idd_f1)
            frac_metrics["ood_acc"].append(odd_acc)
            frac_metrics["ood_f1"].append(odd_f1)

        # 8. Calcular Media y Desviación Estándar para esta fracción
        current_scenario_result = {
            "Fraction": fraction,
            "IDD_Acc_Mean": np.mean(frac_metrics["idd_acc"]),
            "IDD_Acc_Std": np.std(frac_metrics["idd_acc"]),
            "IDD_F1_Mean": np.mean(frac_metrics["idd_f1"]),
            "IDD_F1_Std": np.std(frac_metrics["idd_f1"]),
            "OOD_Acc_Mean": np.mean(frac_metrics["ood_acc"]),
            "OOD_Acc_Std": np.std(frac_metrics["ood_acc"]),
            "OOD_F1_Mean": np.mean(frac_metrics["ood_f1"]),
            "OOD_F1_Std": np.std(frac_metrics["ood_f1"])
        }

        results.append(current_scenario_result)

        # Salvado Intermedio de Respaldo
        df_checkpoint = pd.DataFrame([current_scenario_result])
        checkpoint_file = os.path.join(base_save_dir, f"benchmark_results_frac_{fraction}.csv")
        df_checkpoint.to_csv(checkpoint_file, index = False)
        print(f"\n[!] Salvado intermedio exitoso para la fracción {fraction*100}%.")

    # 9. Guardar resultados finales del Benchmark
    print(f"\n{'='*50}\nGenerando Resultados Finales del Benchmark...\n{'='*50}")
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(base_save_dir, "benchmark_summary.csv"), index = False)
    print("\nBenchmark Finalizado Exitosamente. 'benchmark_summary.csv' guardado.")


if __name__ == "__main__":
    main()
 
