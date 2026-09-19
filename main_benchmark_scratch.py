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

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuración del Benchmark
    data_fractions = [1.0, 0.5, 0.25, 0.10, 0.05]
    seeds = [42, 183, 320, 543, 999]
    noise_percentage = 0.0
    base_save_dir = "./MicroViT_Benchmark_Results"

    # Hiperparámetros Base
    EPOCHS = 30
    BASE_LR = 7.5e-4 # ViT: 7.5e-4, CNN: 1e-3, MicroViT: 7.5e-4
    BETA_1 = 0.9 # ViT: 0.9, CNN: 0.9, MicroViT: 0.9
    BETA_2 = 0.999 # ViT: 0.95, CNN: 0.999, MicroViT: 0.999
    WEIGHT_DECAY = 0.01 # ViT 0.05, CNN: 1e-4, MicroViT: 0.01
    MIN_LR = 1e-5
    WARMUP_EPOCHS = 5 # ViT: 5-7, CNN: 2-3, MicroViT: 5
    START_FACTOR = 0.15 # ViT: 0.1, CNN: 0.2, MicroViT: 0.15

    BATCH_SIZE = 192
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 3

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

            # 2. Inicializar Modelo
            # ViT-Tiny
            vit_config = BaseConfig(
                    hidden_size = 192,
                    num_hidden_layers = 12,
                    num_attention_heads = 3,
                    intermediate_size = 768,
                    hidden_dropout_prob = 0.1,
                    attention_probs_dropout_prob = 0.1,
                    image_size = 224,
                    patch_size = 16,
                    num_channels = 3,
                    num_classes = num_classes
            )

            microvit_config = MicroViTConfig(
                    model = "S3",
                    image_size = 224,
                    num_channels = 3,
                    num_classes = num_classes
            )

            mobilenet_config = MobileNetConfig(
                    model = "V2",
                    image_size = 224,
                    num_channels = 3,
                    num_classes = num_classes
            )

            resnet_config = ResNetConfig(
                    image_size = 224,
                    num_channels = 3,
                    num_classes = num_classes
            )

            model = MicroViTModel(microvit_config) # <- Cambiar por modelo deseado
            model = torch.compile(model)

            # 3. Optimizador y LRScheduler
            optimizer = AdamW(model.parameters(), lr = BASE_LR, betas = (BETA_1,
                                                                        BETA_2), weight_decay = WEIGHT_DECAY)
            warmup = LinearLR(optimizer, start_factor = START_FACTOR, total_iters = WARMUP_EPOCHS)
            cosine = CosineAnnealingLR(optimizer, T_max = (EPOCHS - WARMUP_EPOCHS), eta_min = MIN_LR)
            scheduler = SequentialLR(optimizer, schedulers = [warmup, cosine], milestones = [5])

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
                    save_dir = run_dir
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
 
