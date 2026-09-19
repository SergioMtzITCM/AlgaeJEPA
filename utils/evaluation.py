import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, Dict, Union
from torch.utils.data import DataLoader

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.vit_core import ViTModel
from models.microvit_core import MicroViTModel
from models.mobilenet_core import MobileNetModel
from models.resnet_core import ResNetModel



def _extract_features(model: nn.Module,
                      dataloader: DataLoader,
                      device: torch.device) -> Tuple[np.ndarray, np.ndarray]:

    model.eval()
    all_features = []
    all_labels = []

    expected_out_shape = model.get_output_size() if hasattr(model, "get_output_size") else None

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Inference
            outputs = model(images)

            if outputs.ndim == 4:
                # MobileNet / MicroViT shape: (Batch, Channels, H, W)
                gap_features = outputs.mean(dim = (-2, -1))

            elif outputs.ndim == 3:
                # If MicroViT shape without batch (Channels, H, W) or
                # ViT with batch (Batch, Seq_Len, Hidden_Dim)
                if expected_out_shape is not None and len(expected_out_shape) == 3 and outputs.shape == torch.Size(expected_out_shape):
                    outputs_batched = outputs.unsqueeze(0)
                    gap_features = outputs_batched.mean(dim = (-2, -1))
                else:
                    # ViT Case: (Batch, Seq_Len, Hidden_Dim)
                    gap_features = outputs.mean(dim = 1)

            elif outputs.ndim == 2:
                gap_features = outputs
            else:
                raise ValueError(f"Output tensor shape is not supported: {outputs.shape}")


            all_features.append(gap_features.cpu().numpy())

            if isinstance(labels, torch.Tensor):
                all_labels.append(labels.cpu().numpy())
            else:
                all_labels.append(np.array(labels))

    features_np = np.concatenate(all_features, axis = 0)
    labels_np = np.concatenate(all_labels, axis = 0)

    return features_np, labels_np


def evaluate_model_knn(model: nn.Module,
                       train_dataloader: DataLoader,
                       test_dataloader: DataLoader,
                       device: torch.device,
                       n_neighbors: int = 5,
                       metric: str = "euclidean") -> None:

    metric = metric.lower()
    if metric not in ["euclidean", "manhattan", "cosine"]:
        raise ValueError("metric must be 'euclidean', 'manhattan' or 'cosine'")

    X_train, y_train = _extract_features(model, train_dataloader, device)
    X_test, y_test = _extract_features(model, test_dataloader, device)

    print(f"Train KNN (k={n_neighbors}) with {X_train.shape[-1]} feature dim...")
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, metric = metric)
    knn.fit(X_train, y_train)

    print("Making Predictions...")
    y_pred = knn.predict(X_test)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, digits = 4))
    print("="*50)


@torch.no_grad()
def evaluate_and_report(model: Union[ViTModel, MicroViTModel, ResNetModel, MobileNetModel],
                        dataloader: DataLoader,
                        label_map: Dict[str, int],
                        device: torch.device,
                        save_dir: str,
                        prefix: str) -> Tuple[float, float]:

    """
    Evalúa el modelo y genera reportes detallados y matrices de confusión.
    prefix: ejemplo. 'IID_Test' o 'ODD_Test'
    """

    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim = 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 1. Identificar las clases presentes en el conjunto de datos
    present_labels = np.unique(np.concatenate((all_labels, all_preds)))

    # 2. Mapeo inverso ajustado a las clases presentes
    inverse_label_map = {v: k for k, v in label_map.items()}
    target_names = [str(inverse_label_map[i]) for i in present_labels]

    # 3. Classification Report
    report_dict = classification_report(
            all_labels, 
            all_preds, 
            labels = present_labels,
            target_names = target_names, 
            output_dict = True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(save_dir, f"{prefix}_classification_report.csv"))

    # 4. Matriz de Confusion
    cm = confusion_matrix(all_labels, all_preds, labels = present_labels)

    sns.set_theme(
        context = "paper", 
        style = "white",  
        font = "sans-serif", 
        font_scale = 1.1
    )

    # 5. Ajuste dinámico de la figura según la cantidad de clases reales
    fig_size = max(9, len(present_labels) * 0.45)
    fig, ax = plt.subplots(figsize = (fig_size, fig_size * 0.85))

    # 6. Máscara para ocultar celdas en cero
    mask = cm == 0

    sns.heatmap(
            cm, 
            annot = True, 
            fmt = "d", 
            cmap = "Blues", 
            mask = mask,
            xticklabels = target_names, 
            yticklabels = target_names,
            linewidths = 0.5,
            linecolor = "white",
            annot_kws = {"size": 10, "weight": "bold"},
            cbar_kws = {"shrink": 0.8, "label": "Número de Muestras"},
            ax = ax)
    
    ax.set_title(f'Matriz de Confusión - {prefix}', fontweight = "bold", fontsize = 14, pad = 20)
    ax.set_ylabel('Etiqueta Verdadera', fontweight = "bold", fontsize = 12)
    ax.set_xlabel('Predicción del Modelo', fontweight = "bold", fontsize = 12)

    plt.xticks(rotation = 45, ha = "right", fontweight = "bold", fontsize = 10)
    plt.yticks(rotation = 0, fontweight = "bold", fontsize = 10)

    for _, spine in ax.spines.items():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_confusion_matrix.png"), dpi = 300, bbox_inches = 'tight')
    plt.close(fig)
        
    return report_dict["accuracy"], report_dict["macro avg"]["f1-score"]


        
