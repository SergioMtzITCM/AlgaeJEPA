import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from typing import Tuple
from torch.utils.data import DataLoader

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

        
