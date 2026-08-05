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

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Inference
            outputs = model(images)
            gap_features = outputs.mean(dim = 1)

            all_features.append(gap_features.cpu().numpy())
            all_labels.append(labels)

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

    print(f"Train KNN (k={n_neighbors}) with {X_train.shape[-1]} dims...")
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, metric = metric)
    knn.fit(X_train, y_train)

    print("Making Predictions...")
    y_pred = knn.predict(X_test)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, digits = 4))
    print("="*50)

        