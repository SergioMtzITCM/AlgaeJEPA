import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Tuple
from sklearn.model_selection import train_test_split

class _MyDataset(Dataset):
    def __init__(self,
                 h5_path: str,          # Ahora recibe la ruta del archivo .h5
                 target_size: int = None,
                 mode: str = "pretrain",
                 csv_path: str = None) -> None:
        
        self.h5_path = h5_path
        self.target_size = target_size
        self.mode = mode
        self.csv_path = csv_path
        
        # Punteros para carga perezosa (lazy loading) en multiprocessing
        self.h5_file = None
        self.images_dataset = None

        # Determinar el tamaño del dataset de manera rápida leyendo la metadata
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['images'].shape[0]
            
        if self.mode in ["test", "test_distillation"]:
            if csv_path is None:
                raise ValueError(f"In '{self.mode} mode' 'csv_path' must be provided...")
            self.labels_df = pd.read_csv(csv_path)
            
            # Validación de seguridad:
            if len(self.labels_df) != self.length:
                raise ValueError("El número de filas en el CSV no coincide con el número de imágenes en el HDF5.")

        if self.target_size is not None:
            self.resize = transforms.Resize(self.target_size, antialias=True)
        else:
            self.resize = None

    def _open_hdf5(self):
        """Abre el archivo HDF5 solo cuando un worker lo necesita (Lazy Loading)."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.images_dataset = self.h5_file['images']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Asegurar que el archivo HDF5 está abierto en el worker actual
        self._open_hdf5()

        # 2. Leer directamente el array numpy desde el disco a RAM
        image_np = self.images_dataset[idx]

        # 3. Convertir a Tensor
        image_tensor = torch.from_numpy(image_np)

        # 4. Lógica de modos
        if self.mode == "pretrain":
            return image_tensor

        elif self.mode == "test":
            label = self.labels_df["Class"].iloc[idx]
            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor

        elif self.mode == "distillation":
            if self.resize is None:
                raise ValueError("In 'distillation mode' 'target_size' is required...")
            image_resized = self.resize(image_tensor)
            return image_tensor, image_resized
            
        elif self.mode == "test_distillation":
            if self.resize is None:
                raise ValueError("In 'test_distillation mode' 'target_size' is required...")
            label = self.labels_df["Class"].iloc[idx]
            label_tensor = torch.tensor(label, dtype=torch.long)
            image_resized = self.resize(image_tensor)
            return image_resized, label_tensor

        else:
            raise ValueError(f"'{self.mode}' not recognized")


def get_pretrain_dataloaders(h5_path_pretrain: str,
                             h5_path_test: str,
                             test_csv: str,
                             batch_size: int = 32,
                             shuffle_train: bool = True,
                             num_workers: int = 4,
                             prefetch_factor: int = 3) -> Tuple[DataLoader, DataLoader]:

    pretrain_dataset = _MyDataset(h5_path = h5_path_pretrain, mode = "pretrain")
    test_dataset = _MyDataset(h5_path = h5_path_test, mode = "test", csv_path = test_csv)

    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size = batch_size, shuffle = shuffle_train,
        num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor
    )

    test_loader = DataLoader(
        test_dataset, batch_size = batch_size, shuffle = False,
        num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor
    )

    return pretrain_loader, test_loader

def get_distillation_dataloaders(h5_path_pretrain: str,
                                 h5_path_test: str,
                                 test_csv: str,
                                 target_size: int,
                                 batch_size: int = 32,
                                 shuffle_train: bool = True,
                                 num_workers: int = 4,
                                 prefetch_factor: int = 3) -> Tuple[DataLoader, DataLoader]:

    # Pretraining distillation dataset
    distillation_train_dataset = _MyDataset(
        h5_path = h5_path_pretrain,
        target_size = target_size,
        mode = "distillation"
    )

    # Test dataset
    test_dataset = _MyDataset(
        h5_path = h5_path_test,
        mode = "test_distillation",
        target_size = target_size,
        csv_path = test_csv
    )

    train_loader = DataLoader(
        distillation_train_dataset,
        batch_size = batch_size,
        shuffle = shuffle_train,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = prefetch_factor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = prefetch_factor
    )

    return train_loader, test_loader

def get_knn_dataloaders(h5_path_test: str,
                        test_csv: str,
                        num_shots: int,
                        mode: str = "test",
                        target_size: int = None,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        prefetch_factor: int = 3,
                        seed: int = 42,
                        ) -> Tuple[DataLoader, DataLoader]:

    full_dataset = _MyDataset(
        h5_path = h5_path_test,
        mode = mode,
        csv_path = test_csv,
        target_size = target_size
    )

    df = pd.read_csv(test_csv)

    num_classes = df["Class"].nunique()
    total_train_samples = num_shots * num_classes

    train_idx, test_idx = train_test_split(
        df.index.values,
        train_size = total_train_samples,
        stratify = df["Class"].values,
        random_state = seed
    )

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = prefetch_factor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = prefetch_factor
    )

    return train_loader, test_loader

