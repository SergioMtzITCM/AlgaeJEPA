import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Tuple, Optional, Union, Dict
from sklearn.model_selection import train_test_split

class _MyDataset(Dataset):
    def __init__(self,
                 h5_path: str,
                 dataframe: Optional[pd.DataFrame] = None,
                 csv_path: Optional[str] = None,
                 label_map: Optional[Dict[Union[int, str], int]] = None,
                 mode: str = "pretrain",
                 target_size: Optional[Union[int, Tuple[int, int]]] = None,
                 noise_percentage: float = 0.0,
                 noise_seed: int = 42) -> None:
        
        self.h5_path = h5_path
        self.mode = mode
        self.target_size = target_size
        self.noise_percentage = float(noise_percentage)
        self.noise_seed = noise_seed

        # 1. Cargar Metadatos si existen
        if dataframe is not None:
            self.df = dataframe.copy().reset_index(drop=True)
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path).reset_index(drop=True)
        else:
            self.df = None

        # 2. Normalización de Nombre de Columna de Etiquetas
        if self.df is not None:
            label_col = None
            for col in ["label", "Class", "species"]:
                if col in self.df.columns:
                    label_col = col
                    break
            
            if label_col is None and self.mode == "labeled":
                raise KeyError("El DataFrame/CSV debe contener una columna de etiquetas ('label', 'Class' o 'species').")
            
            if label_col is not None and label_col != "label":
                self.df.rename(columns = {label_col: "label"}, inplace = True)

        # 3. Lógica de Mapeo de Etiquetas (IDD / ODD)
        if self.mode == "labeled" and self.df is not None:
            if label_map is None:
                # Modo Entrenamiento (IDD): Construir diccionario ordenado
                unique_labels = sorted(self.df["label"].unique())
                self.label_map = {orig: new_idx for new_idx, orig in enumerate(unique_labels)}
            else:
                # Modo Validación / ODD / Test: Heredar diccionario y filtrar clases no vistas
                self.label_map = label_map
                known_classes = self.df["label"].isin(self.label_map.keys())
                self.df = self.df[known_classes].reset_index(drop = True)

        # 4. Determinar la Longitud del Dataset
        if self.df is not None:
            self.length = len(self.df)
        else:
            with h5py.File(self.h5_path, 'r') as f:
                self.length = f['images'].shape[0]

        # 5. Transformación de Redimensionamiento
        if self.target_size is not None:
            self.resize = transforms.Resize(self.target_size, antialias = True)
        else:
            self.resize = None

        # Punteros para Carga Perezosa (Lazy Loading) en Multiprocessing
        self.h5_file = None
        self.images_dataset = None

    def _open_hdf5(self):
        """Abre el archivo HDF5 solo cuando un worker lo necesita (Lazy Loading)."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.images_dataset = self.h5_file['images']

    def _apply_gaussian_noise(self, tensor: torch.Tensor, sample_idx: int) -> torch.Tensor:
        """Aplica ruido gaussiano aditivo determinista por muestra."""
        if self.noise_percentage <= 0.0:
            return tensor

        # Generador de números aleatorios aislado con semilla fija por muestra
        generator = torch.Generator()
        generator.manual_seed(self.noise_seed + int(sample_idx))

        noise = torch.randn(tensor.shape, generator = generator, dtype = tensor.dtype) * self.noise_percentage
        return tensor + noise

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Asegurar que el archivo HDF5 está abierto en el worker actual
        self._open_hdf5()

        # 2. Obtener Índice Real en el Archivo HDF5
        if self.df is not None and "h5_index" in self.df.columns:
            real_h5_idx = int(self.df.iloc[idx]["h5_index"])
        else:
            real_h5_idx = idx

        # 3. Cargar Imagen desde HDF5
        image_np = self.images_dataset[real_h5_idx]
        image_tensor = torch.from_numpy(image_np).to(torch.float32)

        # 4. Aplicar Ruido Gaussiano (si aplica)
        image_tensor = self._apply_gaussian_noise(image_tensor, sample_idx=real_h5_idx)


        # 5. Modos de Salida
        if self.mode == "pretrain":
            return image_tensor

        elif self.mode == "distillation":
            if self.resize is None:
                raise ValueError("En 'distillation mode' se requiere especificar 'target_size'.")
            image_resized = self.resize(image_tensor)
            return image_tensor, image_resized

        elif self.mode == "labeled":
            orig_label = self.df.iloc[idx]["label"]
            mapped_label = self.label_map[orig_label] if hasattr(self, "label_map") else orig_label
            label_tensor = torch.tensor(mapped_label, dtype = torch.long)

            if self.resize is not None:
                image_resized = self.resize(image_tensor)
                return image_resized, label_tensor
            
            return image_tensor, label_tensor

        else:
            raise ValueError(f"Modo '{self.mode}' no reconocido.")

def get_pretrain_dataloaders(h5_path_pretrain: str,
                             h5_path_test: Optional[str] = None,
                             test_csv: Optional[str] = None,
                             batch_size: int = 32,
                             shuffle_train: bool = True,
                             num_workers: int = 4,
                             prefetch_factor: int = 3) -> Tuple[DataLoader, DataLoader]:
    """Genera DataLoaders para Pre-entrenamiento y visualización opcional de test (IDD o ODD)."""

    pretrain_dataset = _MyDataset(h5_path = h5_path_pretrain, mode = "pretrain")
    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size = batch_size, shuffle = shuffle_train,
        num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor
    )

    test_loader = None
    if h5_path_test is not None and test_csv is not None:
        test_dataset = _MyDataset(h5_path = h5_path_test, mode = "labeled", csv_path = test_csv)
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, shuffle = False,
            num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor
        )

    return pretrain_loader, test_loader

def get_distillation_dataloaders(h5_path_pretrain: str,
                                 target_size: Union[int, Tuple[int, int]],
                                 h5_path_test: Optional[str] = None,
                                 test_csv: Optional[str] = None,
                                 batch_size: int = 32,
                                 shuffle_train: bool = True,
                                 num_workers: int = 4,
                                 prefetch_factor: int = 3) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Genera DataLoaders para Destilación de Conocimiento y evaluación visual opcional."""

    distill_dataset = _MyDataset(h5_path = h5_path_pretrain, mode = "distillation", target_size = target_size)
    train_loader = DataLoader(
        distill_dataset, batch_size = batch_size, shuffle = shuffle_train,
        num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor
    )

    test_loader = None
    if h5_path_test is not None and test_csv is not None:
        test_dataset = _MyDataset(h5_path = h5_path_test, csv_path = test_csv, mode = "labeled", target_size = target_size)
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, shuffle = False,
            num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor
        )

    return train_loader, test_loader

def get_in_distribution_dataset(h5_path: str,
                                csv_path: str,
                                train_size: float = 0.7,
                                train_fraction: float = 1.0,
                                batch_size: int = 32,
                                seed: int = 42,
                                noise_percentage: float = 0.0,
                                noise_seed: int = 42,
                                num_workers: int = 4,
                                prefetch_factor: int = 3) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Genera DataLoaders de Train, Val y Test para In-Distribution (IDD)."""
    df = pd.read_csv(csv_path)

    # Identificar columna de etiquetas
    label_col = next(col for col in ["label", "Class", "species"] if col in df.columns)

    # Partición 1: Train y Val+Test
    train_df, val_test_df = train_test_split(
        df, train_size = train_size, stratify = df[label_col], random_state = seed
    )

    # Partición 2: Val y Test
    val_df, test_df = train_test_split(
        val_test_df, test_size = 0.5, stratify = val_test_df[label_col], random_state = seed
    )

    # Reducción opcional de conjunto de entrenamiento
    if train_fraction < 1.0:
        train_df, _ = train_test_split(
            train_df, train_size = train_fraction, stratify = train_df[label_col], random_state = seed
        )

    # Datasets
    train_dataset = _MyDataset(
        h5_path = h5_path, dataframe = train_df, mode = "labeled",
        noise_percentage = noise_percentage, noise_seed = noise_seed
    )
    master_label_map = train_dataset.label_map

    val_dataset = _MyDataset(
        h5_path = h5_path, dataframe = val_df, label_map = master_label_map, mode = "labeled",
        noise_percentage = noise_percentage, noise_seed = noise_seed
    )
    test_dataset = _MyDataset(
        h5_path = h5_path, dataframe = test_df, label_map = master_label_map, mode = "labeled",
        noise_percentage = noise_percentage, noise_seed = noise_seed
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                              num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False,
                            num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False,
                             num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor)

    return train_loader, val_loader, test_loader, master_label_map


def get_out_distribution_dataset(h5_path: str,
                                 csv_path: str,
                                 label_map: dict,
                                 batch_size: int = 32,
                                 noise_percentage: float = 0.0,
                                 noise_seed: int = 42,
                                 num_workers: int = 4,
                                 prefetch_factor: int = 3) -> DataLoader:
    """Genera el DataLoader para Out-of-Distribution (ODD), alineando etiquetas con IDD."""
    df = pd.read_csv(csv_path)

    ood_dataset = _MyDataset(
        h5_path = h5_path, dataframe = df, label_map = label_map, mode = "labeled",
        noise_percentage = noise_percentage, noise_seed = noise_seed
    )

    return DataLoader(
        ood_dataset, batch_size = batch_size, shuffle = False,
        num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor
    )



def get_knn_dataloaders(h5_path: str,
                        csv_path: str,
                        num_shots: int,
                        label_map: Optional[dict] = None,
                        target_size: Optional[Union[int, Tuple[int, int]]] = None,
                        batch_size: int = 32,
                        noise_percentage: float = 0.0,
                        noise_seed: int = 42,
                        seed: int = 42,
                        num_workers: int = 4,
                        prefetch_factor: int = 3) -> Tuple[DataLoader, DataLoader]:
    """
    Genera DataLoaders Few-Shot para evaluación kNN (funciona tanto con IDD como con ODD).
    """
    full_dataset = _MyDataset(
        h5_path = h5_path, csv_path = csv_path, label_map = label_map, mode = "labeled",
        target_size = target_size, noise_percentage = noise_percentage, noise_seed = noise_seed
    )

    df = full_dataset.df
    label_col = "label"

    num_classes = df[label_col].nunique()
    total_train_samples = num_shots * num_classes

    train_idx, test_idx = train_test_split(
        df.index.values,
        train_size = total_train_samples,
        stratify = df[label_col].values,
        random_state = seed
    )

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                              num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False,
                             num_workers = num_workers, pin_memory = True, prefetch_factor = prefetch_factor)

    return train_loader, test_loader

