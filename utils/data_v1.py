import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Tuple

from sklearn.model_selection import train_test_split

class _MyDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 target_size: int = None,
                 mode: str = "pretrain",
                 csv_path: str = None) -> None:
        
        self.data_dir = data_dir
        self.target_size = target_size
        self.mode = mode
        self.csv_path = csv_path

        # Obtain the .npy files in the directory
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        
        if mode == "test":
            self.filenames.sort(key = lambda x: int(x.split(".")[0]))
            if csv_path is None:
                raise ValueError("In 'test mode' 'cvs_path' must be provided...")

            # Load the csv file
            self.labels_df = pd.read_csv(csv_path)

        elif mode == "test_distillation":
            self.filenames.sort(key = lambda x: int(x.split(".")[0]))

            if csv_path is None:
                raise ValueError("In 'test mode' 'cvs_path' must be provided...")

            # Load the csv file
            self.labels_df = pd.read_csv(csv_path)
            

        if self.target_size is not None:
            self.resize = transforms.Resize(self.target_size)
        else:
            self.resize = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load .npy file
        fname = self.filenames[idx]
        filepath = os.path.join(self.data_dir, fname)
        image = np.load(filepath)

        # Convert to pytorch tensor (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        # Output modes
        if self.mode == "pretrain":
            return image_tensor

        elif self.mode == "test":
            label = self.labels_df["Class"][idx]
            label_tensor = torch.tensor(label, dtype = torch.int32)
            return image_tensor, label_tensor

        elif self.mode == "distillation":
            if self.resize is None:
                raise ValueError("In 'distillation mode' 'target_size' must be required...")

            image_resized = self.resize(image_tensor)
            return image_tensor, image_resized
            
        elif self.mode == "test_distillation":
            if self.resize is None:
                raise ValueError("In 'distillation mode' 'target_size' must be required...")

            label = self.labels_df["Class"][idx]
            label_tensor = torch.tensor(label, dtype = torch.int32)
            image_resized = self.resize(image_tensor)
            return image_resized, label_tensor

        else:
            raise ValueError(f"'{self.mode}' not recognized")

def get_pretrain_dataloaders(pretrain_dir: str,
                             test_dir: str,
                             test_csv: str,
                             batch_size: int = 32,
                             shuffle_train: bool = True,
                             num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:

    # Pretrain dataset
    pretrain_dataset = _MyDataset(
        data_dir = pretrain_dir,
        mode = "pretrain"
    )

    # Test dataset
    test_dataset = _MyDataset(
        data_dir = test_dir,
        mode = "test",
        csv_path = test_csv
    )

    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size = batch_size,
        shuffle = shuffle_train,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = 3
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = 3
    )

    return pretrain_loader, test_loader

def get_distillation_dataloaders(pretrain_dir: str,
                                 test_dir: str,
                                 test_csv: str,
                                 target_size: int,
                                 batch_size: int = 32,
                                 shuffle_train: bool = True,
                                 num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:

    # Pretraining distillation dataset
    distillation_train_dataset = _MyDataset(
        data_dir = pretrain_dir,
        target_size = target_size,
        mode = "distillation"
    )

    # Test dataset
    test_dataset = _MyDataset(
        data_dir = test_dir,
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
        prefetch_factor = 3
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = 2
    )

    return train_loader, test_loader


def get_knn_dataloaders(test_dir: str,
                        test_csv: str,
                        num_shots: int,
                        mode: str = "test",
                        target_size: int = None,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        seed: int = 42,
                        ) -> Tuple[DataLoader, DataLoader]:

    full_dataset = _MyDataset(
        data_dir = test_dir,
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
        prefetch_factor = 3
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = 3
    )

    return train_loader, test_loader
