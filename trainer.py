import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from typing import Optional

from torch.cuda.amp import autocast, GradScaler

from losses.loss import IJEPA_Loss

from models.vit_core import ViTModel
from models.microvit_core import MicroViTModel

import math

class SIGReg_IJEPA_Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 optimizer: optim.Optimizer,
                 lr_scheduler: LRScheduler,
                 device: torch.device,
                 epochs: int,
                 save_dir: str = "outputs") -> None:

        self.model = model.to(device)
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epochs = epochs

        self.animation_duration = 0.3        # <-- NUEVO

        # AMP
        self.scaler = GradScaler()

        # Metrics Traking
        self.history = {
            "total_loss": [],
            "rec_loss": [],
            "sigreg_loss": []
        }

        self.global_step = 0
        self.best_loss = float("inf")

        # Directories
        self.checkpoint_dir = os.path.join(save_dir, "checkpoints")
        self.figures_dir = os.path.join(save_dir, "figures")
        self.losses_dir = os.path.join(save_dir, "losses")

        for d in [self.checkpoint_dir, self.figures_dir, self.losses_dir]:
            os.makedirs(d, exist_ok = True)

    def train(self) -> None:
        print(f"Initiating Pre-Training in {self.device} for {self.epochs} epochs...")

        # Visualize Latent Space
        self._visualize_latent_space(0)
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_total_loss = 0.0
            epoch_rec_loss = 0.0
            epoch_sigreg_loss = 0.0

            pbar = tqdm(self.train_loader, desc = f"Epoch {epoch}/{self.epochs}", unit = "batch")

            for batch in pbar:

                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward Pass (AMP)
                with autocast():
                    rec_loss, sigreg_loss, total_loss = self.model(batch, self.global_step)

                # Scale, Backward Pass and Optimization - AMP
                self.scaler.scale(total_loss).backward()

                # Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

                # Optimizer step using the scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update Metrics
                epoch_total_loss += total_loss.item()
                epoch_rec_loss += rec_loss.item()
                epoch_sigreg_loss += sigreg_loss.item()

                pbar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "Rec": f"{rec_loss.item():.4f}",
                    "SIGReg": f"{sigreg_loss.item():.4f}"
                })

                self.global_step += 1

            # Update the Scheduler
            self.lr_scheduler.step()

            # Average the Metrics
            avg_total = epoch_total_loss / len(self.train_loader)
            avg_rec = epoch_rec_loss / len(self.train_loader)
            avg_sigreg = epoch_sigreg_loss / len(self.train_loader)

            self.history["total_loss"].append(avg_total)
            self.history["rec_loss"].append(avg_rec)
            self.history["sigreg_loss"].append(avg_sigreg)

            # Save Checkpoints
            self._save_checkpoint(epoch, avg_total, is_best = (avg_total < self.best_loss))
            if avg_total < self.best_loss:
                self.best_loss = avg_total

            # Visualize Latent Space
            self._visualize_latent_space(epoch)
            self.model.train()

        # Plot Convergence Graphic
        self._plot_losses()

        # Make GIF Animation
        self._create_animation()
        print("Pre-Training Completed Successfully")

    @torch.no_grad()
    def _visualize_latent_space(self,
                                epoch: int) -> None:
        """Extract Test Embeddings and plots its distribution"""
        
        self.model.eval()
        all_embeddings = []
        all_labels = []

        for batch in self.test_loader:
            
            images, labels = batch[0].to(self.device), batch[1].tolist()
            all_labels.extend(labels)

            # Using Only the Encoder
            embeddings = self.model.encoder(images)
            gap_embeddings = embeddings.mean(dim = 1)
            all_embeddings.append(gap_embeddings.cpu())

        embeddings_tensor = torch.cat(all_embeddings, dim = 0)

        try:
            fig, ax = plt.subplots(figsize = (12, 12))
            embeds_np = embeddings_tensor.numpy()

            import umap
            reducer = umap.UMAP(n_components = 2, random_state = 32, n_jobs = 3)
            proj = reducer.fit_transform(embeds_np)

            if all_labels:
                scatter = ax.scatter(proj[:, 0], proj[:, 1], c = all_labels, cmap = "tab10", s = 15, alpha = 0.8)
                plt.colorbar(scatter, ax = ax, label = "Classes")
            else:
                ax.scatter(proj[:, 0], proj[:, 1], alpha = 0.7, s = 15)

            ax.set_title(f"Latent Space (UMAP) - Epoch {epoch}")
            ax.grid(True, alpha = 0.3)

            save_path = os.path.join(self.figures_dir, f"latent_epoch_{epoch}.png")
            plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
            plt.close(fig)

        except Exception as e:
            print(f"Warning: UMAP graphic could not be generated ({e})")

    def _plot_losses(self):
        """Graphs the Loss Evolution During Pre-Training"""

        epochs_range = range(1, self.epochs + 1)

        fig, ax = plt.subplots(figsize = (12, 12))
        ax.plot(epochs_range, self.history["total_loss"], label = "Total Loss", color = "black", linewidth = 2)
        ax.plot(epochs_range, self.history["rec_loss"], label = "Reconstruction Loss", linestyle = "--")
        ax.plot(epochs_range, self.history["sigreg_loss"], label = "SIGReg Loss", linestyle = ":")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Evolution During Pre-Training")
        ax.legend()
        ax.grid(True, alpha = 0.4)

        save_path = os.path.join(self.losses_dir, "training_losses.png")
        plt.savefig(save_path, dpi = 300)
        plt.close(fig)

    def _save_checkpoint(self,
                         epoch: int,
                         current_loss: float,
                         is_best: bool) -> None:
        """Saves a Model Checkpoint"""

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": current_loss,
            "config": self.model.config
        }

        # Save Latest
        last_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, last_path)

        # Saves if is the best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"-> New Best Model Saved")

    def _create_animation(self) -> None:
        try:
            import imageio.v2 as imageio
            from PIL import Image
        except ImportError:
            print("'imageio' and 'PIL' must be installed...")
            return

        png_files = [f for f in os.listdir(self.figures_dir) if f.startswith("latent_epoch_") and
                     f.endswith(".png")]

        if not png_files:
            print()
            return

        # Sort 
        def epoch_number(filename):
            try:
                return int(filename.split("_")[-1].split(".")[0])
            except:
                return 0

        png_files.sort(key = epoch_number)

        # Load images in order
        frames = []
        target_size = None
        for fname in png_files:
            path = os.path.join(self.figures_dir, fname)
            img = Image.open(path)

            if target_size is None:
                target_size = img.size
            else:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
            frames.append(img)

        # Frame duration in milis
        duration_ms = int(self.animation_duration * 1000)

        # Save GIF
        gif_path = os.path.join(self.figures_dir, "latent_space_evolution.gif")
        frames[0].save(
            gif_path,
            save_all = True,
            append_images = frames[1:],
            duration = duration_ms,
            loop = 0,
            optimize = True
        )
        print(f"Animation saved in: {gif_path}")




class KD_Trainer:
    def __init__(self,
                 teacher: ViTModel,
                 student: nn.Module,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 optimizer: optim.Optimizer,
                 lr_scheduler: LRScheduler,
                 device: torch.device,
                 epochs: int,
                 loss_type: str = "mse",
                 save_dir: str = "kd_outputs") -> None:

        self.device = device

        # Freeze the Teacher
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = student.to(self.device)

        # Tensor Sizes
        self.t_size = self.teacher.get_output_size()
        self.s_size = self.student.get_output_size()

        # Extract Channel Dims and Spatial Resolutions
        self.t_dim = self.t_size[-1] if len(self.t_size) == 2 else self.t_size[0]
        self.s_dim = self.s_size[-1] if len(self.s_size) == 2 else self.s_size[0]

        # Compute Spatial Resolutions N_patch or (H, W)
        self.t_spatial = math.isqrt(self.t_size[0]) if len(self.t_size) == 2 else self.t_size[1]
        self.s_spatial = math.isqrt(self.s_size[0]) if len(self.s_size) == 2 else self.s_size[1]

        # Channels Projector Configuration (Master -> Student)
        if self.s_dim != self.t_dim:
            self.projector = nn.Conv2d(self.s_dim, self.t_dim, kernel_size = 1).to(self.device)
            optimizer.add_param_group({"params": self.projector.parameters()})
        else:
            self.projector = nn.Identity().to(self.device)

        # Spatial Adapter Configuration (Master -> Student)
        if self.t_spatial != self.s_spatial:
            self.spatial_pooler = nn.AdaptiveAvgPool2d((self.s_spatial, self.s_spatial)).to(self.device)
        else:
            self.spatial_pooler = nn.Identity().to(self.device)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # AMP
        self.scaler = GradScaler()

        # Data and train configuration
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.global_step = 0
        self.best_loss = float("inf")
        self.animation_duration = 0.3

        self.loss_criterion = IJEPA_Loss(loss_type)

        # Metric tracking
        self.history = {"kd_loss": []}

        # Directories
        self.checkpoint_dir = os.path.join(save_dir, "checkpoints")
        self.figures_dir = os.path.join(save_dir, "figures")
        self.losses_dir = os.path.join(save_dir, "losses")

        for d in [self.checkpoint_dir, self.figures_dir, self.losses_dir]:
            os.makedirs(d, exist_ok = True)

    def _standardize_tensor(self,
                            x: torch.Tensor,
                            size_tuple: tuple) -> torch.Tensor:
        
        if len(size_tuple) == 2:
            # Input: (B, Seq_Len, Hidden_Dim) -> (B, N_patch, N_patch, Hidden_Dim)
            B, S, C = x.shape
            N_patch = math.isqrt(S)
            return x.view(B, N_patch, N_patch, C).permute(0, 3, 1, 2)

        elif len(size_tuple) == 3:
            # Input: (B, Hidden_Dim, H, W)
            return x

        return x

    def train(self) -> None:
        print(f"Initiating Distillation in {self.device} por {self.epochs} epochs...")

        # Visualize Student's Latent Space
        self._visualize_latent_space(0)

        for epoch in range(1, self.epochs + 1):
            self.student.train()

            if isinstance(self.projector, nn.Module):
                self.projector.train()

            epoch_loss = 0.0

            pbar = tqdm(self.train_dataloader, desc = f"Epoch {epoch}/{self.epochs}", unit = "batch")

            for batch in pbar:
                img_teacher = batch[0].to(self.device)
                img_student = batch[1].to(self.device)

                self.optimizer.zero_grad()

                # Teacher's Forward Pass - AMP
                with torch.no_grad(), autocast():
                    teacher_out = self.teacher(img_teacher)

                # Student's Forward Pass - AMP
                with autocast():
                    student_out = self.student(img_student)

                    # Tensor Standarization (B, C, H, W)
                    teacher_out_std = self._standardize_tensor(teacher_out, self.t_size)
                    student_out_std = self._standardize_tensor(student_out, self.s_size)

                    # Dimentional and Spatial Alignment
                    student_projected = self.projector(student_out_std)
                    teacher_pooled = self.spatial_pooler(teacher_out_std)

                    # Backward Pass and Optimization
                    loss = self.loss_criterion(student_projected, teacher_pooled)
                
                # Scale, backward, Optimization and Update the Scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update Metrics
                epoch_loss += loss.item()
                pbar.set_postfix({
                    "KD Loss": f"{loss.item():.4f}"
                })

                self.global_step += 1

            # Update the Scheduler
            self.lr_scheduler.step()

            # Average the Metrics
            avg_loss = epoch_loss / len(self.train_dataloader)
            self.history["kd_loss"].append(avg_loss)

            # Save Checkpoints
            self._save_checkpoint(epoch, avg_loss, is_best = (avg_loss < self.best_loss))
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss

            # Visualize Latent Space
            self._visualize_latent_space(epoch)

        # Plot Convergence Graphic
        self._plot_losses()

        # Make GIF Animation
        self._create_animation()
        print("Knowledge Distillation Completed Successfully")

    @torch.no_grad()
    def _visualize_latent_space(self, epoch: int) -> None:
        """Extracts Student's Test Embeddings and plots its distribution."""
        self.student.eval()
        if isinstance(self.projector, nn.Module):
            self.projector.eval()

        all_embeddings = []
        all_labels = []

        for batch in self.test_dataloader:
            images, labels = batch[0].to(self.device), batch[1].tolist()
            all_labels.extend(labels)

            # Using Only the Student
            embeddings = self.student(images)

            if embeddings.dim() == 3:
                # ViT Case: (B, Seq_Len, C) -> mean(Seq_Len)
                gap_embeddings = embeddings.mean(dim = 1)
            
            elif embeddings.dim() == 4:
                # CNN/MicroViT Case: (B, C, H, W) -> mean(H, W)
                gap_embeddings = embeddings.mean(dim = [2, 3])
            
            else:
                raise ValueError("Output dim is not allowed for visualization: {embeddings.shape}")

            all_embeddings.append(gap_embeddings.cpu())

        embeddings_tensor = torch.cat(all_embeddings, dim = 0)

        try:
            fig, ax = plt.subplots(figsize = (12, 12))
            embeds_np = embeddings_tensor.numpy()

            import umap
            reducer = umap.UMAP(n_components = 2, random_state = 32, n_jobs = 3)
            proj = reducer.fit_transform(embeds_np)

            if all_labels:
                scatter = ax.scatter(proj[:, 0], proj[:, 1], c = all_labels, cmap = "tab10", s = 15, alpha = 0.8)
                plt.colorbar(scatter, ax = ax, label = "Classes")
            else:
                ax.scatter(proj[:, 0], proj[:, 1], alpha = 0.7, s = 15)

            ax.set_title(f"Student Latent Space (UMAP) - Epoch {epoch}")
            ax.grid(True, alpha = 0.3)

            save_path = os.path.join(self.figures_dir, f"latent_epoch_{epoch}.png")
            plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
            plt.close(fig)

        except Exception as e:
            print(f"Warning: UMAP graphic could not be generated ({e})")

    def _plot_losses(self):
        """Graphs the Loss Evolution During Distillation"""
        epochs_range = range(1, self.epochs + 1)
        fig, ax = plt.subplots(figsize = (10, 6))
        
        ax.plot(epochs_range, self.history["kd_loss"], label = "Distillation Loss", color = "blue", linewidth = 2)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Evolution During Distillation")
        ax.legend()
        ax.grid(True, alpha = 0.4)

        save_path = os.path.join(self.losses_dir, "distillation_loss.png")
        plt.savefig(save_path, dpi = 300)
        plt.close(fig)

    def _save_checkpoint(self, epoch: int, current_loss: float, is_best: bool) -> None:
        """Saves a Model Checkpoint"""
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "student_state_dict": self.student.state_dict(),
            "projector_state_dict": self.projector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": current_loss,
            "config": self.student.config
        }

        last_path = os.path.join(self.checkpoint_dir, "latest_student.pth")
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_student.pth")
            torch.save(checkpoint, best_path)
            print("-> New Best Student Saved")

    def _create_animation(self) -> None:
        """Makes a GIF of Latent Space Evolution."""
        try:
            import imageio.v2 as imageio
            from PIL import Image
        except ImportError:
            print("'imageio' and 'PIL' must be installed...")
            return

        png_files = [f for f in os.listdir(self.figures_dir) if f.startswith("latent_epoch_") and f.endswith(".png")]
        if not png_files:
            return

        def epoch_number(filename):
            try:
                return int(filename.split("_")[-1].split(".")[0])
            except:
                return 0

        png_files.sort(key=epoch_number)
        frames = []
        target_size = None
        
        for fname in png_files:
            path = os.path.join(self.figures_dir, fname)
            img = Image.open(path)

            if target_size is None:
                target_size = img.size
            else:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
            frames.append(img)

        duration_ms = int(self.animation_duration * 1000)
        gif_path = os.path.join(self.figures_dir, "student_latent_evolution.gif")
        frames[0].save(
            gif_path,
            save_all = True,
            append_images = frames[1:],
            duration = duration_ms,
            loop = 0,
            optimize = True
        )
        print(f"Animation saved in: {gif_path}")
        
