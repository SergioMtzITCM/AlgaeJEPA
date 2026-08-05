import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

from typing import List, Optional

class LatentVisualizer:
    @staticmethod
    def plot_embeddings_evolution(embeddings_list: List[torch.Tensor], labels: Optional[List[int]] = None, method: str = 'umap'):
        """
        Visualiza la evolución de los embeddings a través de las épocas.
        embeddings_list: Lista de tensores [N_muestras, Dim] obtenidos en distintas etapas.
        """
        fig, axes = plt.subplots(1, len(embeddings_list), figsize=(5 * len(embeddings_list), 5))
        if len(embeddings_list) == 1: axes = [axes]
            
        for i, embeds in enumerate(embeddings_list):
            embeds_np = embeds.detach().cpu().numpy()
            
            if method.lower() == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                reducer = PCA(n_components=2)
                
            proj = reducer.fit_transform(embeds_np)
            
            ax = axes[i]
            if labels is not None:
                scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='viridis', s=10)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(proj[:, 0], proj[:, 1], alpha=0.7, s=10)
                
            ax.set_title(f'Espacio Latente ({method.upper()}) - Etapa {i+1}')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()


def attention_visualizer(model: nn.Module,
                         tensor_image: torch.Tensor,
                         original_image: np.ndarray,
                         patch_size: int,
                         reference_patch_idx: int = None,
                         layer_index: int = -1) -> None:

    model.eval()

    with torch.no_grad():
        _, attn_weights = model(tensor_image, output_attentions = True)

    layer_attention = attn_weights[layer_index]

    avg_attention = torch.mean(layer_attention, dim = 1)

    attention_matrix = avg_attention[0] # [N_patches, N_patches]

    # Compute Grid Dims
    B, C, H, W = tensor_image.shape
    grid_h = H // patch_size
    grid_w = W // patch_size

    if reference_patch_idx is None:
        reference_patch_idx = (grid_h // 2) * grid_w + (grid_w // 2)

    patch_attention = attention_matrix[reference_patch_idx]

    attention_map_2d = patch_attention.reshape(grid_h, grid_w).cpu().numpy()

    # Min Max Norm
    min_val = attention_map_2d.min()
    max_val = attention_map_2d.max()
    normalized_attention_map = (attention_map_2d - min_val) / (max_val - min_val)

    scaled_map = cv2.resize(
        normalized_attention_map, 
        (original_image.shape[1], original_image.shape[0]), 
        interpolation = cv2.INTER_CUBIC)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))

    # Imagen Original con el punto de referencia
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Calculamos la coordenada en píxeles del parche de referencia para dibujarlo
    ref_y = (reference_patch_idx // grid_w) * patch_size + (patch_size // 2)
    ref_x = (reference_patch_idx % grid_w) * patch_size + (patch_size // 2)
    axes[0].plot(ref_x, ref_y, marker = 'x', color = 'red', markersize = 15, markeredgewidth = 3, label = "Reference Patch")
    axes[0].legend()

    # Heatmap superpuesto
    heatmap = axes[1].imshow(original_image)
    axes[1].imshow(scaled_map, cmap = 'jet', alpha = 0.5) # alpha controla la transparencia
    axes[1].set_title(f"Attention of the Layer {layer_index if layer_index != -1 else len(attn_weights)}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()