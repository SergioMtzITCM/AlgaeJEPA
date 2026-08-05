import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

import torch

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