import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

from typing import List, Optional

import os 
import pandas as pd
import seaborn as sns

sns.set_theme(
    context = "paper", 
    style = "ticks", 
    palette = "deep", 
    font = "sans-serif", 
    font_scale = 1.2,
    rc = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "figure.dpi": 300
    }
)


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


def generate_degradation_plots(dir_paths: List[str],
                               csv_filename: str = "benchmark_summary.csv") -> pd.DataFrame:
    """
    Lee los archivos 'benchmark_summary.csv de una lista de directorios, extrae la
    arquitectura del nombre de cada carpeta y genera gráficos de degradación.
    """

    combined_data = []

    # 1. Extracción de datos por ruta
    for path in dir_paths:
        csv_path = os.path.join(path, csv_filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[Error] No se encontró el archivo: {csv_path}.")

        # Extraer nombre de la arquitectura desde el nombre del directorio
        dir_name = os.path.basename(os.path.normpath(path))
        model_name = (
                dir_name.replace("_Benchmark_Results", "")
                .replace("_results", "")
                .replace("_benchmark", "")
        )

        df = pd.read_csv(csv_path)
        df["Model"] = model_name
        combined_data.append(df)

    if not combined_data:
        raise ValueError("No se cargaron datos válidos desde las rutas especificadas.")

    full_df = pd.concat(combined_data, ignore_index = True)

    # Asegurar el orden decreciente de escasez de datos (100% -> 5%)
    full_df = full_df.sort_values(by = "Fraction", ascending = False) 

    # Formatear la fracción como porcentaje
    full_df["Fraction_Pct"] = full_df["Fraction"].apply(lambda x: f"{int(round(x * 100))}")
    x_labels = [f"{int(round(f * 100))}%" for f in sorted(full_df["Fraction"].unique(), reverse=True)]

    # 2. Establecer estilo de gráfico
    sns.set_theme(style = "ticks", context = "paper", font = "sans-serif", font_scale = 1.3)
    palette = sns.color_palette("husl", n_colors = full_df["Model"].nunique())
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    models = full_df["Model"].unique()
    model_style_map = {m: (palette[i], markers[i % len(markers)]) for i, m in enumerate(models)}

    # Métricas a graficar: (Columna Media, Columna Std, Título Eje Y, Título Gráfico)
    metrics_config = [
        ("IDD_F1_Mean", "IDD_F1_Std", "Macro F1-Score", "Degradación en Test IDD (F1-Score)"),
        ("IDD_Acc_Mean", "IDD_Acc_Std", "Accuracy", "Degradación en Test IDD (Accuracy)"),
        ("OOD_F1_Mean", "OOD_F1_Std", "Macro F1-Score", "Degradación en Test OOD (F1-Score)"),
        ("OOD_Acc_Mean", "OOD_Acc_Std", "Accuracy", "Degradación en Test OOD (Accuracy)")
    ]

    # Helper para trazar cada curva y su banda de error
    def plot_single_curve(ax, mean_col, std_col, y_label, title):
        for model in models:
            sub_df = full_df[full_df["Model"] == model]
            color, marker = model_style_map[model]

            x_vals = range(len(sub_df))
            y_mean = sub_df[mean_col].values
            y_std = sub_df[std_col].values

            # Línea de tendencia principal
            ax.plot(
                    x_vals, y_mean,
                    label = model,
                    color = color,
                    marker = marker,
                    linewidth = 2.5,
                    markersize = 8,
                    markeredgecolor = 'white',
                    markeredgewidth = 1
            )

            # Sombreado de la desviación estándar
            ax.fill_between(
                    x_vals,
                    y_mean - y_std,
                    y_mean + y_std,
                    color = color,
                    alpha = 0.15
            )

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontweight = "bold")
        ax.set_xlabel("Escenario de Escasez de Datos", fontsize = 11, fontweight = "bold")
        ax.set_ylabel(ylabel, fontsize = 11, fontweight = "bold")
        ax.set_title(title, fontsize = 12, fontweight = "bold", pad = 10)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle = "--", alpha = 0.5)

        ax.grid(True, linestyle = ":", alpha = 0.4)
        sns.despine(ax = ax)

    # 3. Generación y guardado de gráficos individuales
    individual_filenames = [
        "degradation_idd_f1.png",
        "degradation_idd_acc.png",
        "degradation_ood_f1.png",
        "degradation_ood_acc.png"
    ]

    for (mean_col, std_col, ylabel, title), fname in zip(metrics_config, individual_filenames):
        fig, ax = plt.subplots(figsize = (7, 5))
        plot_single_curve(ax, mean_col, std_col, ylabel, title)
        ax.legend(
                title = "Arquitectura", 
                loc = "upper right", 
                frameon = True,
                framealpha = 0.9,
                edgecolor = "gray"
        )

        current_ylim = ax.get_ylim()
        ax.set_ylim(0.0, current_ylim[1] * 1.15)

        plt.tight_layout()

        # Guardar una copia en cada directorio de origen
        for path in dir_paths:
            fig.savefig(os.path.join(path, fname), dpi = 300, bbox_inches = "tight")
        plt.close(fig)

    # 4. Generación y guardado de la Figura Compuesta
    fig_grid, axes = plt.subplots(2, 2, figsize = (14, 10))
    axes_flat = axes.flatten()

    for idx, (mean_col, std_col, ylabel, title) in enumerate(metrics_config):
        plot_single_curve(axes_flat[idx], mean_col, std_col, ylabel, title)

    # Leyenda unificada para el panel completo
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig_grid.legend(
            handles, labels,
            loc = "lower center",
            bbox_to_anchor = (0.5, 1.0),
            ncol = len(models),
            title = "Arquitectura de Deep Learning",
            frameon = True,
            fontsize = 11,
            title_fontsize = 12,
            edgecolor = "black"
    )
    plt.tight_layout()
    fig_grid.subplots_adjust(top = 0.90)

    grid_filename = "degradation_composite_panel.png"
    for path in dir_paths:
        fig_grid.savefig(os.path.join(path, grid_filename), dpi = 300, bbox_inches = "tight")
    plt.close(fig_grid)

    print(f"Proceso completado. Se generaron las gráficas en los {len(dir_paths)} directorios.")
    
    return full_df