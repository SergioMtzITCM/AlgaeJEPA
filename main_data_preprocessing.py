import os
import cv2
import numpy as np
import h5py
from h5py import Dataset
import pandas as pd
from typing import Tuple, List, Dict, Optional

def process_single_image(img_path: str, apply_imagenet_norm: bool = False) -> Optional[np.ndarray]:
    """Lee, estandariza y formatea una única imagen."""

    img = cv2.imread(img_path)
    if img is None:
        return None

    # Asegurar espacio de color RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Asegurar 3 canales (RGB)
    if len(img_rgb.shape) == 2:               # (alto, ancho)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 1:               # (alto, ancho, 1)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    # Si ya tiene 3 canales, no se modifica

    # Redimensionar
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation = cv2.INTER_AREA)

    # Escalar a [0,1] y convertir a float32
    img_float = img_resized.astype(np.float32) / 255.0

    # Estandarización de ImageNet
    if apply_imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # Broadcasting de NumPy para restar media y dividir por std en el último eje
        img_float = (img_float - mean) / std

    # Transponer a (Canales, Alto, Ancho) - (C, H, W)
    img_chw = np.transpose(img_float, (2, 0, 1))

    return img_chw

def build_unlabeled_h5_dataset(lista_directorios: List[str],
                               output_h5_path: str,
                               apply_imagenet_norm: bool = True) -> int:
    """Crea un HDF5 para pre-entrenamiento escaneando directorios planos."""

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    total_images = 0

    print(f"\n* [Pre-entrenamiento] Calculando espacio para {output_h5_path}...")
    for directorio in lista_directorios:
        if os.path.isdir(directorio):
            total_images += len([f for f in os.listdir(directorio) if f.lower().endswith(extensions)])

    if total_images == 0:
        print("No hay imágenes para procesar. Abortando.")
        return 0


    # Crea el archivo HDF5
    os.makedirs(os.path.dirname(output_h5_path) or '.', exist_ok = True)

    with h5py.File(output_h5_path, "w") as h5f:
        # Crear un dataset pre-asignado. Shape: (N, C, H, W)
        # chunks = True optimiza las lecturas parciales en disco.
        dataset = h5f.create_dataset(
            'images',
            shape = (total_images, 3, 224, 224),
            dtype = np.float32,
            chunks = (1, 3, 224, 224) # Cada imagen es un chunk, ideal para acceso aleatorio
        )

        current_idx = 0
        for directorio in lista_directorios:
            if not os.path.isdir(directorio):
                continue

            archives = [f for f in os.listdir(directorio) if f.lower().endswith(extensions)]
            saved_in_dir = 0

            for archive in archives:
                img_path = os.path.join(directorio, archive)
                img_chw = process_single_image(img_path, apply_imagenet_norm)

                if img_chw is not None:
                    dataset[current_idx] = img_chw
                    current_idx += 1
                    saved_in_dir += 1

            print(f"** Procesado {directorio}: {saved_in_dir} imágenes.")

    print(f"* [Pre-entrenamiento] Completado. Total en HDF5: {current_idx}")
    return current_idx


def get_leaf_directories_with_images(base_dirs: List[str]) -> List[str]:
    leaf_dirs = []
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"*** Advertencia: El directorio {base_dir} no existe.")
            continue
        for root, dirs, files in os.walk(base_dir):
            if any(f.lower().endswith(extensions) for f in files):
                leaf_dirs.append(root)

    return leaf_dirs

def generate_global_species_map(base_directories: List[str]) -> Dict[str, int]:
    species_set = set()
    leaf_dirs = get_leaf_directories_with_images(base_directories)
    
    for leaf in leaf_dirs:
        species = os.path.basename(os.path.normpath(leaf))
        species_set.add(species)

    sorted_species = sorted(list(species_set))

    return {species: idx + 1 for idx, species in enumerate(sorted_species)}

def get_dynamic_structure(target_dirs: List[str], species_map: Dict[str, int]) -> List[Dict]:
    directory_info = []
    leaf_dirs = get_leaf_directories_with_images(target_dirs)

    for leaf in leaf_dirs:
        path_parts = os.path.normpath(leaf).split(os.sep)
        species = path_parts[-1]
        instrument = path_parts[-2] if len(path_parts) > 1 else "Unknown_Inst"
        env_name = path_parts[-3] if len(path_parts) > 2 else "Unknown_Env"

        if species in species_map:
            directory_info.append({
                "path": leaf,
                "environment": env_name,
                "instrument": instrument,
                "species": species
            })

    return directory_info

def build_labeled_h5_dataset(target_dirs: List[str],
                             output_h5_path: str,
                             global_species_map: Dict[str, int],
                             apply_imagenet_norm: bool = False) -> None:
    """Crea un HDF5 y su respectivo CSV escaneando jerarquías de directorios"""

    dir_info_list = get_dynamic_structure(target_dirs, global_species_map)

    if not dir_info_list:
        print(f"*** No se encontraron datos en {target_dirs}.")
        return

    total_images = 0
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    for info in dir_info_list:
        total_images += len([f for f in os.listdir(info['path']) if f.lower().endswith(extensions)])
        
    if total_images == 0:
        return

    os.makedirs(os.path.dirname(output_h5_path) or '.', exist_ok=True)
    all_csv_rows = []

    print(f"\n* [Etiquetado] Creando {output_h5_path} para {total_images} imágenes...")
    with h5py.File(output_h5_path, "w") as h5f:
        dataset = h5f.create_dataset(
            'images',
            shape=(total_images, 3, 224, 224),
            dtype=np.float32,
            chunks=(1, 3, 224, 224)
        )

        current_idx = 0
        for info in dir_info_list:
            label = global_species_map[info['species']]
            directory = info['path']
            archives = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
            
            for archive in archives:
                img_path = os.path.join(directory, archive)
                img_chw = process_single_image(img_path, apply_imagenet_norm)
                
                if img_chw is not None:
                    dataset[current_idx] = img_chw
                    
                    # Registrar metadatos
                    all_csv_rows.append({
                        'h5_index': current_idx,
                        'image_path': img_path,
                        'environment': info['environment'],
                        'instrument': info['instrument'],
                        'species': info['species'],
                        'label': label
                    })
                    current_idx += 1

    # Generar el CSV en la misma ruta que el HDF5
    base_name = os.path.splitext(os.path.basename(output_h5_path))[0]
    output_dir = os.path.dirname(output_h5_path) or '.'
    csv_path = os.path.join(output_dir, f'{base_name}_labels.csv')
    
    df = pd.DataFrame(all_csv_rows)
    df.to_csv(csv_path, index=False)
    
    print(f"* [Etiquetado] Proceso completado. Total guardadas: {current_idx}")
    print(f"* [Etiquetado] Metadatos guardados en: {csv_path}")


if __name__ == "__main__":

    USE_IMAGENET_NORM = False

    # 1. GENERACIÓN DEL CONJUNTO DE PRE-ENTRENAMIENTO
    build_unlabeled_h5_dataset(
        lista_directorios = [
            "./Datasets/Phytoplankton/PMID2019_(phytoplankton)/data/",
            "./Datasets/Phytoplankton/SYKE-Plankton-IFCB_2022_(phytoplankton)/data/",
            "./Datasets/Phytoplankton/UDE_Diatoms_in_the_Wild_2024_(phytoplankton)/data/",
            "./Datasets/Phytoplankton/DAPlankton/data/",
            "./Datasets/Zooplankton/Kaggle_Plankton_Planktonset_1.0_(zooplankton)/data/",
            "./Datasets/Zooplankton/Lake_Zooplankton/data/",
            "./Datasets/Zooplankton/SYKE-plankton_ZooScan_2024_(zooplankton)/data/"
        ], 
        output_h5_path = "Pretrain_Data/Pretrain_data.h5",
        apply_imagenet_norm = USE_IMAGENET_NORM
    )

    # 2. MAPEO GLOBAL DE CLASES PARA IDD Y ODD
    dir_sea = "./Datasets/Phytoplankton/DAPlankton/DAPlankton_sea"
    dir_lab = "./Datasets/Phytoplankton/DAPlankton/DAPlankton_lab"

    print("\nGenerando mapeo global de especies...")
    global_map = generate_global_species_map([dir_sea, dir_lab])
    print(f"Mapeo generado. {len(global_map)} especies únicas encontradas.")

    # 3. GENERACIÓN DEL CONJUNTO IN-DISTRIBUTION (IDD)
    build_labeled_h5_dataset(
        target_dirs=["./Datasets/Phytoplankton/DAPlankton/DAPlankton_sea/IFCB/"],
        output_h5_path="./In_Distribution_Dataset/IDD.h5",
        global_species_map = global_map,
        apply_imagenet_norm = USE_IMAGENET_NORM
    )

    # 4. GENERACIÓN DEL CONJUNTO OUT-OF-DISTRIBUTION (ODD)
    build_labeled_h5_dataset(
        target_dirs=["./Datasets/Phytoplankton/DAPlankton/DAPlankton_sea/CS/"],
        output_h5_path="./Out_Distribution_Dataset/ODD.h5",
        global_species_map=global_map,
        apply_imagenet_norm=USE_IMAGENET_NORM
    )
   
