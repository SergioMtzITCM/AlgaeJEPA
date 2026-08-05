import os
from os.path import isdir
import cv2
import numpy as np
import h5py
from h5py import Dataset


def preprocessing_images_to_h5(directory: str, h5_dataset: Dataset, 
                               current_idx: int, is_test_set: bool) -> int:
    """
    Lee todas las imágenes de un directorio, las preprocesa (224x224, RGB, float32 [0,1])
    y las guarda individualmente en output_dir con nombres secuenciales.
    
    Args:
        directory (str): Carpeta de origen con las imágenes.
        h5_dataset (Dataset): REferencia aldataset dentro del archivo HDF5.
        current_idx (int): índice actual en el dataset HDF5 donde se empezará a insertar.
        is_test_set (bool): Bandera que indica si se debe ordenar numéricamente para (Test).
    
    Returns:
        int: Número de imágenes guardadas desde este directorio.
    """
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    
    # Listar archivos de imagen
    archives = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]

    if is_test_set:
        archives.sort(key = lambda x: int(x.split(".")[0]))
    
    if not archives:
        print(f"No se encontraron imágenes en {directory}")
        return 0
    
    saved_count = 0
    
    for archive in archives:
        root = os.path.join(directory, archive)
        
        # Leer imagen
        img = cv2.imread(root)

        if img is None:
            print(f"Advertencia: no se pudo leer {archive}")
            continue
        
        # Convertir BGR a RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Asegurar 3 canales (RGB)
        if len(img_rgb.shape) == 2:               # (alto, ancho)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 1:               # (alto, ancho, 1)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        # Si ya tiene 3 canales, no se modifica
        
        # Redimensionar a 224x224
        img_resized = cv2.resize(img_rgb, (224, 224), interpolation = cv2.INTER_AREA)
        
        # Escalar a [0,1] y convertir a float32
        img_float = img_resized.astype(np.float32) / 255.0

        # Transponer a (Canales, Alto, Ancho) - (C, H, W)
        img_chw = np.transpose(img_float, (2, 0, 1))

        # Insertar en el archivo HDF5 en la posición correspondiente
        h5_dataset[current_idx + saved_count] = img_chw
        saved_count += 1
        
            
    return saved_count


def build_h5_dataset(lista_directorios: list, output_h5_path: str,
                     is_test_set: bool = False) -> int:
    """
    Pre-calcula el total de imágenes, crea un archivo HDF5 optimizado y
    procesa los directorios inyectando los datos.
    """

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    total_images = 0

    print(f"Calculando espacio total necesario para {output_h5_path}...")
    for directorio in lista_directorios:
        if os.path.isdir(directorio):
            total_images += len([f for f in os.listdir(directorio) if f.lower().endswith(extensions)])

    if total_images == 0:
        print("No hay imágenes para procesar...")
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

            print(f"\nProcesando directorio: {directorio}")
            guardadas = preprocessing_images_to_h5(directorio, dataset, current_idx, is_test_set)
            current_idx += guardadas
            print(f"  Imágenes guardadas desde este directorio: {guardadas}")

    print(f"\nProceso completado. Total guardadas en HDF5: {current_idx}")
    
    return current_idx


if __name__ == "__main__":

    # Pre-entrenamiento (NO requiere ordenamiento estricto numérico)
    build_h5_dataset(
        lista_directorios=[
            "./Datasets/Phytoplankton/PMID2019_(phytoplankton)/data/",
            "./Datasets/Phytoplankton/SYKE-Plankton-IFCB_2022_(phytoplankton)/data/",
            "./Datasets/Phytoplankton/UDE_Diatoms_in_the_Wild_2024_(phytoplankton)/data/",
            "./Datasets/Phytoplankton/DAPlankton/data/",
            "./Datasets/Zooplankton/Kaggle_Plankton_Planktonset_1.0_(zooplankton)/data/",
            "./Datasets/Zooplankton/Lake_Zooplankton/data/",
            "./Datasets/Zooplankton/SYKE-plankton_ZooScan_2024_(zooplankton)/data/"
        ], 
        output_h5_path="Preprocessed_data/Preprocessed_data.h5",
        is_test_set = False
    )

    # Test (SÍ requiere ordenamiento estricto para coincidir con el CSV)
    build_h5_dataset(
        lista_directorios=[
            "./Datasets/WHOI-Plankton/data/"
        ], 
        output_h5_path="Test_data_preprocessed/Test_data_preprocessed.h5",
        is_test_set = True
    )
