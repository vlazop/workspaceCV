"""
MÓDULO: data_loader.py
OBJETIVO:
    Carga robusta y estructurada de imágenes de retina DRIVE (resolución nativa 565x584),
    extracción del canal verde y lectura binarizada de máscaras GIF (FOV y Ground Truth).
RESTRICCIONES:
    - Sin redimensionamiento (preservar 584x565).
    - Canales binarios estrictos en uint8 {0, 255}.
"""
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage



# 1. Función para cargar una imagen:
#    - Leer la imagen en formato BGR/RGB con cv2.imread o PIL.Image.
#    - Extraer el plano verde (img[:, :, 1] en BGR/RGB).
#    - Verificar que la resolución sea estrictamente (584, 565).

def cargar_retina_image(image_path):

    # 1. Verificando la existencia de la imagen en la ruta especificada
    path_str = str(image_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"No existe la imagen en {path_str}")
    
    # Para forzar si esta en RGBA o paleta
    with PILImage.open(path_str) as img:
        img_rgb = img.convert('RGB') 
        rgb_image = np.array(img_rgb, dtype=np.uint8)

    # 2. Extraer canal ver (índice 1 en RGB)
    canal_verde = rgb_image[ :, :, 1].copy()

    # 3. Validación de resolución (584 x 565)
    if canal_verde.shape != (584,565):
        raise ValueError(f"Dimensión erronea:{canal_verde.shape}. Debería ser (584, 565).")

    return rgb_image, canal_verde


# 2. Función para cargar máscaras (.gif):
#    - cv2.imread nativo no suele soportar GIF con canal alfa/paleta adecuadamente.
#    - Utilizar PIL (Image.open) y convertir a escala de grises np.array(..., dtype=np.uint8).
#    - Binarizar: todo píxel > 0 se convierte en 255; el resto en 0.

def cargar_mascara_gif(mask_path):

    # 1. Verificando la existencia de la imagen en la ruta especificada
    path_str = str(mask_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"No existe la máscara en {path_str}")
    
    # Cargar con PIL y convertir a escala de grises ('L')
    with PILImage.open(path_str) as img:
        img_gray = img.convert('L') # Para convertir a gray
        gray_array = np.array(img_gray, dtype=np.uint8)

    # 2. Binario estricto {0, 255}.
    # Si p(x, y) > 0 se asigna a 255, el resto a 0
    binario_mascara = (gray_array > 0).astype(np.uint8) * 255

    # 3. Validación de dimensiones nativas de DRIVE (584 x 565)
    if binario_mascara.shape != (584,565):
        raise ValueError(
            f"Dimensión erronea de la máscara:{path_str}: {binario_mascara.shape}. "
            f"Se esperaba (584, 565)."
        )

    return binario_mascara



# 3. Función para escanear directorios (training / test):
#    - Emparejar por ID numérico (ej. '21_training.tif' con '21_training_mask.gif' y '21_manual1.gif').
#    - Retornar diccionarios o listas estructuradas con paths listos para iterar.

def obtener_dataset_registros(base_dir, split='training'):

    split = split.lower()
    if split not in('training', 'test'):
        raise ValueError(f"El split debe ser 'training' o 'test', se ingresó: '{split}'")

    base_path = Path(base_dir) / split

    images_dir = base_path / 'images'
    mask_dir = base_path / 'mask'
    manual_dir = base_path / '1st_manual'

    if not images_dir.is_dir():
        raise FileNotFoundError(f"No se encontró el directorio de imágenes: {images_dir}")

    # 1. Obtener y ordenar los archivos .tif

    image_files = sorted(list(images_dir.glob('*.tif')),key=lambda x: x.name)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No se encontraron imágenes .tif en: {images_dir}")

    registros = []

    for img_file in image_files:
        # Extración del ID númerico (e.g. 21_training.tif se extrae 21)
        file_id = img_file.name.split('_')[0]

        # 2. Localizar la máscara FOV correspondiente
        # Buscamos el archivo que comience con el id y termina con mask.gif
        fov_matches = list(mask_dir.glob(f"{file_id}_*_mask.gif"))

        if not fov_matches:
            # Contigencia si falla o se llama '{file_id}_mask.gif'
            fov_matches = list(mask_dir.glob(f"{file_id}*mask*.gif"))

        if not fov_matches:
            raise FileNotFoundError(f"Máscara FOV no encontrada para el ID {file_id} en {mask_dir}")
        fov_mask_file = fov_matches[0]

        # 3. Localizar Ground Truth (solo si estamos en 'training')
        # El término ground truth se refiere a la información real y verificada que se utiliza como comparación estandar para medir la precisión del modelo

        manual_file = None

        if split == 'training':
            manual_matches = list(manual_dir.glob(f"{file_id}_manual1.gif"))
            
            if not manual_matches:
                # Contigencia si falla o se llama '{file_id}_manual1.gif'
                manual_matches = list(manual_dir.glob(f"{file_id}*.gif"))

            if not manual_matches:
                raise FileNotFoundError(f"Ground Truth no encontrada para el ID {file_id} en {manual_dir}")
            manual_file = manual_matches[0]
        
        registros.append({
            'id': file_id,
            'image_path': img_file,
            'fov_mask_path': fov_mask_file,
            'manual_path': manual_file
        })
    
    return registros


            








    return registros