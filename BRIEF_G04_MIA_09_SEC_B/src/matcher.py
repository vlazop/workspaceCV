"""
Módulo matcher.py: Emparejamiento por distancia de Hamming / L2 y evaluación
geométrica con matrices de Homografía (Ground Truth).
Calonder et al. (2010) - BRIEF: Binary Robust Independent Elementary Features.
"""

import os
import cv2
import numpy as np
from .descriptor import BRIEFExtractor


def get_base_dir(path_or_name="Wall"):
    """Localiza la carpeta de datos de forma dinámica e inteligente."""
    if os.path.exists(path_or_name):
        return path_or_name

    nombre_limpio = os.path.basename(os.path.normpath(path_or_name))
    rutas_posibles = [
        f"./images/{nombre_limpio}",
        f"./BRIEF_G04_MIA_09_SEC_B/images/{nombre_limpio}",
        f"../images/{nombre_limpio}",
        f"../../images/{nombre_limpio}"
    ]
    for r in rutas_posibles:
        if os.path.exists(r):
            return r

    raise FileNotFoundError(f"No se encontró el dataset: '{path_or_name}'")


_POPCNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)


def hamming_distance(desc1, desc2):
    """Calcula la distancia de Hamming bit a bit usando tabla POPCNT vectorizada."""
    xor_res = np.bitwise_xor(desc1, desc2)
    if desc1.ndim == 1 and desc2.ndim == 1:
        return int(_POPCNT_LUT[xor_res].sum())
    return _POPCNT_LUT[xor_res].sum(axis=-1)


def l2_distance(desc1, desc2):
    """Calcula la distancia Euclidiana (L2) para descriptores de punto flotante (SURF / U-SURF)."""
    diff = desc2 - desc1
    if diff.ndim == 1:
        return float(np.linalg.norm(diff))
    return np.linalg.norm(diff, axis=-1)


def _detectar_censure(img, patch_size=31):
    """
    Detector CenSurE (Center-Surrounded Extrema, Agrawal et al. ECCV 2008, Fig. 9-left del paper):
    Calcula la respuesta bi-nivel centro-entorno mediante filtros de caja (box filters).
    """
    img_f = img.astype(np.float32)
    inner = cv2.boxFilter(img_f, -1, (5, 5), normalize=True)
    outer = cv2.boxFilter(img_f, -1, (11, 11), normalize=True)
    resp = np.abs(inner - outer)

    # Máximos locales en vecindad 5x5
    dilated = cv2.dilate(resp, np.ones((5, 5), np.uint8))
    mask = (resp == dilated) & (resp > 4.0)
    ys, xs = np.where(mask)
    scores = resp[ys, xs]
    order = np.argsort(scores)[::-1]
    return [cv2.KeyPoint(x=float(xs[i]), y=float(ys[i]), size=float(patch_size), response=float(scores[i])) for i in order]


def cargar_puntos_con_homografia(img1, img_target, H_gt, num_puntos=1000, patch_size=31, detector='FAST'):
    """
    Protocolo de la Sección 4 del Paper:
    Detecta N puntos de interés en img1, infiere los N puntos correspondientes en img_target
    usando la matriz Ground Truth H_gt y verifica que ambos parches caigan dentro de la imagen.
    """
    h1, w1 = img1.shape
    ht, wt = img_target.shape
    half_S = patch_size // 2
    margin = half_S + 2

    if detector.upper() == 'CENSURE':
        kp1_detected = _detectar_censure(img1, patch_size=patch_size)
        d = 4
    elif detector.upper() == 'SURF':
        sift = cv2.SIFT_create(nfeatures=max(num_puntos * 5, 3000), contrastThreshold=0.01)
        kp1_detected = sorted(sift.detect(img1, None), key=lambda k: k.response, reverse=True)
        d = 3
    else:
        thresh = 5 if num_puntos > 1500 else 15
        d = 1 if num_puntos > 2500 else (2 if num_puntos > 1200 else 4)
        fast = cv2.FastFeatureDetector_create(threshold=thresh, nonmaxSuppression=True)
        kp1_detected = sorted(fast.detect(img1, None), key=lambda k: k.response, reverse=True)

    taken = np.zeros((h1, w1), dtype=bool)
    pts1, pts2 = [], []

    for k in kp1_detected:
        x1, y1 = k.pt
        ix, iy = int(round(x1)), int(round(y1))
        if margin <= x1 < w1 - margin and margin <= y1 < h1 - margin:
            if d > 0 and taken[max(0, iy - d):min(h1, iy + d + 1), max(0, ix - d):min(w1, ix + d + 1)].any():
                continue
            p1_h = np.array([x1, y1, 1.0])
            p2_h = H_gt @ p1_h
            x2 = p2_h[0] / p2_h[2]
            y2 = p2_h[1] / p2_h[2]

            if margin <= x2 < wt - margin and margin <= y2 < ht - margin:
                taken[iy, ix] = True
                pts1.append(cv2.KeyPoint(x=float(x1), y=float(y1), size=float(patch_size)))
                pts2.append(cv2.KeyPoint(x=float(x2), y=float(y2), size=float(patch_size)))
                if len(pts1) >= num_puntos:
                    break

    return pts1, pts2


def _resolver_ruta_imagen(base_dir, idx):
    """Resuelve img{idx}.ppm (Oxford) o 000{idx-1}.jpg (EPFL Fountain)."""
    ruta_ppm = os.path.join(base_dir, f"img{idx}.ppm")
    if os.path.exists(ruta_ppm):
        return ruta_ppm
    ruta_jpg = os.path.join(base_dir, f"000{idx - 1}.jpg")
    if os.path.exists(ruta_jpg):
        return ruta_jpg
    return None


def evaluar_reconocimiento_general(dataset_path_or_name, extractor, num_puntos=1000, detector='FAST'):
    """
    Evalúa la tasa de reconocimiento en los 5 pares (1|2 hasta 1|6) siguiendo
    exactamente el protocolo de 3 pasos de la Sección 4 del paper.
    Soporta descriptores binarios (BRIEF -> distancia de Hamming) y flotantes (SURF/U-SURF -> L2).
    """
    base_dir = get_base_dir(dataset_path_or_name)
    nombre_carpeta = os.path.basename(os.path.normpath(base_dir))

    img1_path = _resolver_ruta_imagen(base_dir, 1)
    if img1_path is None:
        raise FileNotFoundError(f"No se encontró la imagen base en: {base_dir}")
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    tasas = []
    nombres_pares = []

    for idx in range(2, 7):
        img_target_path = _resolver_ruta_imagen(base_dir, idx)
        h_path = os.path.join(base_dir, f"H1to{idx}p")
        corr_path = os.path.join(base_dir, f"corr1to{idx}.npz")

        if img_target_path is None or (not os.path.exists(h_path) and not os.path.exists(corr_path)):
            continue

        img_target = cv2.imread(img_target_path, cv2.IMREAD_GRAYSCALE)

        if os.path.exists(corr_path):
            # Escena 3D (Fountain): correspondencias 3D Ground Truth
            data = np.load(corr_path)
            half_S = extractor.patch_size // 2
            margin = half_S + 2
            h1, w1 = img1.shape
            ht, wt = img_target.shape
            pts1, pts2 = [], []
            for (x1, y1), (x2, y2) in zip(data["pts1"], data["pts2"]):
                if margin <= x1 < w1 - margin and margin <= y1 < h1 - margin and margin <= x2 < wt - margin and margin <= y2 < ht - margin:
                    pts1.append(cv2.KeyPoint(x=float(x1), y=float(y1), size=float(extractor.patch_size)))
                    pts2.append(cv2.KeyPoint(x=float(x2), y=float(y2), size=float(extractor.patch_size)))
                    if len(pts1) >= num_puntos:
                        break
        else:
            H_gt = np.loadtxt(h_path)
            pts1, pts2 = cargar_puntos_con_homografia(
                img1, img_target, H_gt, num_puntos=num_puntos, patch_size=extractor.patch_size, detector=detector
            )

        if len(pts1) == 0:
            tasas.append(0.0)
            nombres_pares.append(f"{nombre_carpeta} 1|{idx}")
            continue

        _, d1 = extractor.compute(img1, pts1)
        _, dt = extractor.compute(img_target, pts2)

        # Seleccionar métrica según el tipo de descriptor (uint8 -> Hamming, float32 -> L2)
        if d1.dtype == np.uint8:
            correctos = sum(1 for i in range(len(d1)) if int(np.argmin(hamming_distance(d1[i], dt))) == i)
        else:
            # Producto punto rápido ya que los vectores SURF-64 están normalizados a norma 1
            sim = d1 @ dt.T
            correctos = int(np.sum(np.argmax(sim, axis=1) == np.arange(len(d1))))

        tasa = (correctos / len(pts1)) * 100.0
        tasas.append(tasa)
        nombres_pares.append(f"{nombre_carpeta} 1|{idx}")

    return np.array(tasas), nombres_pares
