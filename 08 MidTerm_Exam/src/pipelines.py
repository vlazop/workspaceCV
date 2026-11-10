import cv2
import numpy as np


def filtrar_componentes_pequenas(mascara_binaria, area_minima = 25):
    num_etiquetas, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara_binaria, connectivity = 8)
    mascara_limpia = np.zeros_like(mascara_binaria)

    for i in range(1, num_etiquetas):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_minima:
            mascara_limpia[etiquetas == i] = 255
    
    return mascara_limpia

# Pipeline A: Umbral Adaptativo Local Gaussiano + Grandiente Sobel

def pipeline_a_adaptativo_sobel(
    img_suavizada,
    fov_segura,
    block_size = 19,
    c_offset = 2.0,
    sobel_ksize = 3,
    sobel_threshold = 35,
    min_area = 30
):
    if block_size % 2 == 0:
        block_size += 1

    mascara_adapt = cv2.adaptiveThreshold(
        img_suavizada,
        maxValue = 255,
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType= cv2.THRESH_BINARY,
        blockSize = block_size,
        C = c_offset 
    )

    gx = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize = sobel_ksize)
    gy = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize = sobel_ksize)
    
    magnitud = np.sqrt(gx**2 + gy**2)

    # magnitud_norm = np.uint8(255 * (magnitud / np.max(magnitud))) if np.max(magnitud) > 0 else np.zeros_like(img_suavizada)

    if np.max(magnitud) > 0:

        magnitud_norm = np.uint8(255 * (magnitud / np.max(magnitud))) 
    else:

        np.uint8(np.zeros_like(img_suavizada))
    
    _, mascara_sobel = cv2.threshold(magnitud_norm, sobel_threshold, 255, cv2.THRESH_BINARY)

    mascara_fusion = cv2.bitwise_and(mascara_adapt, mascara_sobel)

    mascara_fov = cv2.bitwise_and(mascara_fusion, fov_segura)

    pred_mask = filtrar_componentes_pequenas(mascara_fov, area_minima = min_area)
        
    return pred_mask

def pipeline_b_region_growing(
    img_suavizada,
    fov_segura,
    radio_disco=11,
    factor_semilla=1.0,     # 1.0 usa exactamente el umbral de Otsu de la clase (Pág. 86)
    lo_diff=15,             # Tolerancia calibrada
    up_diff=15,
    min_area=25
):
    alto, ancho = img_suavizada.shape

    kernel_disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radio_disco, radio_disco))
    tophat = cv2.morphologyEx(img_suavizada, cv2.MORPH_TOPHAT, kernel_disco)
    tophat_en_fov = cv2.bitwise_and(tophat, fov_segura)

    pixeles_validos = tophat_en_fov[fov_segura > 0]
    th_otsu, _ = cv2.threshold(pixeles_validos, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    umbral_semillas = int(th_otsu * factor_semilla)
    _, mascara_semillas = cv2.threshold(tophat_en_fov, umbral_semillas, 255, cv2.THRESH_BINARY)

    centro_x, centro_y = ancho // 2, alto // 2
    cv2.circle(mascara_semillas, (centro_x, centro_y), radius=35, color=0, thickness=-1)

    coords_y, coords_x = np.where(mascara_semillas > 0)
    paso = 4
    puntos_semilla = list(zip(coords_x[::paso], coords_y[::paso]))

    mascara_flood = np.zeros((alto + 2, ancho + 2), dtype=np.uint8)
    mascara_flood[1:alto + 1, 1:ancho + 1][fov_segura == 0] = 1

    flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    for pt in puntos_semilla:
        if mascara_flood[pt[1] + 1, pt[0] + 1] == 0:
            cv2.floodFill(
                tophat_en_fov,
                mascara_flood,
                seedPoint=pt,
                newVal=255,
                loDiff=lo_diff,
                upDiff=up_diff,
                flags=flags
            )

    region_crecida = mascara_flood[1:alto + 1, 1:ancho + 1]
    mascara_valida = cv2.bitwise_and(region_crecida, fov_segura)

    pred_mask = filtrar_componentes_pequenas(mascara_valida, area_minima=min_area)

    return pred_mask