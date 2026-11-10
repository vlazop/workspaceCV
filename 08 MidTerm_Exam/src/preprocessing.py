import cv2
import numpy as np

# 1. Inversión:
#    - I_inv = 255 - I_green

# 2. Ecualización Adaptativa (CLAHE):
#    - Instanciar cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#    - Aplicar sobre I_inv o I_green según la convención elegida.


# 3. Reducción de ruido:
#    - cv2.GaussianBlur con kernel pequeño (3x3 o 5x5, sigma ~ 1.0) para no borrar microvasos.


# 4. Máscara de FOV segura (Erosión del borde):
#    - El borde del campo de visión genera un salto abrupto de intensidad (borde artificial).
#    - Aplicar cv2.erode a la fov_mask con un elemento estructurante elíptico (5x5 a 9x9)
#      para que los operadores de gradiente no detecten el borde de la cámara como un vaso.


def preprocesar_canal_verde(canal_verde, clip_limit = 3.0, tile_grid_size = (8, 8), blur_ksize = 3, blur_sigma = 1.0):

    # 1. Inversión fotométrica: convertir vasos (oscuros) en crestas brillantes (máximos locales)

    img_invertida = cv2.bitwise_not(canal_verde)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_grid_size)
    img_clahe = clahe.apply(img_invertida)

    # 3. Filtrado Gaussiano para atenuar ruido térmico/sensor post-CLAHE
    if blur_ksize > 1:
        

        blur_ksize = int(blur_ksize) # Asegurarnos que sea entereo estricto

        if blur_ksize % 2 == 0:
            blur_ksize += 1
        img_suavizada = cv2.GaussianBlur(
            img_clahe, 
            ksize=(blur_ksize, blur_ksize), 
            sigmaX = blur_sigma, 
            sigmaY = blur_sigma
            )
    else:
        img_suavizada = img_clahe
    
    return {
        'invertida': img_invertida,
        'clahe': img_clahe,
        'suavizada': img_suavizada
        }

def erosionar_mascara_fov(mascara_fov, tam_kernel = 7):

    if tam_kernel % 2 == 0:
        tam_kernel +=1 # Garantizar simetría central con tamaño impar

    # Elemento estructurante elíptico (aproxima la concavidad del ojo)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_kernel, tam_kernel))  
    mascara_erosionada = cv2.erode(mascara_fov, kernel, iterations = 1)

    return mascara_erosionada