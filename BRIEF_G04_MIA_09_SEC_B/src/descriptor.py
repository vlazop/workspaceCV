import cv2
import numpy as np
from .sampling import generate_brief_pairs


class BRIEFExtractor:
    def __init__(self, patch_size=31, num_pairs=256, strategy='G_II', sigma=2.0, pairs=None):
        """
        Inicializa el extractor BRIEF según Calonder et al. (2010).
        Args:
            patch_size (int): Tamaño del parche S x S (por defecto 31).
            num_pairs (int): Longitud del descriptor en bits n_d (128, 256 o 512).
            strategy (str): Geometría de muestreo ('G_I' a 'G_V'). Por defecto 'G_II'.
            sigma (float): Desviación estándar del filtro Gaussiano 9x9 (por defecto 2.0).
            pairs (tuple): Pares (X, Y) pre-generados opcionales.
        """
        self.patch_size = patch_size
        self.num_pairs = num_pairs
        self.strategy = strategy
        self.sigma = sigma
        self.half_S = patch_size // 2

        if pairs is not None:
            self.X, self.Y = pairs
        else:
            self.X, self.Y = generate_brief_pairs(patch_size, num_pairs, strategy)

        self.X_int = np.round(self.X).astype(int)
        self.Y_int = np.round(self.Y).astype(int)

    def compute(self, image, keypoints, return_bits=False):
        """
        Calcula los descriptores binarios BRIEF para una lista de puntos clave.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Pre-suavizado Gaussiano (kernel 9x9, Sección 3.1 del paper)
        if self.sigma <= 0:
            smoothed = gray
        else:
            smoothed = cv2.GaussianBlur(gray, (9, 9), self.sigma)

        h, w = gray.shape
        valid_keypoints = []
        descriptors = []

        for kp in keypoints:
            pt = kp.pt if isinstance(kp, cv2.KeyPoint) else kp
            cx, cy = int(round(pt[0])), int(round(pt[1]))

            # 2. Control de bordes: descartar puntos cercanos a los márgenes
            if (cx - self.half_S < 0 or cx + self.half_S >= w or
                cy - self.half_S < 0 or cy + self.half_S >= h):
                continue

            # 3. Muestreo de intensidades p(x) y p(y)
            val_X = smoothed[cy + self.X_int[:, 1], cx + self.X_int[:, 0]]
            val_Y = smoothed[cy + self.Y_int[:, 1], cx + self.Y_int[:, 0]]

            # 4. Prueba binaria tau: 1 si p(x) < p(y), sino 0 (Ecuación 1)
            bits = (val_X < val_Y).astype(np.uint8)

            # 5. Empaquetar bits a arreglo de bytes uint8 (Ecuación 2)
            descriptors.append(bits if return_bits else np.packbits(bits))
            valid_keypoints.append(kp)

        if len(descriptors) == 0:
            return [], np.array([], dtype=np.uint8)

        return valid_keypoints, np.array(descriptors, dtype=np.uint8)


class OBRIEFExtractor(BRIEFExtractor):
    """
    O-BRIEF (Orientation-corrected BRIEF) descrito en la Sección 4 (Orientation Sensitivity, Fig. 9-right):
    Rota las coordenadas de prueba (X, Y) según el ángulo de orientación dominante del punto clave.
    """
    def compute(self, image, keypoints):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        smoothed = gray if self.sigma <= 0 else cv2.GaussianBlur(gray, (9, 9), self.sigma)
        gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

        h, w = gray.shape
        valid_keypoints = []
        descriptors = []

        for kp in keypoints:
            pt = kp.pt if isinstance(kp, cv2.KeyPoint) else kp
            cx, cy = int(round(pt[0])), int(round(pt[1]))

            if (cx - self.half_S < 0 or cx + self.half_S >= w or
                cy - self.half_S < 0 or cy + self.half_S >= h):
                continue

            # Estimar orientación dominante en ventana local (como SURF)
            r = 10
            sub_gx = gx[cy - r:cy + r + 1, cx - r:cx + r + 1]
            sub_gy = gy[cy - r:cy + r + 1, cx - r:cx + r + 1]
            theta = np.arctan2(sub_gy.sum(), sub_gx.sum())
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

            X_rot = np.clip(np.round(self.X @ R.T).astype(int), -self.half_S, self.half_S)
            Y_rot = np.clip(np.round(self.Y @ R.T).astype(int), -self.half_S, self.half_S)

            val_X = smoothed[cy + X_rot[:, 1], cx + X_rot[:, 0]]
            val_Y = smoothed[cy + Y_rot[:, 1], cx + Y_rot[:, 0]]

            bits = (val_X < val_Y).astype(np.uint8)
            descriptors.append(np.packbits(bits))
            valid_keypoints.append(kp)

        if len(descriptors) == 0:
            return [], np.array([], dtype=np.uint8)

        return valid_keypoints, np.array(descriptors, dtype=np.uint8)


class SURF64Extractor:
    """
    Implementación de SURF-64 y U-SURF-64 (Bay et al., 2008) de 64 flotantes (256 bytes)
    para las comparaciones de la Sección 4 (Figuras 6, 7, 8, 9 y Tabla de tiempos).
    Divide el parche en 4x4 subregiones y acumula (sum(dx), sum(dy), sum(|dx|), sum(|dy|)).
    """
    def __init__(self, upright=False, patch_size=31):
        self.upright = upright
        self.patch_size = patch_size
        self.half_S = patch_size // 2

        # Grilla 4x4 de subregiones con 5x5 puntos de muestreo por subregión (20x20 muestras)
        coords = np.linspace(-10, 10, 20)
        ug, vg = np.meshgrid(coords, coords)
        self.u_flat = ug.reshape(4, 5, 4, 5).transpose(0, 2, 1, 3).reshape(16, 25)
        self.v_flat = vg.reshape(4, 5, 4, 5).transpose(0, 2, 1, 3).reshape(16, 25)

    def compute(self, image, keypoints):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        smoothed = cv2.GaussianBlur(gray, (5, 5), 1.2)
        # Respuestas wavelet de Haar horizontales (dx) y verticales (dy)
        dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

        h, w = gray.shape
        valid_keypoints = []
        descriptors = []

        u_int_0 = np.round(self.u_flat).astype(int)
        v_int_0 = np.round(self.v_flat).astype(int)

        for kp in keypoints:
            pt = kp.pt if isinstance(kp, cv2.KeyPoint) else kp
            cx, cy = int(round(pt[0])), int(round(pt[1]))

            if (cx - self.half_S < 0 or cx + self.half_S >= w or
                cy - self.half_S < 0 or cy + self.half_S >= h):
                continue

            if self.upright:
                # U-SURF: sin corrección de orientación
                rx = dx[cy + v_int_0, cx + u_int_0]
                ry = dy[cy + v_int_0, cx + u_int_0]
            else:
                # SURF-64: estima orientación dominante y rota las muestras
                r = 10
                sub_dx = dx[cy - r:cy + r + 1, cx - r:cx + r + 1]
                sub_dy = dy[cy - r:cy + r + 1, cx - r:cx + r + 1]
                theta = np.arctan2(sub_dy.sum(), sub_dx.sum())
                cos_t, sin_t = np.cos(theta), np.sin(theta)

                u_rot = np.clip(np.round(cos_t * self.u_flat - sin_t * self.v_flat).astype(int), -self.half_S, self.half_S)
                v_rot = np.clip(np.round(sin_t * self.u_flat + cos_t * self.v_flat).astype(int), -self.half_S, self.half_S)

                gx_s = dx[cy + v_rot, cx + u_rot]
                gy_s = dy[cy + v_rot, cx + u_rot]
                rx = cos_t * gx_s + sin_t * gy_s
                ry = -sin_t * gx_s + cos_t * gy_s

            # Vector 4D por cada una de las 16 subregiones -> 64 floats
            s_dx = rx.sum(axis=1)
            s_dy = ry.sum(axis=1)
            s_abs_dx = np.abs(rx).sum(axis=1)
            s_abs_dy = np.abs(ry).sum(axis=1)

            vec64 = np.column_stack((s_dx, s_dy, s_abs_dx, s_abs_dy)).ravel().astype(np.float32)
            norm = np.linalg.norm(vec64) + 1e-7
            descriptors.append(vec64 / norm)
            valid_keypoints.append(kp)

        if len(descriptors) == 0:
            return [], np.array([], dtype=np.float32)

        return valid_keypoints, np.array(descriptors, dtype=np.float32)
