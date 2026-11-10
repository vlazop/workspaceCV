import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def generar_mapa_error(pred_mask, gt_mask, fov_mask):

    alto, ancho = pred_mask.shape
    mapa_rgb = np.zeros((alto, ancho, 3), dtype = np.uint8)

    idx_fov = (fov_mask > 0)

    # Sea p(x,y) e {0, 1} predicha por pipeline y g(x, y) e {0, 1} la anotación manual experta 
    # (Ground Truth), que han sido restrigidas al FOV

    p = (pred_mask > 0)
    g = (gt_mask > 0)

    tp_mask = p & g & idx_fov  # Verdaderos positivos (TP)
    fp_mask = p & (~g) & idx_fov # Falsos positivos (FP)
    fn_mask = (~p) & g & idx_fov # Falsos negativos (FN)

    mapa_rgb[tp_mask] = [0, 255, 0] # Verde
    mapa_rgb[fp_mask] = [255, 0, 0] # Rojo
    mapa_rgb[fn_mask] = [0, 128, 255] # Azul

    return mapa_rgb


def visualizar_resultado_pipeline(
    imagen_rgb, 
    gt_mask, 
    pred_mask, 
    fov_mask, 
    metricas = None,
    titulo_pipeline = "Pipeline de Segmentación"
):

    mapa_error = generar_mapa_error(pred_mask, gt_mask, fov_mask)

    fig, axes = plt.subplots(1, 4, figsize = (20,5))

    axes[0].imshow(imagen_rgb)
    axes[0].set_title("1. Retina Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap = "gray")
    axes[1].set_title("2. Ground Truth (Manual)")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap = "gray")
    axes[2].set_title(f"3. Predicción ({titulo_pipeline})")
    axes[2].axis("off")

    axes[3].imshow(mapa_error)
    subtitulo_metricas = ""

    if metricas:
        subtitulo_metricas = f"\nDice: {metricas['dice']:.4f} | IoU: {metricas['iou']:.4f} | Rec: {metricas['sensibilidad']:.4f}"
    axes[3].set_title(f"4. Mapa de Error Clínico{subtitulo_metricas}", fontsize = 10, pad = 10)
    axes[3].axis("off")

    elementos_leyenda = [
        Patch(facecolor = 'green', label = 'TP (Acierto)'),
        Patch(facecolor = 'red', label = 'FP (Ruido/Falso)'),
        Patch(facecolor = (0/255, 128/255, 255/255), label = 'FN (Vaso Omitido)'),
    ]

    axes[3].legend(handles = elementos_leyenda, loc = 'lower right', fontsize = 8, framealpha = 0.7)

    plt.tight_layout()
    plt.show()
    


