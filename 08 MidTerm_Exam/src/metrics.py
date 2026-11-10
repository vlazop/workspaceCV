import numpy as np

def calcular_matriz_confusion_fov(pred_mask, gt_mask,fov_mask):    
    

    idx_fov = (fov_mask > 0)

    # Sea p(x,y) e {0, 1} predicha por pipeline y g(x, y) e {0, 1} la anotación manual experta 
    # (Ground Truth), que han sido restrigidas al FOV

    p = (pred_mask[idx_fov] > 0)
    g =(gt_mask[idx_fov] > 0)

    tp = int(np.sum(p & g)) # Verdaderos positivos (TP)
    fp = int(np.sum(p & ~g)) # Falsos positivos (FP)
    fn = int(np.sum(~p & g)) # Falsos negativos (FN)
    tn = int(np.sum(~p & ~g)) # Verdaderos negativos (TN)

    
    return tp, fp, fn, tn

def evaluar_segmentacion(pred_mask, gt_mask, fov_mask, eps = 1e-7):

    tp, fp, fn, tn = calcular_matriz_confusion_fov(pred_mask, gt_mask,fov_mask)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

    iou = (float(tp) + eps) / (tp + fp + fn + eps)

    sensibilidad = (float(tp) + eps) / (tp + fn + eps)

    especificidad = (float(tn) + eps) / (tn + fp + eps)

    total_pixeles = tp + tn + fp + fn

    accuracy = (float(tp) + tn + eps) / (total_pixeles+ eps)

    return {

        'dice': float(dice),
        'iou': float(iou),
        'sensibilidad': float(sensibilidad),
        'especificidad': float(especificidad),
        'accuracy': float(accuracy),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }