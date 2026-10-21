import numpy as np

def generate_brief_pairs(patch_size, num_pairs, strategy):
    """
    Genera los pares de coordenadas relativas (X, Y) dentro del parche de imagen
    siguiendo las 5 geometrías de muestreo descritas en Calonder et al. (2010), Sección 3.2.
    """
    S = patch_size
    half_patch = S / 2.0

    # G I: Distribución Uniforme Continua U(-S/2, S/2)
    if strategy == 'G_I':
        X = np.random.uniform(-half_patch, half_patch, (num_pairs, 2))
        Y = np.random.uniform(-half_patch, half_patch, (num_pairs, 2))

    # G II: Distribución Gaussiana Isotrópica N(0, (S^2)/25) -> sigma = S / 5
    elif strategy == 'G_II':
        sigma = S / 5.0
        X = np.random.normal(0, sigma, (num_pairs, 2))
        Y = np.random.normal(0, sigma, (num_pairs, 2))

    # G III: Gaussiana en 2 pasos -> X ~ N(0, (S^2)/25), Y ~ N(x_i, (S^2)/100)
    elif strategy == 'G_III':
        sigma_X = S / 5.0
        sigma_Y = S / 10.0
        X = np.random.normal(0, sigma_X, (num_pairs, 2))
        Y = np.random.normal(X, sigma_Y, (num_pairs, 2))

    # G IV: Muestreo aleatorio sobre ubicaciones discretas de una grilla polar gruesa (Sección 3.2)
    elif strategy == 'G_IV':
        radii_levels = np.linspace(half_patch / 4.0, half_patch - 1, 4)
        angles_levels = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        polar_pts = [(0.0, 0.0)]
        for r in radii_levels:
            for a in angles_levels:
                polar_pts.append((r * np.cos(a), r * np.sin(a)))
        polar_pts = np.array(polar_pts)

        idx_X = np.random.randint(0, len(polar_pts), size=num_pairs)
        idx_Y = np.random.randint(0, len(polar_pts), size=num_pairs)
        X = polar_pts[idx_X]
        Y = polar_pts[idx_Y]

    # G V: X fijo en el origen (0,0)^T; Y toma todos los valores de una grilla polar gruesa (Sección 3.2)
    elif strategy == 'G_V':
        X = np.zeros((num_pairs, 2))
        num_rings = 8
        pts_per_ring = max(1, int(np.ceil(num_pairs / num_rings)))
        radii_levels = np.linspace(half_patch / num_rings, half_patch - 1, num_rings)
        angles_levels = np.linspace(0, 2 * np.pi, pts_per_ring, endpoint=False)
        polar_grid = []
        for r in radii_levels:
            for a in angles_levels:
                polar_grid.append((r * np.cos(a), r * np.sin(a)))
        Y = np.array(polar_grid[:num_pairs])

    else:
        raise ValueError(f"Estrategia '{strategy}' no válida")

    # Asegurar que las coordenadas estén dentro del parche [-half_patch+1, half_patch-1]
    X = np.clip(X, -half_patch + 1, half_patch - 1)
    Y = np.clip(Y, -half_patch + 1, half_patch - 1)

    return np.round(X).astype(int), np.round(Y).astype(int)
