# Copyright 2017 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
import numpy as np
from time import time

import im_util
import interest_point
import geometry

class RANSAC:
  """
  Find 2-view consistent matches using RANSAC
  """
  def __init__(self):
    self.params={}
    self.params['num_iterations']=500
    self.params['inlier_dist']=10
    self.params['min_sample_dist']=2

  def consistent(self, H, p1, p2):
    """
    Find interest points that are consistent with 2D transform H

    Inputs: H=homography matrix (3,3)
            p1,p2=corresponding points in images 1,2 of shape (2, N)

    Outputs: cons=list of inliers indicated by true/false (num_points)

    Assumes that H maps from 1 to 2, i.e., hom(p2) ~= H hom(p1)
    """

    cons = np.zeros((p1.shape[1]))
    inlier_dist = self.params['inlier_dist']

    """
    ************************************************
    *** TODO: write code to check consistency with H
    ************************************************
    """
    # 1. Convertir p1 a coordenadas homogéneas (3, N)
    num_puntos = p1.shape[1]
    p1_hom = np.vstack([p1, np.ones((1, num_puntos))])
    # 2. Aplicar la transformación H: (3, 3) @ (3, N) -> (3, N)
    p2_est_hom = H @ p1_hom
    # 3. Normalizar por la coordenada homogénea z (deshomogeneizar a shape (2, N))
    p2_est = p2_est_hom[:2, :] / p2_est_hom[2:3, :]
    # 4. Calcular la distancia euclidiana entre la proyección y el p2 real (N,)
    distancias = np.linalg.norm(p2 - p2_est, axis=0)
    # 5. Los inliers son aquellos cuya distancia es menor al umbral
    cons = distancias < inlier_dist


    """
    ************************************************
    """

    return cons

  def compute_similarity(self,p1,p2):
    """
    Compute similarity transform between pairs of points

    Input: p1,p2=arrays of coordinates (2, 2)

    Output: Similarity matrix S (3, 3)

    Assume S maps from 1 to 2, i.e., hom(p2) = S hom(p1)
    """

    S = np.eye(3,3)

    """
    ****************************************************
    *** TODO: write code to compute similarity transform
    ****************************************************
    """
    x = p1[0, :]
    y = p1[1, :]
    u = p2[0, :]
    v = p2[1, :]
    num_puntos = len(x)
    A = np.zeros((2 * num_puntos, 4))
    b = np.zeros(2 * num_puntos)
    # Ecuaciones para u (filas pares: 0, 2, 4...)
    A[0::2, 0] = x
    A[0::2, 1] = -y
    A[0::2, 2] = 1
    b[0::2] = u
    # Ecuaciones para v (filas impares: 1, 3, 5...)
    A[1::2, 0] = y
    A[1::2, 1] = x
    A[1::2, 3] = 1
    b[1::2] = v
    # Resolvemos el sistema de ecuaciones lineales Ah = b
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_val, tx, ty = h
    # Construimos la matriz de similitud S (3, 3)
    S = np.array([
        [a, -b_val, tx],
        [b_val, a,  ty],
        [0.0,   0.0, 1.0]
    ])
    """
    ****************************************************
    """

    return S

  def ransac_similarity(self, ip1, ipm):
    """
    Find 2-view consistent matches under a Similarity transform

    Inputs: ip1=interest points (2, num_points)
            ipm=matching interest points (2, num_points)
            ip[0,:]=row coordinates, ip[1, :]=column coordinates

    Outputs: S_best=Similarity matrix (3,3)
             inliers_best=list of inliers indicated by true/false (num_points)
    """
    S_best = np.eye(3, 3)
    num_points = ip1.shape[1]
    inliers_best = np.zeros(num_points, dtype=bool)

    """
    *****************************************************
    *** TODO: use ransac to find a similarity transform S
    *****************************************************
    """

    if num_points < 2:
      return S_best, inliers_best
    num_iterations = self.params['num_iterations']
    min_sample_dist = self.params['min_sample_dist']
    for _ in range(num_iterations):
      # 1. Seleccionar 2 puntos aleatorios
      idx = np.random.choice(num_points, 2, replace=False)
      p1 = ip1[:, idx]
      pm = ipm[:, idx]
      # 2. Verificar que los puntos no estén pegados en NINGUNA de las dos imágenes
      if np.linalg.norm(p1[:, 0] - p1[:, 1]) < min_sample_dist:
        continue
      if np.linalg.norm(pm[:, 0] - pm[:, 1]) < min_sample_dist:
        continue
      # 3. Calcular la matriz de similitud S
      S = self.compute_similarity(p1, pm)
      # 4. FILTRO FÍSICO: La escala en una panorámica debe ser cercana a 1 (ej. entre 0.7 y 1.4)
      # s = sqrt(a^2 + b^2)
      scale = np.sqrt(S[0, 0]**2 + S[1, 0]**2)
      if scale < 0.7 or scale > 1.4:
        continue
      # 5. Probar consistencia
      inliers = self.consistent(S, ip1, ipm)
      # 6. Guardar si es el mejor modelo
      if np.sum(inliers) > np.sum(inliers_best):
        inliers_best = inliers
        S_best = S
    """
    *****************************************************
    """

    return S_best, inliers_best
