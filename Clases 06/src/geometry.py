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

import numpy as np

def compute_rotation(ip1, ip2, K1, K2):
  """
  Find rotation matrix R such that |r2 - R*r1|^2 is minimised

  Inputs: ip1,ip2=corresponding interest points (2,num_points),
          K1,K2=camera calibration matrices (3,3)

  Outputs: R=rotation matrix R (3,3)

  r1,r2 are corresponding rays (unit normalised camera coordinates) in image 1,2
  """

  R = H = np.eye(3, 3)

  # 1. Mapear puntos a rayos en el espacio de la camara
  x = np.linalg.inv(K1) @ hom(ip1)
  y = np.linalg.inv(K2) @ hom(ip2)

  # 2. Normalizacion unitaria (|r| = 1) para el algoritmo de Kabsch
  x = x / np.linalg.norm(x, axis=0, keepdims=True)
  y = y / np.linalg.norm(y, axis=0, keepdims=True)

  # 3. Algoritmo de Kabsch / Procrustes ortogonal
  W = y @ x.T
  U, S, Vt = np.linalg.svd(W)
  d = np.linalg.det(U) * np.linalg.det(Vt)

  # 4. Matriz de rotacion pura (det = +1)
  R = U @ np.diag([1.0, 1.0, d]) @ Vt

  # 5. Homografia inducida
  H = K2 @ R @ np.linalg.inv(K1)

  return R, H

def get_calibration(imshape, fov_degrees):
  """
  Return calibration matrix K given image shape and field of view
  """
  H, W, _ = imshape
  f = max(H, W) / (2 * np.tan((fov_degrees / 2) * np.pi / 180))
  
  # Construccion directa sin riesgo de colision de variables
  K1 = np.zeros((3, 3))
  K1[0, 1] = K1[1, 0] = f
  K1[0, 2] = H / 2
  K1[1, 2] = W / 2
  K1[2, 2] = 1.0
  return K1

def K(f, H, W):
  """
  Return camera calibration matrix given focal length and image size
  """
  K1 = np.zeros((3, 3))
  K1[0, 1] = K1[1, 0] = f
  K1[0, 2] = H / 2
  K1[1, 2] = W / 2
  K1[2, 2] = 1.0
  return K1

def hom(p):
  """
  Convert points to homogeneous coordiantes
  """
  ph = np.concatenate((p, np.ones((1, p.shape[1]))))
  return ph

def unhom(ph):
  """
  Convert points from homogeneous to regular coordinates
  """
  p = ph / ph[2, :]
  p = p[0:2, :]
  return p