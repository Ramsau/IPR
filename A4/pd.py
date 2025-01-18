import os
import numpy as np
import scipy.sparse as sp
from typing import Callable
import imageio.v2 as imageio

from math_tools import spnabla


def pdhg(
    x0: np.ndarray,
    K: sp.spmatrix,
    tau: float,
    prox_tG: Callable[[np.ndarray], np.ndarray],
    sigma: float,
    prox_sFstar: Callable[[np.ndarray], np.ndarray],
    max_iter: int = 200,
) -> np.ndarray:
    x = x0.copy()
    y = K @ x0

    for _ in range(max_iter):
        x_prev = x.copy()
        x = prox_tG(x - tau * K.T @ y)
        y = prox_sFstar(y + sigma * K @ (2 * x - x_prev))

    return x


if __name__ == '__main__':
    # All images are 320 x 640
    nabla = spnabla(320, 640)

    # Operator norm of nabla and standard choices for tau, sigma
    L = 8
    tau = 1 / np.sqrt(L)
    sigma = 1 / np.sqrt(L)

    # Tradeoff parameter
    lamda = 10
    max_iter = 200

    imnames = [f'img{i}' for i in range(1, 5)]
    for name in imnames:
        g = imageio.imread('./test/' + name + '.png') / 255.

        def prox_tG(u: np.ndarray) -> np.ndarray:
            '''
            Todo: Implement (16)
            '''
            return (u + tau * lamda * g.ravel()) / (1 + tau * lamda)

        def prox_sFs(p: np.ndarray) -> np.ndarray:
            '''
            Todo: Implement (17)
            '''
            p_reshaped = p.reshape(2, -1)
            norms = np.linalg.norm(p_reshaped, axis=0)
            scale = np.maximum(1, norms)
            p_reshaped /= scale
            return p_reshaped.ravel()


        denoised = pdhg(
            g.ravel(), nabla, tau, prox_tG, sigma, prox_sFs, max_iter
        ).reshape(g.shape)
        # Code with which the reference images were written to disk
        imageio.imsave(
            f'./output/{name}_ref_{lamda=}.png',
            (denoised.reshape(g.shape) * 255.).astype(np.uint8)
        )
