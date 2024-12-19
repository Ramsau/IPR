import sys
import imageio.v3 as imageio
import math_tools
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve


def diffusion_tensor(
    u: np.ndarray,
    sigma_g: float,
    sigma_u: float,
    alpha: float,
    gamma: float,
    nabla: sp.csr_matrix,
    mode: str,
):
    # Implement the diffusion tensor (9)
    # Keep in mind which operations require a flattened image, and which don't
    return sp.eye(2 * u.size)


def nonlinear_anisotropic_diffusion(
    image: np.ndarray,
    sigma_g: float,
    sigma_u: float,
    alpha: float,
    gamma: float,
    tau: float,
    T: float,
    mode: str,
):
    t = 0.
    U_t = image.ravel()
    nabla = math_tools.spnabla_hp(*image.shape)
    id = sp.eye(U_t.shape[0], format="csc")
    while t < T:
        print(f'{t=}')
        D = diffusion_tensor(
            U_t.reshape(image.shape), sigma_g, sigma_u, alpha, gamma, nabla,
            mode
        )
        U_t = spsolve(id + tau * nabla.T @ D @ nabla, U_t)
        t += tau
    return U_t.reshape(image.shape)


params = {
    'ced': {
        'sigma_g': 1.5,
        'sigma_u': 0.7,
        'alpha': 0.0005,
        'gamma': 1e-4,
        'tau': 5.,
        'T': 100.,
    },
    'eed': {
        'sigma_g': 0.,
        'sigma_u': 10,
        'alpha': 0.,
        # This is delta in the assignment sheet, for the sake of an easy
        # implementation we use the same name as in CED
        'gamma': 1e-4,
        'tau': 1.,
        'T': 10.,
    },
}

inputs = {
    'ced': 'starry_night.png',
    'eed': 'fir.png',
}

if __name__ == "__main__":
    mode = sys.argv[1]
    input = imageio.imread(inputs[mode]) / 255.
    output = nonlinear_anisotropic_diffusion(input, **params[mode], mode=mode)
    imageio.imwrite(
        f'./{mode}_out.png', (output.clip(0., 1.) * 255.).astype(np.uint8)
    )
