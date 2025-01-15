import sys
import imageio.v3 as imageio
import math_tools
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve
import scipy


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

    # m, n
    u_tilde = gaussian_filter(u, sigma_u)
    u_tilde_x = scipy.signal.convolve2d(u_tilde, np.array([[-1, 0, 1]]), mode="same")
    u_tilde_y = scipy.signal.convolve2d(u_tilde, np.array([[-1], [0], [1]]), mode="same")

    # m, n
    S_xx = gaussian_filter(u_tilde_x * u_tilde_x, sigma_g, axes=(0, 1))
    S_yy = gaussian_filter(u_tilde_y * u_tilde_y, sigma_g, axes=(0, 1))
    S_xy = gaussian_filter(u_tilde_x * u_tilde_y, sigma_g, axes=(0, 1))

    # m, n, 2, 2
    S = np.array([
        [S_xx, S_xy],
        [S_xy, S_yy],
    ]).transpose(2, 3, 0, 1)
    S_eig = np.linalg.eig(S)

    # m*n
    eig_vals = S_eig.eigenvalues.reshape(u.size, 2)
    eig_vectors = S_eig.eigenvectors.reshape(u.size, 2, 2)
    # resort
    idx = eig_vals.argsort()
    mu_1 = eig_vals[np.arange(eig_vals.shape[0]), idx[:, 1]]
    mu_2 = eig_vals[np.arange(eig_vals.shape[0]), idx[:, 0]]
    v_1 = eig_vectors[np.arange(eig_vectors.shape[0]), idx[:, 1]]
    v_2 = eig_vectors[np.arange(eig_vectors.shape[0]), idx[:, 0]]

    if mode == 'ced':
        g = np.exp(-np.pow(mu_1 - mu_2, 2) / (2 * np.pow(gamma, 2)))
        lambda_1 = alpha
        lambda_2 = alpha + (1 - alpha) * (1 - g)
    elif mode == 'eed':
        lambda_1 = np.pow(1 + mu_1 / np.pow(gamma, 2), -0.5) # gamma=delta according to code template
        lambda_2 = 1
    else:
        raise Exception("Unknown mode")

    D_1 = lambda_1 * np.power(v_1[:, 0], 2) + lambda_2 * np.power(v_2[:, 0], 2)
    D_2 = lambda_1 * np.power(v_1[:, 1], 2) + lambda_2 * np.power(v_2[:, 1], 2)
    D_3 = lambda_1 * v_1[:, 0] * v_1[:, 1] + lambda_2 * v_2[:, 0] * v_2[:, 1]

    # ok, but i can't create np.diag(D_1) - that would be 1.68 TiB...
    D_U_1 = scipy.sparse.diags(D_1 * u.ravel())
    D_U_2 = scipy.sparse.diags(D_2 * u.ravel())
    D_U_3 = scipy.sparse.diags(D_3 * u.ravel())

    D_U = scipy.sparse.vstack([
        scipy.sparse.hstack([D_U_1, D_U_3]),
        scipy.sparse.hstack([D_U_3, D_U_2]),
    ])


    return D_U


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
