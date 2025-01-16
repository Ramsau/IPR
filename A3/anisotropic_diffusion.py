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

    u_tilde = gaussian_filter(u, sigma_u)
    u_deriv = nabla @ u_tilde.ravel()
    u_tilde_x = u_deriv[:u_tilde.size].reshape(u.shape)
    u_tilde_y = u_deriv[u_tilde.size:].reshape(u.shape)

    S_xx = gaussian_filter(u_tilde_x * u_tilde_x, sigma_g)
    S_yy = gaussian_filter(u_tilde_y * u_tilde_y, sigma_g)
    S_xy = gaussian_filter(u_tilde_x * u_tilde_y, sigma_g)

    S = np.array([
        [S_xx, S_xy],
        [S_xy, S_yy],
    ])
    eig_vals = np.zeros((u.shape[0], u.shape[1], 2))
    eig_vecs = np.zeros((u.shape[0], u.shape[1], 2, 2))

    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            # Extract the 2x2 matrix for pixel (i, j)
            S_ij = np.array([[S_xx[i, j], S_xy[i, j]], [S_xy[i, j], S_yy[i, j]]])
            vals, vecs = np.linalg.eigh(S_ij)
            # Sort eigenvalues in descending order
            order = np.argsort(vals)[::-1]
            eig_vals[i, j] = vals[order]
            eig_vecs[i, j] = vecs[:, order]

    mu_1 = eig_vals[:, :, 0]
    mu_2 = eig_vals[:, :, 1]
    v_1 = eig_vecs[:, :, :, 0]
    v_2 = eig_vecs[:, :, :, 1]

    # Compute λ1 and λ2 based on the mode
    if mode == 'ced':
        g = np.exp(-((mu_1 - mu_2)**2) / (2 * gamma**2))
        lambda_1 = alpha
        lambda_2 = alpha + (1 - alpha) * (1 - g)
    elif mode == 'eed':
        lambda_1 = (1 + mu_1 / gamma**2)**-0.5
        lambda_2 = 1
    else:
        raise ValueError("Unknown mode. Use 'ced' or 'eed'.")

    D_1 = lambda_1 * (v_1[..., 0]**2) + lambda_2 * (v_2[..., 0]**2)
    D_2 = lambda_1 * (v_1[..., 1]**2) + lambda_2 * (v_2[..., 1]**2)
    D_3 = lambda_1 * (v_1[..., 0] * v_1[..., 1]) + lambda_2 * (v_2[..., 0] * v_2[..., 1])

    diagD1 = sp.diags(D_1.ravel())
    diagD2 = sp.diags(D_2.ravel())
    diagD3 = sp.diags(D_3.ravel())

    D_U = sp.bmat([
        [diagD1, diagD3],
        [diagD3, diagD2]
    ], format='csr')

    return D_U

def psnr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    mse = np.mean((x - y) ** 2)
    m = 1
    return 10 * np.log10(m ** 2 / mse)

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
    reference_img = imageio.imread(mode + "_out_reference.png") / 255.
    print(psnr(reference_img, output))
