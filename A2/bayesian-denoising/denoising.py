import numpy as np
import utils
from numpy import newaxis as na


def expectation_maximization(
    X: np.ndarray,
    K: int,
    max_iter: int = 50,
    plot: bool = False,
    show_each: int = 5,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Number of data points, features
    N, m = X.shape
    # Init: Uniform weights, first K points as means, identity covariances
    alphas = np.full((K,), 1. / K)
    mus = X[:K]
    sigmas = np.tile(np.eye(m)[None], (K, 1, 1))

    for it in range(max_iter):
        print(it)
        # TODO: Implement (9) - (11)
        L = np.linalg.cholesky(np.linalg.inv(sigmas)) #K, m, m
        x_mu = X[:, na, :] - mus[np.newaxis, :, :] # N, K, m
        slogdet = np.linalg.slogdet(L) # K
        logdet = slogdet.logabsdet * slogdet.sign # K
        L_x_mu = L[na, :, :].transpose(0, 1, 3, 2) @ x_mu[:, :, :, na] # N, K, m, m
        x_mu_norm = -0.5 * (
            np.linalg.norm(L_x_mu, ord=2, axis=(2, 3)) ** 2 + m * np.log(2 * np.pi)
        ) # N, K
        Beta = x_mu_norm + logdet[na, :] + np.log(alphas)[na, :] # N, K

        max_Beta = np.max(Beta, axis=1) # N
        logsumexp = max_Beta + np.log( # N
            np.sum( # N
                np.exp(Beta - max_Beta[:, na]), # N, K
                axis = 1
            )
        )

        gamma = np.exp(Beta - logsumexp[:, na]) # N, K

        gamma_sum = gamma.sum(0) # K
        alphas = gamma_sum / N # K
        mus = np.sum( # K, m
            gamma[:, :, na] * X[:, na, :], # N, K, m
            0
        ) / gamma_sum[:, na]
        x_mu_next = (X[:, na, :] - mus[na, :, :])[:, :, :, na] # N, K, m, 1
        x_mu_square =  x_mu_next @ x_mu_next.transpose(0, 1, 3, 2) # N, K, m, m
        sigma_tilde = np.sum( # K, m, m
            gamma[:, :, na, na] * x_mu_square.transpose(0, 1, 3, 2), # N, K, m, m
            0
        ) / gamma_sum[:, na, na]
        sigmas = sigma_tilde + np.identity(m)[na, :, :] * epsilon

        del L, x_mu, slogdet, logdet, L_x_mu, x_mu_norm, Beta, max_Beta, logsumexp, gamma, gamma_sum, x_mu_next, x_mu_square, sigma_tilde

        if it % show_each == 0 and plot:
            utils.plot_gmm(X, alphas, mus, sigmas)

    return alphas, mus, sigmas


def denoise(
    index: int = 1,
    K: int = 10,
    w: int = 5,
    alpha: float = 0.5,
    max_iter: int = 30,
    test: bool = False,
    sigma: float = 0.1
):
    alphas, mus, sigmas = utils.load_gmm(K, w)
    precs = np.linalg.inv(sigmas)
    precs_chol = np.linalg.cholesky(precs)  # "L" in the assignment sheet
    if test:
        # This is really two times `y` since we dont have access to `x` here
        x, y = utils.test_data(index)
    else:
        x, y = utils.validation_data(index, sigma=sigma, seed=1, w=w)
    # x is image-shaped, y is patch-shaped
    # Initialize the estimate with the noisy patches
    x_est = y.copy()
    m = w ** 2
    lamda = 1 / sigma ** 2
    E = np.eye(m) - np.full((m, m), 1 / m)

    # TODO: Precompute A, b (26)

    for it in range(max_iter):
        # TODO: Implement Line 3, Line 4 of Algorithm 1
        x_tilde = x_est
        x_est = alpha * x_est + (1 - alpha) * x_tilde

        if not test:
            u = utils.patches_to_image(x_est, x.shape, w)
            print(f"it: {it+1:03d}, psnr(u, y)={utils.psnr(u, x):.2f}")

    return utils.patches_to_image(x_est, x.shape, w)


def benchmark(K: int = 10, w: int = 5):
    for i in range(1, 5):
        utils.imsave(f'./test/img{i}_out.png', denoise(i, K, w, test=True))


def train(use_toy_data: bool = True, K: int = 2, w: int = 5):
    data = np.load('./toy.npy') if use_toy_data else utils.train_data(w)
    # Plot only if we use toy data
    alphas, mus, sigmas = expectation_maximization(data, K=K, plot=use_toy_data)
    # Save only if we dont use toy data
    if not use_toy_data:
        utils.save_gmm(K, w, alphas, mus, sigmas)


if __name__ == "__main__":
    do_training = True
    # Use the toy data to debug your EM implementation
    use_toy_data = False
    # Parameters for the GMM: Components and window size, m = w ** 2
    # Use K = 2 for toy/debug model
    K = 2
    w = 5
    if do_training:
        train(use_toy_data, K, w)
    else:
        for i in range(1, 6):
            denoise(i, K, w, test=False)

    # If you want to participate in the challenge, you can benchmark your model
    # Remember to upload the images in the submission.
    # benchmark(K, w)
