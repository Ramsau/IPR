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
        Beta = np.zeros((N, K))
        for k in range(K):
            L = np.linalg.cholesky(np.linalg.inv(sigmas[k])) #m, m
            x_mu = X[:, :] - mus[na, k, :] # N, m
            slogdet = np.linalg.slogdet(L) # scalar
            logdet = slogdet.logabsdet * slogdet.sign # scalar
            L_x_mu = L.T[na, :, :] @ x_mu[:, :, na] # N, m, m
            x_mu_norm = -0.5 * (
                np.linalg.norm(L_x_mu, ord=2, axis=(1, 2)) ** 2 + m * np.log(2 * np.pi)
            ) # N
            Beta[:, k] = x_mu_norm + logdet + np.log(alphas[k]) # N

        max_Beta = np.zeros(N)
        logsumexp = np.zeros(N)
        for i in range(N):
            max_Beta[i] = np.max(Beta[i])
            logsumexp[i] = max_Beta[i] + np.log(
                np.sum(
                    np.exp(Beta[i] - max_Beta[i]), # N, K
                )
            )

        for k in range(K):
            gamma = np.exp(Beta[:, k] - logsumexp[:]) # N
            gamma_sum = gamma.sum(0)
            alphas[k] = gamma_sum / N
            mus[k] = np.sum( # m
                gamma[:, na] * X[:, :], # N, m
                0
            ) / gamma_sum
            x_mu_next = (X[:, :] - mus[na, k, :])[:, :, na] # N, m, 1
            x_mu_square =  x_mu_next @ x_mu_next.transpose(0, 2, 1) # N, m, m
            sigma_tilde = np.sum( # m, m
                gamma[:, na, na] * x_mu_square.transpose(0, 2, 1), # N, m, m
                0
            ) / gamma_sum
            sigmas[k] = sigma_tilde + np.identity(m) * epsilon

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
        x, y = utils.test_data(index, w=w)
    else:
        x, y = utils.validation_data(index, sigma=sigma, seed=1, w=w)
    # x is image-shaped, y is patch-shaped
    # Initialize the estimate with the noisy patches
    x_est = y.copy()
    m = w ** 2
    lamda = 1 / sigma ** 2
    E = np.eye(m) - np.full((m, m), 1 / m)

    # TODO: Precompute A, b (26)
    sigma_inverse = np.linalg.inv(sigmas) # K, m, m
    A = np.linalg.inv(
        (lamda * np.eye(m))[na, :, :] + E.T[na, :, :] @ sigma_inverse @ E[na, :, :]
    ) # K, m, m
    b = (sigma_inverse @ E[na, :, :] @ mus[:, :, na]) # K, m
    N = x_est.shape[0]

    for it in range(max_iter):
        # TODO: Implement Line 3, Line 4 of Algorithm 1
        max_val = float('-inf')
        Beta = np.zeros((N, K))
        X = (E[na, :, :] @ x_est[:, :, na])[:, :, 0]
        for k in range(K):
            L = np.linalg.cholesky(np.linalg.inv(sigmas[k])) #m, m
            x_mu = X[:, :] - mus[na, k, :] # N, m
            slogdet = np.linalg.slogdet(L) # scalar
            logdet = slogdet.logabsdet * slogdet.sign # scalar
            L_x_mu = L.T[na, :, :] @ x_mu[:, :, na] # N, m, m
            x_mu_norm = -0.5 * (
                    np.linalg.norm(L_x_mu, ord=2, axis=(1, 2)) ** 2 + m * np.log(2 * np.pi)
            ) # N
            Beta[:, k] = x_mu_norm + logdet + np.log(alphas[k]) # N

        max_Beta = np.zeros(N)
        k_max = np.zeros(N, dtype=int)
        for i in range(N):
            max_Beta[i] = np.max(Beta[i])
            k_max[i] = np.argmax(Beta[i])


        x_tilde = (A[k_max] @ (lamda * y[:, :, na] + b[k_max]))[:, :, 0]
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
    do_training = False
    # Use the toy data to debug your EM implementation
    use_toy_data = False
    # Parameters for the GMM: Components and window size, m = w ** 2
    # Use K = 2 for toy/debug model
    K = 10
    w = 5
    if do_training:
        train(use_toy_data, K, w)
    else:
        for i in range(1, 6):
            denoise(i, K, w, test=False)
            pass
    # benchmark(K, w)

    # If you want to participate in the challenge, you can benchmark your model
    # Remember to upload the images in the submission.
    # benchmark(K, w)
