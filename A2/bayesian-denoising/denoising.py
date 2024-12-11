import numpy as np
import utils


def logsumexp(log_vals, axis=None, keepdims=False):
    """Compute log-sum-exp in a numerically stable way."""
    max_vals = np.max(log_vals, axis=axis, keepdims=True)
    stable_vals = log_vals - max_vals
    sum_exp = np.sum(np.exp(stable_vals), axis=axis, keepdims=True)
    result = max_vals + np.log(sum_exp)
    return result if keepdims else np.squeeze(result, axis=axis)


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

        # alphas = alphas
        # mus = mus
        # sigmas = sigmas

        log_resps = np.zeros((N, K))
        for k in range(K):
            diff = X - mus[k]
            log_prob = -0.5 * (np.sum(diff @ np.linalg.inv(sigmas[k]) * diff, axis=1) +
                            np.linalg.slogdet(sigmas[k])[1] + m * np.log(2 * np.pi))
            log_resps[:, k] = np.log(alphas[k]) + log_prob

        log_resps -= logsumexp(log_resps, axis=1, keepdims=True)  # utils.logsumexp handles stability
        gamma = np.exp(log_resps)

        Nk = gamma.sum(axis=0)
        alphas = Nk / N
        mus = (gamma.T @ X) / Nk[:, None]
        sigmas = np.zeros((K, m, m))
        for k in range(K):
            diff = X - mus[k]
            sigmas[k] = (gamma[:, k, None] * diff).T @ diff / Nk[k]
            sigmas[k] += epsilon * np.eye(m)


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
    A = np.zeros((K, m, m))
    b = np.zeros((K, m))
    for k in range(K):
        Ek = E.T @ precs[k] @ E
        A[k] = np.linalg.inv(lamda * np.eye(m) + Ek)
        b[k] = precs[k] @ E @ mus[k]

    for it in range(max_iter):
        # TODO: Implement Line 3, Line 4 of Algorithm 1
        x_tilde = np.zeros_like(x_est)
        for i, patch in enumerate(y):
            proj_patch = E @ patch
            kmax = np.argmax([np.log(alphas[k]) - 0.5 * (proj_patch - mus[k]).T @ precs[k] @ (proj_patch - mus[k])
                            for k in range(K)])
            x_tilde[i] = A[kmax] @ (lamda * patch + b[kmax])

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
    benchmark(K, w)