import numpy as np
import numpy.lib.stride_tricks as ns
import imageio
import skimage.color as skc
import skimage.filters as skf


def edge_detection(im):
    '''Implement DoG smooth edge detection (Eq. 6)'''
    S_e = skf.gaussian(im, sigma=sigma_e, mode='constant')
    S_f = skf.gaussian(im, sigma=np.sqrt(1.6) * sigma_e, mode='constant')
    diff = S_e - (tau * S_f)
    edge = np.tanh(diff * phi_e) + 1
    edge = np.minimum(edge, 1)
    return edge

def luminance_quantization(im):
    '''Implement luminance quantization (Eq. 8)'''
    lmax = 100.0
    delta_l = lmax / n_bins
    Q = np.linspace(0, lmax, n_bins + 1) 

    def smooth_step(x):
        i_hat = np.argmin(np.abs(x - Q))
        Q_i_hat = Q[i_hat]
        return Q_i_hat + (delta_l / 2) * np.tanh(phi_q * (x - Q_i_hat))

    quantized_luminance = np.vectorize(smooth_step)(im)
    quantized_luminance = np.clip(quantized_luminance, 0, 255)

    return quantized_luminance

def bilateral_gaussian(im):
    # Radius of the Gaussian filter
    r = int(2 * sigma_s) + 1
    padded = np.pad(im, ((r, r), (r, r), (0, 0)), 'edge')
    '''
    Implement the bilateral Gaussian filter (Eq. 3).
    Apply it to the padded image.
    '''
    filtered = np.zeros_like(im)
    kernel = np.zeros((r * 2, r * 2))
    kernel[r][r] = 1
    kernel = skf.gaussian(kernel, sigma=sigma_s)
    # i'm not even gonna attempt making a bilateral gaussian efficient
    x_size, y_size, channels = im.shape
    for x in range(x_size):
        for y in range(y_size):
            window = padded[x:(x + 2 * r), y:(y + 2 * r)]
            diff = window - im[x, y]
            diff_norm = np.linalg.norm(diff, ord=2, axis=2)
            diff_gauss = np.exp(-(diff_norm ** 2 )/ (2 * sigma_r ** 2))

            weights = kernel * diff_gauss
            weights /= np.sum(weights)

            filtered[x][y][0] = np.sum(window[:,:,0] * weights)
            filtered[x][y][1] = np.sum(window[:,:,1] * weights)
            filtered[x][y][2] = np.sum(window[:,:,2] * weights)

    return filtered


def abstraction(im):
    filtered = skc.rgb2lab(im)
    for _ in range(n_e):
        filtered = bilateral_gaussian(filtered)
    imageio.imwrite(f'bilateral1.png',(skc.lab2rgb(filtered)*255).astype(np.uint8))
    edges = edge_detection(filtered[:, :, 0])
    imageio.imsave('edges.png', (edges * 255).astype(np.uint8))

    for _ in range(n_b - n_e):
        filtered = bilateral_gaussian(filtered)
    imageio.imwrite(f'bilateral2.png',(skc.lab2rgb(filtered)*255).astype(np.uint8))
    luminance_quantized = luminance_quantization(filtered[:, :, 0])
    imageio.imsave('luminance_quantized.png', (luminance_quantized * 255/100).astype(np.uint8))

    '''Get the final image by merging the channels properly'''
    filtered[:, :, 0] = np.multiply(edges, luminance_quantized)
    return skc.lab2rgb(filtered)


def psnr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    mse = np.mean((x - y) ** 2)
    m = 1
    return 10 * np.log10(m ** 2 / mse)

if __name__ == '__main__':
    # Algorithm
    n_e = 2
    n_b = 4
    # Bilateral Filter
    sigma_r = 4.25  # "Range" sigma
    sigma_s = 3.5  # "Spatial" sigma
    # Edge Detection
    sigma_e = 1
    tau = 0.98
    phi_e = 5
    # Luminance Quantization
    n_bins = 10
    phi_q = 0.7

    im = imageio.imread('./girl.png') / 255.
    abstracted = abstraction(im)
    
    im_ref = imageio.imread('./reference.png') / 255.
    print(psnr(im_ref, abstracted))
    
    imageio.imsave('abstracted.png', (np.clip(abstracted, 0, 1) * 255).astype(np.uint8))
    
