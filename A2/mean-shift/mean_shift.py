import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch as th

# It is highly recommended to set up pytorch to take advantage of CUDA GPUs!
device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

# Choose the size of the image here. Prototyping is easier with 128.
M = 128
# The reference implementation works in chunks. Set this as high as possible
# while fitting the intermediary calculations in memory.
simul = M ** 2 // 1

im = th.from_numpy(imageio.imread(f'./{M}.png') / 255.).to(device)

# Visualization
_, ax = plt.subplots(1, 2)
ax[0].imshow(im.cpu().numpy())
artist = ax[1].imshow(im.cpu().numpy())

# zeta_values = [1, 4]
# h_values = [0.1, 0.3]

# # best values for Distinguish 2 classes: Sky and non-sky.
# zeta_values = [0.4]
# h_values = [0.5]

# best values for Distinguish 8 classes: Sky, the six clearly visible houses, and the pavement.
zeta_values = [1]
h_values = [0.4]

for zeta in zeta_values:
    y, x = th.meshgrid(
        th.linspace(-zeta, zeta, M, device=device),
        th.linspace(-zeta, zeta, M, device=device),
        indexing='xy'
    )
    features = th.cat((im, y[..., None], x[..., None]), dim=-1).reshape(-1, 5)
    for h in h_values:
        # The `shifted` array contains the iteration variables
        shifted = features.clone()
        # The `to_do` array contains the indices of the pixels for which the
        # stopping criterion is _not_ yet met.
        to_do = th.arange(M ** 2, device=device)
        while len(to_do):
            # We walk through the points in `shifted` in chunks of `simul`
            # points. Note that for each point, you should compute the distance
            # to _all_ other points, not only the points in the current chunk.
            chunk_size = min(simul, len(to_do))
            chunk_indices = to_do[:chunk_size]
            chunk = shifted[chunk_indices].clone()

            # Mean shift iterations (15)
            distances = th.cdist(chunk, features) ** 2
            weights = (distances <= h**2).double()
            numerator = th.sum(weights[:, :, None] * features[None, :, :], dim=1)
            denominator = th.sum(weights, dim=1, keepdim=True)
            shifted[chunk_indices] = numerator / denominator

            # Termination criterion (17)
            cond = th.norm(shifted[chunk_indices] - chunk, dim=1)**2 < 1e-6
            # We only keep the points for which the stopping criterion is not met.
            # `cond` should be a boolean array of length `simul` that indicates
            # which points should not be kept.
            to_do = to_do[chunk_size:]
            to_do = th.cat((to_do, chunk_indices[~cond]))

            artist.set_data(shifted.view(M, M, 5).cpu()[:, :, :3])
            # plt.show()
            plt.pause(0.01)
            print(f"Zeta={zeta}, h={h}: {len(to_do)} Pixels left.")
        # Reference images were saved using this code.
        print(f"--Saving zeta={zeta}, h={h}--")
        imageio.imsave(
            f'./implementation/{M}/zeta_{zeta:1.1f}_h_{h:.2f}.png',
            (shifted.reshape(M, M, 5)[..., :3].clone().cpu().numpy()*255).astype("uint8")
        )


