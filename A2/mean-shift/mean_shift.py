import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch as th

# It is highly recommended to set up pytorch to take advantage of CUDA GPUs!
device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

# Choose the size of the image here. Prototyping is easier with 128.
M = 256
# The reference implementation works in chunks. Set this as high as possible
# while fitting the intermediary calculations in memory.
simul = M ** 2 // 10

im = th.from_numpy(imageio.imread(f'./{M}.png') / 255.).to(device)

# Visualization
_, ax = plt.subplots(1, 2)
ax[0].imshow(im.cpu().numpy())
artist = ax[1].imshow(im.cpu().numpy())

for zeta in [1, 4]:
    y, x = th.meshgrid(
        th.linspace(-zeta, zeta, M, device=device),
        th.linspace(-zeta, zeta, M, device=device),
        indexing='xy'
    )
    features = th.cat((im, y[..., None], x[..., None]), dim=-1).reshape(-1, 5)
    for h in [0.1, 0.3]:
        # The `shifted` array contains the iteration variables
        shifted = features.clone()
        # The `to_do` array contains the indices of the pixels for which the
        # stopping criterion is _not_ yet met.
        to_do = th.arange(M ** 2, device=device)
        while len(to_do):
            # We walk through the points in `shifted` in chunks of `simul`
            # points. Note that for each point, you should compute the distance
            # to _all_ other points, not only the points in the current chunk.
            chunk = shifted[to_do[:simul]].clone()

            # Mean shift iterations (15)
            distances = th.cdist(chunk[:, :3], features[:, :3]) ** 2 + \
                       th.cdist(chunk[:, 3:], features[:, 3:]) ** 2
            weights = (distances <= h**2).float()
            numerator = th.sum(weights[:, :, None] * features[None, :, :], dim=1)
            denominator = th.sum(weights, dim=1, keepdim=True)
            shifted[to_do[:simul]] = numerator / denominator

            # Termination criterion (17)
            cond = th.norm(shifted[to_do[:simul]] - chunk, dim=1) >= 1e-6
            # We only keep the points for which the stopping criterion is not met.
            # `cond` should be a boolean array of length `simul` that indicates
            # which points should be kept.
            to_do = to_do[th.cat((
                cond, th.zeros(to_do.shape[0] - cond.shape[0], device=device).to(th.bool)
            ))]
            artist.set_data(shifted.view(M, M, 5).cpu()[:, :, :3])
            plt.pause(0.01)
        # Reference images were saved using this code.
        print(f"Saving zeta={zeta}, h={h}")
        imageio.imsave(
            f'./implementation/{M}/zeta_{zeta:1.1f}_h_{h:.2f}.png',
            (shifted.reshape(M, M, 5)[..., :3].clone().cpu().numpy()*255).astype("uint8")
        )


