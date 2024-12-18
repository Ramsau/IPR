import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch as th

# It is highly recommended to set up pytorch to take advantage of CUDA GPUs!
device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

# Choose the size of the image here. Prototyping is easier with 128.
M = 128
# The reference implementation works in chunks. Set this as high as possible
# while fitting the intermediary calculations in memory.
simul = 50 * 50

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
            chunk_before = chunk.clone()
            # TODO: Mean shift iterations (15), writing back the result into shifted.
            I = th.where( # N, simul
                th.norm(
                    (shifted[:, None] - chunk[None, :]) / h, # N, simul, m
                    dim=(-1)
                ) ** 2 <= 1,
                True,
                False
            )
            card = I.count_nonzero(dim=0)
            for i in range(len(chunk)):
                chunk[i] = shifted[I[:, i]].sum(0) / card[i]
                pass

            shifted[to_do[:simul]] = chunk.clone()
            # TODO: Termination criterion (17). cond should be True for samples
            # that need updating. Note that chunk contains the 'old' values.
            cond = th.where(
                th.norm(chunk - chunk_before, dim=1) ** 2 < 1e-6,
                False,
                True
            )
            # We only keep the points for which the stopping criterion is not met.
            # `cond` should be a boolean array of length `simul` that indicates
            # which points should be kept.
            to_do = to_do[th.cat((
                cond, cond.new_ones(to_do.shape[0] - cond.shape[0])
            ))]
            artist.set_data(shifted.view(M, M, 5).cpu()[:, :, :3])
            plt.show()
            plt.pause(0.01)
            print(f"Zeta={zeta}, h={h}: {len(to_do)} Pixels left.")
        # Reference images were saved using this code.
        imageio.imsave(
            f'./implementation/{M}/zeta_{zeta:1.1f}_h_{h:.2f}.png',
            (shifted.reshape(M, M, 5)[..., :3].clone().cpu().numpy()*255).astype("uint8")
        )
