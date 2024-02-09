import os
import numpy as np
import torch

from skimage import io

from examples.tinyhome import TinyHomeEngineV1, print_grid, print_act
from examples.buffer import ReplayBuffer

from transformers import AutoTokenizer
from torch.nn import functional as F

from conv_mamba_lm import from_pretrained

from conv_mamba_lm import MambaLM, MambaLMConfig

from glob import glob

from matplotlib import pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

L = 5
num_actions = 5
num_obs_type = 4

nb_instances = 512
steps = 10000


# files = glob(os.path.join("/media/data_cifs/pathfinder_small/curv_contour_length_14/imgs/**/*"))
meta = np.load("/media/data_cifs/pathfinder_small/curv_contour_length_14/metadata/combined.npy")
root = "/media/data_cifs/pathfinder_small/curv_contour_length_14"

meta = meta[:100]

images, files, labels = [], [], []
meta = meta.astype(str)
for ro in meta:
    fl = os.path.join(root, str(ro[0]), str(ro[1]))
    la = int(ro[3])
    images.append(io.imread(fl))
    files.append(fl)
    labels.append(la)

H, W = 80, 80
images = np.asarray(images).astype(np.float32) / 255.
labels = np.asarray(labels)
images = torch.from_numpy(images)
labels = torch.from_numpy(labels)
images = F.interpolate(images[:, None], [H, W], mode="bicubic").squeeze(1)

bs = 6
d_model = 16  # Dimensionality
config = MambaLMConfig(
    d_model=d_model,
    n_layers=2,
    pscan=False,
    vocab_size=num_actions+num_obs_type,
    pad_vocab_size_multiple=num_actions+num_obs_type)
config.in_dim = 1
config.out_dim = 2
config.timesteps = 1
N, C, T, H, W = bs, d_model, config.timesteps, H, W

model = MambaLM(config).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

"""
mask = np.copy(images)
pt = [30, 30]
prop = 0.5
size = 30
stride = 30

# mask = mask.reshape(mask.shape[0], pt * pt, mask.shape[1] // pt, mask.shape[2] // pt)
mask = torch.ones_like(images).unfold(1, size, stride).unfold(2, size, stride).unfold(3, size, stride)
mask = mask.reshape(mask.shape[0], -1, size, size)

# mask = np.copy(images).reshape(mask.shape[0], mask.shape[1] // pt, mask.shape[2] // pt, pt * pt)
idx = (torch.rand(mask.shape[0], mask.shape[1]) > prop).float()[:, :, None, None]
import pdb;pdb.set_trace()
tokens = (mask * idx).reshape(*images.shape)
import pdb;pdb.set_trace()
"""
optim.zero_grad()
tokens = images[:, None, None].to(device)  # Add singleton dims for channel and time
labels = labels.to(device)
tokens = tokens[:bs]
y = labels[:bs]
logits = model(tokens)
loss = F.cross_entropy(logits, y)  # , reduction="none").mean()
loss.backward()
optim.step()

print("Loss: {}".format(loss))
ps = {k: v for k, v in model.named_parameters()}
print("Grad: {}".format(model.mamba.layers[0].mixer.A_log.grad.sum()))

