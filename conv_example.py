import torch

from examples.tinyhome import TinyHomeEngineV1, print_grid, print_act
from examples.buffer import ReplayBuffer

from transformers import AutoTokenizer
from torch.nn import functional as F

from conv_mamba_lm import from_pretrained

from conv_mamba_lm import MambaLM, MambaLMConfig


device = "cuda" if torch.cuda.is_available() else "cpu"

L = 5
num_actions = 5
num_obs_type = 4

nb_instances = 512
steps = 10000

envs = TinyHomeEngineV1(B=nb_instances, h=L, w=L)
buffer = ReplayBuffer(num_envs=nb_instances, capacity=int(1e6), obs_dim=L*L, act_dim=num_actions)

obs = envs.reset()

for _ in range(steps):
    a = torch.randint(low=0, high=num_actions, size=(nb_instances,))
    next_obs, rew = envs.step(a)

    buffer.store(obs.view(-1, L*L), a, rew.squeeze(1))
    obs = next_obs

d_model = 6  # Dimensionality
N, C, T, H, W = 10, d_model, 4, 24, 24
config = MambaLMConfig(
    d_model=d_model,
    n_layers=2,
    pscan=False,
    vocab_size=num_actions+num_obs_type,
    pad_vocab_size_multiple=num_actions+num_obs_type)
model = MambaLM(config).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

tokens = torch.randn([N, C, T, H, W]).to(device)
logits = model(tokens[:, :, :-1]) # (B, 26T-1, vocab_size)
output = tokens[:, :, 1:]
loss = F.cross_entropy(logits, output)  # , reduction="none").mean()

optim.zero_grad()
loss.backward()
optim.step()

print("Loss: {}".format(loss))
ps = {k: v for k, v in model.named_parameters()}
print("Grad: {}".format(model.mamba.layers[0].mixer.A_log.grad.sum()))

