import torch

from examples.tinyhome import TinyHomeEngineV1, print_grid, print_act
from examples.buffer import ReplayBuffer

from transformers import AutoTokenizer
from torch.nn import functional as F

from mamba_lm import from_pretrained

from mamba_lm import MambaLM, MambaLMConfig


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


config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=num_actions+num_obs_type, pad_vocab_size_multiple=num_actions+num_obs_type, pscan=False)
model = MambaLM(config).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-3)


B, T = 64, 10
batch = buffer.sample(B, T)

obs = torch.tensor(batch['obs']).long().to(device)
act = torch.tensor(batch['act']).long().to(device)

tokens = torch.cat([obs, torch.zeros(B, T, 1, dtype=torch.int, device='cuda')], dim=2).view(B, 26*T) # (B, 26T)
tokens[:, 25::26] = act+4

output = tokens[:, 1:].reshape(-1)
import pdb;pdb.set_trace()
logits = model(tokens[:, :-1]) # (B, 26T-1, vocab_size)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), output)

optim.zero_grad()
loss.backward()
optim.step()

print("Loss: {}".format(loss))
ps = {k: v for k, v in model.named_parameters()}
print("Grad: {}".format(model.mamba.layers[0].mixer.A_log.grad.sum()))

