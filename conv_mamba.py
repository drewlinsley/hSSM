import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_pscan import pscan
from einops import einsum

from fft import fft_conv, FFTConv1d, FFTConv2d, FFTConv3d


"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison.

- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


def rfs(k, n):
    rs = [k]
    for i in range(n - 1):
        rs.append(rs[-1] - 1 + k)
    return rs


def msconv(x, kern, bias, T=8, d=1, cat_dim=2):
    """Apply a multiscale 3d conv.

    Changing dilation would be faster, but also is suboptimal for approximating a spatial RNN.
    """
    p = [kern.shape[2] // 2, kern.shape[3] // 2, kern.shape[4] // 2]

    # Precompute scales
    _, _, t, osh, _ = x.shape
    scales = rfs(kern.shape[-1], T)
    scales = [osh - s for s in scales]

    # Downsample
    zs = []
    for idx, s in enumerate(scales):
        if idx > 0:
            # zx = F.interpolate(x, scale_factor=[1, 1. / 2 ** s, 1. / 2 ** s], mode="nearest")
            zx = F.interpolate(x, size=[t, s, s], mode="nearest")
        else:
            zx = x
        zx = F.conv3d(zx, kern, bias=bias, dilation=d, padding=p)
        if idx > 0:
            # zx = F.interpolate(zx, scale_factor=[1, 2 ** s, 2 ** s], mode="nearest")
            zx = F.interpolate(zx, size=[t, osh, osh], mode="nearest")
        zs.append(zx)
    return torch.stack(zs, cat_dim)  # Scales=Timesteps


@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        #self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # Only propogate final step
            x = x[:, :, [0]]

        #x = self.norm_f(x)

        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        # self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.in_proj = nn.Conv3d(config.d_model, 2 * config.d_inner, kernel_size=[1, 3, 3], padding=[1//2, 3//2, 3//2], bias=config.bias)

        # self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
        #                       kernel_size=config.d_conv, bias=config.conv_bias, 
        #                       groups=config.d_inner,
        #                       padding=config.d_conv - 1)
        self.conv1d = nn.Conv3d(in_channels=config.d_inner, out_channels=config.d_inner,
                              kernel_size=[config.d_conv, 1, 1], bias=config.conv_bias,
                              groups=config.d_inner,
                              padding=[config.d_conv - 1, 1//2, 1//2])

        # projects x to input-dependent Δ, B, C
        # self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.x_proj = nn.Conv3d(config.d_inner, config.dt_rank + 2 * config.d_state, kernel_size=[1, 1, 1], padding=[1//2, 1//2, 1//2], bias=False)

        # projects Δ from dt_rank to d_inner
        # self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        self.dt_proj = nn.Conv3d(config.dt_rank, config.d_inner, kernel_size=[1, 1, 1], padding=[1//2, 1//2, 1//2], bias=True)

        k = torch.randn(config.d_inner, config.d_inner, 1, 7, 7)
        self.spatial_kernel_A = nn.Parameter(k)
        self.spatial_bias_A = nn.Parameter(torch.rand(config.d_inner))
        # k = torch.randn(config.d_inner, config.d_inner, 1, 7, 7)
        # self.spatial_kernel_B = nn.Parameter(k)
        # self.spatial_bias_B = nn.Parameter(torch.rand(config.d_inner))

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            # self.dt_bias.copy_(inv_dt)

        self.dt_proj = nn.Conv3d(config.dt_rank, config.d_inner, kernel_size=[1, 1, 1], padding=[1//2, 1//2, 1//2], bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        # self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.out_proj = nn.Conv3d(config.d_inner, config.d_model, kernel_size=[1, 1, 1], padding=[1//2, 1//2, 1//2], bias=config.bias)

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)

        N, C, L, H, W = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, L, ED), (B, L, ED)

        # x branch
        # x = x.transpose(1, 2) # (B, ED, L)
        # x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        # x = x.transpose(1, 2) # (B, L, ED)
        x = self.conv1d(x)[:, :, :L]

        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()
        # TODO remove .float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        # CHANGE X_PROJ -- need n-scales of these

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        # delta needs to be precomputed multiscale conv
        delta = F.softplus(self.dt_proj(delta)) # (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A[None, :, None, None, None, :]) # (B, L, C, H, W, N)
        deltaB = delta[..., None] * B[:, None].permute((0, 1, 3, 4, 5, 2))

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, C, H, W, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)


        # deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        # deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)
        # At delta, if we multiscale conv above, then we need to integrate below in the timestep loop
        deltaA = torch.exp(delta.unsqueeze(-1) * A[None, :, None, None, None, :]) # (B, L, C, H, W, N)
        deltaB = delta[..., None] * B[:, None].permute((0, 1, 3, 4, 5, 2))

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, C, H, W, N)

        # Spatial convs
        # BX = BX.squeeze(2).permute(0, 1, 4, 2, 3)
        # BX = msconv(BX, self.spatial_kernel_B, bias=self.spatial_bias_B, cat_dim=-1)
        deltaA = deltaA.squeeze(2).permute(0, 1, 4, 2, 3)
        deltaA = msconv(deltaA, self.spatial_kernel_A, bias=self.spatial_bias_A, cat_dim=2)
        deltaA = deltaA.permute(0, 1, 2, 4, 5, 3)

        # B = fft_conv(B, self.spatial_kernel, bias=self.spatial_bias)
        # h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        N, K, L, H, W, Hds = deltaA.shape
        # N, K, L, H, W, Hds = BX.shape

        h = torch.zeros(x.size(0), self.config.d_inner, H, W, self.config.d_state, device=deltaA.device)
        hs = []

        # BX is fixed in time
        # deltaA is time-varying
        BX = BX.squeeze(2)
        for t in range(0, L):
            h = deltaA[:, :, t] * h + BX
            hs.append(h)
        hs = torch.stack(hs, dim=2) # (B, L, ED, N)

        # y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        y = einsum(hs, C, "B K L H W N, B N L H W -> B K L H W")  # CHECK THIS
        y = y + D[None, :, None, None, None] * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        # TODO remove .float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        # todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones([1, d_model, 1, 1, 1]))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
