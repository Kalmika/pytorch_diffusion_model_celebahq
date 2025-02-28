import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm

device = 1

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def calc_t_emb(ts, t_emb_dim):
    """
    Embed time steps into a higher dimension space
    """
    assert t_emb_dim % 2 == 0

    half_dim = t_emb_dim // 2
    t_emb = np.log(10000) / (half_dim - 1)
    t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
    t_emb = t_emb.to(device)
    t_emb = ts * t_emb
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)
    
    return t_emb

def flatten(v):
    """
    Flatten a list of lists/tuples
    """
    return [x for y in v for x in y]

def rescale(x):
    """
    Rescale a tensor to 0-1
    """
    return (x - x.min()) / (x.max() - x.min())

def find_max_epoch(path, ckpt_name):
    """
    Find max epoch in path, formatted ($ckpt_name)_$epoch.pkl, such as unet_ckpt_30.pkl
    """
    files =  os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= len(ckpt_name) + 5:
            continue
        if f[:len(ckpt_name)] == ckpt_name and f[-4:]  == '.pkl':
            number = f[len(ckpt_name)+1:-4]
            try:
                epoch = max(epoch, int(number))
            except:
                continue
    return epoch

def print_size(net):
    """
    Print the number of parameters of a network
    """
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)

def std_normal(size):
    """
    Generate a standard Gaussian of a given size
    """
    return torch.normal(0, 1, size=size).to(device)

def tmp_ti_remove(self, n_sample: int, size, device) -> torch.Tensor:
    x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

    for i in range(self.n_T, 1, -1):
        z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
        eps = self.eps_model(x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1))
        x0_t = (x_i - eps * (1 - self.alphabar_t[i]).sqrt()) / self.alphabar_t[i].sqrt()
        c1 = self.eta * ((1 - self.alphabar_t[i] / self.alphabar_t[i - 1]) * (1 - self.alphabar_t[i - 1]) / (
                1 - self.alphabar_t[i])).sqrt()
        c2 = ((1 - self.alphabar_t[i - 1]) - c1 ** 2).sqrt()
        x_i = self.alphabar_t[i - 1].sqrt() * x0_t + c1 * z + c2 * eps

    return x_i

def sampling_ddim(net, size, T, Alpha, Alpha_bar, output_directory, eta = 0.0):
    """
    Perform the complete sampling step according to DDIM p(x_0|x_T)
    """
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(size) == 4
    print('begin sampling, total steps = %s' % T)

    x = std_normal(size)
    with torch.no_grad():
        for t in range(T-1,-1,-1):
            if t % 10 == 1:                
                for i in range(x.shape[0]):
                    save_image(rescale(x[i]), os.path.join(output_directory, 'sampling-{}_t={}.jpg'.format(i,t)))
                print('reverse step:', t)


            ts = (t * torch.ones((size[0], 1))).to(device)
            noise_pred = net((x, ts,))

            # DDIM
            # Instead, let's do it ourselves:
            prev_t = max(1, t-1) # t-1

            # Noise adding
            sigma = 0 if eta == 0 else eta * ((1 - Alpha_bar[prev_t]) / (1 - Alpha_bar[t]) * (1 - Alpha_bar[t] / Alpha_bar[prev_t])).sqrt()

            predicted_x0 = (x - (1-Alpha_bar[t]).sqrt()*noise_pred) / Alpha_bar[t].sqrt()
            direction_pointing_to_xt = (1 - Alpha_bar[prev_t] - sigma**2 ).sqrt()*noise_pred
            x = Alpha_bar[prev_t].sqrt() * predicted_x0 + direction_pointing_to_xt

            if t > 0:
                x = x + sigma * std_normal(size)
            if t < 15:                
                for i in range(x.shape[0]):
                    save_image(rescale(x[i]), os.path.join(output_directory, 'sampling-{}_t={}_.jpg'.format(i,t)))
                print('reverse step:', t)
    return x
    

def sampling(net, size, T, Alpha, Alpha_bar, Sigma):
    """
    Perform the complete sampling step according to p(x_0|x_T)
    """
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 4
    print('begin sampling, total steps = %s' % T)

    x = std_normal(size)
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1)):
            if t % 100 == 0:                
                for i in range(x.shape[0]):
                    save_image(rescale(x[i]), os.path.join('plots/celebahq_2/', 'sampling-{}_t={}.jpg'.format(i,t)))
                # print('reverse step:', t)
            
            ts = (t * torch.ones((size[0], 1))).to(device)
            noise_pred = net((x,ts,))
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * noise_pred) / torch.sqrt(Alpha[t])
            if t > 0:
                if t < 15:
                    x = x + Sigma[t] * std_normal(size)

            if t < 15:   
                for i in range(x.shape[0]):                        
                    save_image(rescale((Sigma[t] * std_normal(size)[i])), os.path.join('plots/celebahq_2/', 'sampling-{}_t={}_SIGMA.jpg'.format(i,t)))
                    save_image(rescale(x[i]), os.path.join('plots/celebahq_2/', 'sampling-{}_t={}_XXX.jpg'.format(i,t)))
                print("Sigma :,",Sigma[t])             
                for i in range(x.shape[0]):
                    save_image(rescale(x[i]), os.path.join('plots/celebahq_2/', 'sampling-{}_t={}____.jpg'.format(i,t)))
                # print('reverse step:', t)
                                
    for i in range(x.shape[0]):
        save_image(rescale(x[i]), os.path.join('plots/celebahq_2/', 'img2-{}_t={}_FINAL.jpg'.format(i,t)))
    return x

def training_loss(net, loss_fn, T, X, Alpha_bar):
    """
    Compute the loss_fn (default is \ell_2) loss of (epsilon - epsilon_theta)
    """
    B, C, H, W = X.shape
    ts = torch.randint(T, size=(B,1,1,1)).to(device)
    z = std_normal(X.shape)
    xt = torch.sqrt(Alpha_bar[ts]) * X + torch.sqrt(1-Alpha_bar[ts]) * z
    epsilon_theta = net((xt, ts.view(B,1),))
    return loss_fn(epsilon_theta, z)

    