import torch
from neural_network import *
import matplotlib.pyplot as plt
import math
import tqdm
import numpy as np
from optimizer import Adam, GlobalConfigAdam, MConfig, LocalConfigAdam, CosineAnnealingLR
from scipy.stats import qmc
import random
import json
import copy

from hydra.utils import to_absolute_path

# sample data
class BurgersDataset:

    def __init__(self, seed, use_lhs=True):
        
        self.seed = seed
        self.use_lhs = use_lhs

        self.sampler_lhs_1d = qmc.LatinHypercube(d=1, seed=seed)
        self.sampler_lhs_2d = qmc.LatinHypercube(d=2, seed=seed)

    def sample_boundary(self, n_boundary = 50):

        if self.use_lhs:
            sample = self.sampler_lhs_1d.random(n=n_boundary)
            
            t_np = sample[:, 0]
            t = torch.from_numpy(t_np).float()
        else:
            t = torch.rand(size=(n_boundary, 1))

        # half half
        x = torch.cat([-1.0 * torch.ones(size=(n_boundary // 2, 1)), 1.0 * torch.ones(size=(n_boundary // 2, 1))], dim=0)
        
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)

        return x, t

    def sample_initial(self, n_initial = 50):
        
        if self.use_lhs:
            
            sample = self.sampler_lhs_1d.random(n=n_initial)
            x_np = 2.0 * sample[:, 0:1] - 1.0
            x = torch.from_numpy(x_np).float()
        else:
            # u -> x
            # x = a * u + b
            # b = -1
            # a = 2
            x =  2.0 * torch.rand(size=(n_initial, 1)) - 1.0
        
        t = 0 * x

        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)

        return x, t

    def sample_interior(self, n_interior = 2_000):

        if self.use_lhs:
            
            sample = self.sampler_lhs_2d.random(n=n_interior)
        
            x_np = 2 * sample[:, 0:1] - 1
            t_np = sample[:, 1:2]
            
            x = torch.from_numpy(x_np).float()
            t = torch.from_numpy(t_np).float()

        else:

            x =  2 * torch.rand(size=(n_interior, 1)) - 1
            t  = torch.rand(size=(n_interior, 1))

        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)

        return x, t

    def get_sampling_functions(self):
        return [self.sample_initial, self.sample_boundary, self.sample_interior]

# show the domain samples
def draw_domain_samples(burgers_dataset: BurgersDataset):

    x_boundary, t_boundary = burgers_dataset.sample_boundary()
    x_initial, t_initial = burgers_dataset.sample_initial()
    x_interior, t_interior = burgers_dataset.sample_interior()

    plt.scatter(t_boundary, x_boundary, s=2, label="boundary")
    plt.scatter(t_initial, x_initial, s=2, label="initial")
    plt.scatter(t_interior, x_interior, s=2, label="interior")

    plt.legend()
    plt.xlabel("t")
    plt.ylabel("x")

    plt.savefig("burgers_domain_samples.png", dpi=300)

# calculate losses
def initial_loss(model, sample_initial_fn):

    x, t = sample_initial_fn()

    device = next(model.parameters()).device

    x = x.to(device)
    t = t.to(device)

    pred = model(x, t)

    target = -torch.sin(np.pi * x)

    return ((pred - target)**2).mean()

def boundary_loss(model, sample_boundary_fn):

    x, t = sample_boundary_fn()

    device = next(model.parameters()).device

    x = x.to(device)
    t = t.to(device)

    pred = model(x, t)

    # MSE with target 0
    return (pred**2).mean()

def pde_loss(model, sample_interior_fn):

    x, t = sample_interior_fn()

    device = next(model.parameters()).device

    x = x.to(device)
    t = t.to(device)

    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    # du/dt + u du /dx = nu d^2 u /dx^2

    # m x m -> m
    # jacobian is m x m (diagonal matrix) since one time only influences one output
    du_dt = torch.autograd.grad(outputs=u, inputs=t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # also an m dimensional vector
    du_dx = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    ddu_dxx = torch.autograd.grad(outputs=du_dx, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    visc = 0.01 / np.pi

    residual = du_dt + (u * du_dx) - (visc * ddu_dxx)

    return (residual**2).mean()

# EVERY PROBLEM NEEDS THESE
def get_dataset(seed, use_lhs):
    return BurgersDataset(seed, use_lhs)

def get_loss_functions():
    return [initial_loss, boundary_loss, pde_loss]

class Tester:

    def __init__(self, device):
    
        u_numerical = np.load(to_absolute_path("sols/burgers_numerical_solution.npz"))["u"]
        u_numerical = torch.from_numpy(u_numerical).to(device)
        N_x_val, N_t_val = u_numerical.shape

        x_val = torch.linspace(-1, 1, N_x_val).to(device)
        t_val = torch.linspace(0, 1, N_t_val).to(device)

        X_val, T_val = torch.meshgrid(x_val, t_val, indexing="ij")

        self.x_val = X_val.reshape(-1, 1)
        self.t_val = T_val.reshape(-1, 1)
        self.u_numerical = u_numerical.reshape(-1, 1)

    def compute_val_loss(self, model):
        
        model.eval()
        u_pred_val = model(self.x_val, self.t_val)
        val_mse = ((u_pred_val - self.u_numerical)**2).mean().item()

        return val_mse

def get_tester(device):
    return Tester(device)

def create_graphs(model, device):
    
    N_x = 200
    N_t = 200

    x_test = torch.linspace(-1, 1, N_x).to(device)
    t_test = torch.linspace(0, 1, N_t).to(device)

    X_test, T_test = torch.meshgrid(x_test, t_test, indexing="ij")

    x_test = X_test.reshape(-1, 1)
    t_test = T_test.reshape(-1, 1)

    u_pred = model(x_test, t_test)

    u_pred = u_pred.reshape(N_x, N_t).detach().cpu().numpy()
    u_numerical = np.load(to_absolute_path("sols/burgers_numerical_solution.npz"))["u"]

    plt.figure(figsize=(5, 5))
    plt.imshow(u_pred, extent=[0, 1, -1,  1], origin="lower", aspect="auto", cmap="coolwarm")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.savefig(f"pred.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow((u_pred - u_numerical)**2, extent=[0, 1, -1,  1], origin="lower", aspect="auto", cmap="Reds")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.savefig(f"se_pred.png", dpi=300, bbox_inches='tight')
    plt.close()

    # render out three times
    times = [0.25, 0.5, 0.75]

    for t in times:
        
        t_idx = int(t * N_t)

        plt.title(f"t = {t}")
        plt.plot(np.linspace(-1, 1, N_x), u_numerical[:, t_idx], label="data")
        plt.scatter(np.linspace(-1, 1, N_x), u_pred[:, t_idx], label="pinn", marker="x", color='#FF458A')
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.savefig(f"t_{t}.png", dpi=300)
        plt.close()