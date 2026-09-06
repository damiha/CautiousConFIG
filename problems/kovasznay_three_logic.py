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
import multiprocessing
from tqdm.contrib.concurrent import process_map
import copy
from hydra.utils import to_absolute_path

class KovasznayThreeLossesDataset:

    def __init__(self, seed, use_lhs=True):
        
        self.seed = seed
        self.use_lhs = use_lhs

        self.sampler_lhs_1d = qmc.LatinHypercube(d=1, seed=seed)
        self.sampler_lhs_2d = qmc.LatinHypercube(d=2, seed=seed)

        self.x_start = -0.5
        self.x_end = 1
        self.y_start = -0.5
        self.y_end = 1.5

    def sample_boundary(self, n_boundary = 500):

        n_group = n_boundary // 4

        x_left_right = torch.cat([self.x_start * torch.ones((n_group, )), self.x_end * torch.ones((n_group, ))])
        y_top_bottom = torch.cat([self.y_start * torch.ones((n_group, )), self.y_end * torch.ones((n_group, ))])

        # x_start, y random
        # x end, y random

        # x random, y start
        # x random, y end

        if self.use_lhs:

            sample_y_left = self.sampler_lhs_1d.random(n=n_group)
            sample_y_right = self.sampler_lhs_1d.random(n=n_group)
            
            sample_x_top = self.sampler_lhs_1d.random(n=n_group)
            sample_x_bottom = self.sampler_lhs_1d.random(n=n_group)

            sample_x_combined = np.concatenate([sample_x_top, sample_x_bottom])
            sample_y_combined = np.concatenate([sample_y_left, sample_y_right])

            x_random_np = (self.x_end - self.x_start) * sample_x_combined[:, 0] + self.x_start
            y_random_np = (self.y_end - self.y_start) * sample_y_combined[:, 0] + self.y_start

            x_random = torch.from_numpy(x_random_np).float()
            y_random = torch.from_numpy(y_random_np).float()

        else:
            x_random = (self.x_end - self.x_start) * torch.rand((2 * n_group, )) + self.x_start
            y_random = (self.y_end - self.y_start) * torch.rand((2 * n_group, )) + self.y_start

        #print(f"{x_left_right.shape=}")
        #print(f"{x_random.shape=}")
        #print(f"{y_top_bottom.shape=}")
        #print(f"{y_random.shape=}")

        x = torch.cat([x_left_right, x_random])
        y = torch.cat([y_random, y_top_bottom])

        #print("boundary")
        #print(f"{x.shape=}, {y.shape=}")

        return x, y

    def sample_interior(self, n_interior = 20_000):

        if self.use_lhs:
            
            sample = self.sampler_lhs_2d.random(n=n_interior)
        
            x_np = (self.x_end - self.x_start) * sample[:, 0] + self.x_start
            y_np = (self.y_end - self.y_start) * sample[:, 1] + self.y_start
            
            x = torch.from_numpy(x_np).float()
            y = torch.from_numpy(y_np).float()

        else:

            x =  (self.x_end - self.x_start) * torch.rand(size=(n_interior, )) + self.x_start
            y  = (self.y_end - self.y_start) * torch.rand(size=(n_interior, )) + self.y_start

        #print("interior")
        #print(f"{x.shape=}, {y.shape=}")

        return x, y

    def get_sampling_functions(self):
        
        return [self.sample_boundary, self.sample_interior, self.sample_interior]

def draw_domain_samples(dataset: KovasznayThreeLossesDataset):

    x_boundary, y_boundary = dataset.sample_boundary()
    x_interior, y_interior = dataset.sample_interior()

    plt.scatter(x_boundary, y_boundary, s=2, label="boundary")
    plt.scatter(x_interior, y_interior, s=2, label="interior")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig("kovasznay_domain_samples.png", dpi=300)
    plt.close()

def boundary_loss(model, sample_boundary_fn):

    x, y = sample_boundary_fn()

    device = next(model.parameters()).device

    x = x.to(device)
    y = y.to(device)

    n_boundary = len(x)
    n_group = n_boundary // 4

    # x in {-0.5, 1}
    # y in [-0.5, 1.5]
    '''
    x_left = x[:n_group]
    x_right = x[n_group:(2 * n_group)]

    y_random_for_x_left = y[:n_group]
    y_random_for_x_right = y[n_group:(2 * n_group)]

    y_top = y[(2 * n_group):(3 * n_group)]
    y_bottom = y[(3 * n_group):]
    x_random_for_y_top = x[(2 * n_group):(3 * n_group)]
    x_random_for_y_bottom = x[(3 * n_group):]
    '''

    nu = 1 / 40
    lambda_val = 1 / (2 * nu) - np.sqrt(1 / (4 * nu**2) + 4 * np.pi**2)

    u_target = 1 - torch.exp(lambda_val * x) * torch.cos(2 * np.pi * y)
    v_target = lambda_val / (2 * torch.pi) * torch.exp(lambda_val * x) * torch.sin(2 * torch.pi * y)
    p_target = 0.5 * (1 - torch.exp(2 * lambda_val * x))

    pred = model(x, y)
    u_pred = pred[:, 0]
    v_pred = pred[:, 1]
    p_pred = pred[:, 2]

    return torch.mean((u_pred - u_target)**2) + torch.mean((v_pred - v_target)**2) + torch.mean((p_pred - p_target)**2)

def momentum_loss(model, sample_interior_fn):

    # Reynolds number
    Re = 40

    x, y = sample_interior_fn()

    device = next(model.parameters()).device

    x = x.to(device).requires_grad_(True)
    y = y.to(device).requires_grad_(True)

    pred = model(x, y)

    u = pred[:, 0]
    v = pred[:, 1]
    p = pred[:, 2]

    '''
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    dp_dx = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]

    du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    dp_dy = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

    du_dxx = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    du_dyy = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]

    dv_dxx = torch.autograd.grad(dv_dx, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    dv_dyy = torch.autograd.grad(dv_dy, y, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    '''

    grads_u = torch.autograd.grad(
        outputs=u,
        inputs=[x, y],
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )
    du_dx, du_dy = grads_u[0], grads_u[1]

    grads_v = torch.autograd.grad(
        outputs=v,
        inputs=[x, y],
        grad_outputs=torch.ones_like(v),
        create_graph=True,
    )
    dv_dx, dv_dy = grads_v[0], grads_v[1]

    grads_p = torch.autograd.grad(
        outputs=p,
        inputs=[x, y],
        grad_outputs=torch.ones_like(p),
        create_graph=True,
    )
    dp_dx, dp_dy = grads_p[0], grads_p[1]

    du_dxx = torch.autograd.grad(
        outputs=du_dx, inputs=x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]
    du_dyy = torch.autograd.grad(
        outputs=du_dy, inputs=y, grad_outputs=torch.ones_like(du_dy), create_graph=True
    )[0]

    dv_dxx = torch.autograd.grad(
        outputs=dv_dx, inputs=x, grad_outputs=torch.ones_like(dv_dx), create_graph=True
    )[0]
    dv_dyy = torch.autograd.grad(
        outputs=dv_dy, inputs=y, grad_outputs=torch.ones_like(dv_dy), create_graph=True
    )[0]

    momentum_x_residual = (u * du_dx) + (v * du_dy) + dp_dx - (1 / Re) * (du_dxx + du_dyy)
    momentum_y_residual = (u * dv_dx) + (v * dv_dy) + dp_dy - (1 / Re) * (dv_dxx + dv_dyy)

    #continuity_residual = du_dx + dv_dy

    return torch.mean(momentum_x_residual**2) + torch.mean(momentum_y_residual**2)

def continuity_loss(model, sample_interior_fn):

    # Reynolds number
    Re = 40

    x, y = sample_interior_fn()

    device = next(model.parameters()).device

    x = x.to(device).requires_grad_(True)
    y = y.to(device).requires_grad_(True)

    pred = model(x, y)

    u = pred[:, 0]
    v = pred[:, 1]
    p = pred[:, 2]

    '''
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    dp_dx = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]

    du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    dp_dy = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

    du_dxx = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    du_dyy = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]

    dv_dxx = torch.autograd.grad(dv_dx, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    dv_dyy = torch.autograd.grad(dv_dy, y, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    '''

    du_dx = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    dv_dy = torch.autograd.grad(
        outputs=v,
        inputs=y,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
    )[0]

    '''
    grads_p = torch.autograd.grad(
        outputs=p,
        inputs=[x, y],
        grad_outputs=torch.ones_like(p),
        create_graph=True,
    )
    dp_dx, dp_dy = grads_p[0], grads_p[1]

    du_dxx = torch.autograd.grad(
        outputs=du_dx, inputs=x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]
    du_dyy = torch.autograd.grad(
        outputs=du_dy, inputs=y, grad_outputs=torch.ones_like(du_dy), create_graph=True
    )[0]

    dv_dxx = torch.autograd.grad(
        outputs=dv_dx, inputs=x, grad_outputs=torch.ones_like(dv_dx), create_graph=True
    )[0]
    dv_dyy = torch.autograd.grad(
        outputs=dv_dy, inputs=y, grad_outputs=torch.ones_like(dv_dy), create_graph=True
    )[0]

    momentum_x_residual = (u * du_dx) + (v * du_dy) + dp_dx - (1 / Re) * (du_dxx + du_dyy)
    momentum_y_residual = (u * dv_dx) + (v * dv_dy) + dp_dy - (1 / Re) * (dv_dxx + dv_dyy)
    '''

    continuity_residual = du_dx + dv_dy

    return torch.mean(continuity_residual**2)

# EVERY PROBLEM NEEDS THESE FOUR
def get_dataset(seed, use_lhs):
    return KovasznayThreeLossesDataset(seed, use_lhs)

def get_loss_functions():
    return [boundary_loss, momentum_loss, continuity_loss]

class Tester:

    def __init__(self, device):
    
        sol_analytical = np.load(to_absolute_path("sols/kovasznay_analytical_solution.npz"))
        u_x_sol = sol_analytical["u_x"]
        u_y_sol = sol_analytical["u_y"]
        p_sol = sol_analytical["p"]

        N_y, N_x = u_x_sol.shape

        sol_analytical = np.concatenate([u_x_sol.reshape(N_y, N_x, 1), u_y_sol.reshape(N_y, N_x, 1), p_sol.reshape(N_y, N_x, 1)], axis=-1)
        sol_analytical = torch.from_numpy(sol_analytical).to(device)
        
        x_val = torch.linspace(-0.5, 1, N_x)
        y_val = torch.linspace(-0.5, 1.5, N_y)
        Y_val, X_val = torch.meshgrid(y_val, x_val, indexing="ij")

        self.x_val = X_val.reshape(-1, 1).to(device)
        self.y_val = Y_val.reshape(-1, 1).to(device)
        self.sol_analytical = sol_analytical.reshape(-1, 3)

    def compute_val_loss(self, model):
        
        model.eval()
        pred = model(self.x_val, self.y_val)
        val_mse = ((pred - self.sol_analytical)**2).mean().item()
        return val_mse

def get_tester(device):
    return Tester(device)

def create_graphs(model, device):

    sol_analytical = np.load(to_absolute_path("sols/kovasznay_analytical_solution.npz"))
    u_x_sol = sol_analytical["u_x"]
    u_y_sol = sol_analytical["u_y"]
    p_sol = sol_analytical["p"]

    # for inference to PINN
    N_y, N_x = u_x_sol.shape

    x_test = torch.linspace(-0.5, 1, N_x)
    y_test = torch.linspace(-0.5, 1.5, N_y)
    Y_test, X_test = torch.meshgrid(y_test, x_test, indexing="ij")

    x_test = X_test.reshape(-1, 1).to(device)
    y_test = Y_test.reshape(-1, 1).to(device)

    pred = model(x_test, y_test)
    
    pred_u_x = pred[:, 0].reshape(N_y, N_x).detach().cpu().numpy()
    pred_u_y = pred[:, 1].reshape(N_y, N_x).detach().cpu().numpy()
    pred_p = pred[:, 2].reshape(N_y, N_x).detach().cpu().numpy()

    plt.title("u_x")
    plt.imshow(pred_u_x, origin="lower", extent=[-0.5, 1, -0.5, 1.5], cmap="coolwarm")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("pred_u_x.png", dpi=300)
    plt.close()

    plt.title("u_y")
    plt.imshow(pred_u_y, origin="lower", extent=[-0.5, 1, -0.5, 1.5], cmap="coolwarm")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("pred_u_y.png", dpi=300)
    plt.close()

    plt.title("p")
    plt.imshow(pred_p, origin="lower", extent=[-0.5, 1, -0.5, 1.5], cmap="coolwarm")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("pred_p.png", dpi=300)
    plt.close()

    # squared errors

    #plt.title("u_x")
    plt.imshow((pred_u_x - u_x_sol)**2, origin="lower", extent=[-0.5, 1, -0.5, 1.5], cmap="Reds")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("se_u_x.png", dpi=300)
    plt.close()

    #plt.title("u_y")
    plt.imshow((pred_u_y - u_y_sol)**2, origin="lower", extent=[-0.5, 1, -0.5, 1.5], cmap="Reds")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("se_u_y.png", dpi=300)
    plt.close()

    #plt.title("p.png")
    plt.imshow((pred_p - p_sol)**2, origin="lower", extent=[-0.5, 1, -0.5, 1.5], cmap="Reds")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("se_p.png", dpi=300)
    plt.close()

    # show the distribution for u_x (just like in the config paper)
    x_vals = [-0.25, 0, 0.5]

    for x_val in x_vals:

        # how to find the right index?
        # first map to [0, 1]
        x_val_in_zero_one = (x_val -(-0.5)) / (1 -(-0.5))
        x_val_idx = int(x_val_in_zero_one * N_x)
        
        plt.title(f"x = {x_val}")

        plt.plot(np.linspace(-0.5, 1, N_x), u_x_sol[:, x_val_idx], label="data")
        plt.scatter(np.linspace(-0.5, 1, N_x), pred_u_x[:, x_val_idx], marker="x", color="#FF458A", label="pinn")
        plt.xlabel("x")
        plt.ylabel(r"$u_x$")
        plt.legend()
        plt.savefig(f"u_x_at_{x_val}.png", dpi=300)
        plt.close()