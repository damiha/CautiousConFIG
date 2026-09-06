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
import torch.quasirandom
from hydra.utils import to_absolute_path

# we are training on rtx 3090
# no real speedup
#torch.set_float32_matmul_precision('high')

# from the paper
class SchroedingerDataset:

    def __init__(self, seed, use_lhs=True):
        
        self.seed = seed
        self.use_lhs = use_lhs

        self.sampler_lhs_1d = qmc.LatinHypercube(d=1, seed=seed)
        self.sampler_lhs_2d = qmc.LatinHypercube(d=2, seed=seed)

    def sample_boundary(self, n_boundary = 500):

        # x in {-5, 5}
        # t in [0, pi / 2]
        # first and second half of t must match
        n_pairs = n_boundary // 2

        if self.use_lhs:
            sample = self.sampler_lhs_1d.random(n=n_pairs)
            t_np = (np.pi / 2) * sample[:, 0]
            t_half = torch.from_numpy(t_np).float()
        else:
            t_half = (np.pi / 2) * torch.rand(size=(n_pairs, ))

        # half half
        t = torch.cat([t_half, t_half], dim=0)
        x = torch.cat([-5.0 * torch.ones(size=(n_boundary // 2, )), 5.0 * torch.ones(size=(n_boundary // 2, ))], dim=0)

        #print("boundary")
        #print(f"{x.shape=}, {t.shape=}")

        return x, t

    def sample_initial(self, n_initial = 500):

        # t = 0
        # x in [-5, 5]
        
        if self.use_lhs:
            
            sample = self.sampler_lhs_1d.random(n=n_initial)
            x_np = 10.0 * sample[:, 0] - 5.0
            x = torch.from_numpy(x_np).float()
        else:
            x =  10.0 * torch.rand(size=(n_initial, )) - 5.0
        
        t = 0 * x

        #print("initial")
        #print(f"{x.shape=}, {t.shape=}")

        return x, t

    def sample_interior(self, n_interior = 20_000):

        if self.use_lhs:
            
            sample = self.sampler_lhs_2d.random(n=n_interior)
        
            x_np = 10 * sample[:, 0] - 5
            t_np = (np.pi / 2) * sample[:, 1]
            
            x = torch.from_numpy(x_np).float()
            t = torch.from_numpy(t_np).float()

        else:

            x =  10 * torch.rand(size=(n_interior, )) - 5
            t  = (np.pi / 2) * torch.rand(size=(n_interior, ))

        #print("interior")
        #print(f"{x.shape=}, {t.shape=}")

        return x, t

    def get_sampling_functions(self):
        return [self.sample_initial, self.sample_boundary, self.sample_interior]

def initial_loss(model, sample_initial_fn):

    x, t = sample_initial_fn()

    device = next(model.parameters()).device

    x = x.to(device)
    t = t.to(device)

    pred = model(x, t)

    u_pred = pred[:, 0]
    v_pred = pred[:, 1]

    # 2 * sech
    u_target = (2 / torch.cosh(x)).view(-1)

    return ((u_pred - u_target)**2 + v_pred**2).mean()

def boundary_loss(model, sample_boundary_fn):

    x, t = sample_boundary_fn()

    device = next(model.parameters()).device
    x = x.to(device)
    t = t.to(device)

    # not needed because we don't compare first derivatives at the boundary
    # see below
    
    #x.requires_grad_(True)
    #t.requires_grad_(True)

    # first half for -5, second half for 5
    n_half = len(x) // 2

    x_left = x[:n_half]
    x_right = x[n_half:]
    t_left = t[:n_half]
    t_right = t[n_half:]

    h_left = model(x_left, t_left)
    h_right = model(x_right, t_right)

    u_left = h_left[:, 0]
    v_left = h_left[:, 1]

    u_right = h_right[:, 0]
    v_right = h_right[:, 1]

    '''
    u_x_left = torch.autograd.grad(
        outputs=u_left, inputs=x_left,
        grad_outputs=torch.ones_like(x_left),
        create_graph=True,
        retain_graph=True
    )[0]

    u_x_right = torch.autograd.grad(
        outputs=u_right, inputs=x_right,
        grad_outputs=torch.ones_like(x_right),
        create_graph=True,
        retain_graph=True
    )[0]
    
    v_x_left = torch.autograd.grad(
        outputs=v_left,
        inputs=x_left,
        grad_outputs=torch.ones_like(x_left),
        create_graph=True,
        retain_graph=True
    )[0]

    v_x_right = torch.autograd.grad(
        outputs=v_right,
        inputs=x_right,
        grad_outputs=torch.ones_like(x_right),
        create_graph=True,
        retain_graph=True
    )[0]
    '''

    loss_matching_h_val = ((u_left - u_right)**2 + (v_left - v_right)**2).mean()

    #loss_matching_h_der = ((u_x_left - u_x_right)**2 + (v_x_left - v_x_right)**2).mean()

    #return loss_matching_h_val + loss_matching_h_der
    return loss_matching_h_val

def pde_loss(model, sample_interior_fn):

    x, t = sample_interior_fn()
    device = next(model.parameters()).device

    x = x.to(device).requires_grad_(True)
    t = t.to(device).requires_grad_(True)

    pred = model(x, t)

    u_pred = pred[:, 0]
    v_pred = pred[:, 1]

    u_sq_plus_v_sq = u_pred**2 + v_pred**2

    # u derivatives
    '''
    u_pred_x = torch.autograd.grad(
        outputs = u_pred,
        inputs = x,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        #retain_graph=True
    )[0]

    u_pred_xx = torch.autograd.grad(
        outputs = u_pred_x,
        inputs = x,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        #retain_graph=True
    )[0]

    u_pred_t = torch.autograd.grad(
        outputs = u_pred,
        inputs = t,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        #retain_graph=True
    )[0]

    # v derivatives
    v_pred_x = torch.autograd.grad(
        outputs = v_pred,
        inputs = x,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        #retain_graph=True
    )[0]

    v_pred_xx = torch.autograd.grad(
        outputs = v_pred_x,
        inputs = x,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        #retain_graph=True
    )[0]

    v_pred_t = torch.autograd.grad(
        outputs = v_pred,
        inputs = t,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        #retain_graph=True
    )[0]
    '''

    # 4 instead of 6 autograd.grad calls
    grad_u = torch.autograd.grad(outputs=u_pred.sum(), inputs=[x, t], create_graph=True)
    u_pred_x = grad_u[0]
    u_pred_t = grad_u[1]

    # u_xx
    u_pred_xx = torch.autograd.grad(outputs=u_pred_x.sum(), inputs=x, create_graph=True)[0]

    # v derivatives (x and t simultaneously)
    grad_v = torch.autograd.grad(outputs=v_pred.sum(), inputs=[x, t], create_graph=True)
    v_pred_x = grad_v[0]
    v_pred_t = grad_v[1]

    # v_xx
    v_pred_xx = torch.autograd.grad(outputs=v_pred_x.sum(), inputs=x, create_graph=True)[0]

    #r_u = 0.5 * u_pred_xx + u_sq_plus_v_sq * u_pred - v_pred_t

    #r_v = 0.5 * v_pred_xx + u_sq_plus_v_sq * v_pred + u_pred_t

    # implement this
    #return (r_u**2 + r_v**2).mean()
    f_u = u_pred_t + 0.5 * v_pred_xx + u_sq_plus_v_sq * v_pred
    f_v = v_pred_t - 0.5 * u_pred_xx - u_sq_plus_v_sq * u_pred

    return (f_u**2 + f_v**2).mean()

def get_dataset(seed, use_lhs):
    return SchroedingerDataset(seed, use_lhs)

def get_loss_functions():
    return [initial_loss, boundary_loss, pde_loss]

class Tester:

    def __init__(self, device):
        
        h_numerical = np.load(to_absolute_path("sols/schroedinger_numerical_solution.npz"))["h"]
        h_numerical = torch.from_numpy(h_numerical).to(device)
        N_t_val, N_x_val  = h_numerical.shape

        # linspace with endpoint=False not in torch
        x_val_np = np.linspace(-5.0, 5.0, N_x_val, endpoint=False)
        x_val = torch.from_numpy(x_val_np).float().to(device)

        dt_solver = 0.001
        t_val_np = np.arange(N_t_val) * dt_solver
        t_val = torch.from_numpy(t_val_np).float().to(device)
        #t_val = torch.linspace(0, np.pi / 2, N_t_val).to(device)

        #print("t_val")
        #print(t_val[-1])

        T_val, X_val  = torch.meshgrid(t_val, x_val, indexing="ij")

        self.x_val = X_val.reshape(-1, 1)
        self.t_val = T_val.reshape(-1, 1)
        self.h_numerical = h_numerical.reshape(-1, 1)

    def compute_val_loss(self, model):

        h_pred_val = model(self.x_val, self.t_val)

        u_pred_val = h_pred_val[:, 0:1]
        v_pred_val = h_pred_val[:, 1:2]
        h_pred_complex = torch.complex(u_pred_val, v_pred_val)

        val_mse = ((h_pred_complex - self.h_numerical).abs()**2).mean().item()

        return val_mse
    
def get_tester(device):
    return Tester(device)
    
# everything numpy (needed torch only for inference)
def create_graphs(model, device):

    h_numerical = np.load(to_absolute_path("sols/schroedinger_numerical_solution.npz"))["h"]
    
    N_t_test, N_x_test = h_numerical.shape

    x_test_np = np.linspace(-5.0, 5.0, N_x_test, endpoint=False)
    x_test = torch.from_numpy(x_test_np).float().to(device)

    t_test_np = np.linspace(0, np.pi / 2, N_t_test, endpoint=False)
    t_test = torch.from_numpy(t_test_np).float().to(device)

    T_test, X_test = torch.meshgrid(t_test, x_test, indexing="ij")

    x_test = X_test.reshape(-1, 1)
    t_test = T_test.reshape(-1, 1)

    h_pred_test = model(x_test, t_test)
    u_pred_test = h_pred_test[:, 0:1]
    v_pred_test = h_pred_test[:, 1:2]

    h_pred_complex = torch.complex(u_pred_test, v_pred_test).detach().cpu().numpy()
    h_pred_complex = h_pred_complex.reshape(N_t_test, N_x_test)

    # predict u
    plt.figure(figsize=(5, 5))
    plt.imshow(h_pred_complex.real, cmap="coolwarm",  aspect="auto", extent=[0, np.pi / 2, -5, 5], origin="lower")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar(location="top", pad=0.08)
    plt.savefig(f"pred_u.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # squared error in u
    plt.figure(figsize=(5, 5))
    plt.imshow((h_pred_complex.real - h_numerical.real)**2, cmap="Reds",  aspect="auto", extent=[0, np.pi / 2, -5, 5], origin="lower")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar(location="top", pad=0.08)
    plt.savefig(f"se_u.png", dpi=300, bbox_inches="tight")
    plt.close()

    # predict v
    plt.figure(figsize=(5, 5))
    plt.imshow(h_pred_complex.imag, cmap="coolwarm",  aspect="auto", extent=[0, np.pi / 2, -5, 5], origin="lower")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar(location="top", pad=0.08)
    plt.savefig(f"pred_v.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # squared error in v
    plt.figure(figsize=(5, 5))
    plt.imshow((h_pred_complex.imag - h_numerical.imag)**2, cmap="Reds",  aspect="auto", extent=[0, np.pi / 2, -5, 5], origin="lower")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar(location="top", pad=0.08)
    plt.savefig(f"se_v.png", dpi=300, bbox_inches="tight")
    plt.close()

    # predict the absolute value
    plt.figure(figsize=(5, 5))
    plt.imshow(np.absolute(h_pred_complex), cmap="Reds",  aspect="auto", extent=[0, np.pi / 2, -5, 5], origin="lower")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar(location="top", pad=0.08)
    plt.savefig(f"pred_h_mag.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # squared error in absolute value prediction
    plt.figure(figsize=(5, 5))
    plt.imshow((np.absolute(h_pred_complex) - np.absolute(h_numerical))**2, cmap="Reds",  aspect="auto", extent=[0, np.pi / 2, -5, 5], origin="lower")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar(location="top", pad=0.08)
    plt.savefig(f"se_h_mag.png", dpi=300, bbox_inches="tight")
    plt.close()

    dt = t_test_np[1] - t_test_np[0]

    # predict h mag profile (like in ConFIG paper)
    for t_selected in [0.59, 0.79, 0.98]:

        t_step_selected = int(t_selected / dt)

        h_numerical_selected = h_numerical[t_step_selected, :]
        h_pred_selected = h_pred_complex[t_step_selected, :]

        print(f"{x_test.shape=}, {h_numerical_selected.shape=}, {h_pred_selected.shape=}")

        plt.plot(x_test_np, np.absolute(h_numerical_selected), label="data")
        plt.scatter(x_test_np, np.absolute(h_pred_selected), label="pinn", marker="x", color="#FF458A")
        plt.xlim(-5, 5)
        plt.ylim(0, 6)
        plt.title(f"t = {t_selected}")
        plt.xlabel("x")
        plt.ylabel("|h|")
        plt.legend()
        plt.savefig(f"mag_{t_selected}.png", dpi=300)
        plt.close()

