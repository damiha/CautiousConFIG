import torch
import matplotlib.pyplot as plt
import math

from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector

def get_param_vec(model):
    return torch.cat([p.view(-1) for pname, p in model.named_parameters()])

'''
def get_grad_vec(model):
    return torch.cat([p.grad.view(-1) for pname, p in model.named_parameters()])
'''
def get_grad_vec(model):
    grads = []
    for pname, p in model.named_parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros_like(p).view(-1))
    return torch.cat(grads)

# for measurement config
def apply_param_vec(model, theta):

    param_idx = 0

    for pname, p in model.named_parameters():
                
        numel = p.numel()
        theta_for_p = theta[param_idx:(param_idx + numel)]

        p.copy_(theta_for_p.view_as(p))

        param_idx += numel

# new class
class CosineAnnealingLR:

    def __init__(self, optimizer, max_lr, min_lr, n_epochs):
        
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.n_epochs = n_epochs

        self.curr_epochs = 0

    # after every epoch
    # sets the new learning rate
    def step(self):

        self.optimizer.lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos((self.curr_epochs / self.n_epochs) * math.pi))
        
        self.curr_epochs += 1

class Adam:

    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8, use_caution=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            self.m = 0 * theta_vec
            self.v = 0 * theta_vec
        
        self.t = 0
        self.use_caution = use_caution
        self.d = len(self.m)

        self.loss_history = []
        self.loss_histories = [[] for _ in loss_fns]

        # TODO: support more trackers (like ConFIG)
        self.trackers = trackers

    def plot_loss_history(self, filename):

        # default labels
        if len(self.loss_histories) == 3:
            loss_labels = [
                "initial", "boundary", "pde"
            ]
        elif len(self.loss_histories) == 2:
            loss_labels = ["boundary", "pde"]

        for i, loss_i_history in enumerate(self.loss_histories):
            
            # tensors up until now
            y_vals = [val.item() if torch.is_tensor(val) else val for val in loss_i_history]
            
            plt.title(f"{loss_labels[i]} loss")
            plt.plot(y_vals)
            plt.xlabel("#epochs")
            plt.ylabel("loss")
            plt.savefig(f"{filename}_{i}.png", dpi=300)
            plt.close()

        y_total = [val.item() if torch.is_tensor(val) else val for val in self.loss_history]
        plt.plot(y_total)
        
        plt.xlabel("#epochs")
        plt.ylabel("loss")
        plt.savefig(filename, dpi=300)
        plt.close()

    def train_one_epoch(self):
        
        self.t += 1
        
        loss = 0
        losses = []

        for i, (loss_fn, sample_fn) in enumerate(zip(self.loss_fns, self.sample_fns)):

            loss_i = loss_fn(self.model, sample_fn)

            # don't use .item()
            self.loss_histories[i].append(loss_i.detach())

            losses.append(loss_i.detach())

            loss += loss_i

        for tracker in self.trackers:
            tracker.track_losses(self.t, losses)

        # don't use .item()
        self.loss_history.append(loss.detach())

        self.model.zero_grad()

        loss.backward()

        with torch.no_grad():

            # optimizer step
            g = get_grad_vec(self.model)

            self.m = self.beta_1 * self.m + (1 - self.beta_1) * g
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g**2

            hat_m = self.m / (1 - self.beta_1**self.t)
            hat_v = self.v / (1 - self.beta_2**self.t)

            u =  hat_m / (torch.sqrt(hat_v) + self.eps)

            if self.use_caution:

                phi = ((u * g) > 0).float()
                rescale_lr = self.d / (1 + phi.sum())

                u_cautious = phi * u

                delta_theta = -(self.lr * rescale_lr) * u_cautious
            else:
                delta_theta = -self.lr * u

            # apply
            param_idx = 0

            for pname, p in self.model.named_parameters():
                
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel

class Lion:

    def __init__(self, model, lr, loss_fns, sample_fns, beta_1=0.9, beta_2=0.99, use_caution=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        # Lion Standard-Hyperparameter (beta_1=0.9, beta_2=0.99)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            # Lion benötigt NUR den ersten Moment (m), keinen zweiten Moment (v)!
            self.m = 0 * theta_vec
        
        self.t = 0
        self.use_caution = use_caution
        self.d = len(self.m)

        self.loss_history = []
        self.loss_histories = [[] for _ in loss_fns]

        # TODO: support more trackers (like ConFIG)
        self.trackers = trackers

    def plot_loss_history(self, filename):

        # default labels
        if len(self.loss_histories) == 3:
            loss_labels = [
                "initial", "boundary", "pde"
            ]
        elif len(self.loss_histories) == 2:
            loss_labels = ["boundary", "pde"]

        for i, loss_i_history in enumerate(self.loss_histories):
            
            # tensors up until now
            y_vals = [val.item() if torch.is_tensor(val) else val for val in loss_i_history]
            
            plt.title(f"{loss_labels[i]} loss")
            plt.plot(y_vals)
            plt.xlabel("#epochs")
            plt.ylabel("loss")
            plt.savefig(f"{filename}_{i}.png", dpi=300)
            plt.close()

        y_total = [val.item() if torch.is_tensor(val) else val for val in self.loss_history]
        plt.plot(y_total)
        
        plt.xlabel("#epochs")
        plt.ylabel("loss")
        plt.savefig(filename, dpi=300)
        plt.close()

    def train_one_epoch(self):
        
        self.t += 1
        
        loss = 0
        losses = []

        for i, (loss_fn, sample_fn) in enumerate(zip(self.loss_fns, self.sample_fns)):

            loss_i = loss_fn(self.model, sample_fn)

            # don't use .item()
            self.loss_histories[i].append(loss_i.detach())

            losses.append(loss_i.detach())

            loss += loss_i

        for tracker in self.trackers:
            tracker.track_losses(self.t, losses)

        # don't use .item()
        self.loss_history.append(loss.detach())

        self.model.zero_grad()

        loss.backward()

        with torch.no_grad():

            # optimizer step
            g = get_grad_vec(self.model)

            # 1. Lion Update Berechnung (Interpolation aus m und neuem Gradient g)
            c = self.beta_1 * self.m + (1 - self.beta_1) * g
            
            # 2. Lion Vektor-Schritt (reines Vorzeichen sign(c))
            u = torch.sign(c)

            # 3. Interne Momentum-Aktualisierung für die NÄCHSTE Epoche (mit beta_2)
            self.m = self.beta_2 * self.m + (1 - self.beta_2) * g

            # Cautious Updates
            if self.use_caution:

                phi = ((u * g) > 0).float()
                rescale_lr = self.d / (1 + phi.sum())

                u_cautious = phi * u

                delta_theta = -(self.lr * rescale_lr) * u_cautious
            else:
                delta_theta = -self.lr * u

            # apply
            param_idx = 0

            for pname, p in self.model.named_parameters():
                
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel

# wrapper for PyTorch's Adam
class OfficialAdam:

    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8, use_caution=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        # TODO: support more trackers (like ConFIG)
        self.trackers = trackers

        self.official_adam = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(beta_1, beta_2), eps=eps)

        self.t = 0

    def train_one_epoch(self):

        # the learning rate scheduler manipulates self.lr
        # we forward it to the official adam
        self.official_adam.param_groups[0]['lr'] = self.lr
        
        self.t += 1
        
        loss = 0
        losses = []

        for i, (loss_fn, sample_fn) in enumerate(zip(self.loss_fns, self.sample_fns)):

            loss_i = loss_fn(self.model, sample_fn)

            losses.append(loss_i.detach())

            loss += loss_i

        for tracker in self.trackers:
            tracker.track_losses(self.t, losses)

        
        # classic PyTorch
        self.official_adam.zero_grad()
        loss.backward()
        self.official_adam.step()

# wrapper for the config version from the official 'conflictfree' python package
class OfficialConfigAdam:

    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8, use_caution=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        # TODO: support more trackers (like ConFIG)
        self.trackers = trackers

        self.official_adam = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(beta_1, beta_2), eps=eps)

        self.t = 0

    def train_one_epoch(self):

        # the learning rate scheduler manipulates self.lr
        # we forward it to the official adam
        self.official_adam.param_groups[0]['lr'] = self.lr
        
        self.t += 1
        
        losses = []
        grads = []

        for i, (loss_fn, sample_fn) in enumerate(zip(self.loss_fns, self.sample_fns)):
            
            self.official_adam.zero_grad()

            loss_i = loss_fn(self.model, sample_fn)
            
            losses.append(loss_i.detach())

            loss_i.backward()
            grads.append(get_gradient_vector(self.model))

        # the least squares solution of the original repo had some problems
        # we also use torch.linalg.pinv
        apply_gradient_vector(self.model,ConFIG_update(grads, use_least_square=False))
        self.official_adam.step()

        # track at the end
        for tracker in self.trackers:
            tracker.track_losses(self.t, losses)

class GlobalConfigAdam:
    
    # previously, use_least_squares = True
    # TODO: change to beta_1 = 0.9 again
    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8, use_least_squares=False,
     use_caution=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            self.m = 0 * theta_vec
            self.v = 0 * theta_vec
        
        self.t = 0
        self.use_least_squares = use_least_squares
        self.use_caution = use_caution

        # number of parameters
        self.d = len(self.m)

        # for analysis
        self.trackers = trackers

        for tracker in self.trackers:
            tracker.register_model(self.model)

    def train_one_epoch(self):
        
        self.t += 1
        
        grads = []
        
        losses = []

        for loss_fn, sample_fn in zip(self.loss_fns, self.sample_fns):
            
            # before every backprop
            self.model.zero_grad()

            loss_i = loss_fn(self.model, sample_fn)
            loss_i.backward()

            losses.append(loss_i.clone().detach())

            grads.append(get_grad_vec(self.model))

        # apply
        with torch.no_grad():

            # config update direction
            uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

            U_T = torch.stack(uniform_grads)
            m = len(grads)
            ones_vec = torch.ones(size=(m, )).to(U_T.device)

            if self.use_least_squares:
                # (U.T)^dagger ones = g_u_tilde
                # U.T g_u_tilde = ones
                # U.T in m x d
                g_u_tilde = torch.linalg.lstsq(U_T, ones_vec).solution

            else:
                g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
            
            g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

            G_T = torch.stack(grads)

            g_config = (G_T @ g_u).sum() * g_u

            self.m = self.beta_1 * self.m + (1 - self.beta_1) * g_config
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_config**2

            hat_m = self.m / (1 - self.beta_1**self.t)
            hat_v = self.v / (1 - self.beta_2**self.t)

            u =  hat_m / (torch.sqrt(hat_v) + self.eps)

            if self.use_caution:
                
                phi = (g_config * u > 0).float()
                rescale_factor = self.d / (phi.sum() + 1)
                u_cautious = u * phi

                delta_theta = -(self.lr * rescale_factor) * u_cautious
            
            else:
                delta_theta = -self.lr * u

            param_idx = 0

            # track the params (not only the changes)
            theta_vec = get_param_vec(self.model)

            for tracker in self.trackers:
                tracker.track_losses(self.t, losses)
                tracker.track_model_weights(self.t, theta_vec)

            for pname, p in self.model.named_parameters():
                
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel

            # update the trackers
            update_dir = (rescale_factor * u_cautious) if self.use_caution else u

            for tracker in self.trackers:

                tracker.track_directions(
                    epoch = self.t,
                    grads = grads,
                    g_config = g_config,
                    adam_out = u,
                    update_dir = update_dir
                )

# just shenanigans
class CautiousConFIGCore:
    
    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8, use_least_squares=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            self.m = 0 * theta_vec
            self.v = 0 * theta_vec
        
        self.t = 0
        self.use_least_squares = use_least_squares

        # number of parameters
        self.d = len(self.m)

        # for analysis
        self.trackers = trackers
        self.last_phi = None

        for tracker in self.trackers:
            tracker.register_model(self.model)

    def train_one_epoch(self):
        
        self.t += 1
        
        grads = []
        losses = []

        for loss_fn, sample_fn in zip(self.loss_fns, self.sample_fns):
            
            self.model.zero_grad()

            loss_i = loss_fn(self.model, sample_fn)
            loss_i.backward()

            losses.append(loss_i.clone().detach())
            grads.append(get_grad_vec(self.model))

        # apply
        with torch.no_grad():

            uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

            U_T = torch.stack(uniform_grads)
            m_len = len(grads)
            ones_vec = torch.ones(size=(m_len, )).to(U_T.device)

            if self.use_least_squares:
                g_u_tilde = torch.linalg.lstsq(U_T, ones_vec).solution
            else:
                g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
            
            g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

            G_T = torch.stack(grads)
            g_config = (G_T @ g_u).sum() * g_u

            self.m = self.beta_1 * self.m + (1 - self.beta_1) * g_config
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_config**2

            hat_m = self.m / (1 - self.beta_1**self.t)
            hat_v = self.v / (1 - self.beta_2**self.t)

            u = hat_m / (torch.sqrt(hat_v) + self.eps)

            # --- Cautious ConFIG Core Logic ---
            phi_current = (g_config * u > 0).float()

            if self.last_phi is None:
                self.last_phi = phi_current.clone().detach()

            # Verundung (Intersection) von aktueller und vorheriger Maske -> Nur Core wird geupdatet
            phi_core = phi_current * self.last_phi

            for tracker in self.trackers:
                tracker.track_mask(epoch = self.t, mask = phi_core)
            
            # Rescaling basiert nun auf der strikten Core-Gruppe
            rescale_factor = self.d / (phi_core.sum() + 1)
            u_cautious_core = u * phi_core

            delta_theta = -(self.lr * rescale_factor) * u_cautious_core

            param_idx = 0
            theta_vec = get_param_vec(self.model)

            for tracker in self.trackers:
                tracker.track_losses(self.t, losses)
                tracker.track_model_weights(self.t, theta_vec)

            for pname, p in self.model.named_parameters():
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]
                p.add_(delta_theta_for_p.view_as(p))
                param_idx += numel

            update_dir = rescale_factor * u_cautious_core

            for tracker in self.trackers:
                tracker.track_directions(
                    epoch = self.t,
                    grads = grads,
                    g_config = g_config,
                    adam_out = u,
                    update_dir = update_dir
                )

            # Masken für den nächsten Schritt austauschen / aktualisieren
            self.last_phi = phi_current.clone().detach()

# new paper
# TODO: fix the tracking
class ConFIGGrams:
    
    # previously, use_least_squares = True
    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8, use_least_squares=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            self.m = 0 * theta_vec
            self.v = 0 * theta_vec
        
        self.t = 0
        self.use_least_squares = use_least_squares

        # number of parameters
        self.d = len(self.m)

        # for analysis
        self.trackers = trackers

    def train_one_epoch(self):
        
        self.t += 1
        
        grads = []
        
        losses = []

        for loss_fn, sample_fn in zip(self.loss_fns, self.sample_fns):
            
            # before every backprop
            self.model.zero_grad()

            loss_i = loss_fn(self.model, sample_fn)
            loss_i.backward()

            losses.append(loss_i.clone().detach())

            grads.append(get_grad_vec(self.model))

        # apply
        with torch.no_grad():

            # config update direction
            uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

            U_T = torch.stack(uniform_grads)
            m = len(grads)
            ones_vec = torch.ones(size=(m, )).to(U_T.device)

            if self.use_least_squares:
                # (U.T)^dagger ones = g_u_tilde
                # U.T g_u_tilde = ones
                # U.T in m x d
                g_u_tilde = torch.linalg.lstsq(U_T, ones_vec).solution

            else:
                g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
            
            g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

            G_T = torch.stack(grads)

            g_config = (G_T @ g_u).sum() * g_u

            self.m = self.beta_1 * self.m + (1 - self.beta_1) * g_config
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_config**2

            hat_m = self.m / (1 - self.beta_1**self.t)
            hat_v = self.v / (1 - self.beta_2**self.t)

            u =  hat_m / (torch.sqrt(hat_v) + self.eps)

            # grams update
            # copy the signs from the gradient and the magnitudes from the u
            u_grams = torch.sign(g_config) * torch.abs(u)

            delta_theta = -self.lr * u_grams

            param_idx = 0

            # track the params (not only the changes)
            theta_vec = get_param_vec(self.model)

            for tracker in self.trackers:
                tracker.track_losses(self.t, losses)
                tracker.track_model_weights(self.t, theta_vec)

            for pname, p in self.model.named_parameters():
                
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel

            for tracker in self.trackers:

                tracker.track_directions(
                    epoch = self.t,
                    grads = grads,
                    g_config = g_config,
                    adam_out = u,
                    update_dir = u_grams
                )

# this rms prop is with debiasing
class GlobalConfigRMSProp:
    
    # previously, use_least_squares = True
    def __init__(self, model, lr, loss_fns, sample_fns, beta_2 = 0.999, eps=1e-8, use_least_squares=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        self.beta_2 = beta_2
        self.eps = eps

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            self.v = 0 * theta_vec
        
        self.t = 0
        self.use_least_squares = use_least_squares

        # number of parameters
        self.d = len(self.v)

        # for analysis
        self.trackers = trackers

    def train_one_epoch(self):
        
        self.t += 1
        
        grads = []
        
        losses = []

        for loss_fn, sample_fn in zip(self.loss_fns, self.sample_fns):
            
            # before every backprop
            self.model.zero_grad()

            loss_i = loss_fn(self.model, sample_fn)
            loss_i.backward()

            losses.append(loss_i.clone().detach())

            grads.append(get_grad_vec(self.model))

        # apply
        with torch.no_grad():

            # config update direction
            uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

            U_T = torch.stack(uniform_grads)
            m = len(grads)
            ones_vec = torch.ones(size=(m, )).to(U_T.device)

            if self.use_least_squares:
                # (U.T)^dagger ones = g_u_tilde
                # U.T g_u_tilde = ones
                # U.T in m x d
                g_u_tilde = torch.linalg.lstsq(U_T, ones_vec).solution

            else:
                g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
            
            g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

            G_T = torch.stack(grads)

            g_config = (G_T @ g_u).sum() * g_u

            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_config**2
            hat_v = self.v / (1 - self.beta_2**self.t)

            u =  g_config / (torch.sqrt(hat_v) + self.eps)

            delta_theta = -self.lr * u

            param_idx = 0

            # track the params (not only the changes)
            theta_vec = get_param_vec(self.model)

            for tracker in self.trackers:
                tracker.track_losses(self.t, losses)
                tracker.track_model_weights(self.t, theta_vec)

            for pname, p in self.model.named_parameters():
                
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel

            # update the trackers
            update_dir = u

            for tracker in self.trackers:

                tracker.track_directions(
                    epoch = self.t,
                    grads = grads,
                    g_config = g_config,
                    adam_out = u,
                    update_dir = update_dir
                )

class GlobalConfigLion:
    
    def __init__(self, model, lr, loss_fns, sample_fns, beta_1=0.9, beta_2=0.99, 
                 use_least_squares=False, use_caution=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        # Lion Standard-Hyperparameter: beta_1=0.9 (Update-Interp.), beta_2=0.99 (Momentum-Track)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            # Lion benötigt NUR den ersten Moment (m), kein zweites Moment (v)!
            self.m = torch.zeros_like(theta_vec)
        
        self.t = 0
        self.use_least_squares = use_least_squares
        self.use_caution = use_caution

        # number of parameters
        self.d = len(self.m)

        # for analysis
        self.trackers = trackers

    def train_one_epoch(self):
        
        self.t += 1
        
        grads = []
        losses = []

        for loss_fn, sample_fn in zip(self.loss_fns, self.sample_fns):
            
            # before every backprop
            self.model.zero_grad()

            loss_i = loss_fn(self.model, sample_fn)
            loss_i.backward()

            losses.append(loss_i.clone().detach())

            grads.append(get_grad_vec(self.model))

        # apply
        with torch.no_grad():

            # 1. ConFIG update direction (identisch zu GlobalConfigAdam)
            uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

            U_T = torch.stack(uniform_grads)
            m_losses = len(grads)
            ones_vec = torch.ones(size=(m_losses, )).to(U_T.device)

            if self.use_least_squares:
                g_u_tilde = torch.linalg.lstsq(U_T, ones_vec).solution
            else:
                g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
            
            g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

            G_T = torch.stack(grads)
            g_config = (G_T @ g_u).sum() * g_u

            # 2. Lion Update Logic
            # a) Interpolation aus bisherigem Momentum und neuem Gradienten für das Update
            c = self.beta_1 * self.m + (1 - self.beta_1) * g_config
            
            # b) Reiner Vorzeichen-Schritt (l_infinity Box)
            # HINWEIS: Keine Bias-Korrektur nötig, da sign(c / (1 - beta_1^t)) == sign(c)!
            u = torch.sign(c)

            # c) Momentum-Speicher für die NÄCHSTE Epoche aktualisieren (nutzt beta_2)
            self.m = self.beta_2 * self.m + (1 - self.beta_2) * g_config

            # 3. Cautious Logic (falls aktiv)
            if self.use_caution:
                phi = (g_config * u > 0).float()
                rescale_factor = self.d / (phi.sum() + 1)
                u_cautious = u * phi

                delta_theta = -(self.lr * rescale_factor) * u_cautious
            else:
                delta_theta = -self.lr * u

            # Track the params before applying updates
            theta_vec = get_param_vec(self.model)

            for tracker in self.trackers:
                tracker.track_losses(self.t, losses)
                tracker.track_model_weights(self.t, theta_vec)

            # Apply delta_theta to model parameters
            param_idx = 0
            for pname, p in self.model.named_parameters():
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel

            # 4. Trackers aktualisieren
            update_dir = (rescale_factor * u_cautious) if self.use_caution else u

            for tracker in self.trackers:
                tracker.track_directions(
                    epoch = self.t,
                    grads = grads,
                    g_config = g_config,
                    adam_out = u,  # 'u' repräsentiert hier den Lion-Output
                    update_dir = update_dir
                )

# for measurements of p, rho and lambda_x, lambda_c
class MeasurementConfigAdam:
    
    # previously, use_least_squares = True
    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8, use_least_squares=False,
     use_caution=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            self.m = 0 * theta_vec
            self.v = 0 * theta_vec
        
        self.t = 0
        self.use_least_squares = use_least_squares
        self.use_caution = use_caution

        # number of parameters
        self.d = len(self.m)

        # for analysis
        self.trackers = trackers

    def train_one_epoch(self):
        
        self.t += 1
        
        grads = []
        
        losses = []

        for loss_fn, sample_fn in zip(self.loss_fns, self.sample_fns):
            
            # before every backprop
            self.model.zero_grad()

            loss_i = loss_fn(self.model, sample_fn)
            loss_i.backward()

            losses.append(loss_i.clone().detach())

            grads.append(get_grad_vec(self.model))

        # apply
        with torch.no_grad():

            # config update direction
            uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

            U_T = torch.stack(uniform_grads)
            m = len(grads)
            ones_vec = torch.ones(size=(m, )).to(U_T.device)

            if self.use_least_squares:
                # (U.T)^dagger ones = g_u_tilde
                # U.T g_u_tilde = ones
                # U.T in m x d
                g_u_tilde = torch.linalg.lstsq(U_T, ones_vec).solution

            else:
                g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
            
            g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

            G_T = torch.stack(grads)

            g_config = (G_T @ g_u).sum() * g_u

            self.m = self.beta_1 * self.m + (1 - self.beta_1) * g_config
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_config**2

            hat_m = self.m / (1 - self.beta_1**self.t)
            hat_v = self.v / (1 - self.beta_2**self.t)

            u =  hat_m / (torch.sqrt(hat_v) + self.eps)

            phi = (g_config * u > 0).float()
            u_cautious = u * phi

            prob = phi.mean()
            delta_theta_P = -self.lr * (u * phi)
            delta_theta_not_P = -self.lr * (u * (1 - phi))

            if self.use_caution:
                
                rescale_factor = self.d / (phi.sum() + 1)
                delta_theta = -(self.lr * rescale_factor) * u_cautious
            
            else:
                delta_theta = -self.lr * u

            param_idx = 0

            # track the params (not only the changes)
            theta_vec = get_param_vec(self.model).detach().clone()

            for pname, p in self.model.named_parameters():
                
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel

        # cautious rescale factor
        alpha = self.d / (phi.sum() + 1)
        delta_theta_cautious = alpha * delta_theta_P

        # make a regular step scale like a cautious step
        s = alpha * torch.linalg.norm(u_cautious) / torch.linalg.norm(u)
        delta_theta_scale_only = -self.lr * s * u

        return theta_vec, delta_theta_P, delta_theta_not_P, delta_theta_cautious, delta_theta_scale_only, prob

# for Ablations
class GlobalConfigAdamForAblations:
    
    # previously, use_least_squares = True
    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8, use_least_squares=False,
     use_caution=False, trackers=[],
     random_mask=False,
     rescale_strategy="cautious",
     some_mask=True,
    ):

        ### for the ablations
        self.random_mask = random_mask
        self.rescale_strategy = rescale_strategy
        self.some_mask = some_mask

        # check whether the combinations make sense
        is_ablation_lr = (not random_mask and rescale_strategy == "adam_output")
        is_ablation_random_mask = (random_mask and rescale_strategy == "cautious")
        is_no_ablation = (not random_mask and rescale_strategy == "cautious")
        is_ablation_no_rescaling = (not random_mask and rescale_strategy == "none")
        is_ablation_cautious_scaling_no_mask = (not random_mask and rescale_strategy == "cautious" and not some_mask)

        assert (is_ablation_cautious_scaling_no_mask or is_ablation_lr or is_ablation_random_mask or is_no_ablation or is_ablation_no_rescaling), \
         f"Needs to be a sound ablation study ({self.random_mask=}, {self.rescale_strategy=})"
        ###

        ### new random element (random mask) needs new random number generator
        self.mask_generator = torch.Generator(device="cpu")
        self.mask_generator.manual_seed(1402)
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            self.m = 0 * theta_vec
            self.v = 0 * theta_vec
        
        self.t = 0
        self.use_least_squares = use_least_squares
        self.use_caution = use_caution

        # number of parameters
        self.d = len(self.m)

        # for analysis
        self.trackers = trackers

        # for noise computation
        self.last_g_config = None

    def train_one_epoch(self):
        
        self.t += 1
        
        grads = []
        
        losses = []

        for loss_fn, sample_fn in zip(self.loss_fns, self.sample_fns):
            
            # before every backprop
            self.model.zero_grad()

            loss_i = loss_fn(self.model, sample_fn)
            loss_i.backward()

            losses.append(loss_i.clone().detach())

            grads.append(get_grad_vec(self.model))

        # apply
        with torch.no_grad():

            # config update direction
            uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

            U_T = torch.stack(uniform_grads)
            m = len(grads)
            ones_vec = torch.ones(size=(m, )).to(U_T.device)

            if self.use_least_squares:
                # (U.T)^dagger ones = g_u_tilde
                # U.T g_u_tilde = ones
                # U.T in m x d
                g_u_tilde = torch.linalg.lstsq(U_T, ones_vec).solution

            else:
                g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
            
            g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

            G_T = torch.stack(grads)

            g_config = (G_T @ g_u).sum() * g_u

            # for analysis purposes
            self.last_g_config = g_config.detach().clone()

            self.m = self.beta_1 * self.m + (1 - self.beta_1) * g_config
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_config**2

            hat_m = self.m / (1 - self.beta_1**self.t)
            hat_v = self.v / (1 - self.beta_2**self.t)

            u =  hat_m / (torch.sqrt(hat_v) + self.eps)

            if self.use_caution:
                
                phi = (g_config * u > 0).float()

                u_cautious = u * phi

                if self.rescale_strategy == "cautious":
                    # from the "cautious optimizers paper"
                    rescale_factor = self.d / (phi.sum() + 1)

                elif self.rescale_strategy == "adam_output":

                    # steps should be as large as adam output
                    norm_u = torch.linalg.norm(u)

                    norm_u_cautious = torch.linalg.norm(u_cautious)

                    rescale_factor = (norm_u / norm_u_cautious)

                elif self.rescale_strategy == "none":

                    rescale_factor = 1.0

                else:
                    raise NotImplementedError(f"Rescale strategy {self.rescale_strategy} unknown!")

                if self.some_mask:

                    if self.random_mask:
                        
                        params_to_pick = int(phi.sum().item())

                        # sampling without replacement
                        random_permutation = torch.randperm(self.d, generator=self.mask_generator).to(u.device)

                        # first (:params_to_pick) are not zeroed out
                        indices_to_zero = random_permutation[params_to_pick:]

                        u_cautious = u.clone().detach()
                        u_cautious[indices_to_zero] = 0

                    # other option is cautious mask

                else:
                    
                    hypothetical_update = rescale_factor * u_cautious

                    rescale_factor = torch.linalg.norm(hypothetical_update) / (torch.linalg.norm(u) + 1e-6)
                    u_cautious = u

                delta_theta = -(self.lr * rescale_factor) * u_cautious
            
            else:
                delta_theta = -self.lr * u

            param_idx = 0

            # track the params (not only the changes)
            theta_vec = get_param_vec(self.model)

            for tracker in self.trackers:
                tracker.track_losses(self.t, losses)
                tracker.track_model_weights(self.t, theta_vec)

            for pname, p in self.model.named_parameters():
                
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel

            # update the trackers
            update_dir = (rescale_factor * u_cautious) if self.use_caution else u

            for tracker in self.trackers:

                tracker.track_directions(
                    epoch = self.t,
                    grads = grads,
                    g_config = g_config,
                    adam_out = u,
                    update_dir = update_dir
                )

class MConfig:

    # previously, use_least_squares = True
    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8, use_least_squares=False, use_caution=False, trackers=[]):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.n_tasks = len(loss_fns)

        with torch.no_grad():

            theta_vec = get_param_vec(model)
            
            # momentum of task gradients
            self.ms = [0 * theta_vec for _ in range(self.n_tasks)]
            self.ts = [0 for _ in range(self.n_tasks)]

            # for final step
            self.m = 0 * theta_vec
            self.v = 0 * theta_vec

        self.d = len(theta_vec)
        
        self.t = 0
        self.use_least_squares = use_least_squares
        self.use_caution = use_caution

        self.trackers = trackers

    def ConFIG_update(self, grads):

        # config update direction
        uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

        U_T = torch.stack(uniform_grads)
        m = len(grads)
        ones_vec = torch.ones(size=(m, )).to(U_T.device)

        if self.use_least_squares:
            # (U.T)^dagger ones = g_u_tilde
            # U.T g_u_tilde = ones
            # U.T in m x d
            g_u_tilde = torch.linalg.lstsq(U_T, ones_vec).solution

        else:
            g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
        
        g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

        G_T = torch.stack(grads)

        g_config = (G_T @ g_u).sum() * g_u

        return g_config

    def train_one_epoch(self):
        
        self.t += 1
        
        # choose one task
        loss_idx = ((self.t - 1) % self.n_tasks)

        loss_fn = self.loss_fns[loss_idx]
        sample_fn = self.sample_fns[loss_idx]

        self.model.zero_grad()
        chosen_loss = loss_fn(self.model, sample_fn)
        chosen_loss.backward()

        # apply
        with torch.no_grad():

            # update one momentum vector
            g_chosen = get_grad_vec(self.model)
            self.ts[loss_idx] = self.ts[loss_idx] + 1
            self.ms[loss_idx] = self.beta_1 * self.ms[loss_idx] + (1 - self.beta_1) * g_chosen

            # debias all non-zero first momenta
            hat_ms = []

            for i, m in enumerate(self.ms):

                if self.ts[i] == 0:
                    continue

                hat_ms.append(m / (1 - self.beta_1**self.ts[i]))

            hat_m = self.ConFIG_update(hat_ms)

            g_config = (hat_m * (1 - self.beta_1**self.t) - (self.beta_1 * self.m)) / (1 - self.beta_1)

            self.m = self.beta_1 * self.m + (1 - self.beta_1) * g_config
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_config**2
            hat_v = self.v / (1 - self.beta_2**self.t)

            u =  hat_m / (torch.sqrt(hat_v) + self.eps)

            if self.use_caution:
                
                phi = (g_config * u > 0).float()
                rescale_factor = self.d / (phi.sum() + 1)
                u_cautious = u * phi

                delta_theta = -(self.lr * rescale_factor) * u_cautious
            
            else:
                delta_theta = -self.lr * u

            param_idx = 0

            for pname, p in self.model.named_parameters():
                
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel

# local = layerwise (conflict resolution)
# can be with local caution, global caution
class LocalConfigAdam:

    # previously, use_least_squares = True
    def __init__(self, model, lr, loss_fns, sample_fns, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8,
        use_least_squares=False,
        use_caution=False,
        caution_type="global"
    ):
        
        self.model = model
        self.lr = lr
        self.loss_fns = loss_fns
        self.sample_fns = sample_fns

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        with torch.no_grad():
            theta_vec = get_param_vec(model)
            self.m = 0 * theta_vec
            self.v = 0 * theta_vec
        
        self.t = 0
        self.use_least_squares = use_least_squares
        self.use_caution = use_caution

        self.caution_type = caution_type

        assert self.caution_type in ["global", "local"], f"Unknown caution type {self.caution_type}"

        # number of parameters
        self.d = len(self.m)

    def ConFIG_update(self, grads):

        # config update direction
        uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

        U_T = torch.stack(uniform_grads)
        m = len(grads)
        ones_vec = torch.ones(size=(m, )).to(U_T.device)

        if self.use_least_squares:
            # (U.T)^dagger ones = g_u_tilde
            # U.T g_u_tilde = ones
            # U.T in m x d
            g_u_tilde = torch.linalg.lstsq(U_T, ones_vec).solution

        else:
            g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
        
        g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

        G_T = torch.stack(grads)

        g_config = (G_T @ g_u).sum() * g_u

        return g_config

    def train_one_epoch(self):
        
        self.t += 1
        
        grads = []

        for loss_fn, sample_fn in zip(self.loss_fns, self.sample_fns):
            
            # before every backprop
            self.model.zero_grad()

            loss_i = loss_fn(self.model, sample_fn)
            loss_i.backward()

            grads.append(get_grad_vec(self.model))

        # apply
        with torch.no_grad():

            config_results = []

            param_idx = 0

            for pname, p in self.model.named_parameters():

                numel = p.numel()

                grads_for_layer = [g[param_idx:(param_idx + numel)] for g in grads]

                g_config_for_layer = self.ConFIG_update(grads_for_layer)

                config_results.append(g_config_for_layer)
                param_idx += numel

            g_config = torch.cat(config_results)

            self.m = self.beta_1 * self.m + (1 - self.beta_1) * g_config
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_config**2

            hat_m = self.m / (1 - self.beta_1**self.t)
            hat_v = self.v / (1 - self.beta_2**self.t)

            u =  hat_m / (torch.sqrt(hat_v) + self.eps)

            if self.use_caution:
                
                # shared by global and local, only the rescale factor changes
                phi = (g_config * u > 0).float()
                u_cautious = u * phi

                if self.caution_type == "global":
                    
                    # just a global scalar
                    rescale_factor = self.d / (phi.sum() + 1)
                
                elif self.caution_type == "local":
                    
                    # vector (different for the different layers)
                    param_idx = 0
                    rescale_factor = torch.ones_like(u)

                    for pname, p in self.model.named_parameters():
                        
                        numel = p.numel()

                        local_phi = phi[param_idx:(param_idx + numel)]

                        local_rescale_factor = numel / (local_phi.sum() + 1)

                        rescale_factor[param_idx:(param_idx + numel)] *= local_rescale_factor

                        param_idx += numel

                else:
                    raise NotImplementedError("Unknown caution type")

                delta_theta = -self.lr * (rescale_factor * u_cautious)
            
            else:
                delta_theta = -self.lr * u

            param_idx = 0

            for pname, p in self.model.named_parameters():
                
                numel = p.numel()
                delta_theta_for_p = delta_theta[param_idx:(param_idx + numel)]

                p.add_(delta_theta_for_p.view_as(p))

                param_idx += numel