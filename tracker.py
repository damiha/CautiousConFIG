import pandas as pd
import torch
from optimizer import get_grad_vec
import numpy as np
from collections import deque

# for tracking everything inside the optimizer
class Tracker:
    
    def __init__(self):
        pass
        self.model = None

    # grads = [g_1, ...., g_m]
    # g_config is the config update direction
    # adam out is whatever comes out of adam
    # theta_new = theta_old - final_lr * update_dir
    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        pass

    def track_model_weights(self, epoch, theta):
        pass

    def track_losses(self, epoch, losses):
        pass

    def register_model(self, model):
        self.model = model

    def track_mask(self, epoch, mask):
        pass

    def get_df(self):
        pass

# how many parameters updated every time?
class UpdatedPercentageTracker(Tracker):

    def __init__(self):
        super().__init__()

        self.epochs = []
        self.updated_percentage_history = []

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():
            
            # epoch is an int
            self.epochs.append(epoch)

            # for cautious config
            if g_config is not None: 
                
                phi = ((g_config * adam_out) > 0).float()

                # parameters in total
                d = len(g_config)

                updated_percentage = phi.sum() / d

                self.updated_percentage_history.append(updated_percentage.detach())

    def get_df(self):

        history = []

        for i, epoch in enumerate(self.epochs):

            history.append({
                "epoch": epoch,
                "updated_percentage": self.updated_percentage_history[i].item()
            })

        return pd.DataFrame(history)

class UpdatedPercentagePerLayerTracker(Tracker):

    def __init__(self):
        super().__init__()
        
        self.history = []

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():

            # for cautious config
            if g_config is not None: 
                
                phi = ((g_config * adam_out) > 0).float()
                param_idx = 0
                record = {
                    "epoch": epoch
                }

                for pname, p in self.model.named_parameters():
                    
                    numel = p.numel()

                    param_phi = phi[param_idx:(param_idx + numel)]

                    param_updated_percentage = param_phi.mean().item()

                    record[f"update_percentage[{pname}]"] = param_updated_percentage

                    param_idx += numel

                self.history.append(record)

    def get_df(self):
        return pd.DataFrame(self.history)

class TopIndexJaccardTracker(Tracker):

    def __init__(self):
        super().__init__()
        
        self.history = []

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():

            # for cautious config
            if g_config is not None: 
                
                # 1. Cautious Mask (boolesch)
                mask_cautious = (g_config * adam_out) > 0
                k = int(mask_cautious.sum().item())

                d = g_config.numel()

                if 0 < k < d:
                    # 2. Top-k Indizes basierend auf den größten Beträgen in g_config
                    _, topk_indices = torch.topk(g_config.abs(), k)
                    
                    mask_topk = torch.zeros_like(mask_cautious, dtype=torch.bool)
                    mask_topk[topk_indices] = True

                    # 3. Jaccard Index (Intersection over Union)
                    intersection = (mask_cautious & mask_topk).sum().item()
                    union = (mask_cautious | mask_topk).sum().item()

                    jaccard = intersection / union if union > 0 else 1.0
                elif k == d:
                    # Wenn alle Elemente ausgewählt werden, sind beide Mengen identisch
                    jaccard = 1.0
                else:
                    # Spezialfall: k = 0 (keine Elemente ausgewählt)
                    jaccard = 1.0

                print(f"Top abs u jaccard: {jaccard}")

                record = {
                    "epoch": epoch,
                    "jaccard_similarity": jaccard,
                    "k_selected": k,
                    "k_percentage": k / d
                }

                self.history.append(record)

    def get_df(self):
        return pd.DataFrame(self.history)

# are the deleted u's any different from the kept u's
class MaskSplitStatsTracker(Tracker):

    def __init__(self):
        super().__init__()
        
        self.history = []

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():

            # for cautious config
            if g_config is not None: 
                
                phi = ((g_config * adam_out) > 0).float()
                param_idx = 0
                record = {
                    "epoch": epoch
                }

                for pname, p in self.model.named_parameters():
                    
                    numel = p.numel()

                    param_phi = phi[param_idx:(param_idx + numel)].bool()
                    param_u = adam_out[param_idx:(param_idx + numel)]
                    param_g_config = g_config[param_idx:(param_idx + numel)]

                    selected_u = param_u[param_phi].abs()
                    dropped_u = param_u[~param_phi].abs()

                    n_selected = selected_u.numel()
                    mean_abs_selected_u = selected_u.mean().item() if n_selected > 0 else 0.0
                    std_abs_selected_u = selected_u.std().item() if n_selected > 1 else 0.0

                    n_dropped = dropped_u.numel()
                    mean_abs_dropped_u = dropped_u.mean().item() if n_dropped > 0 else 0.0
                    std_abs_dropped_u = dropped_u.std().item() if n_dropped > 1 else 0.0

                    record[f"mean_abs_selected_u[{pname}]"] = mean_abs_selected_u
                    record[f"std_abs_selected_u[{pname}]"] = std_abs_selected_u

                    record[f"mean_abs_dropped_u[{pname}]"] = mean_abs_dropped_u
                    record[f"std_abs_dropped_u[{pname}]"] = std_abs_dropped_u

                    selected_g_config = param_g_config[param_phi].abs()
                    dropped_g_config = param_g_config[~param_phi].abs()

                    n_selected_g_config = selected_g_config.numel()
                    mean_abs_selected_g_config = selected_g_config.mean().item() if n_selected_g_config > 0 else 0.0
                    std_abs_selected_g_config = selected_g_config.std().item() if n_selected_g_config > 1 else 0.0

                    n_dropped_g_config = dropped_g_config.numel()
                    mean_abs_dropped_g_config = dropped_g_config.mean().item() if n_dropped_g_config > 0 else 0.0
                    std_abs_dropped_g_config = dropped_g_config.std().item() if n_dropped_g_config > 1 else 0.0

                    record[f"mean_abs_selected_g_config[{pname}]"] = mean_abs_selected_g_config
                    record[f"std_abs_selected_g_config[{pname}]"] = std_abs_selected_g_config

                    record[f"mean_abs_dropped_g_config[{pname}]"] = mean_abs_dropped_g_config
                    record[f"std_abs_dropped_g_config[{pname}]"] = std_abs_dropped_g_config

                    param_idx += numel

                self.history.append(record)

    def get_df(self):
        return pd.DataFrame(self.history)

class SNRTracker(Tracker):

    def __init__(self, window_size=10, beta1=0.9, eps=1e-8):
        super().__init__()
        self.window_size = window_size
        self.beta1 = beta1
        self.eps = eps

        self.grad_buffer = []
        self.history = []

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):

        if g_config is None:
            return

        with torch.no_grad():
            
            self.grad_buffer.append(g_config.detach().clone())
            if len(self.grad_buffer) > self.window_size:
                self.grad_buffer.pop(0)

            if len(self.grad_buffer) < 2:
                return

            K = len(self.grad_buffer)

            weights = torch.tensor(
                [self.beta1 ** (K - 1 - i) for i in range(K)],
                device=g_config.device,
                dtype=g_config.dtype,
            )
            weights = weights / weights.sum()
            weights_2d = weights.view(-1, 1)

            stacked_grads = torch.stack(self.grad_buffer, dim=0)

            weighted_mean = (stacked_grads * weights_2d).sum(dim=0)

            diff_sq = (stacked_grads - weighted_mean) ** 2
            weighted_var = (diff_sq * weights_2d).sum(dim=0)
            weighted_std = torch.sqrt(weighted_var + 1e-12)

            snr = weighted_mean.abs() / (weighted_std + self.eps)

            record = {"epoch": epoch}
            param_idx = 0

            for pname, p in self.model.named_parameters():
                numel = p.numel()
                param_snr = snr[param_idx : (param_idx + numel)]

                mean_snr = param_snr.mean().item()
                std_snr = param_snr.std().item() if numel > 1 else 0.0

                record[f"mean_snr[{pname}]"] = mean_snr
                record[f"std_snr[{pname}]"] = std_snr

                param_idx += numel

            self.history.append(record)

    def get_df(self):
        return pd.DataFrame(self.history)

class GradientSumSNRTracker(Tracker):

    def __init__(self, window_size=10, beta1=0.9, eps=1e-8):
        super().__init__()
        self.window_size = window_size
        self.beta1 = beta1
        self.eps = eps

        self.grad_buffer = []
        self.history = []

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):

        if grads is None:
            return

        with torch.no_grad():

            g_sum = torch.stack(grads, dim=0).sum(dim=0)

            # Detachten Klon in den Ringpuffer schieben
            self.grad_buffer.append(g_sum.detach().clone())
            if len(self.grad_buffer) > self.window_size:
                self.grad_buffer.pop(0)

            # Mindestens 2 Schritte für Varianz schätzen
            if len(self.grad_buffer) < 2:
                return

            K = len(self.grad_buffer)

            # Geometrische Gewichte (EMA für recency bias)
            weights = torch.tensor(
                [self.beta1 ** (K - 1 - i) for i in range(K)],
                device=g_sum.device,
                dtype=g_sum.dtype,
            )
            weights = weights / weights.sum()
            weights_2d = weights.view(-1, 1)

            stacked_grads = torch.stack(self.grad_buffer, dim=0)

            # 1. Gewichtetes Signal
            weighted_mean = (stacked_grads * weights_2d).sum(dim=0)

            # 2. Gewichtetes Rauschen (Std)
            diff_sq = (stacked_grads - weighted_mean) ** 2
            weighted_var = (diff_sq * weights_2d).sum(dim=0)
            weighted_std = torch.sqrt(weighted_var + 1e-12)

            # 3. Elementweises SNR der naiven Summe
            snr = weighted_mean.abs() / (weighted_std + self.eps)

            # 4. Extrahieren pro Layer
            record = {"epoch": epoch}
            param_idx = 0

            for pname, p in self.model.named_parameters():
                numel = p.numel()
                param_snr = snr[param_idx : (param_idx + numel)]

                mean_snr = param_snr.mean().item()
                std_snr = param_snr.std().item() if numel > 1 else 0.0

                record[f"mean_snr_gsum[{pname}]"] = mean_snr
                record[f"std_snr_gsum[{pname}]"] = std_snr

                param_idx += numel

            self.history.append(record)

    def get_df(self):
        return pd.DataFrame(self.history)

# how many percent of the parameters updated how many percent of the epochs
class UpdatedDistributionTracker(Tracker):

    def __init__(self):
        super().__init__()

        self.history = []

        self.n_updates_seen = 0

        self.update_register = 0

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():

            # for cautious config
            if g_config is not None:

                d = len(g_config)

                self.n_updates_seen += 1
                
                phi = ((g_config * adam_out) > 0).float()

                self.update_register += phi

                updated_overall_ratio = (self.update_register / self.n_updates_seen)

                #print(updated_overall_ratio)

                def ratio_updated_more_than(updated_overall_ratio, threshold):
                    return ((updated_overall_ratio > threshold).float().sum() / d).item()
                
                self.history.append({
                    "epoch": epoch,
                    "updated_more_than_90": ratio_updated_more_than(updated_overall_ratio, 0.9),
                    "updated_more_than_75": ratio_updated_more_than(updated_overall_ratio, 0.75),
                    "updated_more_than_50": ratio_updated_more_than(updated_overall_ratio, 0.5),
                    "updated_more_than_25": ratio_updated_more_than(updated_overall_ratio, 0.25),
                    "updated_more_than_10": ratio_updated_more_than(updated_overall_ratio, 0.1),
                })

    def get_df(self):

        return pd.DataFrame(self.history)

class UpdatedDistributionForMaskTracker(Tracker):

    def __init__(self):
        super().__init__()

        self.history = []

        self.n_updates_seen = 0

        self.update_register = 0

    def track_mask(self, epoch, mask):
        
        with torch.no_grad():

            # for cautious config
            if mask is not None:

                d = len(mask)

                self.n_updates_seen += 1
                
                phi = mask

                self.update_register += phi

                updated_overall_ratio = (self.update_register / self.n_updates_seen)

                #print(updated_overall_ratio)

                def ratio_updated_more_than(updated_overall_ratio, threshold):
                    return ((updated_overall_ratio > threshold).float().sum() / d).item()

                record = {
                    "epoch": epoch,
                    "updated_more_than_90": ratio_updated_more_than(updated_overall_ratio, 0.9),
                    "updated_more_than_75": ratio_updated_more_than(updated_overall_ratio, 0.75),
                    "updated_more_than_50": ratio_updated_more_than(updated_overall_ratio, 0.5),
                    "updated_more_than_25": ratio_updated_more_than(updated_overall_ratio, 0.25),
                    "updated_more_than_10": ratio_updated_more_than(updated_overall_ratio, 0.1),
                }

                print(record)
                
                self.history.append(record)

    def get_df(self):

        return pd.DataFrame(self.history)

class SlidingWindowMaskTracker(Tracker):

    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
        # Speichert die Masken der letzten N Schritte
        self.mask_window = deque(maxlen=window_size)
        self.history = []

    def track_mask(self, epoch, mask):
        
        with torch.no_grad():
            if mask is not None:
                d = len(mask)
                # Neue Maske ins Fenster werfen (älteste fällt automatisch raus)
                self.mask_window.append(mask.clone().detach())
                
                # Alle Masken im aktuellen Fenster zu einem Tensor stapeln (shape: [k, d])
                stacked_masks = torch.stack(list(self.mask_window))
                k = stacked_masks.shape[0] # Kann am Anfang kleiner als window_size sein
                
                # Zählen, wie oft jeder Parameter in diesem Fenster aktiv war (Summe über die Zeitachse)
                activity_counts = stacked_masks.sum(dim=0)
                
                # Relativer Anteil im Fenster (Wert zwischen 0.0 und 1.0)
                activity_ratio = activity_counts / k

                def ratio_active_for_fraction(ratio, threshold):
                    return ((ratio >= threshold).float().sum() / d).item()

                record = {
                    "epoch": epoch,
                    "window_size": k,
                    # Wie viel Prozent der Parameter waren im GANZEN Fenster (100 der Zeit) an? -> Der lokale Core!
                    "local_core_100pct": ratio_active_for_fraction(activity_ratio, 1.0),
                    # Wie viel waren zu >= 80% im Fenster an?
                    "local_core_80pct": ratio_active_for_fraction(activity_ratio, 0.8),
                    # Wie viel waren zu >= 50% im Fenster an?
                    "local_core_50pct": ratio_active_for_fraction(activity_ratio, 0.5),
                }

                #print(record)

                self.history.append(record)

    def get_df(self):
        return pd.DataFrame(self.history)

# tracks the cosine similarities
class CosineSimilarityTracker(Tracker):

    def __init__(self):
        
        super().__init__()

        self.epochs = []

        self.loss_names = ["init", "bound", "pde"]

        self.g_config_vs_g_i = {loss_name: [] for loss_name in self.loss_names}
        self.adam_out_vs_g_i = {loss_name: [] for loss_name in self.loss_names}
        self.update_dir_vs_g_i = {loss_name: [] for loss_name in self.loss_names}

    def cosine_sim(self, g_a, g_b):

        norm_g_a = torch.linalg.norm(g_a)
        norm_g_b = torch.linalg.norm(g_b)

        eps = 1e-6

        if norm_g_a < eps or norm_g_b < eps:
            return 0

        return torch.dot(g_a, g_b) / (norm_g_a * norm_g_b + eps)

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        self.epochs.append(epoch)

        with torch.no_grad():

            for i, grad in enumerate(grads):
                
                # two losses -> boundary, pde
                # three losses -> initial, boundary, pde
                loss_name_i = self.loss_names[i] if len(grads) == 3 else self.loss_names[i + 1]

                cosine_sim_g_config_vs_g_i = self.cosine_sim(g_config, grad)
                cosine_sim_adam_out_vs_g_i = self.cosine_sim(adam_out, grad)
                cosine_sim_update_dir_vs_g_i = self.cosine_sim(update_dir, grad)

                self.g_config_vs_g_i[loss_name_i].append(cosine_sim_g_config_vs_g_i.item())
                self.adam_out_vs_g_i[loss_name_i].append(cosine_sim_adam_out_vs_g_i.item())
                self.update_dir_vs_g_i[loss_name_i].append(cosine_sim_update_dir_vs_g_i.item())

    def get_df(self):

        history = []

        for i, epoch in enumerate(self.epochs):

            record = {
                "epoch": epoch
            }
            
            for loss_name in self.loss_names:

                for v, vec_name in enumerate(["g_config", "adam_out", "update_dir"]):

                    dict_selected = [self.g_config_vs_g_i, self.adam_out_vs_g_i, self.update_dir_vs_g_i][v]

                    # no value for combination vec_vs_task_gradient at epoch
                    if i >= len(dict_selected[loss_name]):
                        continue

                    record[f"{vec_name}_vs_g_{loss_name}"] = dict_selected[loss_name][i]

            history.append(record)

        return pd.DataFrame(history)

# tracks the cosine similarities
class CosineSimilarityK6Tracker(Tracker):

    def __init__(self):
        
        super().__init__()

        self.epochs = []

        self.loss_names = ["boundary_x", "bound_y", "bound_p", "momentum_x", "momentum_y", "continuity"]

        self.g_config_vs_g_i = {loss_name: [] for loss_name in self.loss_names}
        self.adam_out_vs_g_i = {loss_name: [] for loss_name in self.loss_names}
        self.update_dir_vs_g_i = {loss_name: [] for loss_name in self.loss_names}

    def cosine_sim(self, g_a, g_b):

        norm_g_a = torch.linalg.norm(g_a)
        norm_g_b = torch.linalg.norm(g_b)

        eps = 1e-6

        if norm_g_a < eps or norm_g_b < eps:
            return 0

        return torch.dot(g_a, g_b) / (norm_g_a * norm_g_b + eps)

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        self.epochs.append(epoch)

        with torch.no_grad():

            for i, grad in enumerate(grads):

                loss_name_i = self.loss_names[i]

                cosine_sim_g_config_vs_g_i = self.cosine_sim(g_config, grad)
                cosine_sim_adam_out_vs_g_i = self.cosine_sim(adam_out, grad)
                cosine_sim_update_dir_vs_g_i = self.cosine_sim(update_dir, grad)

                self.g_config_vs_g_i[loss_name_i].append(cosine_sim_g_config_vs_g_i.item())
                self.adam_out_vs_g_i[loss_name_i].append(cosine_sim_adam_out_vs_g_i.item())
                self.update_dir_vs_g_i[loss_name_i].append(cosine_sim_update_dir_vs_g_i.item())

    def get_df(self):

        history = []

        for i, epoch in enumerate(self.epochs):

            record = {
                "epoch": epoch
            }
            
            for loss_name in self.loss_names:

                for v, vec_name in enumerate(["g_config", "adam_out", "update_dir"]):

                    dict_selected = [self.g_config_vs_g_i, self.adam_out_vs_g_i, self.update_dir_vs_g_i][v]

                    # no value for combination vec_vs_task_gradient at epoch
                    if i >= len(dict_selected[loss_name]):
                        continue

                    record[f"{vec_name}_vs_g_{loss_name}"] = dict_selected[loss_name][i]

            history.append(record)

        return pd.DataFrame(history)

# tracks the cosine similarities between different two consecutive time steps
class StepwiseAlignmentTracker(Tracker):

    def __init__(self):
        
        super().__init__()

        self.epochs = []

        self.loss_names = ["init", "bound", "pde"]

        # this later gets averaged
        self.stepwise_alignment_history = []

        # needed for stepwise alignment
        self.last_update_direction = None

    def cosine_sim(self, g_a, g_b):

        norm_g_a = torch.linalg.norm(g_a)
        norm_g_b = torch.linalg.norm(g_b)

        eps = 1e-6

        if norm_g_a < eps or norm_g_b < eps:
            return 0

        return torch.dot(g_a, g_b) / (norm_g_a * norm_g_b + eps)

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        self.epochs.append(epoch)

        with torch.no_grad():

            if not torch.isnan(update_dir).any():

                if self.last_update_direction is None:

                    # set for next time
                    self.last_update_direction = update_dir.clone().detach()

                else:

                    # must already be set with a sensible value
                    stepwise_alignment = self.cosine_sim(update_dir, self.last_update_direction)
                    self.last_update_direction = update_dir.clone().detach()

                    self.stepwise_alignment_history.append(stepwise_alignment.detach())

    def get_df(self):

        history = []

        for i, epoch in enumerate(self.epochs):

            record = {
                "epoch": epoch,
                "stepwise_alignment": self.stepwise_alignment_history[i - 1].item() # only alignment after the first epoch   
            }

            history.append(record)

        return pd.DataFrame(history)


class ParamGroupTracker(Tracker):

    def __init__(self):
        super().__init__()
        
        self.history = []
        self.last_phi = None

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():

            if g_config is not None: 
                
                current_phi = ((g_config * adam_out) > 0).bool()
                u_abs = adam_out.abs()

                if self.last_phi is None:
                    # Set for the next step
                    self.last_phi = current_phi.clone().detach()
                else:
                    param_idx = 0
                    record = {"epoch": epoch}

                    for pname, p in self.model.named_parameters():
                        
                        numel = p.numel()

                        # Slice masks and updates for the current layer
                        l_current_phi = current_phi[param_idx : param_idx + numel]
                        l_last_phi = self.last_phi[param_idx : param_idx + numel]
                        l_u_abs = u_abs[param_idx : param_idx + numel]

                        # Group definitions
                        m_11 = l_current_phi & l_last_phi       # Core (Kept)
                        m_00 = (~l_current_phi) & (~l_last_phi) # Dead (Ignored)
                        m_01 = l_current_phi & (~l_last_phi)    # Switched ON
                        m_10 = (~l_current_phi) & l_last_phi    # Switched OFF

                        # Compute mean magnitudes
                        record[f"mean_u_11[{pname}]"] = l_u_abs[m_11].mean().item() if m_11.any() else 0.0
                        record[f"mean_u_00[{pname}]"] = l_u_abs[m_00].mean().item() if m_00.any() else 0.0
                        record[f"mean_u_01[{pname}]"] = l_u_abs[m_01].mean().item() if m_01.any() else 0.0
                        record[f"mean_u_10[{pname}]"] = l_u_abs[m_10].mean().item() if m_10.any() else 0.0

                        # Compute group sizes (percentages)
                        record[f"pct_11[{pname}]"] = m_11.sum().item() / numel
                        record[f"pct_00[{pname}]"] = m_00.sum().item() / numel
                        record[f"pct_01[{pname}]"] = m_01.sum().item() / numel
                        record[f"pct_10[{pname}]"] = m_10.sum().item() / numel

                        param_idx += numel

                    self.history.append(record)
                    
                    # Update for next iteration
                    self.last_phi = current_phi.clone().detach()

    def get_df(self):
        return pd.DataFrame(self.history)

# tracks all dot products between gradients
class GradientDotProductTracker(Tracker):

    def __init__(self):
        
        super().__init__()

        self.history = []

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():

            record = {"epoch": epoch}

            if len(grads) == 2:
                loss_names = ["bound", "pde"]

            else:
                loss_names = ["init", "bound", "pde"]

            for i, g_i in enumerate(grads):

                for j, g_j in enumerate(grads):

                    # i <= j allowed
                    # j < i not allowed
                    if j < i:
                        continue

                    g_i_dot_g_j = torch.dot(g_i, g_j)

                    dot_prod_name = f"g_{loss_names[i]}_dot_g_{loss_names[j]}"

                    record[dot_prod_name] = g_i_dot_g_j.item()

            self.history.append(record)

    def get_df(self):
        return pd.DataFrame(self.history)

# ConFIG scales up g_u when all gradients point in the same direction
# and slows down, when the sum of projections is low
# important to track
class ConFIGScalingTracker(Tracker):

    def __init__(self):
        
        super().__init__()

        self.history = []

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():
            
            # g_config = (sum...) g_u
            # the absolute value of the sum (the sum is always positive) can be extracted by just getting the norm
            self.history.append({
                "epoch": epoch,
                "norm_g_config": torch.linalg.norm(g_config).item()
            })

    def get_df(self):
        return pd.DataFrame(self.history)

class SingularValueTracker(Tracker):

    def __init__(self):
        
        super().__init__()

        self.epochs = []

        self.loss_names = ["init", "bound", "pde"]

        self.min_singular_value_history = []
        self.max_singular_value_history = []

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        self.epochs.append(epoch)

        with torch.no_grad():
            
            # code taken from config
            uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]
            U_T = torch.stack(uniform_grads)

            # torch.linalg.pinv requires good conditioned matrices
            _, S, _ = torch.linalg.svd(U_T)

            largest_sv, smallest_sv = S[0], S[-1]

            self.min_singular_value_history.append(smallest_sv)
            self.max_singular_value_history.append(largest_sv)

    def get_df(self):

        history = []

        for i, epoch in enumerate(self.epochs):

            record = {
                "epoch": epoch,
                "min_singular_value": self.min_singular_value_history[i].item(), 
                "max_singular_value": self.max_singular_value_history[i].item() 
            }

            history.append(record)

        return pd.DataFrame(history)

# this tracker is not called by the optimizer
class ValMseTracker(Tracker):

    def __init__(self):
        
        super().__init__()

        self.epochs = []
        self.val_mse_history = []

    def track_mse(self, epoch, val_mse):
        self.epochs.append(epoch)
        self.val_mse_history.append(val_mse)

    def get_df(self):

        history = []

        for i, val_mse in enumerate(self.val_mse_history):

            history.append({
                "epoch": self.epochs[i],
                "val_mse": val_mse
            })

        return pd.DataFrame(history)

# how much of the mask aligns between time steps
class StepwiseMaskTracker(Tracker):

    def __init__(self):
        super().__init__()
        self.history = []
        self.prev_mask = None

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        with torch.no_grad():
            if g_config is not None:
                

                current_mask = ((g_config * adam_out) > 0).float()
                d = current_mask.numel()

                updated_percentage = (current_mask.sum() / d).item()

                if self.prev_mask is not None:
                    # Binarized transition: prev * 2 + current
                    # 0 -> 0 : 0
                    # 0 -> 1 : 1
                    # 1 -> 0 : 2
                    # 1 -> 1 : 3
                    transitions = self.prev_mask * 2 + current_mask

                    n_00 = (transitions == 0).sum().item()
                    n_01 = (transitions == 1).sum().item()
                    n_10 = (transitions == 2).sum().item()
                    n_11 = (transitions == 3).sum().item()

                    # ratios (depending on d)
                    p_00 = n_00 / d
                    p_01 = n_01 / d
                    p_10 = n_10 / d
                    p_11 = n_11 / d

                    # metrics
                    hamming_dist = p_01 + p_10  # Flip Rate
                    p_same = p_00 + p_11        # Stays same

                    # Jaccard Overlap: Intersection / Union
                    union = n_01 + n_10 + n_11
                    jaccard = (n_11 / union) if union > 0 else 1.0

                    # transition probabilities
                    prev_ones = n_10 + n_11
                    prev_zeros = n_00 + n_01
                    p_1_given_1 = (n_11 / prev_ones) if prev_ones > 0 else 0.0
                    p_1_given_0 = (n_01 / prev_zeros) if prev_zeros > 0 else 0.0

                    self.history.append({
                        "epoch": epoch,
                        "p_00": p_00,
                        "p_01": p_01,
                        "p_10": p_10,
                        "p_11": p_11,
                        "p_same": p_same,
                        "flip_rate": hamming_dist,
                        "jaccard_overlap": jaccard,
                        "p_1_given_1": p_1_given_1,
                        "p_1_given_0": p_1_given_0,
                    })

                # save mask for next step
                self.prev_mask = current_mask.clone()

    def get_df(self):
        return pd.DataFrame(self.history)

# track all losses (init, boundary, pde)
class MultiLossTracker(Tracker):

    def __init__(self):
        super().__init__()
        self.history = []

    def track_losses(self, epoch, losses):
        
        # order is always important
        if len(losses) == 2:

            self.history.append({
                "epoch": epoch,
                "L_bound": losses[0].item(),
                "L_pde": losses[1].item()
            })

        elif len(losses) == 3:

            self.history.append({
                "epoch": epoch,
                "L_init": losses[0].item(),
                "L_bound": losses[1].item(),
                "L_pde": losses[2].item()
            })

        # for kovasznay
        elif len(losses) == 6:

            self.history.append({
                "epoch": epoch,
                "L_bound_u_x": losses[0].item(),
                "L_bound_u_y": losses[1].item(),
                "L_bound_p": losses[2].item(),
                "L_momentum_x": losses[3].item(),
                "L_momentum_y": losses[4].item(),
                "L_continuity": losses[5].item()
            })

    def get_df(self):
        return pd.DataFrame(self.history)

# track multi-task curvature (PC grad paper)
class MultiTaskCurvatureTracker(Tracker):

    def __init__(self):
        super().__init__()
        self.history = []

        self.prev_loss = None
        self.prev_theta = None
        self.prev_grad_L = None

        self.curr_loss = None
        self.curr_theta = None

    def track_losses(self, epoch, losses):
        with torch.no_grad():
            self.curr_loss = sum([l.item() for l in losses])

    def track_model_weights(self, epoch, theta):
        with torch.no_grad():
            self.curr_theta = theta.clone().detach()

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        with torch.no_grad():

            grad_L = torch.stack(grads).sum(dim=0).clone().detach()

            if self.prev_theta is not None and self.prev_loss is not None and self.prev_grad_L is not None:
                
                # 1. Schrittvektor Delta theta
                delta_theta = self.curr_theta - self.prev_theta
                
                # 2. Schrittweite Quadrat (||delta_theta||^2)
                step_norm_sq = torch.dot(delta_theta, delta_theta).item()

                # 3. Absolute Krümmungsabweichung (Taylor 2. Ordnung)
                linear_approx = self.prev_loss + torch.dot(self.prev_grad_L, delta_theta)
                H_raw = 2.0 * (self.curr_loss - linear_approx).item()

                # 4. Normierte, intrinsische Richtungskrümmung (Rayleigh-Quotient)
                # Betrag verhindert Negativ-Spitzen durch Rauschen um 0
                H_normalized = abs(H_raw) / (step_norm_sq + 1e-8)

                self.history.append({
                    "epoch": epoch - 1,
                    "multitask_curvature": H_normalized,  # Jetzt normiert & sauber auf Log-Skala
                    "multitask_curvature_raw": H_raw      # Unnormiert (falls du es noch brauchst)
                })

            self.prev_loss = self.curr_loss
            self.prev_theta = self.curr_theta
            self.prev_grad_L = grad_L

    def get_df(self):
        return pd.DataFrame(self.history)

class PathTurtuosityTracker(Tracker):

    def __init__(self):
        super().__init__()
        self.history = []

        self.first_theta = None
        self.prev_theta = None
        self.path_length = 0.0

    def track_model_weights(self, epoch, theta):
        with torch.no_grad():

            if self.first_theta is None: 
                self.first_theta = theta.clone().detach()
                self.prev_theta = theta.clone().detach()
                return

            segment_length = torch.linalg.norm(theta - self.prev_theta)
            self.path_length += segment_length.item()

            dist_to_first = torch.linalg.norm(theta - self.first_theta)

            eps = 1e-8
            turtuosity = (self.path_length / (dist_to_first.item() + eps))

            self.history.append({
                "epoch": epoch,
                "turtuosity": turtuosity
            })

            self.prev_theta = theta.clone().detach()
        
    def get_df(self):
        return pd.DataFrame(self.history)

# TODO: tracking different for config and adam
def ConFIG_update(grads):

    # config update direction
    uniform_grads = [torch.nan_to_num(g / g.norm(), nan=0) for g in grads]

    U_T = torch.stack(uniform_grads)
    m = len(grads)
    ones_vec = torch.ones(size=(m, )).to(U_T.device)

    g_u_tilde = torch.linalg.pinv(U_T) @ ones_vec
    
    g_u = torch.nan_to_num(g_u_tilde / g_u_tilde.norm(), nan=0)

    G_T = torch.stack(grads)

    g_config = (G_T @ g_u).sum() * g_u

    return g_config

# this tracker is not super useful
class NoiseTracker(Tracker):

    def __init__(self, k_batches):
        super().__init__()

        self.k_batches = k_batches
        self.history = []

    def cosine_sim(self, g_a, g_b):

        norm_g_a = torch.linalg.norm(g_a)
        norm_g_b = torch.linalg.norm(g_b)

        eps = 1e-6

        if norm_g_a < eps or norm_g_b < eps:
            return 0

        return torch.dot(g_a, g_b) / (norm_g_a * norm_g_b + eps)

    def track_noise(self, epoch, model, fixed_sampling_fns, loss_fns, first_momentum = None):

        true_g_i_s = []

        input_noise_powers = []

        all_g_i_samples = []

        for i, loss_i_fn in enumerate(loss_fns):

            g_i_samples = []

            sample_i_fn = fixed_sampling_fns[i]

            for _ in range(self.k_batches):

                loss_i_tensor = loss_i_fn(model, sample_i_fn)

                model.zero_grad()
                loss_i_tensor.backward()

                g_i_sample = get_grad_vec(model).detach().clone()
                g_i_samples.append(g_i_sample)

            
            true_g_i = torch.mean(torch.stack(g_i_samples), axis=0)
            true_g_i_s.append(true_g_i)

            input_errors_g_i = [((g_i_sample - true_g_i)**2).sum().item() for g_i_sample in g_i_samples]

            # expected value
            input_noise_power = sum(input_errors_g_i) / len(input_errors_g_i)

            input_noise_powers.append(input_noise_power)

            all_g_i_samples.append(g_i_samples)
        
        # sum over m
        input_noise_power = sum(input_noise_powers)

        g_config_true = ConFIG_update(true_g_i_s)

        output_errors = []

        cosine_sims_to_ground_truth = []
        cosine_sims_masked_to_ground_truth = []

        for batch_idx in range(self.k_batches):

            batch_gs = [all_g_i_samples[i][batch_idx] for i in range(len(loss_fns))]

            g_config_batch = ConFIG_update(batch_gs)

            # we walk in the direction of the masked first moment 
            momentum_masked = (g_config_batch * first_momentum > 0).float() * first_momentum
            #print(momentum_masked)

            output_errors.append(((g_config_true - g_config_batch)**2).sum().item())

            # measure angle dispersion
            cosine_sims_to_ground_truth.append(self.cosine_sim(g_config_true, g_config_batch).item())

            # measure masked angle dispersion
            cosine_sims_masked_to_ground_truth.append(self.cosine_sim(g_config_true, momentum_masked).item())

        output_noise_power = sum(output_errors) / self.k_batches

        naf =  (output_noise_power / input_noise_power)

        # TODO: input snr berechnen

        output_snr = (torch.linalg.norm(g_config_true)**2).item() / output_noise_power

        #print(f"NAF = {naf}, Output SNR = {output_snr}")

        # print angle mean
        angle_mean = np.mean(np.array(cosine_sims_to_ground_truth))
        angle_std = np.std(np.array(cosine_sims_to_ground_truth), ddof=1)

        angle_masked_mean = np.mean(np.array(cosine_sims_masked_to_ground_truth))
        angle_masked_std = np.std(np.array(cosine_sims_masked_to_ground_truth), ddof=1)

        print(f"angle = ({angle_mean} +/- {angle_std})")
        print(f"angle = ({angle_masked_mean} +/- {angle_masked_std})")
        print(f"angle = ({self.cosine_sim(first_momentum, g_config_true).item()})")
        print("---")

        record = {
            "epoch": epoch,
            "input_noise_power": input_noise_power,
            "output_noise_power": output_noise_power,
            "naf": (output_noise_power / input_noise_power),
            "output_snr": output_snr,
            "angle_mean": angle_mean,
            "angle_std": angle_std
        }

        # add input snr
        # add output snr
        self.history.append(record)
        
    def get_df(self):
        return pd.DataFrame(self.history)

class InfinityNormGeometryTracker(Tracker):

    def __init__(self):
        
        super().__init__()

        self.history = []

    def cosine_sim(self, g_a, g_b):

        norm_g_a = torch.linalg.norm(g_a)
        norm_g_b = torch.linalg.norm(g_b)

        eps = 1e-6

        if norm_g_a < eps or norm_g_b < eps:
            return 0

        return torch.dot(g_a, g_b) / (norm_g_a * norm_g_b + eps)

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():

            self.history.append({
                "epoch": epoch,
                "infinity_norm_factor": self.cosine_sim(update_dir, torch.sign(update_dir)).item()
            })

    def get_df(self):
        return pd.DataFrame(self.history)


class GradientAlignmentTracker(Tracker):

    def __init__(self):
        
        super().__init__()

        self.history = []

        self.last_g = None

    def cosine_sim(self, g_a, g_b):

        norm_g_a = torch.linalg.norm(g_a)
        norm_g_b = torch.linalg.norm(g_b)

        eps = 1e-6

        if norm_g_a < eps or norm_g_b < eps:
            return 0

        return torch.dot(g_a, g_b) / (norm_g_a * norm_g_b + eps)

    def track_directions(self, epoch, grads, g_config, adam_out, update_dir):
        
        with torch.no_grad():

            sum_of_normed_gradients = 0

            for g in grads:
                sum_of_normed_gradients += (g / torch.linalg.norm(g))

            n = len(grads)
            
            sq_mean_of_normed_gradients = ((sum_of_normed_gradients / n)**2).sum().item()

            intra_alignment_score = 2 * sq_mean_of_normed_gradients - 1

            # inter alignment
            sum_of_gradients = 0

            for g in grads:
                sum_of_gradients += g

            if self.last_g is not None:

                inter_alignment_score = self.cosine_sim(self.last_g, sum_of_gradients).item()
            else:
                # sensible dummy value
                inter_alignment_score = 1

            self.history.append({
                "epoch": epoch,
                "intra_alignment_score": intra_alignment_score,
                "inter_alignment_score": inter_alignment_score
            })

            self.last_g = sum_of_gradients

    def get_df(self):
        return pd.DataFrame(self.history)
