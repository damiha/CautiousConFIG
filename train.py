import math
import random
import json
import copy
import importlib
import os
import itertools

from hydra.utils import to_absolute_path
import hydra
from omegaconf import DictConfig
from functools import reduce

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_single_seed(cfg: DictConfig):

    # dangerous imports?
    import torch
    #from neural_network import *
    from neural_network import NeuralNetwork, xavier_init_weights # wild card operator was not allowed inside function
    import matplotlib.pyplot as plt
    import tqdm
    import numpy as np
    from optimizer import Adam, OfficialAdam, Lion, \
     GlobalConfigLion, GlobalConfigAdam, MConfig, OfficialConfigAdam, \
      GlobalConfigAdamForAblations, MConfig, LocalConfigAdam, CosineAnnealingLR, GlobalConfigRMSProp, ConFIGGrams, CautiousConFIGCore
    
    from scipy.stats import qmc

    from tracker import UpdatedPercentageTracker, ValMseTracker, CosineSimilarityTracker, \
     StepwiseAlignmentTracker, SingularValueTracker, StepwiseMaskTracker, PathTurtuosityTracker, MultiTaskCurvatureTracker, \
     MultiLossTracker, GradientDotProductTracker, ConFIGScalingTracker, NoiseTracker, InfinityNormGeometryTracker, GradientAlignmentTracker, \
     UpdatedDistributionTracker, UpdatedPercentagePerLayerTracker, MaskSplitStatsTracker, \
      TopIndexJaccardTracker, SNRTracker, GradientSumSNRTracker, ParamGroupTracker,  \
      UpdatedDistributionForMaskTracker, SlidingWindowMaskTracker, CosineSimilarityK6Tracker

    ### force determinism
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    #torch.use_deterministic_algorithms(True)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False

    ### set seed
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.problem.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    ### 

    features = list(cfg.problem.features)
    device = cfg.problem.device
    optimizer_mode = cfg.optimizer.name
    n_epochs = cfg.problem.n_epochs
    use_caution = cfg.optimizer.use_caution

    json_to_write_out = {
        "features": features,
        "device": device,
        "optimizer_mode": optimizer_mode,
        "n_epochs": n_epochs,
        "use_caution": use_caution,
        "use_lhs": cfg.problem.use_lhs
    }

    use_res = False
    if "use_res" in cfg.problem:
        use_res = cfg.problem.use_res

    fixed_batch = False
    if "fixed_batch" in cfg.problem:
        fixed_batch = cfg.problem.fixed_batch
    
    json_to_write_out["fixed_batch"] = fixed_batch

    if fixed_batch:

        assert "k_batches" in cfg.problem, "Need to know how many batches..."

        k_batches = cfg.problem.k_batches
        json_to_write_out["k_batches"] = k_batches

    json_to_write_out["use_res"] = use_res

    if "lr" in cfg.problem:
        lr = cfg.problem.lr
        json_to_write_out["lr"] = lr
    
    # just overwrite
    if "max_lr" in cfg.problem:
        lr = cfg.problem.max_lr

        json_to_write_out["max_lr"] = cfg.problem.max_lr
        json_to_write_out["min_lr"] = cfg.problem.min_lr
        json_to_write_out["lr_scheduler"] = cfg.problem.lr_scheduler

    ### stuff for the ablations
    random_masking=False
    rescale_strategy="cautious"
    some_mask=True

    if "random_masking" in cfg.optimizer:
        random_masking = cfg.optimizer.random_masking
        
    if "rescale_strategy" in cfg.optimizer:
        rescale_strategy = cfg.optimizer.rescale_strategy

    if "some_mask" in cfg.optimizer:
        some_mask = cfg.optimizer.some_mask
        
    if cfg.optimizer.name == "AblatedConFIG":
        json_to_write_out["rescale_strategy"] = rescale_strategy
        json_to_write_out["random_masking"] = random_masking
        json_to_write_out["some_mask"] = some_mask
    ###

    pde_module = importlib.import_module(f"problems.{cfg.problem.name}_logic")

    dataset = pde_module.get_dataset(seed=cfg.seed, use_lhs=cfg.problem.use_lhs)

    tester = pde_module.get_tester(device)

    loss_fns = pde_module.get_loss_functions()

    sampling_fns = dataset.get_sampling_functions()

    noise_tracker = None

    if fixed_batch:

        new_sampling_fns = []

        # use this for noise calculations
        fixed_sampling_fns = []
        
        if "noise" in cfg.trackers:
            noise_tracker = NoiseTracker(k_batches)

        for sample_fn in sampling_fns:

            batches = [sample_fn() for _ in range(k_batches)]

            cycler = itertools.cycle(batches)
            cycler2 = itertools.cycle(batches)

            new_sampling_fns.append(
                lambda c=cycler: tuple(t.clone() for t in next(c))
            )

            fixed_sampling_fns.append(
                lambda c=cycler2: tuple(t.clone() for t in next(c))
            )

        sampling_fns = new_sampling_fns

    model = NeuralNetwork(features, use_res = use_res)
    model = model.to(device)
    model.apply(xavier_init_weights)

    # add trackers only to ConFIG for now
    trackers = []
    val_mse_tracker = None

    if noise_tracker is not None:
        trackers.append(noise_tracker)

    for tracker_name in cfg.trackers:

        if tracker_name == "updated_percentage":
            trackers.append(UpdatedPercentageTracker())

        elif tracker_name == "val_mse":
            val_mse_tracker = ValMseTracker()
            trackers.append(val_mse_tracker)

        elif tracker_name == "cosine_sim":
            trackers.append(CosineSimilarityTracker())

        elif tracker_name == "stepwise_alignment":
            trackers.append(StepwiseAlignmentTracker())

        elif tracker_name == "singular_value":
            trackers.append(SingularValueTracker())

        elif tracker_name == "stepwise_mask":
            trackers.append(StepwiseMaskTracker())

        elif tracker_name == "path_turtuosity":
            trackers.append(PathTurtuosityTracker())

        elif tracker_name == "multi_task_curvature":
            trackers.append(MultiTaskCurvatureTracker())

        elif tracker_name == "multi_loss":
            trackers.append(MultiLossTracker())

        elif tracker_name == "gradient_dot_product":
            trackers.append(GradientDotProductTracker())

        elif tracker_name == "config_scaling":
            trackers.append(ConFIGScalingTracker())

        elif tracker_name == "infinity_norm":
            trackers.append(InfinityNormGeometryTracker())

        elif tracker_name == "gradient_alignment":
            trackers.append(GradientAlignmentTracker())

        elif tracker_name == "updated_distribution":
            trackers.append(UpdatedDistributionTracker())

        elif tracker_name == "updated_distribution_mask":
            trackers.append(UpdatedDistributionForMaskTracker())

        elif tracker_name == "sliding_window_mask":
            trackers.append(SlidingWindowMaskTracker())

        elif tracker_name == "updated_percentage_per_layer":
            trackers.append(UpdatedPercentagePerLayerTracker())

        elif tracker_name == "mask_split_stats":
            trackers.append(MaskSplitStatsTracker())

        elif tracker_name == "snr":
            trackers.append(SNRTracker())

        elif tracker_name == "gsum_snr":
            trackers.append(GradientSumSNRTracker())

        elif tracker_name == "param_group":
            trackers.append(ParamGroupTracker())

        elif tracker_name == "cosine_sim_k6":
            trackers.append(CosineSimilarityK6Tracker())

    # TODO: remove unconditional
    #trackers.append(TopIndexJaccardTracker())

    if optimizer_mode == "Adam":

        optimizer = Adam(model, lr, loss_fns, sampling_fns, use_caution=use_caution, trackers=trackers)

    elif optimizer_mode == "OfficialAdam":

        optimizer = OfficialAdam(model, lr, loss_fns, sampling_fns, use_caution=use_caution, trackers=trackers)

    if optimizer_mode == "Lion":

        optimizer = Lion(model, lr, loss_fns, sampling_fns, use_caution=use_caution, trackers=trackers)

    elif optimizer_mode == "ConFIG":

        optimizer = GlobalConfigAdam(model, lr, loss_fns, sampling_fns, use_caution=use_caution, trackers=trackers)

    elif optimizer_mode == "CautiousConFIGCore":
        optimizer = CautiousConFIGCore(model, lr, loss_fns, sampling_fns, trackers=trackers)

    elif optimizer_mode == "ConFIGGrams":

        optimizer = ConFIGGrams(model, lr, loss_fns, sampling_fns, trackers=trackers)

    elif optimizer_mode == "ConFIGRMSProp":

        optimizer = GlobalConfigRMSProp(model, lr, loss_fns, sampling_fns, trackers=trackers)

    elif optimizer_mode == "ConFIGLion":

        optimizer = GlobalConfigLion(model, lr, loss_fns, sampling_fns, use_caution=use_caution, trackers=trackers)

    elif optimizer_mode == "MConFIG":

        optimizer = MConfig(model, lr, loss_fns, sampling_fns, use_caution=use_caution, trackers=trackers)

    elif optimizer_mode == "OfficialConFIG":

        optimizer = OfficialConfigAdam(model, lr, loss_fns, sampling_fns, use_caution=use_caution, trackers=trackers)

    elif optimizer_mode == "AblatedConFIG":

        optimizer = GlobalConfigAdamForAblations(model, lr, loss_fns, sampling_fns,
         use_caution=use_caution, trackers=trackers,
         random_mask=random_masking, rescale_strategy=rescale_strategy,
         some_mask=some_mask
        )

    elif optimizer_mode == "MConFIG":

        optimizer = MConfig(model, lr, loss_fns, sampling_fns, use_caution=use_caution)

    elif optimizer_mode == "LConFIG":

        optimizer = LocalConfigAdam(model, lr, loss_fns, sampling_fns, use_caution=use_caution, caution_type=cfg.optimizer.caution_type)

    # setup scheduler
    lr_scheduler = None

    if "lr_scheduler" in cfg.problem:

        if cfg.problem.lr_scheduler == "cosine":

            lr_scheduler = CosineAnnealingLR(
                optimizer,
                max_lr = cfg.problem.max_lr,
                min_lr = cfg.problem.min_lr,
                n_epochs = n_epochs
            )

        else:
            raise NotImplementedError(f"Unknonw LR scheduler: {cfg.problem.lr_scheduler}")

    pbar = tqdm.tqdm(range(n_epochs))

    best_mse = float("inf")
    best_model_weights = None

    compute_val_loss_every = 10

    for epoch_idx in pbar:

        optimizer.train_one_epoch()

        if "end_early" in cfg.problem and cfg.problem.end_early and cfg.problem.end_at_epoch < epoch_idx:
            break

        # noise calculation
        if epoch_idx % compute_val_loss_every == 0 and noise_tracker is not None:
            
            first_momentum = None

            if isinstance(optimizer, GlobalConfigAdam):
                first_momentum = optimizer.m

            noise_tracker.track_noise(epoch_idx, model, fixed_sampling_fns, loss_fns, first_momentum)

        # validation loss (only some epochs)
        if epoch_idx % compute_val_loss_every == 0:
        
            with torch.no_grad():

                val_mse = tester.compute_val_loss(model)

                if val_mse_tracker is not None:
                    val_mse_tracker.track_mse(epoch_idx, val_mse)

                if val_mse < best_mse:
                    best_mse = val_mse
                    best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                model.train()

        if lr_scheduler is not None:
            lr_scheduler.step()

        pbar.set_postfix({"curr_mse": f"{val_mse:.5f}", "best_mse": f"{best_mse:.2e}"})

    json_to_write_out["best_mse"] = best_mse
    json_to_write_out["seed"] = seed
    json_to_write_out["filename"] = f"{cfg.problem.name}_{cfg.optimizer.filename}"
    json_to_write_out["compute_val_loss_every"] = compute_val_loss_every

    with open(cfg.optimizer.filename, "w", encoding="utf-8") as file:
        json.dump(json_to_write_out, file, indent=4, ensure_ascii=False)
        print("Results saved in JSON!")

    # collect all trackers information
    dfs = []

    for tracker in trackers:
        dfs.append(tracker.get_df())

    if len(dfs) > 0:

        final_df = reduce(
            lambda left, right: left.merge(right, on="epoch", how="outer"),
            dfs
        )

        final_df = final_df.sort_values("epoch")
        final_df.to_csv("tracked.csv")

    if cfg.problem.test_model_afterwards:

        model = NeuralNetwork(features, use_res=use_res)
        model = model.to(device)
        model.apply(xavier_init_weights)

        model.eval()
        model.load_state_dict(best_model_weights)

        pde_module.create_graphs(model, device)

if __name__ == "__main__":
    run_single_seed()