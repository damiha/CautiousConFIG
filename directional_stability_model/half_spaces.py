import numpy as np
import matplotlib.pyplot as plt
import tqdm

# works
def calculate_alignment_score(U_T):
    return (2 * (np.mean(U_T, axis=0)**2).sum() - 1).item()

def generate_conflicting_gradients(d, m, conflict_strength):

    v_conflict = np.random.randn(d)
    v_conflict /= np.linalg.norm(v_conflict)
    
    gs = []

    for i in range(m):

        direction = 1.0 if i < m // 2 else -1.0
        g_i = direction * conflict_strength * v_conflict + (1 - conflict_strength) * np.random.randn(d)
        gs.append(g_i)

    return gs

def ConFIG_direction(gs, alignment_scores):

    m = len(gs)
    G_T = np.stack(gs)

    uniform_gs = [g / (1e-8 + np.linalg.norm(g)) for g in gs]

    U_T = np.stack(uniform_gs)

    alignment_scores.append(calculate_alignment_score(U_T))

    g_u_tilde = np.linalg.pinv(U_T) @ np.ones(m)

    g_u = g_u_tilde / np.linalg.norm(g_u_tilde)

    g_config = (G_T @ g_u).sum() * g_u

    return g_config

def run_trial(conflict_strength, minibatch_sigma, d, m, n_tries=10_000):

    alignment_scores = []
    scores_u = []
    scores_cautious = []
    update_percentage = []

    n_cautious_in_hull = 0
    n_u_in_hull = 0

    for _ in range(n_tries):

        # conflict model
        gs = generate_conflicting_gradients(d, m, conflict_strength)

        G_T = np.stack(gs)

        g_config = ConFIG_direction(gs, alignment_scores)

        # noise model
        u_raw = g_config  + (minibatch_sigma * np.random.randn(d))

        u_cautious = (g_config * u_raw > 0).astype(float) * u_raw

        update_percentage.append((g_config * u_raw > 0).astype(float).mean())

        u_dot_products = G_T @ u_raw
        u_cautious_dot_products = G_T @ u_cautious

        cos_u = np.mean([np.dot(g, u_raw) / (np.linalg.norm(g) * np.linalg.norm(u_raw)) for g in gs])
        cos_cautious = np.mean([np.dot(g, u_cautious) / (np.linalg.norm(g) * np.linalg.norm(u_cautious)) for g in gs])

        scores_u.append(cos_u)
        scores_cautious.append(cos_cautious)

        if (u_dot_products > 0).all():
            n_u_in_hull += 1

        if (u_cautious_dot_products > 0).all():
            n_cautious_in_hull += 1

    # we can take the mean everywhere, we have a static model
    return {
        "intra_alignment_score": np.mean(np.array(alignment_scores)).item(),
        "cosine_sim_config": np.mean(np.array(scores_u)).item(),
        "cosine_sim_cautious": np.mean(np.array(scores_cautious)).item(),
        "p_conflictfree_config": n_u_in_hull / n_tries,
        "p_conflictfree_cautious": n_cautious_in_hull / n_tries,
        "update_percentage_cautious": np.mean(np.array(update_percentage)).item()
    }

# a study is a list of dictionaries


def run_intra_alignment_study():

    # intra alignment study
    # tests m and conflict strength
    # m -> different curves
    # conflict strength on x-axis
    # d = 1000
    # minibatch_sigma = 0
    intra_alignment_study = []
    for m in range(2, 7):
        for conflict_strength in list(np.linspace(0, 1, 21)):
            
            intra_alignment_study.append({
                "d": 1000,
                "n_tries": 1000,
                "minibatch_sigma": 0,
                "m": m,
                "conflict_strength": conflict_strength
            })

    m_to_intra_alignment_scores = {m: [] for m in range(2, 7)}

    for trial in tqdm.tqdm(intra_alignment_study):

        trial_result = run_trial(**trial)

        m_to_intra_alignment_scores[trial["m"]].append(trial_result["intra_alignment_score"])

    
    for m in range(2, 7):

        plt.plot(np.linspace(0, 1, 21), m_to_intra_alignment_scores[m], label=f"m = {m}")

    plt.xlabel("conflict strength")
    plt.ylabel("intra alignment score")
    plt.legend()
    plt.savefig("intra_alignment.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def run_conflict_strenght_and_m_at_constant_minibatch_noise_study():

    # extract cosine scores and descent successes and masking

    # tests m and conflict strength
    # m -> different curves
    m_range = range(2, 7)
    # conflict strength on x-axis
    conflict_range = list(np.linspace(0, 1, 21))
    d = 1000
    minibatch_sigma = 3

    intra_alignment_study = []
    for m in m_range:
        for conflict_strength in conflict_range:
            
            intra_alignment_study.append({
                "d": d,
                "n_tries": 1000,
                "minibatch_sigma": minibatch_sigma,
                "m": m,
                "conflict_strength": conflict_strength
            })

    # cosine sims
    m_to_cosine_sim_config = {m: [] for m in m_range}
    m_to_cosine_sim_cautious = {m: [] for m in m_range}

    # conflict free
    m_to_p_conflictfree_config = {m: [] for m in m_range}
    m_to_p_conflictfree_cautious = {m: [] for m in m_range}

    m_to_update_percentage_cautious = {m: [] for m in m_range}

    for trial in tqdm.tqdm(intra_alignment_study):

        trial_result = run_trial(**trial)

        m_to_p_conflictfree_config[trial["m"]].append(trial_result["p_conflictfree_config"])
        m_to_p_conflictfree_cautious[trial["m"]].append(trial_result["p_conflictfree_cautious"])

        m_to_cosine_sim_config[trial["m"]].append(trial_result["cosine_sim_config"])
        m_to_cosine_sim_cautious[trial["m"]].append(trial_result["cosine_sim_cautious"])

        m_to_update_percentage_cautious[trial["m"]].append(trial_result["update_percentage_cautious"])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    colors = {
        2: "tab:blue",
        3: "tab:orange",
        4: "tab:green",
        5: "tab:red",
        6: "tab:purple"
    }
    
    # plot the scores
    for m in m_range:

        # left for descent directions
        ax[0].plot(conflict_range, m_to_p_conflictfree_config[m], c=colors[m], linestyle="--")
        ax[0].plot(conflict_range, m_to_p_conflictfree_cautious[m], c=colors[m], label=f"m = {m}")

        # middle for the cosine sims
        ax[1].plot(conflict_range, m_to_cosine_sim_config[m], c=colors[m], linestyle="--")
        ax[1].plot(conflict_range, m_to_cosine_sim_cautious[m], c=colors[m], label=f"m = {m}")

        ax[2].plot(conflict_range, m_to_update_percentage_cautious[m], c=colors[m], label=f"{m=}")

    ax[0].set_xlabel("conflict strength")
    ax[1].set_xlabel("conflict strength")
    ax[2].set_xlabel("conflict strength")

    ax[0].set_ylabel(r"Percentage conflict-free")
    ax[1].set_ylabel(r"avg. $\text{Cos}(g_i, \Delta \theta)$")
    ax[2].set_ylabel("Upate Percentage")

    ax[1].legend()

    fig.suptitle(rf"d = 1000, $\sigma = {minibatch_sigma}$")
    fig.savefig("constant_noise.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

def run_noise_and_m_at_constant_conflict_strength_study():

    # extract cosine scores and descent successes and masking

    # tests m and conflict strength
    # m -> different curves
    m_range = range(2, 7)
    # conflict strength on x-axis

    # this is where everything happens
    conflict_strength = 0.9
    d = 1000

    minibatch_sigma_range = list(np.linspace(0, 10, 21))

    intra_alignment_study = []
    for m in m_range:
        for minibatch_sigma in minibatch_sigma_range:
            
            intra_alignment_study.append({
                "d": d,
                "n_tries": 1000,
                "minibatch_sigma": minibatch_sigma,
                "m": m,
                "conflict_strength": conflict_strength
            })

    # cosine sims
    m_to_cosine_sim_config = {m: [] for m in m_range}
    m_to_cosine_sim_cautious = {m: [] for m in m_range}

    # conflict free
    m_to_p_conflictfree_config = {m: [] for m in m_range}
    m_to_p_conflictfree_cautious = {m: [] for m in m_range}

    m_to_update_percentage_cautious = {m: [] for m in m_range}

    for trial in tqdm.tqdm(intra_alignment_study):

        trial_result = run_trial(**trial)

        m_to_p_conflictfree_config[trial["m"]].append(trial_result["p_conflictfree_config"])
        m_to_p_conflictfree_cautious[trial["m"]].append(trial_result["p_conflictfree_cautious"])

        m_to_cosine_sim_config[trial["m"]].append(trial_result["cosine_sim_config"])
        m_to_cosine_sim_cautious[trial["m"]].append(trial_result["cosine_sim_cautious"])

        m_to_update_percentage_cautious[trial["m"]].append(trial_result["update_percentage_cautious"])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    colors = {
        2: "tab:blue",
        3: "tab:orange",
        4: "tab:green",
        5: "tab:red",
        6: "tab:purple"
    }
    
    # plot the scores
    for m in m_range:

        # left for descent directions
        ax[0].plot(minibatch_sigma_range, m_to_p_conflictfree_config[m], c=colors[m], linestyle="--")
        ax[0].plot(minibatch_sigma_range, m_to_p_conflictfree_cautious[m], c=colors[m], label=f"m = {m}")

        # middle for the cosine sims
        ax[1].plot(minibatch_sigma_range, m_to_cosine_sim_config[m], c=colors[m], linestyle="--")
        ax[1].plot(minibatch_sigma_range, m_to_cosine_sim_cautious[m], c=colors[m], label=f"m = {m}")

        ax[2].plot(minibatch_sigma_range, m_to_update_percentage_cautious[m], c=colors[m], label=f"{m=}")

    ax[0].set_xlabel(r"$\sigma$")
    ax[1].set_xlabel(r"$\sigma$")
    ax[2].set_xlabel(r"$\sigma$")

    ax[0].set_ylabel(r"Percentage conflict-free")
    ax[1].set_ylabel(r"avg. $\text{Cos}(g_i, \Delta \theta)$")
    ax[2].set_ylabel("Upate Percentage")

    ax[1].legend()

    fig.suptitle(rf"d = 1000, conflict_strength = {conflict_strength}")
    fig.savefig("constant_conflict_strength.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

def sweep_model_size():

    # extract cosine scores and descent successes and masking

    # tests m and conflict strength
    # m -> different curves
    m_range = range(2, 7)
    # conflict strength on x-axis

    # this is where everything happens
    conflict_strength = 0.9
    minibatch_sigma = 3

    d_range = list(np.linspace(10, 1000, 21))
    d_range = [int(d) for d in d_range]

    intra_alignment_study = []
    for m in m_range:
        for d in d_range:
            
            intra_alignment_study.append({
                "d": d,
                "n_tries": 1000,
                "minibatch_sigma": minibatch_sigma,
                "m": m,
                "conflict_strength": conflict_strength
            })

    # cosine sims
    m_to_cosine_sim_config = {m: [] for m in m_range}
    m_to_cosine_sim_cautious = {m: [] for m in m_range}

    # conflict free
    m_to_p_conflictfree_config = {m: [] for m in m_range}
    m_to_p_conflictfree_cautious = {m: [] for m in m_range}

    m_to_update_percentage_cautious = {m: [] for m in m_range}

    for trial in tqdm.tqdm(intra_alignment_study):

        trial_result = run_trial(**trial)

        m_to_p_conflictfree_config[trial["m"]].append(trial_result["p_conflictfree_config"])
        m_to_p_conflictfree_cautious[trial["m"]].append(trial_result["p_conflictfree_cautious"])

        m_to_cosine_sim_config[trial["m"]].append(trial_result["cosine_sim_config"])
        m_to_cosine_sim_cautious[trial["m"]].append(trial_result["cosine_sim_cautious"])

        m_to_update_percentage_cautious[trial["m"]].append(trial_result["update_percentage_cautious"])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    colors = {
        2: "tab:blue",
        3: "tab:orange",
        4: "tab:green",
        5: "tab:red",
        6: "tab:purple"
    }
    
    # plot the scores
    for m in m_range:

        # left for descent directions
        ax[0].plot(d_range, m_to_p_conflictfree_config[m], c=colors[m], linestyle="--")
        ax[0].plot(d_range, m_to_p_conflictfree_cautious[m], c=colors[m], label=f"m = {m}")

        # middle for the cosine sims
        ax[1].plot(d_range, m_to_cosine_sim_config[m], c=colors[m], linestyle="--")
        ax[1].plot(d_range, m_to_cosine_sim_cautious[m], c=colors[m], label=f"m = {m}")

        ax[2].plot(d_range, m_to_update_percentage_cautious[m], c=colors[m], label=f"{m=}")

    ax[0].set_xlabel(r"#parameters")
    ax[1].set_xlabel(r"#parameters")
    ax[2].set_xlabel(r"#parameters")

    ax[0].set_ylabel(r"Percentage conflict-free")
    ax[1].set_ylabel(r"avg. $\text{Cos}(g_i, \Delta \theta)$")
    ax[2].set_ylabel("Upate Percentage")

    ax[1].legend()

    fig.suptitle(rf"conflict_strength = {conflict_strength}, $\sigma = {minibatch_sigma}$")
    fig.savefig("sweep_model_size.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":

    #run_intra_alignment_study()
    #run_conflict_strenght_and_m_at_constant_minibatch_noise_study()
    #run_noise_and_m_at_constant_conflict_strength_study()
    sweep_model_size()


