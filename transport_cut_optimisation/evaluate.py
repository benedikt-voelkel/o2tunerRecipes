"""
The cut tuning evaluation run
"""
from os.path import join

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import seaborn as sns

from o2tuner.io import parse_yaml
from o2tuner.config import resolve_path


def plot_base(x, y, ax=None, title=None, **plot_kwargs):
    """
    wrapper around plotting x and y to an axes
    """
    if not ax:
        ax = plt.gca()
    if len(x) != len(y):
        print(f"WARNING: Not going to plot for different lengths of x ({len(x)}) and y ({len(y)})")
        return ax
    ax.plot(x, y, **plot_kwargs)
    if title:
        ax.set_title(title)
    return ax


def plot_steps_hits_loss(inspectors, config, outfile="steps_hits_history.png", best=False):
    """
    Helper function to plot steps and hits
    """

    # get everything we want to plot from the inspector
    insp = inspectors[0]
    if best:
        best_indices = insp.get_best_indices()
    else:
        best_indices = list(range(insp.get_n_trials()))
    steps = [s for i, s in enumerate(insp.get_annotation_per_trial("rel_steps")) if i in best_indices]
    hits = [h for i, h in enumerate(insp.get_annotation_per_trial("rel_hits")) if i in best_indices]
    losses = insp.get_losses(flatten=False)

    x_axis = [n for i, n in enumerate(insp.get_trial_numbers()) if i in best_indices]
    figure, ax = plt.subplots(figsize=(30, 10))
    linestyles = ["--", ":", "-."]
    colors = list(mcolors.TABLEAU_COLORS.values())
    # first iterate through colors, then through lines styles
    line_style_index = 0

    # plot hits
    # loop through all detectors, indexing corresponds to their place in the user configuration
    for i, det in enumerate(config["O2DETECTORS"]):
        if i > 0 and not i % len(colors):
            line_style_index += 1
        y_axis = []
        for yax in hits:
            if yax[i] is None:
                y_axis.append(0)
            y_axis.append(yax[i])

        if not any(y_axis):
            continue
        plot_base(x_axis, y_axis, ax, label=det, linestyle=linestyles[line_style_index], color=colors[i % len(colors)], linewidth=2)

    # add steps to plot
    plot_base(x_axis, steps, ax, linestyle="-", linewidth=2, color="black", label="STEPS")
    # add loss to plot, make new axis to allow different scale
    ax_loss = ax.twinx()
    ax_loss.set_ylabel("LOSS", color="gray", fontsize=20)
    markers = ["+", "x"]
    for i, losses_i in enumerate(losses):
        plot_base(x_axis, [l for i, l in enumerate(losses_i) if i in best_indices], ax_loss, linestyle="", marker=markers[i%len(markers)], markersize=20, linewidth=2, color="gray")
    ax_loss.tick_params(axis="y", labelcolor="gray", labelsize=20)

    ax.set_xlabel("iteration", fontsize=20)
    ax.set_ylabel("rel. value hits, steps", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(loc="best", ncol=4, fontsize=20)

    figure.tight_layout()
    figure.savefig(outfile)
    plt.close(figure)


def plot_hits_param_correlation(inspectors, config, index_to_med_id):
    insp = inspectors[0]

    df = insp._study.trials_dataframe().query("state == 'COMPLETE'")
    steps = insp.get_annotation_per_trial("rel_steps")
    hits = insp.get_annotation_per_trial("rel_hits")
    df["rel_steps"] = steps
    col_keep = ["rel_steps"]
    col_hits = ["rel_steps"]

    for i, det in enumerate(config["O2DETECTORS"]):
        y_axis = []
        for yax in hits:
            if yax[i] is None:
                y_axis.append(0)
            y_axis.append(yax[i])
        #y_axis = [yax[i] for yax in y_axis_all]
        #if None in y_axis:
        #    continue
        if not any(y_axis):
            continue
        df[f"rel_hits_{det}"] = y_axis
        col_keep.append(f"rel_hits_{det}")
        col_hits.append(f"rel_hits_{det}")

    columns = df.columns
    keep = [c for c in columns if "params" in c]
    col_keep.extend(keep)
    df = df[col_keep]

    map_params = {}
    parameters_to_optimise = config["parameters_to_optimise"]
    
    for i, med_id in enumerate(index_to_med_id):
        param = parameters_to_optimise[i % len(parameters_to_optimise)]
        map_params[f"params_{i}"] = f"{param} of {med_id}"

    columns_new = [map_params[k] for k in keep]

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    fig, ax = plt.subplots(figsize=(40, 40))
    corr = df.corr()
    corr.drop(col_hits, inplace=True)

    corr.drop(keep, axis=1, inplace=True)
    corr.drop("rel_hits_MID", axis=1, inplace=True)
    sns.heatmap(corr, ax=ax, cmap=cmap, yticklabels=columns_new, vmin=-1, vmax=1)
    ax.tick_params(axis="both", labelsize=20)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)
    ax.collections[0].colorbar.ax.set_ylabel("correlation", fontsize=40)
    fig.tight_layout()
    fig.savefig("corr.png")
    plt.close(fig)


def evaluate(inspectors, config):

    map_params = {}
    insp = inspectors[0]
    index_to_med_id = insp.get_annotation_per_trial("index_to_medium_id")[0]
    parameters_to_optimise = config["parameters_to_optimise"]
    map_params = {}
    for i, med_id in enumerate(index_to_med_id):
        param = parameters_to_optimise[i % len(parameters_to_optimise)]
        map_params[f"{i}"] = f"{param} of {med_id}"
    insp.set_parameter_name_map(map_params)


    if config:
        plot_steps_hits_loss(inspectors, config, "steps_hits_history.png", False)
        plot_steps_hits_loss(inspectors, config, "steps_hits_history_best.png", True)
        plot_hits_param_correlation(inspectors, config, index_to_med_id)
    else:
        print("WARNING: Cannot do the step and hits history without the user configuration")

    
    for d in range(insp.n_directions):

        # LOSS - FEATURE history
        figure, _ = insp.plot_loss_feature_history(n_most_important=20, objective_number=d)
        figure.tight_layout()
        figure.savefig(f"loss_feature_history_{d}.png")
        plt.close(figure)

        # importance per objective
        figure, _ = insp.plot_importance(n_most_important=50, objective_number=d)
        figure.tight_layout()
        figure.savefig(f"importance_parameters_{d}.png")
        plt.close(figure)

        figure, _ = insp.plot_parallel_coordinates(objective_number=d)
        figure.savefig(f"parallel_coordinates_{d}.png")
        plt.close(figure)

        figure, _ = insp.plot_slices(objective_number=d)
        figure.savefig(f"slices_{d}.png")
        plt.close(figure)

    figure, _ = insp.plot_correlations()
    figure.savefig(f"parameter_correlations.png")
    plt.close(figure)

    figure, _ = insp.plot_pairwise_scatter()
    figure.savefig(f"pairwise_scatter.png")
    plt.close(figure)

    return True


def evaluate_print(inspectors, config):
    insp = inspectors[0]
    steps = insp.get_annotation_per_trial("rel_steps")
    for i, s in enumerate(steps):
        print(i, s)

    return True


