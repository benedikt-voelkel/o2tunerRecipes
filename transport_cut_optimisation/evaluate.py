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


def plot_steps_hits_loss(inspectors, config):
    """
    Helper function to plot steps and hits
    """

    # get everything we want to plot from the inspector
    steps = []
    hits = []
    losses = []
    for insp in inspectors:
        this_steps = insp.get_annotation_per_trial("rel_steps")
        steps.extend(this_steps)
        this_hits = insp.get_annotation_per_trial("rel_hits")
        hits.extend(this_hits)
        this_losses = insp.get_losses()
        losses.extend(this_losses)

    x_axis = range(len(losses))
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
        #y_axis = [yax[i] for yax in y_axis_all]
        #if None in y_axis:
        #    continue
        if not any(y_axis):
            continue
        plot_base(x_axis, y_axis, ax, label=det, linestyle=linestyles[line_style_index], color=colors[i % len(colors)], linewidth=2)

    # add steps to plot
    plot_base(x_axis, steps, ax, linestyle="-", linewidth=2, color="black", label="STEPS")
    # add loss to plot, make new axis to allow different scale
    ax_loss = ax.twinx()
    ax_loss.set_ylabel("LOSS", color="gray", fontsize=20)
    plot_base(x_axis, losses, ax_loss, linestyle="", marker="x", markersize=20, linewidth=2, color="gray")
    ax_loss.tick_params(axis="y", labelcolor="gray", labelsize=20)

    ax.set_xlabel("iteration", fontsize=20)
    ax.set_ylabel("rel. value hits, steps", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(loc="best", ncol=4, fontsize=20)

    figure.tight_layout()
    figure.savefig("steps_hits_history.png")
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
    counter = 0
    for med_id in index_to_med_id:
        for param in config["REPLAY_CUT_PARAMETERS"]:
            map_params[f"params_{str(counter)}"] = f"{param} of {med_id}"
            counter += 1

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

    if config:
        plot_steps_hits_loss(inspectors, config)

        index_to_med_id = parse_yaml(join(resolve_path(f"{config['reference_dir']}_0"), config["index_to_med_id"]))
        plot_hits_param_correlation(inspectors, config, index_to_med_id)
        # at the same time, extract mapping of optuna parameter names to actual meaningful names related to the task at hand
        counter = 0
        for med_id in index_to_med_id:
            for param in config["REPLAY_CUT_PARAMETERS"]:
                map_params[str(counter)] = f"{param} of {med_id}"
                counter += 1
    else:
        print("WARNING: Cannot do the step and hits history without the user configuration")

    for i, insp in enumerate(inspectors):
        figure, _ = insp.plot_importance(map_params=map_params, n_most_important=50)
        figure.tight_layout()
        figure.savefig(f"importance_parameters_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_parallel_coordinates(map_params=map_params)
        figure.savefig(f"parallel_coordinates_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_slices(map_params=map_params)
        figure.savefig(f"slices_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_correlations(map_params=map_params)
        figure.savefig(f"parameter_correlations_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_pairwise_scatter(map_params=map_params)
        figure.savefig(f"pairwise_scatter_{i}.png")
        plt.close(figure)

    return True

