"""
The cut tuning evaluation run
"""
from os.path import join

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    steps = inspectors[0].get_annotation_per_trial("rel_steps")
    y_axis_all = inspectors[0].get_annotation_per_trial("rel_hits")
    losses = inspectors[0].get_losses()
    for insp in inspectors[1:]:
        steps.extend(insp.get_annotation_per_trial("rel_steps"))
        y_axis_all.extend(insp.get_annotation_per_trial("rel_hits"))
        losses.extend(insp.get_losses())

    # X ticks just from 1 to n iterations
    x_axis = range(1, len(steps) + 1)

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
        x_axis = []
        y_axis = []
        for trial, yax in enumerate(y_axis_all):
            if yax[i] is None:
                continue
            x_axis.append(trial)
            y_axis.append(yax[i])
        #y_axis = [yax[i] for yax in y_axis_all]
        #if None in y_axis:
        #    continue
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


def evaluate(inspectors, config):

    map_params = {}
    insp = inspectors[0]

    if config:
        plot_steps_hits_loss(inspectors, config)
        # at the same time, extract mapping of optuna parameter names to actual meaningful names related to the task at hand
        counter = 0
        index_to_med_id = parse_yaml(join(resolve_path(f"{config['reference_dir']}_0"), config["index_to_med_id"]))
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
