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
    for i, h in enumerate(hits):
        if h is None:
            continue
        hits[i] = [d[0] if d is not None else None for d in h]

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

    ax.set_xlabel("trial", fontsize=20)
    ax.set_ylabel("rel. number of kept hits and steps", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncols=4, mode="expand", borderaxespad=0., fontsize=20)

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

    #if config.get("with_distributions", None):
    #      for i, h in enumerate(hits):
    #          if h is None:
    #              continue
    #          hits[i] = [d[0] if d is not None else None for d in h]

    for i, det in enumerate(config["O2DETECTORS"]):
        y_axis = []
        for yax in hits:
            if yax[i] is None:
                y_axis.append(0)
            y_axis.append(yax[i][0] if yax[i] else None)
        #y_axis = [yax[i] for yax in y_axis_all]
        #if None in y_axis:
        #    continue
        if not any(y_axis):
            continue
        df[f"rel_hits_{det}"] = y_axis
        col_keep.append(f"rel_hits_{det}")
        col_hits.append(f"rel_hits_{det}")

    columns = df.columns
    keep = [c for c in columns if "params" in c and not "_do" in c]
    col_keep.extend(keep)
    df = df[col_keep]
    print(df["rel_hits_FDD"])

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
    print(corr)
    corr.drop(col_hits, inplace=True)

    corr.drop(keep, axis=1, inplace=True)
    corr.drop("rel_hits_MID", axis=1, inplace=True)
    sns.heatmap(corr, ax=ax, cmap=cmap, yticklabels=columns_new, vmin=-1, vmax=1)
    ax.set_xlabel("rel. number of hits", fontsize=50)
    ax.set_ylabel("parameter", fontsize=50)
    ax.tick_params(axis="both", labelsize=40)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=40)
    ax.collections[0].colorbar.ax.set_ylabel("correlation", fontsize=50)
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
        insp.set_parameter_name_map(map_params)
    else:
        print("WARNING: Cannot do the step and hits history without the user configuration")

    for i, insp in enumerate(inspectors):
        figure, _ = insp.plot_correlations(params_regex="(^[0-9][0-9]*$)")
        figure.savefig(f"parameter_correlations_{i}.png")
        figure.tight_layout()
        plt.close(figure)
        return True


        figure, _ = insp.plot_loss_feature_history(n_most_important=20)
        figure.tight_layout()
        figure.savefig(f"loss_feature_history_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_importance(n_most_important=None)
        figure.tight_layout()
        figure.savefig(f"importance_parameters_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_parallel_coordinates(n_most_important=20)
        figure.savefig(f"parallel_coordinates_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_slices(n_most_important=20)
        figure.savefig(f"slices_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_correlations(params_regex="(^[0-9][0-9]*)")
        figure.savefig(f"parameter_correlations_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_pairwise_scatter(n_most_important=20)
        figure.savefig(f"pairwise_scatter_{i}.png")
        plt.close(figure)

    return True


def evaluate_simple(inspectors, config):
    insp = inspectors[0]
    losses = insp.get_losses()
    dirs = insp.get_annotation_per_trial("cwd")
    rel_hits = insp.get_annotation_per_trial("rel_hits")
    rel_steps = insp.get_annotation_per_trial("rel_steps")

    filtered = []
    better_loss = losses[0]
    for i, (l, d, rh, rs) in enumerate(zip(losses, dirs, rel_hits, rel_steps)):
        if l < better_loss:
            # sum of relative hits
            rel_hits_sum = [h[0] for h in rh if h is not None and h[0] is not None]
            dets = [det for h, det in zip(rh, config["O2DETECTORS"]) if h is not None and h[0] is not None]
            det_hits = ", ".join([f"{det}: {h}" for det, h in zip(dets, rel_hits_sum)])
            rel_hits_sum = sum(rel_hits_sum) / len(rel_hits_sum)
            filtered.append((i+1, l, d, rel_hits_sum, rs, det_hits))
            better_loss = l
    print("Losses with directories")
    for i, l, d, h, s, det_h in filtered:
        print(f"Trial {i}, Loss {l} is in directory {d}, sum of relative number hits is {h} ({det_h}) and relative number of steps {s}")
    return True
        

