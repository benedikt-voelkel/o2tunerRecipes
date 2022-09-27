from os.path import join

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from o2tuner.config import resolve_path
from o2tuner.io import parse_json, parse_yaml


def param_plots(config):
    opt_dir = resolve_path(config["opt_dir"])
    ref_dir = resolve_path(config["ref_dir"])

    opt_summary = parse_yaml(join(opt_dir, "o2tuner_optimisation_summary.yaml"))
    best_trial_dir = join(opt_dir, opt_summary["best_trial_cwd"])

    opt_params = parse_json(join(best_trial_dir, "cuts_o2.json"))
    ref_params = parse_json(join(ref_dir, config["o2_medium_params_reference"]))

    # We are only interested in those parameters that are present in the optimised JSON,
    # because others are not touched
    default_params = ref_params["default"]["cuts"]
    CUT_ID_TO_NAME = [name for name in default_params if name in config["parameters_to_optimise"]]
    CUT_NAME_TO_ID = {name: i for i, name in enumerate(CUT_ID_TO_NAME)}

    def fill_ratios_single(cuts, cuts_ref):
        cut_list = [0] * len(CUT_ID_TO_NAME)
        for cut_name in CUT_ID_TO_NAME:
            cut_denom = cuts_ref[cut_name]
            if cut_denom <= 0:
                cut_denom = default_params[cut_name]
            cut_num = cuts[cut_name] if cuts[cut_name] > 0 else cut_denom
            cut_list[CUT_NAME_TO_ID[cut_name]] = cut_num / cut_denom
        return cut_list


    def fill_from_defaults(opt_params_module, medium_names, add_to_list):
        for opm in opt_params_module:
            medium_names.append(opm["medium_name"])
            add_to_list.append(fill_ratios_single(opm["cuts"], default_params))

    ratios_list = []
    medium_names = []
    for mod_name, medium_params in opt_params.items():
        if mod_name not in config["modules_to_optimise"]:
            continue
        if mod_name in ["default", "enableSpecialCuts", "enableSpecialProcesses"]:
            continue
        if mod_name not in ref_params:
            fill_from_defaults(medium_params, medium_names, ratios_list)
        current_ref_params = ref_params[mod_name]
        for medium_params_batch in medium_params:
            if "cuts" not in medium_params_batch:
                continue
            found_ind = None
            found_cuts = default_params
            for ind_ref, medium_params_batch_ref in enumerate(current_ref_params):
                if medium_params_batch_ref["local_id"] == medium_params_batch["local_id"]:
                    found_ind = ind_ref
                    found_cuts = medium_params_batch_ref.get("cuts", default_params)
                    break
            ratios_list.append(fill_ratios_single(medium_params_batch["cuts"], found_cuts))
            medium_names.append(medium_params_batch["medium_name"])
            if found_ind is not None:
                # remove this index so we search less and less each time
                del current_ref_params[found_ind]

    figure, ax = plt.subplots(figsize=(30, 30))
    #heatmap(ratios_list, ax, vmin=df.min().min(), vmax=df.max().max(), mask=False, norm=mcolors.LogNorm(), cmap=sns.color_palette("Greens", as_cmap=True))
    sns.heatmap(ratios_list, ax=ax, mask=False, norm=mcolors.LogNorm(), cmap=sns.color_palette("Greens", as_cmap=True), xticklabels=CUT_ID_TO_NAME, yticklabels=medium_names, linewidth=0.5)
    ax.set_xlabel("cut parameter", fontsize=40)
    ax.set_ylabel("medium name", fontsize=40)
    ax.tick_params(axis="both", labelsize=20)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)
    ax.collections[0].colorbar.ax.set_ylabel("ratio opt / ref", fontsize=40)

    figure.tight_layout()
    figure.savefig("param_difference_mod.png")
    plt.close(figure)

    return True
    


