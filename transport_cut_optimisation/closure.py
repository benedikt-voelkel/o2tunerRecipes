from os.path import join
from math import sqrt
from os import environ

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcmap
import seaborn as sns

from ROOT import TGeoManager, TFile

from o2tuner.config import resolve_path
from o2tuner.io import parse_json, parse_yaml
from o2tuner.system import run_command

class ParamHelper:
    def __init__(self, config):
        self.config = config

        self.opt_dir = resolve_path(config["optimisation_dir"])
        self.ref_dir = f"{resolve_path(config['reference_dir'])}_0"

        opt_summary = parse_yaml(join(self.opt_dir, "o2tuner_optimisation_summary.yaml"))
        self.best_trial_dir = join(self.opt_dir, opt_summary["best_trial_cwd"])
        self.opt_params_file = join(self.best_trial_dir, "cuts_o2.json")

        self.opt_params = parse_json(self.opt_params_file)
        self.ref_params = parse_json(join(self.ref_dir, config["o2_medium_params_reference"]))

        # We are only interested in those parameters that are present in the optimised JSON,
        # because others are not touched
        self.default_params = self.ref_params["default"]["cuts"]
        self.cut_id_to_name = [name for name in self.default_params if name in config["parameters_to_optimise"]]
        self.cut_name_to_id = {name: i for i, name in enumerate(self.cut_id_to_name)}

    def fill_ratios_single(self, cuts, cuts_ref):
        cut_list = [0] * len(self.cut_id_to_name)
        for cut_name in self.cut_id_to_name:
            cut_denom = cuts_ref[cut_name]
            if cut_denom <= 0:
                cut_denom = self.default_params[cut_name]
            cut_num = cuts[cut_name] if cuts[cut_name] > 0 else cut_denom
            cut_list[self.cut_name_to_id[cut_name]] = cut_num / cut_denom
        return cut_list

    def fill_from_defaults(self, opt_params_module, medium_names, add_to_list):
        for opm in opt_params_module:
            medium_names.append(opm["medium_name"])
            add_to_list.append(self.fill_ratios_single(opm["cuts"], self.default_params))

    def make_ratios(self):
        ratios_list = []
        medium_names = []
        for mod_name, medium_params in self.opt_params.items():
            if mod_name not in self.config["modules_to_optimise"]:
                continue
            if mod_name in ["default", "enableSpecialCuts", "enableSpecialProcesses"]:
                continue
            if mod_name not in self.ref_params:
                fill_from_defaults(medium_params, medium_names, ratios_list)
            current_ref_params = self.ref_params[mod_name]
            for medium_params_batch in medium_params:
                if "cuts" not in medium_params_batch:
                    continue
                found_ind = None
                found_cuts = self.default_params
                global_id = -1
                for ind_ref, medium_params_batch_ref in enumerate(current_ref_params):
                    if medium_params_batch_ref["local_id"] == medium_params_batch["local_id"]:
                        found_ind = ind_ref
                        found_cuts = medium_params_batch_ref.get("cuts", self.default_params)
                        global_id = medium_params_batch_ref["global_id"]
                        break
                ratios_list.append(self.fill_ratios_single(medium_params_batch["cuts"], found_cuts))
                medium_names.append(f"{medium_params_batch['medium_name']} (ID {global_id})")
                if found_ind is not None:
                    # remove this index so we search less and less each time
                    del current_ref_params[found_ind]

        return medium_names, ratios_list

    def sort_params_by_global_id(self):
        collect_ref = {}
        collect_opt = {}
        for collect, params in zip((collect_ref, collect_opt), (self.ref_params, self.opt_params)):
            for mod_name, medium_params in params.items():
                if mod_name in ["default", "enableSpecialCuts", "enableSpecialProcesses"]:
                    continue
                for batch in medium_params:
                    if "cuts" not in batch:
                        continue
                    cut_list = [0] * len(self.cut_id_to_name)
                    for ind, cut_name in enumerate(self.cut_id_to_name):
                        cut_list[ind] = batch["cuts"][cut_name]
                    collect[batch["global_id"]] = cut_list
        return collect_ref, collect_opt



def param_plots(config):
    param_helper = ParamHelper(config)

    medium_names, ratios_list = param_helper.make_ratios()

    figure, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(ratios_list, ax=ax, mask=False, norm=mcolors.LogNorm(), cmap=sns.color_palette("Greens", as_cmap=True), xticklabels=param_helper.cut_id_to_name, yticklabels=medium_names, linewidth=0.5)
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
    
def param_rz(config):

    param_helper = ParamHelper(config)
    ref_params, opt_params = param_helper.sort_params_by_global_id()

    cut_bins = config.get("cut_bins", [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.])

    # load our geometry
    geo_file = join(param_helper.ref_dir, "o2sim_geometry.root")
    geo_mgr = TGeoManager.Import(geo_file)
    # get the cave dimensions
    cave = geo_mgr.GetVolume("cave")
    cave_box = cave.GetShape()
    x_lim_raw = (-cave_box.GetDX(), cave_box.GetDX())
    y_lim_raw = (-cave_box.GetDY(), cave_box.GetDY())
    x_lim = (-10 / sqrt(2), 10 / sqrt(2))
    y_lim = (-10 / sqrt(2), 10 / sqrt(2))
    z_lim = (-cave_box.GetDZ(), cave_box.GetDZ())

    size_ratio = (z_lim[1] - z_lim[0])  / (sqrt(2) * x_lim[1])

    # define the number of voxels in x, y, z, respectively
    resolution = config.get("n_voxels", 20)
    n_voxels_x = int((x_lim[1] - x_lim[0]) / resolution)
    n_voxels_y = int((y_lim[1] - y_lim[0]) / resolution)
    n_voxels_z = int((z_lim[1] - z_lim[0]) / resolution)
    vmin = config.get("heatmap_min", 0.00001)
    vmax = config.get("heatmap_max", 1)

    def make_rz(param_space, n_voxels, suffix):
        print(f"Make R-Z for suffix {suffix}")

        # The voxels which will each get assigned a bin number according to the cut value
        voxels = np.full((*n_voxels, 5), -1)
        coord_xyz = np.zeros((*n_voxels, 3))

        # step size of x, y, z spacing
        step_x = (x_lim[1] - x_lim[0]) / n_voxels[0]
        step_y = (y_lim[1] - y_lim[0]) / n_voxels[1]
        step_z = (z_lim[1] - z_lim[0]) / n_voxels[2]

        # loop over all voxels
        for i in range(n_voxels[0]):
            x = x_lim[0] + step_x * (1 + 2 * i) / 2
            for j in range(n_voxels[1]):
                y = y_lim[0] + step_y * (1 + 2 * j) / 2
                for k in range(n_voxels[2]):
                    z = z_lim[0] + step_z * (1 + 2 * k) / 2
                    # save the voxel coordinates
                    coord_xyz[i][j][k][:] = np.array([x, y, z])
                    # find the volume at this point
                    geo_mgr.FindNode(x, y, z)
                    vol = geo_mgr.GetCurrentVolume()
                    med = vol.GetMedium()
                    if not med:
                        print(f"No medium found vor volume {vol.GetName()}, continue...")
                        continue
                    med_id = med.GetId()
                    if med_id not in param_space:
                        # This might happen when only certain modules are of interest
                        #print(f"Cannot find medium ID {med_id}, continue...")
                        #print(f"MedId {med_id} not known")
                        continue
                    # get the maximum value because that is what is taken anyway
                    # TODO Take care of that so that we get the fully corrected space when we request the best space
                    for p, cut_name in enumerate(param_helper.cut_id_to_name):
                        value = param_space[med_id][p]
                        if value <= 0:
                            value = param_helper.default_params[cut_name]
                        # Add to voxels
                        voxels[i][j][k][p] = value

        # Here we can cut out slices of voxels now
        r = np.sqrt(coord_xyz[:,:,:,0]**2 + coord_xyz[:,:,:,1]**2).flatten()
        z = coord_xyz[:,:,:,2].flatten()

        cmap_rz = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = mcmap.get_cmap("Greens")

        for i, p in enumerate(param_helper.cut_id_to_name):
            print(f"For parameter {p}")
            colors = [v for v in voxels[:,:,:,i].flatten()]
            figure, ax = plt.subplots(figsize=(40, 10)) #0 / size_ratio))
            ax.scatter(z, r, c=colors, s=2, norm=cmap_rz, cmap=cmap)
            ax.set_xlabel("Z [cm]", fontsize=40)
            ax.set_ylabel("R [cm]", fontsize=40)
            ax.tick_params(axis="both", labelsize=30)
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)
            #ax.set_aspect("equal", adjustable="box")

            #figure.tight_layout()
            save_name = f"params_rz_{p}_{suffix}.png"
            figure.savefig(save_name)
            plt.close(figure)

    make_rz(ref_params, (n_voxels_x, n_voxels_y, n_voxels_z), f"ref")
    make_rz(opt_params, (n_voxels_x, n_voxels_y, n_voxels_z), f"opt")
    return True


def make_sorted_histos(histos):
    x_axis = []

    for h in histos:
      axis = h.GetXaxis()
      for i in range(1, h.GetNbinsX() + 1):
          bl = axis.GetBinLabel(i)
          if bl in x_axis or not bl:
              continue
          x_axis.append(bl)
    lab_to_id = {lab: i for i, lab in enumerate(x_axis)}

    sorted_histos = [[0] * len(x_axis) for _ in histos]

    for i, h in enumerate(histos):
        axis = h.GetXaxis()
        for b in range(1, h.GetNbinsX() + 1):
            bl = axis.GetBinLabel(b)
            if not bl:
                continue
            sorted_histos[i][lab_to_id[bl]] = h.GetBinContent(b)

    return x_axis, sorted_histos


def overlay_histograms(x_axis, sorted_histos, labels, savepath, x_label="x_axis", y_label="y_axis", annotations=None):

    if not labels:
        labels = [f"histo_{i}" for i, _ in enumerate(sorted_histos)]

    fig, ax = plt.subplots(figsize=(40, 25))

    hatches = [None, "/"]
    
    for i, (h, l) in enumerate(zip(sorted_histos, labels)):
        ax.bar(x_axis, h, alpha=0.5, label=l, hatch=hatches[i%len(hatches)])

    ax.set_xlabel(x_label, fontsize=40)
    ax.set_ylabel(y_label, fontsize=40)
    ax.tick_params(axis="both", labelsize=20)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    ax.legend(loc="best", fontsize=40)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)



def step_analysis(config):
    
    MCSTEPLOGGER_ROOT = environ.get("MCSTEPLOGGER_ROOT")
    param_helper = ParamHelper(config)
    events = config["events"]
    generator = config["generator"]
    engine = config["engine"]

    cmd = f'MCSTEPLOG_TTREE=1 LD_PRELOAD={MCSTEPLOGGER_ROOT}/lib/libMCStepLoggerInterceptSteps.so ' \
          f'o2-sim-serial -n {events} -g extkinO2  -e {engine} --extKinFile {join(param_helper.ref_dir, "o2sim_Kine.root")} ' \
          f'--skipModules ZDC --configKeyValues "MaterialManagerParam.inputFile={param_helper.opt_params_file}"'

    run_command(cmd, log_file="steplogging.log")

    cmd = "mcStepAnalysis analyze -f {} -l {} -o {}"

    # Run step analysis for ref
    cmd_ref = cmd.format(join(param_helper.ref_dir, "MCStepLoggerOutput.root"), "ref_cuts", "ref_cuts")
    run_command(cmd_ref)
    # Run step analysis for opt
    cmd_opt = cmd.format("MCStepLoggerOutput.root", "opt_cuts", "opt_cuts")
    run_command(cmd_opt)

    file_ref = TFile(join("ref_cuts", "SimpleStepAnalysis", "Analysis.root"), "READ")
    file_opt = TFile(join("opt_cuts", "SimpleStepAnalysis", "Analysis.root"), "READ")

    histos = [file_ref.Get("MCAnalysisObjects/nStepsPerMod"), file_opt.Get("MCAnalysisObjects/nStepsPerMod")]
    labels = ["reference", "optimised"]

    x_axis, sorted_histos = make_sorted_histos(histos)

    sum_ref = sum(sorted_histos[0])

    for sh in sorted_histos:
        for i, c in enumerate(sh):
            sh[i] = c / sum_ref


    overlay_histograms(x_axis, sorted_histos[:1], labels[:1], "steps_per_mod_ref.png", x_label="modules", y_label="steps / sum(steps(ref))")
    overlay_histograms(x_axis, sorted_histos, labels, "steps_per_mod_ref_opt.png", x_label="modules", y_label="steps / sum(steps(ref))")

    return True



