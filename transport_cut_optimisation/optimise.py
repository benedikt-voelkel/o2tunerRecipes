"""
The cut tuning optimisation run
"""

import sys
import argparse
from os.path import join, abspath
from os import environ

import numpy as np

from o2tuner.system import run_command
from o2tuner.utils import annotate_trial
from o2tuner.optimise import optimise
from o2tuner.io import parse_json, dump_json, dump_yaml, parse_yaml, exists_file
from o2tuner.optimise import needs_cwd
from o2tuner.config import resolve_path

# Get environment variables we need to execute some cmds
O2_ROOT = environ.get("O2_ROOT")
MCSTEPLOGGER_ROOT = environ.get("MCSTEPLOGGER_ROOT")


def extract_hits(path, o2_detectors, det_name_to_id):
    """
    Retrieve the number of hits per detector
    """
    hits = [None] * len(o2_detectors)
    with open(path, "r", encoding="utf8") as hit_file:
        for line in hit_file:
            fields = line.split()
            if not fields:
                continue
            if fields[0] in o2_detectors:
                pot_nan = fields[1].lower()
                if "nan" in pot_nan:
                    hits[det_name_to_id[fields[0]]] = None
                    continue
                # NOTE relying on the first field containing the number of hits, this may change if the O2 macro changes
                hits[det_name_to_id[fields[0]]] = float(fields[1])
        return hits


def extract_avg_steps(path):
    """
    Retrieve the average number of original and skipped steps
    """
    search_string = "Original number, skipped, kept, skipped fraction and kept fraction of steps"
    extract_start = len(search_string.split())
    steps_orig = []
    steps_skipped = []
    with open(path, "r", encoding="utf8") as step_file:
        for line in step_file:
            if search_string in line:
                line = line.split()
                steps_orig.append(int(line[extract_start]))
                steps_skipped.append(int(line[extract_start + 1]))
    if not steps_orig:
        print("ERROR: Could not extract steps")
        sys.exit(1)
    return sum(steps_orig) / len(steps_orig), sum(steps_skipped) / len(steps_skipped)


def compute_metrics(hits_path, hits_ref_path, step_path, o2_detectors):
    """
    Compute the loss and return steps and hits relative to the baseline
    """
    hits_opt = extract_hits(hits_path, o2_detectors, {n: i for i, n in enumerate(o2_detectors)})
    hits_ref = extract_hits(hits_ref_path, o2_detectors, {n: i for i, n in enumerate(o2_detectors)})
    rel_hits = [h / r if (h is not None and r is not None and r > 0) else None for h, r in zip(hits_opt, hits_ref)]

    steps = extract_avg_steps(step_path)
    rel_steps = 1 - (steps[1] / steps[0])

    return rel_steps, rel_hits


def compute_loss(rel_hits, rel_steps, rel_hits_cutoff, penalty_below):
    """
    Compute the loss and return steps and hits relative to the baseline
    """
    rel_hits_valid = [rh for rh in rel_hits if rh is not None]
    loss = rel_steps**2
    penalise = 0
    for rvh in rel_hits_valid:
        if rvh < rel_hits_cutoff:
            penalise += 1
            loss += (penalty_below * (rel_hits_cutoff - rvh))**2

    return loss / (penalise + 1)


def mask_params(params, index_to_med_id, replay_cut_parameters):
    """
    Provide a mask and only enable indices for provided parameters
    (or all, if none are given)
    """
    if not params:
        return np.tile(np.full(len(replay_cut_parameters), True), len(index_to_med_id))
    mask = np.full(len(replay_cut_parameters), False)
    replay_param_to_id = {v: i for i, v in enumerate(replay_cut_parameters)}
    for par in params:
        mask[replay_param_to_id[par]] = True
    return np.tile(mask, len(index_to_med_id))


def unmask_modules(modules, replay_cut_parameters, index_to_med_id, passive_medium_ids_map, detector_medium_ids_map):
    """
    Un-mask all indices for a given list of modules
    """
    mask = np.full(len(index_to_med_id) * len(replay_cut_parameters), False)
    mod_med_map = {**passive_medium_ids_map, **detector_medium_ids_map}
    med_id_to_index = {med_id: i for i, med_id in enumerate(index_to_med_id)}

    for mod, medium_ids in mod_med_map.items():
        if mod not in modules:
            continue
        for mid in medium_ids:
            index = med_id_to_index[mid]
            mask[index * len(replay_cut_parameters):(index + 1) * len(replay_cut_parameters)] = True
    return mask


def arrange_to_space(arr, n_params, index_to_med_id):
    return {med_id: list(arr[i * n_params:((i + 1) * n_params)]) for i, med_id in enumerate(index_to_med_id)}


def make_o2_format(space_drawn, ref_params_json, passive_medium_ids_map, replay_cut_parameters):
    """
    Write the parameters and values in a JSON structure readible by O2MaterialManager
    """
    params = parse_json(ref_params_json)
    for module, batches in params.items():
        if module in ["default", "enableSpecialCuts", "enableSpecialProcesses"]:
            continue
        for batch in batches:
            med_id = batch["global_id"]
            # according to which modules are recognised by this script
            if module in passive_medium_ids_map:
                batch["cuts"] = dict(zip(replay_cut_parameters, space_drawn[med_id]))
    return params


def run_on_batch(batch_id, config):
    reference_dir = resolve_path(f"{config['reference_dir']}_{batch_id}")
    o2_medium_params_reference = join(reference_dir, config["o2_medium_params_reference"])
    ref_params_array = parse_yaml(join(reference_dir, config["reference_params"]))

    # replay the simulation
    baseline_dir = resolve_path(f"{config['baseline_dir']}_{batch_id}")
    kine_file = join(reference_dir, "o2sim_Kine.root")
    steplogger_file = join(reference_dir, "MCStepLoggerOutput.root")
    cut_file_param = ";MCReplayParam.cutFile=cuts.json"
    cmd = f'o2-sim-serial -n {config["events"]} -g extkinO2 --extKinFile {kine_file} -e MCReplay --skipModules ZDC ' \
          f'--configKeyValues="MCReplayParam.stepFilename={steplogger_file}{cut_file_param}"'
    _, sim_file = run_command(cmd, log_file="sim.log")

    # extract the hits using O2 macro and pipe to file
    extract_hits_root = abspath(join(O2_ROOT, "share", "macro", "analyzeHits.C"))
    cmd_extract_hits = f"root -l -b -q {extract_hits_root}"
    _, hit_file = run_command(cmd_extract_hits, log_file="hits.dat")

    # compute the loss and further metrics...
    baseline_hits_file = join(baseline_dir, "hits.dat")
    return compute_metrics(hit_file, baseline_hits_file, sim_file, config["O2DETECTORS"])


@needs_cwd
def objective_default(trial, config):
    """
    The central objective funtion for the optimisation
    """
    # Get some info from the reference dir
    reference_dir_any = resolve_path(f"{config['reference_dir']}_0")
    index_to_med_id = parse_yaml(join(reference_dir_any, config["index_to_med_id"]))
    passive_medium_ids_map = parse_yaml(join(reference_dir_any, config["passive_medium_ids_map"]))
    detector_medium_ids_map = parse_yaml(join(reference_dir_any, config["detector_medium_ids_map"]))
    o2_medium_params_reference = join(reference_dir_any, config["o2_medium_params_reference"])
    ref_params_array = parse_yaml(join(reference_dir_any, config["reference_params"]))

    # Get the new cut parameters for this trial
    previous_opt_values = config.get("previous_opt_dir", None)
    if previous_opt_values:
        best_trial_dir = resolve_path(previous_opt_values)
        best_trial_dir_rel = parse_yaml(join(best_trial_dir, "o2tuner_optimisation_summary.yaml"))["best_trial_cwd"]
        best_trial_dir = join(best_trial_dir, best_trial_dir_rel)
        previous_opt_values = parse_yaml(join(best_trial_dir, config["opt_params"]))

    # make param mask so to set only those parameters requested
    mask = mask_params(config["parameters_to_optimise"], index_to_med_id, config["REPLAY_CUT_PARAMETERS"])
    mask_passive = unmask_modules(config["modules_to_optimise"], config["REPLAY_CUT_PARAMETERS"],
                                  index_to_med_id, passive_medium_ids_map, detector_medium_ids_map)
    mask = mask & mask_passive

    # draw/aka "suggest" parameters
    this_array = np.array(previous_opt_values) if previous_opt_values else np.full((len(mask,)), -1.)
    for i, param in enumerate(mask):
        if not param:
            continue
        low = ref_params_array[i]
        up = config["search_value_up"]
        if low < 0:
            print(f"Lower value was {low}")
            low = config["search_value_low"]
        if low >= up:
            up = low * 10
        this_array[i] = trial.suggest_float(f"{i}", low, up, log=True)
    space_drawn = arrange_to_space(this_array, len(config["REPLAY_CUT_PARAMETERS"]), index_to_med_id)

    # dump the JSONs. The first is digested by the MCReplay engine...
    param_file_path = "cuts.json"
    dump_json(space_drawn, param_file_path)
    # ...and the second can be used to directly to O2 --confKeyValues
    space_drawn_o2 = make_o2_format(space_drawn, o2_medium_params_reference, passive_medium_ids_map, config["REPLAY_CUT_PARAMETERS"])
    param_file_path_o2 = "cuts_o2.json"
    dump_json(space_drawn_o2, param_file_path_o2)
    
    batches = config["batches"]
    rel_steps_avg = 0
    rel_hits_avg = [0] * len(config["O2DETECTORS"])
    if config["use_all_batches"]:
        # EITHER we always run over all produced reference batches...
        rel_hits_has_hits = [0] * len(config["O2DETECTORS"])
        for i in range(batches):
            rel_steps, rel_hits = run_on_batch(i, config)
            rel_steps_avg += rel_steps
            for j, hits in enumerate(rel_hits):
                if hits is not None:
                    rel_hits_avg[j] += hits
                    rel_hits_has_hits[j] += 1
        for j, (hits, has_hits) in enumerate(zip(rel_hits_avg, rel_hits_has_hits)):
            if has_hits:
                rel_hits_avg[j] = hits / has_hits
        rel_steps_avg /= batches
    else:
        # ...OR only one random batch
        batch_id = np.random.randint(0, batches)
        rel_steps_avg, rel_hits_avg = run_on_batch(batch_id, config)
    
    # ...and annotate drawn space and metrics to trial so we can re-use it
    annotate_trial(trial, "space", list(this_array))
    annotate_trial(trial, "rel_steps", rel_steps)
    annotate_trial(trial, "rel_hits", rel_hits)
    dump_yaml([float(value) for value in this_array], config["opt_params"])

    # remove all the artifacts we don't need to keep space
    #remove_dir(cwd, keep=["hits.dat", "cuts.json", "cuts_o2.json", "sim.log"])
    return compute_loss(rel_hits_avg, rel_steps_avg, config["rel_hits_cutoff"], config["penalty_below"])
