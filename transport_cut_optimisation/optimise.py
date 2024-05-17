"""
The cut tuning optimisation run
"""

import sys
import argparse
from os.path import join, abspath
from os import environ
from math import exp

import numpy as np

from o2tuner.system import run_command
from o2tuner.utils import annotate_trial
from o2tuner.optimise import optimise
from o2tuner.io import parse_json, dump_json, dump_yaml, parse_yaml, exists_file
from o2tuner.optimise import needs_cwd, directions
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


def extract_avg_steps_ref_opt(path_ref, path_opt):
    """
    Retrieve the average number of original and skipped steps
    """
    search_string = "This event/chunk did"
    extract_field = 4

    def extract_steps_impl(path):
        steps = []
        with open(path, "r", encoding="utf8") as step_file:
            for line in step_file:
                if search_string in line:
                    line = line.split()
                    steps.append(int(line[extract_start]))
        return steps

    steps_ref = extract_steps_impl(path_ref)
    steps_opt = extract_steps_impl(path_opt)
    steps_skipped = [sr - so for sr, so in zip(steps_ref, steps_opt)]
    if not steps_ref or not steps_opt:
        print("ERROR: Could not extract steps")
        sys.exit(1)

    return sum(steps_ref) / len(steps_ref), sum(steps_skipped) / len(steps_skipped)


def compute_metrics(hits_path, hits_ref_path, step_path, o2_detectors, step_path_ref=None):
    """
    Compute the loss and return steps and hits relative to the baseline
    """
    hits_opt = extract_hits(hits_path, o2_detectors, {n: i for i, n in enumerate(o2_detectors)})
    hits_ref = extract_hits(hits_ref_path, o2_detectors, {n: i for i, n in enumerate(o2_detectors)})
    rel_hits = [h / r if (h is not None and r is not None and r > 0) else None for h, r in zip(hits_opt, hits_ref)]

    if step_path_ref:
      steps = extract_avg_steps_ref_opt(step_path_ref, step_path)
    else:
      steps = extract_avg_steps(step_path)
    rel_steps = 1 - (steps[1] / steps[0])

    return rel_steps, rel_hits


def compute_loss(rel_hits, rel_steps, rel_hits_cutoff, penalty_below, ref_values=None, drawn_values=None):
    """
    Compute the loss and return steps and hits relative to the baseline
    """
    rel_hits_valid = [rh for rh in rel_hits if rh is not None]

    if ref_values is None or drawn_values is None:
        loss = rel_steps**2
    else:
        loss = drawn_values / ref_values - 1
        loss = 10**(-loss)
        loss = (sum(loss) / len(loss))**2

    #loss += np.std(rel_hits_valid)**2
    # scale so that the loss is somewhat comparable when running with different number of detectors
    scaling = 1 / len(rel_hits_valid)
    loss_hits = 0
    #rel_hits_valid = [0.5 * exp(2 * (rel_hits_cutoff - rhv)) for rhv in rel_hits_valid]
    #rel_hits_valid = [exp((rel_hits_cutoff - rhv)) for rhv in rel_hits_valid]
    rel_hits_valid = min(rel_hits_valid)
    scale = 10
    offset = 1
    if rel_hits_valid <= rel_hits_cutoff:
        return loss, 10 * (1 - rel_hits_valid) + 1
    return loss, 1 / (1 - rel_hits_cutoff) * (1 - rel_hits_valid)


    return loss, max(rel_hits_valid)

    for rvh in rel_hits_valid:

        
        pb = penalty_below * (rel_hits_cutoff - rvh) if rvh < rel_hits_cutoff else 0
        #if rvh < rel_hits_cutoff:
            # since it's very low, add a penalty factor
        #    loss_hits += scaling * (penalty_below * (1 - rvh))**2
        #else:
        loss_hits += scaling * (1 - rvh)**2 + pb

    return loss, loss_hits


def make_o2_format(flat_value_list, index_to_med_id, module_medium_ids_map, replay_cut_parameters, ref_params):
    """
    Write the parameters and values in a JSON structure readible by O2MaterialManager
    """

    # get the media per module
    for module, medium_ids in module_medium_ids_map.items():
        media = ref_params[module]
        for medium in media:
            medium_id = medium["global_id"]
            
            # run through until we find this medium ID
            for idx in range(0, len(index_to_med_id), len(replay_cut_parameters)):
                med_id = index_to_med_id[idx]
                if med_id != medium_id:
                    continue
                take_values = flat_value_list[idx:idx+len(replay_cut_parameters)]
                cuts = medium.get("cuts", {})
                for i, param in enumerate(replay_cut_parameters):
                    cuts[param] = take_values[i]
                medium["cuts"] = cuts
    return ref_params


def flatten_o2_dict(o2_cuts_dict, replay_cut_parameters, o2_modules):
    """
    Derive lists and dicts from O2 format of medium parameters.

    The flat list will only be extended for values of interest.
    """
    # parameters just as a list
    params_flat = []
    # map the index of the above list to the medium ID
    index_to_med_id = []
    # map list of medium IDs to passivle module
    module_medium_ids_map = {}
    # map the medium name to the index
    index_to_med_name = []
    # map module name to index
    index_to_mod_name = []
    # take the defaults, these will be the lowest values to start from when tuning
    default_cuts = o2_cuts_dict["default"]["cuts"]

    # make sure that we always loop through the modules in the same order
    for idx_mod, module in enumerate(sorted(list(o2_cuts_dict.keys()))):
        if module in ["default", "enableSpecialCuts", "enableSpecialProcesses"]:
            # not actually modules, skip
            continue
        if module not in o2_modules:
            # not interested in that
            continue

        module_medium_map = module_medium_ids_map.get(module, [])

        # the media for this module
        media = o2_cuts_dict[module]

        for medium in media:
            # loop through all media of this module
            med_id = medium["global_id"]
            cuts = medium.get("cuts", {})
            cuts_append = []
            for rcp in replay_cut_parameters:
                # Set values of this paramter.
                # This medium might have an empty cut dict, but since it is requested to be studied, we fill the list
                value = cuts.get(rcp, default_cuts[rcp])
                cuts_append.append(value)

            index_to_med_id.extend([med_id] * len(cuts_append))
            index_to_med_name.extend(medium["medium_name"] * len(cuts_append))
            index_to_mod_name.extend(module * len(cuts_append))
            params_flat.extend(cuts_append)
            module_medium_map.append(med_id)

        # set the medium IDs for this module
        module_medium_ids_map[module] = module_medium_map

    return params_flat, index_to_med_id, index_to_med_name, index_to_mod_name, module_medium_ids_map


def run_on_batch(batch_id, config):
    """
    Run over one batch

    There might be various reference runs identified by their batch ID.
    Run optimisation wrt a specific batch ID
    """
    # Reference parameters of this batch
    reference_dir = resolve_path(f"{config['reference_dir']}_{batch_id}")

    # replay the simulation
    baseline_dir = resolve_path(f"{config['baseline_dir']}_{batch_id}")
    kine_file = join(reference_dir, "o2sim_Kine.root")
    steplogger_file = join(reference_dir, "MCStepLoggerOutput.root")
    cut_file_param = ";MaterialManagerParam.inputFile=cuts_o2.json"
    cmd = f'o2-sim-serial -n {config["events"]} -g extkinO2 --extKinFile {kine_file} -e MCReplay --skipModules ZDC ' \
          f'--configKeyValues="MCReplayParam.stepFilename={steplogger_file}{cut_file_param}"'
    #cmd = f'o2-sim-serial --seed 1 -n {config["events"]} -g extkinO2 --extKinFile {kine_file} --skipModules ZDC ' \
    #      f'--configKeyValues="SimCutParams.trackSeed=true{cut_file_param}"'
    _, sim_file = run_command(cmd, log_file="sim.log")

    # extract the hits using O2 macro and pipe to file
    extract_hits_root = abspath(join(O2_ROOT, "share", "macro", "analyzeHits.C"))
    cmd_extract_hits = f"root -l -b -q {extract_hits_root}"
    _, hit_file = run_command(cmd_extract_hits, log_file="hits.dat")

    # compute the loss and further metrics...
    baseline_hits_file = join(baseline_dir, "hits.dat")
    sim_file_ref = None
    #baseline_hits_file = join(reference_dir, "hits.dat")
    #sim_file_ref = join(reference_dir, "sim.log")
    return compute_metrics(hit_file, baseline_hits_file, sim_file, config["O2DETECTORS"], sim_file_ref)


@needs_cwd
@directions(['minimize', 'minimize'])
def objective_default(trial, config):
    """
    The central objective funtion for the optimisation
    """
    # Get some info from the reference dir
    reference_dir_any = resolve_path(f"{config['reference_dir']}_0")
    o2_medium_params_reference = parse_json(join(reference_dir_any, config["o2_medium_params_reference"]))
    parameters_to_optimise = config["parameters_to_optimise"]
    modules_to_optimise = config["modules_to_optimise"]

    ref_params_array, index_to_med_id, index_to_med_name, index_to_mod_name, module_medium_ids_map = flatten_o2_dict(o2_medium_params_reference, parameters_to_optimise, modules_to_optimise)
    params_array = np.copy(ref_params_array)

    # draw/aka "suggest" parameters
    for i, value in enumerate(ref_params_array):
        # the flattened lists are of n_media * n_parameters and every len(parameters_to_optimise) belongs to another medium.
        # So first find out via modulo, which cut parameter this is
        cut_parameter_name = parameters_to_optimise[i % len(parameters_to_optimise)]
        low = value
        up = config["search_value_up"]
        if low < 0:
            print(f"Lower value was {low}")
            low = config["search_value_low"]
        if low >= up:
            up = low * 10
        params_array[i] = trial.suggest_float(f"{i}", low, up, log=True)

    # put everything back together so that we can pass a new JSON to the MaterialManager via --confKeyValues MaterialManagerParam.inputFile=...
    space_drawn_o2 = make_o2_format(params_array, index_to_med_id, module_medium_ids_map, parameters_to_optimise, o2_medium_params_reference)
    # write that file for later usage during the replay
    param_file_path_o2 = "cuts_o2.json"
    dump_json(space_drawn_o2, param_file_path_o2)

    batches = config["batches"]
    # the batches to run over
    run_on_batches = config.get("use_all_batches", None)
    if run_on_batches is None:
        run_on_batches = list(range(batches))

    rel_steps_avg = 0
    rel_hits_avg = [None] * len(config["O2DETECTORS"])
    # count in how many batches a detector had hits so that we can build the correct average
    rel_hits_has_hits = [0] * len(config["O2DETECTORS"])
    this_batches = config["use_all_batches"]
    for i in run_on_batches:
        print(f"Using batch {i}")
        rel_steps, rel_hits = run_on_batch(i, config)
        rel_steps_avg += rel_steps
        for j, hits in enumerate(rel_hits):
            if hits is not None:
                if rel_hits_avg[j] is None:
                    rel_hits_avg[j] = 0
                rel_hits_avg[j] += hits
                rel_hits_has_hits[j] += 1
    for j, (hits, has_hits) in enumerate(zip(rel_hits_avg, rel_hits_has_hits)):
        if has_hits:
            rel_hits_avg[j] = hits / has_hits
    rel_steps_avg /= len(run_on_batches)

    # ...and annotate drawn space and metrics to trial so we can re-use it
    annotate_trial(trial, "rel_steps", rel_steps_avg)
    annotate_trial(trial, "rel_hits", rel_hits_avg)
    annotate_trial(trial, "index_to_medium_id", index_to_med_id)
    annotate_trial(trial, "index_to_medium_name", index_to_med_name)
    annotate_trial(trial, "index_to_module_name", index_to_mod_name)

    # remove all the artifacts we don't need to keep space
    #remove_dir(cwd, keep=["hits.dat", "cuts_o2.json", "sim.log"])

    #return compute_loss(rel_hits_avg, rel_steps_avg, config["rel_hits_cutoff"], config["penalty_below"], ref_params_array, params_array)
    steps_loss, hits_loss = compute_loss(rel_hits_avg, rel_steps_avg, config["rel_hits_cutoff"], config["penalty_below"])
    return steps_loss, hits_loss

