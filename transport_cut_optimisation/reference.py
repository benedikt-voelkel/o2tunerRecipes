"""
The cut tuning reference/basline run
"""

import sys
from os.path import join
from os import environ
from platform import system as os_system

from o2tuner.system import run_command
from o2tuner.io import parse_json, dump_yaml
from o2tuner.config import resolve_path


MCSTEPLOGGER_ROOT = environ.get("MCSTEPLOGGER_ROOT")

def run_reference(config):
    """
    Called after arguments have been parsed
    """

    gen_config_file = "genConfig.cfg"
    with open(gen_config_file, "w") as f:
        f.write(f"Random:seed {config['seed']}\n")
        f.write("Random:setSeed on\n")
    events = config["events"]
    generator = config["generator"]
    engine = config["engine"]
    o2_medium_params_reference = config["o2_medium_params_reference"]
    o2_passive = config["O2PASSIVE"]
    o2_detectors = config["O2DETECTORS"]
    replay_cut_parameters = config["REPLAY_CUT_PARAMETERS"]

    lib_extension = ".dylib" if os_system() == "Darwin" else ".so"
    preload = "DYLD_INSERT_LIBRARIES" if os_system() == "Darwin" else "LD_PRELOAD"

    cmd = f'MCSTEPLOG_TTREE=1 {preload}={MCSTEPLOGGER_ROOT}/lib/libMCStepLoggerInterceptSteps{lib_extension} ' \
          f'o2-sim-serial -n {events} -g {generator} -e {engine} ' \
          f'--skipModules ZDC --configKeyValues "MaterialManagerParam.outputFile={o2_medium_params_reference};GeneratorPythia8.config={gen_config_file}"'
    run_command(cmd, log_file=config["o2_sim_log"])

    reference_params = []
    params = parse_json(o2_medium_params_reference)

    index_to_med_id = []
    passive_medium_ids_map = {}
    # pylint: disable=duplicate-code
    detector_medium_ids_map = {}
    default_cuts = params["default"]["cuts"]
    for module, batches in params.items():
        if module in ["default", "enableSpecialCuts", "enableSpecialProcesses"]:
            continue

        for batch in batches:
            med_id = batch["global_id"]
            # according to which modules are recognised by this script
            if module in o2_passive:
                if module not in passive_medium_ids_map:
                    passive_medium_ids_map[module] = []
                passive_medium_ids_map[module].append(med_id)
            elif module in o2_detectors:
                if module not in detector_medium_ids_map:
                    detector_medium_ids_map[module] = []
                detector_medium_ids_map[module].append(med_id)
            else:
                continue

            cuts_read = batch.get("cuts", {})
            cuts_append = []
            for rcp in replay_cut_parameters:
                value = cuts_read.get(rcp, -1)
                if value < 0:
                    value = default_cuts[rcp]
                cuts_append.append(value)

            index_to_med_id.append(med_id)
            reference_params.extend(cuts_append)

    dump_yaml(reference_params, config["reference_params"])
    dump_yaml(index_to_med_id, config["index_to_med_id"])
    dump_yaml(passive_medium_ids_map, config["passive_medium_ids_map"])
    dump_yaml(detector_medium_ids_map, config["detector_medium_ids_map"])
    return True


def run_baseline(config):
    reference_dir = resolve_path(config["reference_dir"])
    kine_file = join(reference_dir, "o2sim_Kine.root")
    steplogger_file = join(reference_dir, "MCStepLoggerOutput.root")

    cmd = f'o2-sim-serial -n {config["events"]} -g extkinO2 --extKinFile {kine_file} -e MCReplay --skipModules ZDC ' \
          f'--configKeyValues="MCReplayParam.stepFilename={steplogger_file}"'
    run_command(cmd, log_file="sim.log")
    return True

