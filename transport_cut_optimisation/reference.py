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

    seed = config['seed']
    gen_config_file = "genConfig.cfg"
    with open(gen_config_file, "w") as f:
        f.write(f"Random:seed {seed}\n")
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
          f'o2-sim-serial -n {events} -g {generator} -e {engine} --seed {seed} ' \
          f'--skipModules ZDC --configKeyValues "MaterialManagerParam.outputFile={o2_medium_params_reference};GeneratorPythia8.config={gen_config_file}"'
    #cmd = f'o2-sim-serial -n {events} -g {generator} -e {engine} ' \
    #      f'--skipModules ZDC --configKeyValues "MaterialManagerParam.outputFile={o2_medium_params_reference};GeneratorPythia8.config={gen_config_file}"'
    run_command(cmd, log_file=config["o2_sim_log"])

    return True


def run_baseline(config):
    reference_dir = resolve_path(config["reference_dir"])
    kine_file = join(reference_dir, "o2sim_Kine.root")
    steplogger_file = join(reference_dir, "MCStepLoggerOutput.root")

    cmd = f'o2-sim-serial -n {config["events"]} -g extkinO2 --extKinFile {kine_file} -e MCReplay --skipModules ZDC ' \
          f'--configKeyValues="MCReplayParam.stepFilename={steplogger_file}"'
          #f'--configKeyValues="MCReplayParam.stepFilename={steplogger_file};MCReplayParam.blockProcessesCuts=true"'
    run_command(cmd, log_file="sim.log")
    return True

