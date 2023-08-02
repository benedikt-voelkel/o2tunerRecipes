"""
The cut tuning reference/basline run
"""

from os.path import join, abspath
from os import environ
from platform import system as os_system

from o2tuner.system import run_command
from o2tuner.config import resolve_path

O2_ROOT = environ.get("O2_ROOT")
MCSTEPLOGGER_ROOT = environ.get("MCSTEPLOGGER_ROOT")

def reference(inspectors, config):
    """
    Called after arguments have been parsed
    """
    events = config["events"]
    generator = config["generator"]
    engine = config["engine"]

    lib_extension = ".dylib" if os_system() == "Darwin" else ".so"
    preload = "DYLD_INSERT_LIBRARIES" if os_system() == "Darwin" else "LD_PRELOAD"

    cmd = f'MCSTEPLOG_NO_MAGFIELD=1 MCSTEPLOG_TTREE=1 {preload}={MCSTEPLOGGER_ROOT}/lib/libMCStepLoggerInterceptSteps{lib_extension} ' \
          f'o2-sim-serial -n {events} -g {generator} -e {engine} '
    run_command(cmd, log_file=config["o2_sim_log"])
    return True


def baseline(inspectors, config):
    reference_dir = resolve_path(config["reference_dir"])
    kine_file = join(reference_dir, "o2sim_Kine.root")
    steplogger_file = join(reference_dir, "MCStepLoggerOutput.root")

    cmd = f'o2-sim-serial -n {config["events"]} -g extkinO2 --extKinFile {kine_file} -e MCReplay --configKeyValues="MCReplayParam.allowStopTrack=true;MCReplayParam.stepFilename={steplogger_file}"'
    run_command(cmd, log_file=config["o2_sim_log"])
    extract_hits_root = abspath(join(O2_ROOT, "share", "macro", "analyzeHits.C"))
    cmd_extract_hits = f"root -l -b -q {extract_hits_root}"
    run_command(cmd_extract_hits, log_file="hits.dat")
    return True

def baseline_hits(inspectors, config):
    extract_hits_root = abspath(join(O2_ROOT, "share", "macro", "analyzeHits.C"))
    cmd_extract_hits = f"root -l -b -q {extract_hits_root}"
    run_command(cmd_extract_hits, log_file="hits.dat")
    return True
