import sys
from os.path import exists, join, dirname
import argparse
from copy import deepcopy, copy
from o2tuner.io import dump_yaml, parse_yaml

def generate_config(config_in, n_batches, recipe_dir=None):
    """
    Generate the config for a certain number of batches
    """

    if not exists(config_in):
        print(f"ERROR: Cannot find {config_in}")
        return 1

    if n_batches <= 0:
        print(f"ERROR: Number of batches must be at least 1, however, found {n_batches}")
        return 1

    if not recipe_dir:
        recipe_dir = dirname(config_in)

    config_in = parse_yaml(config_in)

    # the global static configuration
    config_global = config_in["config"]
    # immediately set the number of batches
    config_global["batches"] = n_batches

    # python settings for the initial reference run
    stage_reference_python = {"python": {"file": "reference.py",
                                     "entrypoint": "run_reference"}}
    # python settings for the baseline run
    # remeber: the baseline runs are needed to ensure that we compare the same number of hits at all times
    stage_baseline_in = {"python": {"file": "reference.py",
                                    "entrypoint": "run_baseline"}}
    # cmd and redirecting the output to a certain filename so we know where it will go
    stage_baseline_hits_in = {"cmd": "root -l -b -q ${O2_ROOT}/share/macro/analyzeHits.C",
                              "log_file": config_global["hits_log_file"]}
    # collect the dependencies for the optimisation
    opt_deps = []

    # all stages defined by the user
    user_stages = config_in.get("stages_user", {})

    for i in range(n_batches):
        # add the batch suffix
        ref_name = f"reference_sim_{i}"
        # copy the python settings
        stage_reference = deepcopy(stage_reference_python)
        # specify the output directory depending on the batch
        ref_dir = f"{config_global['reference_dir']}_{i}"
        # now set the directory
        stage_reference["cwd"] = ref_dir
        # set a specific seed depending on the batch to ensure reproducibility on the one hand and to use different seeds in every batch
        stage_reference["config"] = {"seed": i + 1}

        # similar steps as for the reference run
        baseline_name = f"baseline_sim_{i}"
        baseline_dir = f"{config_global['baseline_dir']}_{i}"
        stage_baseline = copy(stage_baseline_in)
        stage_baseline["config"] = {"reference_dir": ref_dir}
        stage_baseline["cwd"] = baseline_dir
        # obviously, this depends on the reference stage
        stage_baseline["deps"] = [ref_name]

        # similar steps as for the previous stages
        stage_hits = copy(stage_baseline_hits_in)
        stage_hits["cwd"] = baseline_dir
        # obviously, this depends on the baseline stage
        stage_hits["deps"] = [baseline_name]

        # finally, add all configrued stages for this batch
        user_stages[f"reference_sim_{i}"] = stage_reference
        user_stages[f"baseline_sim_{i}"] = stage_baseline
        user_stages[f"baseline_hits_{i}"] = stage_hits

        # for the optimisation, we need to take care and add each hit stage
        opt_deps.append(f"baseline_hits_{i}")

    # now add all user stages to the overall config
    config_in["stages_user"] = user_stages

    # for each defined optimisation, add the corresponding dependencies
    for name, stage in config_in["stages_optimisation"].items():
        config = stage.get("config", {})
        stage["config"] = config
        if not "deps" in stage:
            stage["deps"] = opt_deps

    # write and return
    dump_yaml(config_in, join(recipe_dir, "config.yaml"))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batches", default=1, type=int, help='how many simulation batches to produce')
    parser.add_argument("-i", "--in", help="full path to the in file. Usually called config.yaml.ini", required=True)
    parser.add_argument('-r', '--recipe-dir', dest='recipe_dir', help='where to find the entire recipe aka all python files')
    args = parser.parse_args()
    sys.exit(generate_config(args.in, args.batches, args.recipe_dir))

