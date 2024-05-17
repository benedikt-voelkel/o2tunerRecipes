import sys
from os.path import exists, join
import argparse
from copy import deepcopy, copy
from o2tuner.io import dump_yaml, parse_yaml

def generate_config(recipe_dir, n_batches):
    """
    Generate the config for a certain number of batches
    """

    config_in = join(recipe_dir, "config.yaml.in")
    if not exists(config_in):
        print(f"ERROR: Cannot find {config_in}")
        return 1

    config_in = parse_yaml(config_in)

    config_default = config_in["global_config"]
    config_default["batches"] = n_batches

    stage_reference_in = {"python": {"file": "reference.py",
                                     "entrypoint": "run_reference"}}
    stage_baseline_in = {"python": {"file": "reference.py",
                                    "entrypoint": "run_baseline"}}
    stage_baseline_hits_in = {"cmd": "root -l -b -q ${O2_ROOT}/share/macro/analyzeHits.C",
                              "log_file": config_default["hits_log_file"]}

    opt_deps = []

    user_stages = config_in.get("stages_user", {})
    for i in range(n_batches):
        ref_name = f"reference_sim_{i}"
        stage_reference = deepcopy(stage_reference_in)
        ref_dir = f"{config_default['reference_dir']}_{i}"
        stage_reference["cwd"] = ref_dir
        stage_reference["config"]["seed"] = i + 1

        baseline_name = f"baseline_sim_{i}"
        baseline_dir = f"{config_default['baseline_dir']}_{i}"
        stage_baseline = copy(stage_baseline_in)
        stage_baseline["config"] = {"reference_dir": ref_dir, "events": config_default["events"]}
        stage_baseline["cwd"] = baseline_dir
        stage_baseline["deps"] = [ref_name]

        stage_hits = copy(stage_baseline_hits_in)
        stage_hits["cwd"] = baseline_dir
        stage_hits["deps"] = [baseline_name]

        user_stages[f"reference_sim_{i}"] = stage_reference
        user_stages[f"baseline_sim_{i}"] = stage_baseline
        user_stages[f"baseline_hits_{i}"] = stage_hits

        opt_deps.append(f"baseline_hits_{i}")

    config_in["stages_user"] = user_stages

    for name, stage in config_in["stages_optimisation"].items():
        config = stage.get("config", {})
        config["batches"] = n_batches
        stage["config"] = config
        if not "deps" in stage:
            stage["deps"] = opt_deps

    dump_yaml(config_in, join(recipe_dir, "config.yaml"))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batches", default=1, type=int)
    parser.add_argument("-d", "--dir", default="./", help="recipe directory")
    args = parser.parse_args()
    sys.exit(generate_config(args.dir, args.batches))

