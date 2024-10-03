# Cut tuning for O2 (target GEANT4)

This targets the energy cut optimisation of GEANT4(VMC).

Each medium can be associated with energy cuts specific to various different particles, e.g. photons, electrons/positrons, hadrons etc.
The aim is to transport as little particles as possible, hence to increase energy cuts as much as possible while keeping the physics accuracy high.

This collection of python and `yaml` files serves as the input or recipe for the o2tuner tool.

## On the shoulders of

The application of this recipe stands on the shoulders of the `MCStepLogger` and the `MCReplayEngine` (see their [repository](https://github.com/AliceO2Group/VMCStepLogger)).
Also, it needs `o2tuner` installed, see the [o2tuner documentation](https://github.com/AliceO2Group/o2tuner/blob/master/README.md#build)

## Step-by-step guide

### Generate the initial config
A suitable configuration must be created first. This is done with [`generate_config.py`](generate_config.py) as follows
```bash
</your/path/to/o2tunerRecipes/transport_cut_optimisation>/generate_config.py -b <n_batches> -i </your/path/to/o2tunerRecipes/transport_cut_optimisation>/config.yaml.in
```
Since the produced step files of the MCStepLogger are usually large, one can choose to produce multiple batches `<n_batches>`.
It is this batch configuration that makes the generation necessary.
The created configuration is written to `</your/path/to/o2tunerRecipes/transport_cut_optimisation>/config.yaml`. Once created, it is possible to add some fine tuning.
For more information on the configuration, please check out the [o2tuner documentation](https://github.com/AliceO2Group/o2tuner/blob/master/README.md#configuration)

### Passive modules to optimise
The focus lies on the cut parameters of passive modules, such as the beam pipe (`PIPE`), the absorber (`ABSO`) and others.
The full list of passive modules is defined in the global `config` dictionary under the key `O2PASSIVE`. This is a static list so it should not be changed. To *choose* certain modules, add them to the list under the key `modules_to_optimise`.

### Parameters to optimise
A full list of cut parameters is defined in the global `config` dictionary `REPLAY_CUT_PARAMETERS`. This is a static list so it should not be changed. To choose *certain* parameters to be taken into account during the optimisation, add them to the list `parameters_to_optimise`.
**NOTE** that currently only `CUTGAM` and `CUTELE` are treated in the `MCReplayEngine`. The reason is the subtle treatment of the parameters by the original GEANT4_VMC engine implementation.

### Other parameters in `config.yaml.in`
Here are some other parameters that can be fine-tuned in the `config.yaml` after its generation:
```yaml
engine: TGeant4                                             # engine to be used (for which to optimise); this should not be changed
generator: pythia8pp                                        # generator to be used
events: 50                                                  # number of event to be simulated per batch
seed: 624                                                   # seed passed to o2-sim-serial
o2_medium_params_reference: o2_medium_params_reference.json # the name of the file where the O2 reference parameters will be found
o2_sim_log: sim.log                                         # the file name to be used to pipe the output of each simulation into
passive_medium_ids_map: passive_medium_ids_map.yaml         # YAML to serialise the mapping of medium IDs to each passive module
detector_medium_ids_map: detector_medium_ids_map.yaml       # YAML to serialise the mapping of medium IDs to each detector
reference_params: reference_params.yaml                     # YAML to serialise the numpy array of parameters into
index_to_med_id: index_to_med_id.yaml                       # YAML to serialise and map each medium ID to a global ID
opt_params: current_params.yaml                             # current params just as a list in YAML
rel_hits_cutoff: 0.95                                       # desired minimum drop of relative ratio of hits
search_value_low: 0.00001                                   # the lower bound of values to be drawn for each parameter
search_value_up: 1.                                         # the upper bound of values to be drawn for each parameter
penalty_below: 2                                            # apply an additional penalty factor when the hits are below, this is smooth
opt_values: opt_values.yaml                                 # simply the list of suggested values of a trial
hits_log_file: &hits_file hits.dat                          # common name of file to pipe hit evaluation into (when using O2's analyzeHits.C macro)
reference_dir: &reference_dir reference                     # prefix for the reference directories (batch number added per batch), directory where the very original simulation is done and steps are collected
baseline_dir: &baseline_dir baseline                        # prefix for the baseline directory (batch number added per batch), directory where the original simulation is replayed once more so that RNG effects are taken care of
optimisation_dir: &opt_name optimisation                    # the optimisation directory and name
optimise_on_batches:                                        # list of batch numbers to be used for optimisation; if not set or None, use all batches
  - 0
```
Some of the keys have anchors and are used in other places (see [`config.yaml.in`](config.yaml.in)).

### The optimisation stage
For information on how an optimisation stage is defined in general, please check out the [o2tuner documentation](https://github.com/AliceO2Group/o2tuner/blob/master/README.md#define-optimisation-stages).
At this point, we are more interested in some specific implementations in the [optimisation python code](optimise.py).

The objective is defined in `objective_default`. First of all, some paths and names are extracted to know where the reference data has been placed, which passive modules and cut parameters should be taken into account and so on.
After that, the values of the parameters are drawn for a given trial. These parameter values are written to a `JSON` file in the trial's directory in the format that is understood by the O2 simulation.

Originally, a certain number of batches has been created for the reference and baseline stages. For the optimisation, it is possible to choose the number of batches that should be used. That can be useful to do faster tests by setting that to only one batch. For the final optimisation run, all batches can then be chosen. To change this, set the `optimise_on_batches` in the `config.yaml`. It should either be `None` (or not defined) or a list of integers that refer to the batches (`0...N-1`). In the latter case, exactly those batches will be used while otherwise, all batches will be used which implies higher statistics for the optimisation run. At the same time, the optimisation will take longer.

### Run everything
To run, change into a suitable working directory and execute
```bash
o2tuner -w <run_dir> -c </your/path/to/o2tunerRecipes/transport_cut_optimisation>/config.yaml -s optimisation
```

This runs everything up to and including the `optimisation` stage that is defined in the `config.yaml`. All artifacts are dumped in directories under `<run_dir>`.
The `sqlite` database file is located under `<run__dir>/optimisation/opt.db`

#### Rerun or try another optimisation
If you want to test different optimisations (better ideas for metrics calculations, try out multi-objective etc.), change the value of the `cwd` key under the `optimisation` stage and run
```bash
o2tuner -w <run_dir> -c </your/path/to/o2tunerRecipes/transport_cut_optimisation>/config.yaml -s optimisation
```
again which will now run in a new directory `<run_dir>/<new_optimisation>`.

#### Resume the optimisation run
You want to add more trials to a previously finished optimisation? No problem! Simply specify in the `config.yaml` how many `trials` should be run and without changing anything else, run the same command-line as before.
The optimisation will be initiated from the stage in the previously created `<run_dir>/optimisation/opt.db` and from there the optimisation will go on.
Please see also the [o2tuner documentation](https://github.com/AliceO2Group/o2tuner/blob/master/README.md#abortcontinue-an-optimisation).

#### Evaluation stages
Once an optimisation is finished, there are various other stages that can be done to evaluate and get some insight.
* `evaluate`, among other things:
    1. history of steps and hits per trial
    1. importance of parameters
    1. correlation between parameters
    1. correlation between parameters and relative number of hits
* `evaluate_print`: print trial numbers and their losses to the terminal
* `evaluate_params`: ratio of parameter values before and after optimisation
* `rz_params`: `r-z` plot of the optimised modules showing a heatmap indicating the size of the cut parameter values
* `step_analysis`: step analysis of reference and optimisation run creating overlay plot of steps per module.
