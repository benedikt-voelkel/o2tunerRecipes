# Cut tuning for O2 (target GEANT4)

This directory contains scripts and the example configuration to run the transport cut tuning optimisation.

## Brief overview

The entire optimisation includes the creation of the reference data, followed by the optimisation which is followed by an evaluation stage. To run everything, please follow this README.


## Generate the configuration file

The `config.yaml` needs to be generated first. The reason for this is that the reference stage is quite heavy given that steps of all generated events are written out. Therefore, a configuration file is generated with [`generate_config.py`](generate_config.py) as follows
```bash
generate_config.py <n_batches> -d <path/to/recipe_dir>
```

The number of batches means that the reference is repeated that many times so that there are `n_batches x events` available in the end. The number of events can still be tweaked in the generated `path/to/recipe_dir/config.yaml`.

## Run

Done with
```bash
o2tuner -c <path/to/recipe_dir/config.yaml> -w <workdir>
```

