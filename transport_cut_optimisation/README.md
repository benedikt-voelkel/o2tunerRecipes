# Cut tuning for O2 (target GEANT4)

This directory contains scripts and the configuration to run the transport cut tuning optimisation.

## Brief overview

The entire optimisation includes the creation of the reference data, followed by the optimisation which is followed by an evaluation stage. To run everything, please follow the steps in the following.

## Generate the configuration file

The configuration needs to be generated first. The reason for this is that the reference stage and overall configuration is quite complex. But ince created, everything works out of the box. The configuration file is generated with [`generate_config.py`](generate_config.py) as follows
```bash
generate_config.py -b <n_batches> -d <your/path/to/o2tunerRecipes/transport_cut_optimisation>
```

This configures the executed scripts at a later stage to create `<n_batches>` reference runs with different generators seeds. The number of events can still be tweaked in the generated `/your/path/to/o2tunerRecipes/transport_cut_optimisation/config.yaml`.

## Run

You will see that there are multiple stages defined in the configuration file. As seen in the following, the stage to be run can explicitly be specified. If run without arguments, all steps will be done. Note, that steps will be executed according to their dependencies.

Done with
```bash
o2tuner -c </your/path/to/o2tunerRecipes/transport_cut_optimisation/config.yaml> -w <workdir> [-s <stage_to_be_run>] 
```

The python scripts that are loctaed next to the configuration file will be found automatically.
