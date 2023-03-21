"""
A skeleton for optimisations
"""

from o2tuner.utils import annotate_trial
from o2tuner.optimise import needs_cwd

# With the @needs_cwd decorator, each trial is executed in its own working directory.
# This can be useful in case the objective produces artifacts. Uncomment, if needed
#@needs_cwd
def objective(trial, config):
    """
    The central objective funtion for the optimisation
    """

    value1 = config["parameter1"]
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -20, 20)

    # annotate the trial with some further computations, if desired. These annotations can be read later
    annotate_trial(trial, "sum", x + y)
    annotate_trial(trial, "parameter1", value1)

    return (x - 7.5)**2 + (y + 10)**2


# Another objective, might have a different search range
def objective_larger_range(trial, config):
    """
    The central objective funtion for the optimisation
    """

    value1 = config["parameter1"]
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -20, 20)

    # annotate the trial with some further computations, if desired. These annotations can be read later
    annotate_trial(trial, "sum", x + y)
    annotate_trial(trial, "parameter1", value1)

    return (x - 7.5)**2 + (y + 10)**2
