"""This module contains utilities for building and executing unit tests, as well as for reporting
the results

Related
-------
:mod:`hyperparameter_hunter.tests`
    This module contains all unit tests for the library, and it uses
    :mod:`hyperparameter_hunter.utils.test_utils` throughout"""
##################################################
# Import Miscellaneous Assets
##################################################
from pathlib import Path


def has_experiment_result_file(results_dir, experiment_id, result_type=None):
    """Check if the specified result files exist in `results_dir` for Experiment `experiment_id`

    Parameters
    ----------
    results_dir: String
        HyperparameterHunterAssets directory in which to search for experiment result files
    experiment_id: String or `experiments.BaseExperiment` descendant instance
        The ID of the experiment whose result files should be searched for in `results_dir`. If not
        string, should be an instance of a descendant of :class:`experiments.BaseExperiment` that
        has :attr:`experiment_id`
    result_type: List, string, or None, default=None
        Result file types for which to check. Valid values include any subdirectory name that can be
        included in "HyperparameterHunterAssets/Experiments" by default: ["Descriptions",
        "Heartbeats", "PredictionsOOF", "PredictionsHoldout", "PredictionsTest", "ScriptBackups"].
        If string, should be one of the aforementioned strings, or "ALL" to use all of the results.
        If list, should be a subset of the aforementioned list of valid values. Else, default is
        ["Descriptions", "Heartbeats", "PredictionsOOF", "ScriptBackups"]. The returned boolean
        signifies whether ALL of the `result_type` files were found, not whether ANY of were found

    Returns
    -------
    Boolean
        True if all result files specified by `result_type` exist in `results_dir` for the
        Experiment specified by `experiment_id`. Else, False"""
    experiment_id = experiment_id if isinstance(experiment_id, str) else experiment_id.experiment_id

    #################### Format `result_type` ####################
    if not result_type:
        result_type = ["Descriptions", "Heartbeats", "PredictionsOOF", "ScriptBackups"]
    elif result_type == "ALL":
        result_type = [
            "Descriptions",
            "Heartbeats",
            "PredictionsOOF",
            "PredictionsHoldout",
            "PredictionsTest",
            "ScriptBackups",
        ]
    if isinstance(result_type, str):
        result_type = [result_type]

    for subdir in result_type:
        #################### Select Result File Suffix ####################
        if subdir == "Descriptions":
            suffix = ".json"
        elif subdir == "Heartbeats":
            suffix = ".log"
        elif subdir == "ScriptBackups":
            suffix = ".py"
        elif subdir.startswith("Predictions"):
            suffix = ".csv"
        else:
            raise ValueError(f"Cannot resolve suffix for subdir `result_type`: {subdir}")

        #################### Check "Experiments" Directory ####################
        if results_dir.endswith("HyperparameterHunterAssets"):
            experiments_dir = Path(results_dir) / "Experiments"
        else:
            experiments_dir = Path(results_dir) / "HyperparameterHunterAssets" / "Experiments"

        if not (experiments_dir / subdir / f"{experiment_id}{suffix}").exists():
            return False

    return True
