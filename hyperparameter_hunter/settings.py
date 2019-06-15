"""This module is the doorway for other modules to access the information set by the active
:class:`hyperparameter_hunter.environment.Environment`, and to access the appropriate logging
methods. Specifically, other modules will most often use :class:`hyperparameter_hunter.settings.G`
to access the aforementioned information. Additionally, this module defines several variables to
assist in navigating the 'HyperparameterHunterAssets' directory structure

Related
-------
:mod:`hyperparameter_hunter.environment`
    This module sets :attr:`hyperparameter_hunter.settings.G.Env` to itself, creating the primary
    gateway used by other modules to access the active Environment's information"""
##################################################
# Import Miscellaneous Assets
##################################################
import os.path
import warnings


##################################################
# Result File Paths
##################################################
ASSETS_DIRNAME = "HyperparameterHunterAssets"

ASSETS_EXPERIMENTS_DIRNAME = "Experiments"
ASSETS_TESTED_KEYS_DIRNAME = "TestedKeys"
ASSETS_KEY_ATTRIBUTE_LOOKUP_DIRNAME = "KeyAttributeLookup"
ASSETS_LEADERBOARDS_DIRNAME = "Leaderboards"

RESULT_FILE_SUB_DIR_PATHS = {
    #################### Experiments ####################
    "checkpoint": "{}/Checkpoints".format(ASSETS_EXPERIMENTS_DIRNAME),
    "description": "{}/Descriptions".format(ASSETS_EXPERIMENTS_DIRNAME),
    "heartbeat": "{}/Heartbeats".format(ASSETS_EXPERIMENTS_DIRNAME),
    "predictions_holdout": "{}/PredictionsHoldout".format(ASSETS_EXPERIMENTS_DIRNAME),
    "predictions_in_fold": "{}/PredictionsInFold".format(ASSETS_EXPERIMENTS_DIRNAME),
    "predictions_oof": "{}/PredictionsOOF".format(ASSETS_EXPERIMENTS_DIRNAME),
    "predictions_test": "{}/PredictionsTest".format(ASSETS_EXPERIMENTS_DIRNAME),
    "script_backup": "{}/ScriptBackups".format(ASSETS_EXPERIMENTS_DIRNAME),
    #################### Tested Keys ####################
    "tested_keys": "{}".format(ASSETS_TESTED_KEYS_DIRNAME),
    #################### Key Attribute Lookup ####################
    "key_attribute_lookup": "{}".format(ASSETS_KEY_ATTRIBUTE_LOOKUP_DIRNAME),
    #################### Leaderboards ####################
    "leaderboards": "{}".format(ASSETS_LEADERBOARDS_DIRNAME),
    "global_leaderboard": "{}/GlobalLeaderboard.csv".format(ASSETS_LEADERBOARDS_DIRNAME),
    #################### Other ####################
    "current_heartbeat": "Heartbeat.log",
    # 'analytics': '{}'.format(),
    # 'ensembles': '{}'.format(),
    # 'optimization_rounds': '{}'.format(),
}

##################################################
# Temporary File Paths
##################################################
TEMP_MODULES_DIR_NAME = "__temp_files"
TEMP_MODULES_DOT_PATH = f"hyperparameter_hunter.library_helpers.{TEMP_MODULES_DIR_NAME}"
TEMP_MODULES_DIR_PATH = f"{os.path.split(__file__)[0]}/library_helpers/{TEMP_MODULES_DIR_NAME}"


##################################################
# Global Settings
##################################################
class G(object):
    """This class defines global attributes that are set upon instantiation of
    :class:`environment.Environment`. All attributes contained herein are class variables (not
    instance variables) because the expectation is for the attributes of this class to be set only
    once, then referenced by operations that may be executed after instantiating a
    :class:`environment.Environment`. This allows functions to be called or classes to be initiated
    without passing a reference to the currently active Environment, because they check the
    attributes of this class, instead

    Attributes
    ----------
    Env: None
        This is set to "self" in :meth:`environment.Environment.__init__`. This fact allows other
        modules to check if :attr:`settings.G.Env` is None. If None, a
        :class:`environment.Environment` has not yet been instantiated. If not None, any attributes
        or methods of the instantiated Env may be called
    save_transformed_predictions: False
        Declares format in which a model's predictions should be saved, with regard to
        :class:`feature_engineering.FeatureEngineer` transformations. If no transformation of the
        target variable takes place (either through :class:`feature_engineering.FeatureEngineer`,
        :class:`feature_engineering.EngineerStep`, or otherwise), then this setting can be ignored.

        If `save_transformed_predictions` is True, and target transformation does occur, then
        experiment predictions are saved in the same form as the transformed target, which is the
        form returned directly by a fitted model's `predict` method. For example, if target data is
        label-encoded, and an :class:`feature_engineering.EngineerStep` is used to one-hot encode
        the target, then one-hot-encoded predictions will be saved.

        Conversely, if `save_transformed_predictions` is False (default), and target transformation
        does occur, then experiment predictions are saved in the inverted form of the transformed
        target, which is the same form as the original target data. Continuing the example of
        label-encoded target data, and an :class:`feature_engineering.EngineerStep` to one-hot
        encode the target, in this case, label-encoded predictions will be saved.
    save_transformed_metrics: True
        Declares manner in which a model's predictions should be evaluated through the provided
        :attr:`environment.Environment.metrics`, with regard to
        :class:`feature_engineering.FeatureEngineer` transformations. If no transformation of the
        target variable takes place (either through :class:`feature_engineering.FeatureEngineer`,
        :class:`feature_engineering.EngineerStep`, or otherwise), then this setting can be ignored.

        A more descriptive name for this may be "calculate_metrics_using_transformed_predictions",
        but that's a bit verbose, even by my standards.

        If `save_transformed_metrics` is True (default), and target transformation does occur, then
        experiment metrics are calculated using the transformed targets and predictions, which is
        the form returned directly by a fitted model's `predict` method. For example, if target data
        is label-encoded, and an :class:`feature_engineering.EngineerStep` is used to one-hot encode
        the target, then metrics functions will receive the following as input:
        (one-hot-encoded targets, and one-hot-encoded predictions).

        Conversely, if `save_transformed_metrics` is False, and target transformation does occur,
        then experiment metrics are calculated using the inverse of the transformed targets and
        predictions, which is same form as the original target data. Continuing the example of
        label-encoded target data, and an :class:`feature_engineering.EngineerStep` to one-hot
        encode the target, in this case, metrics functions will receive the following as input:
        (label-encoded targets, and label-encoded predictions).
    priority_callbacks: Tuple
        Intended for internal use only. The contents of this tuple are inserted at the front of an
        Experiment's list of callback bases via :class:`experiment_core.ExperimentMeta`, ahead of
        even the Experiment's original base classes. This is used primarily for testing callbacks,
        but it can also be used if you absolutely need a callback to be placed before the
        Experiment's other ancestors in its MRO
    log_: print
        ...
    debug_: print
        ...
    warn_: print
        ...
    import_hooks: List
        ...
    sentinel_registry: List
        ...
    mirror_registry: List
        ...
    """

    Env = None

    #################### Miscellaneous Settings ####################
    save_transformed_predictions = False
    save_transformed_metrics = True

    #################### Internal Settings ####################
    priority_callbacks = tuple()

    #################### Standard Logging Set by :class:`environment.Environment` ####################
    @staticmethod
    def log(content, *args, **kwargs):
        """Set in :meth:`environment.Environment.initialize_reporting` to the updated version of
        :meth:`reporting.ReportingHandler.log`"""
        print(content, *args, **kwargs)

    @staticmethod
    def debug(content, *args, **kwargs):
        """Set in :meth:`environment.Environment.initialize_reporting` to the updated version of
        :meth:`reporting.ReportingHandler.debug`"""
        print(content, *args, **kwargs)

    @staticmethod
    def warn(content, *args, **kwargs):
        """Set in :meth:`environment.Environment.initialize_reporting` to the updated version of
        :meth:`reporting.ReportingHandler.warn`"""
        warnings.warn(content, *args, **kwargs)

    #################### Optimization Logging Set by :class:`optimization_core.BaseOptimizationProtocol` ####################
    log_ = print
    debug_ = print
    warn_ = warnings.warn

    import_hooks = []
    sentinel_registry = []
    mirror_registry = []

    @classmethod
    def reset_attributes(cls):
        """Return the attributes of :class:`settings.G` to their original values"""
        cls.Env = None
        cls.log = print
        cls.debug = print
        cls.warn = warnings.warn
