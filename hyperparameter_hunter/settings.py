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
import warnings


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
    # 'analytics': '{}'.format(),
    # 'ensembles': '{}'.format(),
    # 'optimization_rounds': '{}'.format(),
}


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
    log: print
        This is set in :meth:`environment.Environment.initialize_reporting` to the updated version
        of :meth:`reporting.ReportingHandler.log`
    debug: print
        This is set in :meth:`environment.Environment.initialize_reporting` to the updated version
        of :meth:`reporting.ReportingHandler.debug`
    warn: warnings.warn
        This is set in :meth:`environment.Environment.initialize_reporting` to the updated version
        of :meth:`reporting.ReportingHandler.warn`
    log_: print
        ... # TODO: ...
    debug_: print
        ... # TODO: ...
    warn_: print
        ... # TODO: ...
    import_hooks: List
        ... # TODO: ...
    sentinel_registry: List
        ... # TODO: ...
    """

    Env = None

    #################### Standard Logging Set by :class:`environment.Environment` ####################
    log = print
    debug = print
    warn = warnings.warn

    #################### Optimization Logging Set by :class:`optimization_core.BaseOptimizationProtocol` ####################
    log_ = print
    debug_ = print
    warn_ = warnings.warn

    import_hooks = []
    sentinel_registry = []

    @classmethod
    def reset_attributes(cls):
        """Return the attributes of :class:`settings.G` to their original values"""
        cls.Env = None
        cls.log = print
        cls.debug = print
        cls.warn = warnings.warn
