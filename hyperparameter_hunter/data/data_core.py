"""This module defines mechanisms for managing an experiment's various datasets, and each datasets's
inputs, targets, and predictions.

**Important Contents**

In order to maintain the states of different datasets across all divisions of an experiment and
amid transformations that may be applied to the data via
:mod:`~hyperparameter_hunter.feature_engineering`, two main classes are defined herein:

1. :class:`BaseDataChunk`:

    * Logical separations between "columns" of data for a given :class:`BaseDataset`
    * Held and maintained by :class:`BaseDataset` and its descendants
    * Three primary descendants of :class:`BaseDataChunk`:

        1. :class:`InputChunk`: Maintains a dataset's input data (and transformations)
        2. :class:`TargetChunk`: Maintains a dataset's target data (and transformations)
        3. :class:`PredictionChunk`: Maintains a dataset's predictions (and transformations)

    * Descendants of :class:`BaseDataChunk` should implement the eight "on_<division>_<point>"
      callback methods defined by :class:`~hyperparameter_hunter.callbacks.bases.BaseCallback`

        * Because :class:`BaseDataChunk` subclasses are isolated from the experiment, these methods
          need not invoke their `super` methods, although they are allowed to if necessary

    * :class:`NullDataChunk` does nothing but mimic the normal :class:`BaseDataChunk` child structure

        * Used for :class:`BaseDataset` subclasses lacking a particular data chunk, such as:

            1) `TestDataset`'s `TargetChunk`, because the targets for a test dataset are unknown, or
            2) `TrainDataset`'s `PredictionChunk`, because predictions are not made on training data

2. :class:`BaseDataset`:

    # TODO: ...

**Dataset Attribute Syntax**

The intricate subclass network bolstering the module's predominant :class:`BaseDataset` subclasses
may be intimidating at first, but don't worry; there's a shortcut. Follow these steps to ensure
proper syntax and a valid result when accessing data from a
:class:`~hyperparameter_hunter.experiments.CVExperiment`:

1. {`data_train`, `data_oof`, `data_holdout`, `data_test`} - Dataset attribute
2. {`input`, `target`, `prediction`} - Data chunk
3. [`T`] - Optional transformation
4. {`d`, `run`, `fold`, `rep`, `final`} - Division, initial (`d`) or `final` data

By stacking three values (four if following optional step "3") from the above formula, you can
access all of the interesting stuff stored in the datasets from the comfort of your experiment or
:func:`~hyperparameter_hunter.callbacks.bases.lambda_callback`.

Related
-------
:mod:`hyperparameter_hunter.callbacks.bases`
    This module defines the core callback method structure mirrored by :class:`BaseDataCore`.
    Despite the strong logical connection to this module, it is important to remember that the only
    actual connection between the two modules is in :mod:`hyperparameter_hunter.callbacks.wranglers`
:mod:`hyperparameter_hunter.callbacks.wranglers`
    # TODO: ... Handlers for the `Dataset`s to invoke callback methods with required parameters
    This module defines the callback classes that act as handlers for the descendants of
    :class:`BaseDataset`
:mod:`hyperparameter_hunter.experiments`
    # TODO: ...
"""
##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd
from typing import List, Optional

##################################################
# Global Variables
##################################################
OptionalDF = Optional[pd.DataFrame]


##################################################
# Data Core (Both `BaseDataChunk` and `BaseDataset`)
##################################################
class BaseDataCore:
    """Basic building block class for both :class:`BaseDataChunk` and :class:`BaseDataset`

    Notes
    -----
    Defines core callback method scaffolding, characterized by the following default behaviors:
        * Eight core callback methods call :meth:`BaseDataCore._on_call_default`
        * :meth:`BaseDataCore._on_call_default` does nothing
        * :meth:`BaseDataCore._do_something` calls the appropriate core callback method"""

    #################### Primary Callback Methods ####################
    def on_experiment_start(self, *args, **kwargs):
        self._on_call_default("experiment", "start", *args, **kwargs)

    def on_repetition_start(self, *args, **kwargs):
        self._on_call_default("repetition", "start", *args, **kwargs)

    def on_fold_start(self, *args, **kwargs):
        self._on_call_default("fold", "start", *args, **kwargs)

    def on_run_start(self, *args, **kwargs):
        self._on_call_default("run", "start", *args, **kwargs)

    def on_run_end(self, *args, **kwargs):
        self._on_call_default("run", "end", *args, **kwargs)

    def on_fold_end(self, *args, **kwargs):
        self._on_call_default("fold", "end", *args, **kwargs)

    def on_repetition_end(self, *args, **kwargs):
        self._on_call_default("repetition", "end", *args, **kwargs)

    def on_experiment_end(self, *args, **kwargs):
        self._on_call_default("experiment", "end", *args, **kwargs)

    #################### Internal Methods ####################
    def _on_call_default(self, division: str, point: str, *args, **kwargs):
        """Basic/fallback logic for invocation of primary callback methods

        Parameters
        ----------
        division: {"experiment", "repetition", "fold", "run"}
            Time span division identifier of the primary callback method to be invoked
        point: {"start", "end"}
            Time span point identifier in `division` of the primary callback method to be invoked
        *args: List
            Positional arguments to provide on invocation of the specified primary callback method
        **kwargs: Dict
            Keyword arguments to provide on invocation of the specified primary callback method

        Notes
        -----
        This method's primary utility is providing a unified handler for any unmodified invocation
        of a primary callback method. This is possible because the default behavior of the primary
        callback methods defined herein is to simply call :meth:`BaseDataCore._on_call_default`

        This method should not be invoked anywhere other than in the primary callback methods
        defined in this :class:`BaseDataCore`. In general, it should only appear elsewhere if it
        is being overridden by a subclass of :class:`BaseDataCore`"""
        ...

    def _do_something(self, division: str, point: str, *args, **kwargs):
        """Invoke one of the defined primary callback methods according to `division` and `point`

        Parameters
        ----------
        division: {"experiment", "repetition", "fold", "run"}
            Time span division identifier of the primary callback method to be invoked
        point: {"start", "end"}
            Time span point identifier in `division` of the primary callback method to be invoked
        *args: List
            Positional arguments to provide on invocation of the specified primary callback method
        **kwargs: Dict
            Keyword arguments to provide on invocation of the specified primary callback method

        Notes
        -----
        This method is a simple mechanism for calling a defined primary callback method using the
        strings `division` and `point`, rather than explicitly naming the method"""
        getattr(self, f"on_{division}_{point}")(*args, **kwargs)


##################################################
# Data Chunks (Inputs, Targets, Predictions)
##################################################
class _BaseDataChunk(BaseDataCore):
    def __init__(self, d: OptionalDF):
        """Helper superclass for :class:`BaseDataChunk`

        Parameters
        ----------
        d: pd.DataFrame, or None
            Raw data representing the initial state of the data to be handled by this chunk"""
        self.d: OptionalDF = d
        self.run: OptionalDF = None
        self.fold: OptionalDF = None
        self.rep: OptionalDF = None
        self.final: OptionalDF = None

    def __eq__(self, other):
        for attr in ["d", "run", "fold", "rep", "final"]:
            if getattr(self, attr) is None:
                if getattr(other, attr) is not None:
                    return False
            elif not getattr(self, attr).equals(getattr(other, attr)):
                return False
        return True


class BaseDataChunk(_BaseDataChunk):
    def __init__(self, d: OptionalDF):
        """Create logical separations between "columns" of data for a :class:`BaseDataset`

        Parameters
        ----------
        d: pd.DataFrame, or None
            Raw data representing the initial state of the data to be handled by this chunk, and its
            transformed self (:attr:`BaseDataChunk.T`)

        Attributes
        ----------
        T: _BaseDataChunk
            Extra data chunk tracking transformations/inversions applied to :class:`_BaseDataChunk`
            attributes via :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`. If
            no feature engineering is performed, `T` can be ignored"""
        super().__init__(d=d)
        self.T: _BaseDataChunk = _BaseDataChunk(d=d)

    def __eq__(self, other):
        return super().__eq__(other) and self.T.__eq__(getattr(other, "T", object()))


class NullDataChunk(BaseDataChunk):
    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwargs):
        """I'm useless. I don't do anything - ever"""
        super().__init__(d=None)

    def _on_call_default(self, division, point, *args, **kwargs):
        """Reinforce uselessness by doing nothing"""
        return None


##################################################
# Datasets (Train, OOF/Validation, Holdout, Test)
##################################################
class BaseDataset(BaseDataCore):
    #################### Chunk Initializers ####################
    _input_type: type = NullDataChunk
    _target_type: type = NullDataChunk
    _prediction_type: type = NullDataChunk

    def __init__(
        self,
        data: OptionalDF = None,
        feature_selector: List[str] = None,
        target_column: List[str] = None,
        require_data: bool = False,
    ):
        """Base class for organizing entire datasets into three :class:`BaseDataChunk` subclasses

        Parameters
        ----------
        data: pd.DataFrame, or None, default=None
            Initial whole dataset, comprising both input and target data
        feature_selector: List, or None, default=None
            Column names to include as input data for the dataset
        target_column: List, or None, default=None
            Column name(s) in the dataset that contain the target output data
        require_data: Boolean, default=False
            If True, `data` must be provided as a pandas DataFrame

        Notes
        -----
        Subclasses of `BaseDataset` should override the three chunk initializer attributes
        (`_input_type`, `_target_type`, `_prediction_type`) to a :class:`BaseDataChunk` subclass
        in order to establish callback method behavior for the data chunk attributes. Note that
        :class:`NullDataChunk` is also an acceptable value for any of the chunk initializers"""
        try:
            input_ = data.copy().loc[:, feature_selector]
        except AttributeError:
            if require_data or data is not None:
                raise
            input_ = None

        try:
            target = data.copy().loc[:, target_column]
        except (AttributeError, KeyError, TypeError):
            target = None

        #################### Data Chunks ####################
        self.input: BaseDataChunk = self._input_type(input_)
        self.target: BaseDataChunk = self._target_type(target)
        self.prediction: BaseDataChunk = self._prediction_type(None)

    def _on_call_default(self, division: str, point: str, *args, **kwargs):
        """Central logic update for invocation of core callback methods

        This means that when, for example, :meth:`BaseDataset.on_fold_end` is called, the
        `on_fold_end` methods of :attr:`input`, :attr:`target`, and :attr:`prediction` will be
        invoked by way of those attributes' `_do_something` methods"""
        self.input._do_something(division, point, *args, **kwargs)
        self.target._do_something(division, point, *args, **kwargs)
        self.prediction._do_something(division, point, *args, **kwargs)
