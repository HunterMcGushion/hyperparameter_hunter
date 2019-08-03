"""This module organizes and executes feature engineering/preprocessing step functions. The central
components of the module are :class:`FeatureEngineer` and :class:`EngineerStep` - everything else
is built to support those two classes. This module works with a very broad definition of
"feature engineering". The following is a non-exhaustive list of transformations that are
considered valid for `FeatureEngineer` step functions:

* Manual feature creation
* Input data scaling/normalization/standardization
* Target data transformation
* Re-sampling
* Data imputation
* Feature selection/elimination
* Encoding (one-hot, label, etc.)
* Binarization/binning/discretization
* Feature extraction (as for NLP/image recognition tasks)
* Feature shuffling

Related
-------
:mod:`hyperparameter_hunter.space`
    Only related when optimizing `FeatureEngineer` steps within an Optimization Protocol, but
    defines :class:`~hyperparameter_hunter.space.dimensions.Categorical`, which is the mechanism for
    defining a feature engineer step search space, and
    :class:`~hyperparameter_hunter.space.dimensions.RejectedOptional`, which is used to represent
    the absence of a feature engineer step, when labeled as `optional`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.keys.hashing import make_hash_sha256
from hyperparameter_hunter.space.dimensions import Categorical, RejectedOptional
from hyperparameter_hunter.utils.boltons_utils import remap, default_visit, default_enter
from hyperparameter_hunter.utils.general_utils import subdict

##################################################
# Import Miscellaneous Assets
##################################################
import ast
from contextlib import suppress
from inspect import getsource
import numpy as np
import pandas as pd
from typing import List, Callable, Dict, Union, Tuple

##################################################
# Global Variables
##################################################
EMPTY_SENTINEL = type("EMPTY_SENTINEL", tuple(), {})
DFDict = Dict[str, pd.DataFrame]
DescendantsType = Dict[str, Union["DescendantsType", None]]

N_DATASET_TRAIN = ["train_data", "train_inputs", "train_targets"]
N_DATASET_VALIDATION = ["validation_data", "validation_inputs", "validation_targets"]
N_DATASET_HOLDOUT = ["holdout_data", "holdout_inputs", "holdout_targets"]
N_DATASET_TEST = ["test_inputs"]

N_DATASET_ALL = ["all_data", "all_inputs", "all_targets"]
N_DATASET_NON_TRAIN = ["non_train_data", "non_train_inputs", "non_train_targets"]

STANDARD_DATASET_NAMES = N_DATASET_TRAIN + N_DATASET_VALIDATION + N_DATASET_HOLDOUT + N_DATASET_TEST
MERGED_DATASET_NAMES = N_DATASET_ALL + N_DATASET_NON_TRAIN

COUPLED_DATASET_CANDIDATES = [
    N_DATASET_TRAIN,
    N_DATASET_VALIDATION,
    N_DATASET_HOLDOUT,
    N_DATASET_ALL,
    N_DATASET_NON_TRAIN,
]


class DatasetNameReport:
    def __init__(self, params: Tuple[str], stage: str):
        """Characterize the relationships between the dataset names `params`

        Parameters
        ----------
        params: Tuple[str]
            Dataset names requested by a feature engineering step callable. Must be a subset of
            {"train_data", "train_inputs", "train_targets", "validation_data", "validation_inputs",
            "validation_targets", "holdout_data", "holdout_inputs", "holdout_targets",
            "test_inputs", "all_data", "all_inputs", "all_targets", "non_train_data",
            "non_train_inputs", "non_train_targets"}
        stage: String in {"pre_cv", "intra_cv"}
            Feature engineering stage during which the datasets `params` are requested

        Attributes
        ----------
        merged_datasets: List[tuple]
            Tuples of strings denoting paths to datasets that represent a merge between multiple
            datasets. Merged datasets are those prefixed with either "all" or "non_train". These
            paths are locations in `descendants`
        coupled_datasets: List[tuple]
            Tuples of strings denoting paths to datasets that represent a coupling of "inputs" and
            "targets" datasets. Coupled datasets are those suffixed with "data". These paths are
            locations in `descendants`, and the values at each path should be a dict containing keys
            with "inputs" and "targets" suffixes
        leaves: Dict[tuple, str]
            Mapping of full path tuples in `descendants` to their leaf values. Tuple paths represent
            the steps necessary to reach the standard dataset leaf value in `descendants` by
            traversing merged and coupled datasets. Values in `leaves` should be identical to the
            last element of the corresponding tuple key
        descendants: DescendantsType
            Nested dict in which all keys are dataset name strings, and all leaf values are `None`.
            Represents the structure of the requested dataset names, traversing over merged and
            coupled datasets (if necessary) in order to reach the standard dataset leaves"""
        self.params: Tuple[str] = params
        self.stage: str = stage

        self.merged_datasets: List[tuple] = []
        self.coupled_datasets: List[tuple] = []
        self.leaves: Dict[tuple, str] = dict()
        self.descendants: DescendantsType = remap(
            {_: _ for _ in self.params}, visit=self._visit, enter=self._enter, use_registry=False
        )

    @staticmethod
    def _visit(path, key, value):
        """If `key` == `value`, return tuple of (`key`, None). Else `default_visit`"""
        if key and key == value:
            return (key, None)
        return default_visit(path, key, value)

    def _enter(self, path, key, value):
        """Update contents of `merged_datasets`, `coupled_datasets`, and `leaves` and direct
        traversal of the sub-datasets that compose the current dataset name"""
        #################### Merged Datasets ####################
        if value in MERGED_DATASET_NAMES:
            self.merged_datasets.append(path + (key,))
            _names_for_merge = names_for_merge(value, self.stage)
            return dict(), zip(_names_for_merge, _names_for_merge)

        #################### Coupled Datasets ####################
        for coupled_candidate in COUPLED_DATASET_CANDIDATES:
            if value == coupled_candidate[0]:
                self.coupled_datasets.append(path + (key,))
                return dict(), zip(coupled_candidate[1:], coupled_candidate[1:])

        #################### Leaf Datasets ####################
        if key:
            self.leaves[path + (key,)] = key

        return default_enter(path, key, value)


def names_for_merge(merge_to: str, stage: str) -> List[str]:
    """Retrieve the names of the standard datasets that are allowed to be included in a merged
    DataFrame of type `merge_to` at stage `stage`

    Parameters
    ----------
    merge_to: String
        Type of merged dataframe to produce. Should be one of the following: {"all_data",
        "all_inputs", "all_targets", "non_train_data", "non_train_inputs", "non_train_targets"}
    stage: String in {"pre_cv", "intra_cv}
        Feature engineering stage for which the merged dataframe is requested. The results produced
        with each option differ only in that a `merged_df` created with `stage="pre_cv"` will never
        contain "validation" data because it doesn't exist before cross-validation has begun.
        Conversely, a `merged_df` created with `stage="intra_cv"` will contain the appropriate
        "validation" data if it exists

    Returns
    -------
    names: List
        Subset of {"train_data", "train_inputs", "train_targets", "validation_data",
        "validation_inputs", "validation_targets", "holdout_data", "holdout_inputs",
        "holdout_targets", "test_inputs"}

    Examples
    --------
    >>> names_for_merge("all_data", "intra_cv")
    ['train_data', 'validation_data', 'holdout_data']
    >>> names_for_merge("all_inputs", "intra_cv")
    ['train_inputs', 'validation_inputs', 'holdout_inputs', 'test_inputs']
    >>> names_for_merge("all_targets", "intra_cv")
    ['train_targets', 'validation_targets', 'holdout_targets']
    >>> names_for_merge("all_data", "pre_cv")
    ['train_data', 'holdout_data']
    >>> names_for_merge("all_inputs", "pre_cv")
    ['train_inputs', 'holdout_inputs', 'test_inputs']
    >>> names_for_merge("all_targets", "pre_cv")
    ['train_targets', 'holdout_targets']
    >>> names_for_merge("non_train_data", "intra_cv")
    ['validation_data', 'holdout_data']
    >>> names_for_merge("non_train_inputs", "intra_cv")
    ['validation_inputs', 'holdout_inputs', 'test_inputs']
    >>> names_for_merge("non_train_targets", "intra_cv")
    ['validation_targets', 'holdout_targets']
    >>> names_for_merge("non_train_data", "pre_cv")
    ['holdout_data']
    >>> names_for_merge("non_train_inputs", "pre_cv")
    ['holdout_inputs', 'test_inputs']
    >>> names_for_merge("non_train_targets", "pre_cv")
    ['holdout_targets']"""
    merge_type, data_group = merge_to.rsplit("_", 1)
    names = [_ for _ in STANDARD_DATASET_NAMES if _.endswith(data_group)]

    if stage == "pre_cv":
        names = [_ for _ in names if _ not in N_DATASET_VALIDATION]
    if merge_type == "non_train":
        names = [_ for _ in names if not _.startswith("train")]

    return names


def merge_dfs(merge_to: str, stage: str, dfs: DFDict) -> pd.DataFrame:
    """Construct a multi-indexed DataFrame containing the values of `dfs` deemed necessary by
    `merge_to` and `stage`. This is the opposite of `split_merged_df`

    Parameters
    ----------
    merge_to: String
        Type of `merged_df` to produce. Should be one of the following: {"all_data", "all_inputs",
        "all_targets", "non_train_data", "non_train_inputs", "non_train_targets"}
    stage: String in {"pre_cv", "intra_cv}
        Feature engineering stage for which `merged_df` is requested
    dfs: Dict
        Mapping of dataset names to their DataFrame values. Keys in `dfs` should be a subset of
        {"train_data", "train_inputs", "train_targets", "validation_data", "validation_inputs",
        "validation_targets", "holdout_data", "holdout_inputs", "holdout_targets", "test_inputs"}

    Returns
    -------
    merged_df: pd.DataFrame
        Multi-indexed DataFrame, in which the first index is a string naming the dataset in `dfs`
        from which the corresponding data originates. The following index(es) are the original
        index(es) from the dataset in `dfs`. All primary indexes in `merged_df` will be one of the
        strings considered to be valid keys for `dfs`

    Raises
    ------
    ValueError
        If all the DataFrames that would have been used in `merged_df` are None. This can happen if
        requesting `merge_to="non_train_targets"` during `stage="pre_cv"` when there is no holdout
        dataset available. Under these circumstances, the holdout dataset targets would be the sole
        contents of `merged_df`, rendering `merged_df` invalid since the data is unavailable

    See Also
    --------
    names_for_merge: Describes how `stage` values differ"""
    df_names = names_for_merge(merge_to, stage)
    df_names = [_ for _ in df_names if isinstance(dfs.get(_, None), pd.DataFrame)]
    try:
        merged_df = pd.concat([dfs[_] for _ in df_names], keys=df_names)
    except ValueError as _ex:
        raise ValueError(f"Merging {df_names} into {merge_to} does not produce DataFrame") from _ex
        # TODO: Add more specific error message for below scenario?
        # Tricky: This will be raised when `stage`="pre_cv" and `merge_to`="non_train..." if
        #   holdout/test data not available in `dfs`. May occur, for example, when using a step
        #   function that requests "non_train_inputs", with an `Environment` that has neither
        #   `holdout_dataset` nor `test_dataset` IF attempting to force the `EngineerStep`'s
        #   `stage`="pre_cv", instead of its default "intra_cv". This is correct behavior because
        #   "non_train_inputs" cannot be made under these circumstances; however, the precise cause
        #   of the problem may not be immediately apparent
    return merged_df


def split_merged_df(merged_df: pd.DataFrame) -> DFDict:
    """Separate a multi-indexed DataFrame into a dict mapping primary indexes in `merged_df` to
    DataFrames containing one fewer dimension than `merged_df`. This is the opposite of `merge_dfs`

    Parameters
    ----------
    merged_df: pd.DataFrame
        Multi-indexed DataFrame of the form returned by :func:`merge_dfs` to split into the separate
        DataFrames named by the primary indexes of `merged_df`

    Returns
    -------
    dfs: Dict
        Mapping of dataset names to their DataFrame values. Keys in `dfs` will be a subset of
        {"train_data", "train_inputs", "train_targets", "validation_data", "validation_inputs",
        "validation_targets", "holdout_data", "holdout_inputs", "holdout_targets", "test_inputs"}
        containing only those values that are also primary indexes in `merged_df`"""
    dfs = dict()
    for df_index in merged_df.index.levels[0]:
        dfs[df_index] = merged_df.loc[df_index, :].copy()
    return dfs


def validate_dataset_names(params: Tuple[str], stage: str) -> List[str]:
    """Produce the names of merged datasets in `params` and verify there are no duplicate references
    to any datasets in `params`

    Parameters
    ----------
    params: Tuple[str]
        Dataset names requested by a feature engineering step callable. Must be a subset of
        {"train_data", "train_inputs", "train_targets", "validation_data", "validation_inputs",
        "validation_targets", "holdout_data", "holdout_inputs", "holdout_targets",
        "test_inputs", "all_data", "all_inputs", "all_targets", "non_train_data",
        "non_train_inputs", "non_train_targets"}
    stage: String in {"pre_cv", "intra_cv}
        Feature engineering stage for which `merged_df` is requested

    Returns
    -------
    List[str]
        Names of merged datasets in `params`

    Raises
    ------
    ValueError
        If requested `params` contain a duplicate reference to any dataset, either by way of
        merging/coupling or not"""
    report = DatasetNameReport(params, stage)

    reverse_multidict = dict()
    for leaf_path, leaf_name in report.leaves.items():
        reverse_multidict.setdefault(leaf_name, set()).add(leaf_path)
    for leaf_name, leaf_paths in reverse_multidict.items():
        if len(leaf_paths) > 1:
            err_str = f"Requested params include duplicate references to `{leaf_name}` by way of:"
            err_str += "".join([f"\n   - {a_path}" for a_path in sorted(leaf_paths)])
            err_str += "\nEach dataset may only be requested by a single param for each function"
            raise ValueError(err_str)

    return [_[0] if len(_) == 1 else _ for _ in report.merged_datasets]


class EngineerStep:
    def __init__(self, f: Callable, stage=None, name=None, params=None, do_validate=False):
        """Container for individual :class:`FeatureEngineer` step functions

        Compartmentalizes functions of singular engineer steps and allows for greater customization
        than a raw engineer step function

        Parameters
        ----------
        f: Callable
            Feature engineering step function that requests, modifies, and returns datasets `params`

            Step functions should follow these guidelines:

                1. Request as input a subset of the 11 data strings listed in `params`
                2. Do whatever you want to the DataFrames given as input
                3. Return new DataFrame values of the input parameters in same order as requested

            If performing a task like target transformation, causing predictions to be transformed,
            it is often desirable to inverse-transform the predictions to be of the expected form.
            This can easily be done by returning an extra value from `f` (after the datasets) that
            is either a callable, or a transformer class that was fitted during the execution of `f`
            and implements an `inverse_transform` method. This is the only instance in which it is
            acceptable for `f` to return values that don't mimic its input parameters. See the
            engineer function definition using SKLearn's `QuantileTransformer` in the Examples
            section below for an actual inverse-transformation-compatible implementation
        stage: String in {"pre_cv", "intra_cv"}, or None, default=None
            Feature engineering stage during which the callable `f` will be given the datasets
            `params` to modify and return. If None, will be inferred based on `params`.

                * "pre_cv" functions are applied only once in the experiment: when it starts
                * "intra_cv" functions are reapplied for each fold in the cross-validation splits

            If `stage` is left to be inferred, "pre_cv" will *usually* be selected. However, if
            any `params` (or parameters in the signature of `f`) are prefixed with "validation..."
            or "non_train...", then `stage` will inferred as "intra_cv". See the Notes section
            below for suggestions on the `stage` to use for different functions
        name: String, or None, default=None
            Identifier for the transformation applied by this engineering step. If None,
            `f.__name__` will be used
        params: Tuple[str], or None, default=None
            Dataset names requested by feature engineering step callable `f`. If None, will be
            inferred by parsing the signature of `f`. Must be a subset of the following 11 strings:

            Input Data

            1. "train_inputs"
            2. "validation_inputs"
            3. "holdout_inputs"
            4. "test_inputs"
            5. "all_inputs"
                ``("train_inputs" + ["validation_inputs"] + "holdout_inputs" + "test_inputs")``
            6. "non_train_inputs"
                ``(["validation_inputs"] + "holdout_inputs" + "test_inputs")``

            Target Data

            7. "train_targets"
            8. "validation_targets"
            9. "holdout_targets"
            10. "all_targets"
                ``("train_targets" + ["validation_targets"] + "holdout_targets")``
            11. "non_train_targets"
                ``(["validation_targets"] + "holdout_targets")``

            As an alternative to the above list, just remember that the first half of all parameter
            names should be one of {"train", "validation", "holdout", "test", "all", "non_train"},
            and the second half should be either "inputs" or "targets". The only exception to this
            rule is "test_targets", which doesn't exist.

            Inference of "validation" `params` is affected by `stage`. During the "pre_cv" stage,
            the validation dataset has not yet been created and is still a part of the train
            dataset. During the "intra_cv" stage, the validation dataset is created by removing a
            portion of the train dataset, and their values passed to `f` reflect this fact. This
            also means that the values of the merged ("all"/"non_train"-prefixed) datasets may or
            may not contain "validation" data depending on the `stage`; however, this is all handled
            internally, so you probably don't need to worry about it.

            `params` may not include multiple references to the same dataset, either directly or
            indirectly. This means `("train_inputs", "train_inputs")` is invalid due to duplicate
            direct references. Less obviously, `("train_inputs", "all_inputs")` is invalid because
            "all_inputs" includes "train_inputs"
        do_validate: Boolean, or "strict", default=False
            ... Experimental...
            Whether to validate the datasets resulting from feature engineering steps. If True,
            hashes of the new datasets will be compared to those of the originals to ensure they
            were actually modified. Results will be logged. If `do_validate` = "strict", an
            exception will be raised if any anomalies are found, rather than logging a message. If
            `do_validate` = False, no validation will be performed

        See Also
        --------
        :class:`FeatureEngineer`
            The container for `EngineerStep` instances - `EngineerStep`s should always be provided
            to HyperparameterHunter through a `FeatureEngineer`
        :class:`~hyperparameter_hunter.space.dimensions.Categorical`
            Can be used during optimization to search through a group of `EngineerStep`s given as
            `categories`. The `optional` kwarg of `Categorical` designates a `FeatureEngineer` step
            that may be one of the `EngineerStep`s in `categories`, or may be omitted entirely
        :func:`get_engineering_step_stage`
            More information on `stage` inference and situations where overriding it may be prudent

        Notes
        -----
        `stage`: Generally, feature engineering conducted in the "pre_cv" stage should regard each
        sample/row as independent entities. For example, steps like converting a string day of the
        week to one-hot encoded columns, or imputing missing values by replacement with -1 might be
        conducted "pre_cv", since they are unlikely to introduce an information leakage. Conversely,
        steps like scaling/normalization, whose results for the data in one row are affected by the
        data in other rows should be performed "intra_cv" in order to recalculate the final values
        of the datasets for each cross validation split and avoid information leakage.

        `params`: In the list of the 11 valid `params` strings, "test_inputs" is notably missing the
        "..._targets" counterpart accompanying the other datasets. The "targets" suffix is missing
        because test data targets are never given. Note that although "test_inputs" is still
        included in both "all_inputs" and "non_train_inputs", its lack of a target column means that
        "all_targets" and "non_train_targets" may have different lengths than their
        "inputs"-suffixed counterparts

        Examples
        --------
        >>> from sklearn.preprocessing import StandardScaler, QuantileTransformer
        >>> def s_scale(train_inputs, non_train_inputs):
        ...     s = StandardScaler()
        ...     train_inputs[train_inputs.columns] = s.fit_transform(train_inputs.values)
        ...     non_train_inputs[train_inputs.columns] = s.transform(non_train_inputs.values)
        ...     return train_inputs, non_train_inputs
        >>> # Sensible parameter defaults inferred based on `f`
        >>> es_0 = EngineerStep(s_scale)
        >>> es_0.stage
        'intra_cv'
        >>> es_0.name
        's_scale'
        >>> es_0.params
        ('train_inputs', 'non_train_inputs')
        >>> # Override `stage` if you want to fit your scaler on OOF data like a crazy person
        >>> es_1 = EngineerStep(s_scale, stage="pre_cv")
        >>> es_1.stage
        'pre_cv'

        *Watch out for multiple requests to the same data*

        >>> es_2 = EngineerStep(s_scale, params=("train_inputs", "all_inputs"))
        Traceback (most recent call last):
            File "feature_engineering.py", line ? in validate_dataset_names
        ValueError: Requested params include duplicate references to `train_inputs` by way of:
           - ('all_inputs', 'train_inputs')
           - ('train_inputs',)
        Each dataset may only be requested by a single param for each function

        *Error is the same if `(train_inputs, all_inputs)` is in the actual function signature*

        *EngineerStep functions aren't just limited to transformations. Make your own features!*

        >>> def sqr_sum(all_inputs):
        ...     all_inputs["square_sum"] = all_inputs.agg(
        ...         lambda row: np.sqrt(np.sum([np.square(_) for _ in row])), axis="columns"
        ...     )
        ...     return all_inputs
        >>> es_3 = EngineerStep(sqr_sum)
        >>> es_3.stage
        'pre_cv'
        >>> es_3.name
        'sqr_sum'
        >>> es_3.params
        ('all_inputs',)

        *Inverse-transformation Implementation:*

        >>> def q_transform(train_targets, non_train_targets):
        ...     t = QuantileTransformer(output_distribution="normal")
        ...     train_targets[train_targets.columns] = t.fit_transform(train_targets.values)
        ...     non_train_targets[train_targets.columns] = t.transform(non_train_targets.values)
        ...     return train_targets, non_train_targets, t
        >>> # Note that `train_targets` and `non_train_targets` must still be returned in order,
        >>> #   but they are followed by `t`, an instance of `QuantileTransformer` we just fitted,
        >>> #   whose `inverse_transform` method will be called on predictions
        >>> es_4 = EngineerStep(q_transform)
        >>> es_4.stage
        'intra_cv'
        >>> es_4.name
        'q_transform'
        >>> es_4.params
        ('train_targets', 'non_train_targets')
        >>> # `params` does not include any returned transformers - Only data requested as input
        """
        self._f = f
        self._name = name
        self.params = params
        self._stage = stage
        self.do_validate = do_validate

        self.inversion = None
        self.merged_datasets: List[str] = validate_dataset_names(self.params, self.stage)
        self.original_hashes = dict()
        self.updated_hashes = dict()

    def __call__(self, **datasets: DFDict) -> DFDict:
        """Apply :attr:`f` to `datasets` to produce updated datasets. If `f` requests any
        merged/coupled datasets (as reflected by :attr:`params`), conversions to accommodate those
        requests will take place here

        Parameters
        ----------
        **datasets: DFDict
            Original dict of datasets, containing all datasets, some of which may be superfluous, or
            may require additional processing to resolve merged/coupled datasets

        Returns
        -------
        new_datasets: DFDict
            Dict of datasets, which have been updated by :attr:`f`. Any datasets that may have been
            merged prior to being given to :attr:`f` have been split back into the original
            datasets, with the updates made by :attr:`f`"""
        if self.do_validate:
            self.original_hashes = hash_datasets(datasets)

        datasets_for_f = self.get_datasets_for_f(datasets)
        step_result = self.f(**datasets_for_f)
        step_result = (step_result,) if not isinstance(step_result, tuple) else step_result

        if len(step_result) == len(self.params) + 1:
            self.inversion, step_result = step_result[-1], step_result[:-1]

        new_datasets = dict(zip(self.params, step_result))
        for dataset_name, dataset_value in new_datasets.items():
            if dataset_name in self.merged_datasets:
                new_datasets = dict(new_datasets, **split_merged_df(dataset_value))
        new_datasets = dict(datasets, **new_datasets)

        if self.do_validate:
            self.updated_hashes = hash_datasets(new_datasets)
        # TODO: Check `self.do_validate` here to decide whether to `compare_dataset_columns`
        return new_datasets

    def inverse_transform(self, data):
        """Perform the inverse transformation for this engineer step (if it exists)

        Parameters
        ----------
        data: Array-like
            Data to inverse transform with :attr:`inversion` or :attr:`inversion.inverse_transform`

        Returns
        -------
        Array-like
            If :attr:`inversion` is None, return `data` unmodified. Else, return the result of
            :attr:`inversion` or :attr:`inversion.inverse_transform`, given `data`"""
        if not self.inversion:
            return data
        elif callable(getattr(self.inversion, "inverse_transform", None)):
            return self.inversion.inverse_transform(data)
        elif callable(self.inversion):
            return self.inversion(data)
        raise TypeError(
            f"`inversion` must be callable, or class with `inverse_transform`, not {self.inversion}"
        )

    def get_datasets_for_f(self, datasets: DFDict) -> DFDict:
        """Produce a dict of DataFrames containing only the merged datasets and standard datasets
        requested in :attr:`params`. In other words, add the requested merged datasets and remove
        unnecessary standard datasets

        Parameters
        ----------
        datasets: DFDict
            Original dict of datasets, containing all datasets provided to
            :meth:`EngineerStep.__call__`, some of which may be superfluous, or may require
            additional processing to resolve merged/coupled datasets

        Returns
        -------
        DFDict
            Updated version of `datasets`, in which unnecessary datasets have been filtered out, and
            the requested merged datasets have been added"""
        datasets_for_f = datasets

        for _dataset_name in self.merged_datasets:
            datasets_for_f[_dataset_name] = merge_dfs(_dataset_name, self.stage, datasets)

        return subdict(datasets_for_f, keep=self.params)

    def get_key_data(self) -> dict:
        """Produce a dict of critical attributes describing the :class:`EngineerStep` instance for
        use by key-making classes

        Returns
        -------
        Dict
            Important attributes describing this :class:`EngineerStep` instance"""
        return dict(
            name=self.name,
            f=self.f,
            params=self.params,
            stage=self.stage,
            do_validate=self.do_validate,
            original_hashes=self.original_hashes,
            updated_hashes=self.updated_hashes,
        )

    ##################################################
    # Properties
    ##################################################
    @property
    def f(self) -> Callable:
        """Feature engineering step callable that requests, modifies, and returns datasets"""
        return self._f

    @property
    def name(self) -> str:
        """Identifier for the transformation applied by this engineering step"""
        if self._name is None:
            self._name = self.f.__name__
        return self._name

    @property
    def params(self) -> tuple:
        """Dataset names requested by feature engineering step callable :attr:`f`. See documentation
        in :meth:`EngineerStep.__init__` for more information/restrictions"""
        return self._params

    @params.setter
    def params(self, value):
        self._params = value if value is not None else get_engineering_step_params(self.f)

    @property
    def stage(self) -> str:
        """Feature engineering stage during which the `EngineerStep` will be executed"""
        if self._stage is None:
            self._stage = get_engineering_step_stage(self.params)
        return self._stage

    ##################################################
    # Comparison Methods
    ##################################################
    def __hash__(self):
        return hash((self.name, self.f, self.params, self.stage, self.do_validate))

    def __eq__(self, other):
        """Check whether `other` is equal to `self`

        The two are considered equal if `other` has the following attributes and their values
        are equal to those of `self`: :attr:`name`, :attr:`f`, :attr:`params`, :attr:`stage`, and
        :attr:`do_validate`. The values of all the aforementioned attributes will have been set on
        initialization (either explicitly or by inference), and they should never be altered

        Parameters
        ----------
        other: EngineerStep, dict, str
            Object to compare to `self`. If dict, the critical attributes mentioned above are
            regarded as keys of `other`, and `other` should be of the form returned by
            :meth:`EngineerStep.get_comparison_attrs`. If str, `other` will be compared to the
            result of `self`'s :meth:`EngineerStep.stringify`

        Returns
        -------
        Boolean
            True if `other` is equal to `self`, else False

        Examples
        --------
        >>> def dummy_f(train_inputs, non_train_inputs):
        ...     return train_inputs, non_train_inputs
        >>> es_0 = EngineerStep(dummy_f)
        >>> assert es_0 == EngineerStep(dummy_f)
        >>> assert es_0 == EngineerStep.get_comparison_attrs(es_0)
        >>> assert es_0 == es_0.stringify()
        """
        if isinstance(other, str):
            return self.stringify() == other
        elif isinstance(other, (dict, EngineerStep)):
            # Collect dicts of attributes for comparison
            other_attrs = self.get_comparison_attrs(other)
            own_attrs = self.get_comparison_attrs(self)
            # If `other_attrs["f"]` is str, should be SHA256 - Use hash of `self.f` to compare
            if isinstance(other_attrs["f"], str):
                own_attrs["f"] = make_hash_sha256(own_attrs["f"])

            return own_attrs == other_attrs

        return False

    @staticmethod
    def get_comparison_attrs(step_obj: Union["EngineerStep", dict]) -> dict:
        """Build a dict of critical :class:`EngineerStep` attributes

        Parameters
        ----------
        step_obj: EngineerStep, dict
            Object for which critical :class:`EngineerStep` attributes should be collected

        Returns
        -------
        attr_vals: Dict
            Critical :class:`EngineerStep` attributes. If `step_obj` does not have a necessary
            attribute (for `EngineerStep`) or a necessary key (for dict), its value in `attr_vals`
            will be a placeholder object. This is to facilitate comparison, while also ensuring
            missing values will always be considered unequal to other values

        Examples
        --------
        >>> def dummy_f(train_inputs, non_train_inputs):
        ...     return train_inputs, non_train_inputs
        >>> es_0 = EngineerStep(dummy_f)
        >>> EngineerStep.get_comparison_attrs(es_0)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'name': 'dummy_f',
         'f': <function dummy_f at ...>,
         'params': ('train_inputs', 'non_train_inputs'),
         'stage': 'intra_cv',
         'do_validate': False}
        >>> EngineerStep.get_comparison_attrs(
        ...     dict(foo="hello", f=dummy_f, params=["all_inputs", "all_targets"], stage="pre_cv")
        ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'name': <object object at ...>,
         'f': <function dummy_f at ...>,
         'params': ('all_inputs', 'all_targets'),
         'stage': 'pre_cv',
         'do_validate': <object object at ...>}
        """
        # Attributes necessary for equality
        attr_names = ("name", "f", "params", "stage", "do_validate")
        if isinstance(step_obj, dict):
            attr_vals = {_: step_obj.get(_, object()) for _ in attr_names}
        else:
            attr_vals = {_: getattr(step_obj, _, object()) for _ in attr_names}

        # Ensure :attr:`params` is always a tuple, not a list
        attr_vals["params"] = tuple(attr_vals["params"])
        return attr_vals

    def stringify(self) -> str:
        """Make a stringified representation of `self`, compatible with :meth:`EngineerStep.__eq__`

        Returns
        -------
        String
            String describing all critical attributes of the :class:`EngineerStep` instance. This
            value is not particularly human-friendly due to both its length and the fact that
            :attr:`EngineerStep.f` is represented by its hash

        Examples
        --------
        >>> def dummy_f(train_inputs, non_train_inputs):
        ...     return train_inputs, non_train_inputs
        >>> EngineerStep(dummy_f).stringify()  # doctest: +ELLIPSIS
        "EngineerStep(dummy_f, ..., ('train_inputs', 'non_train_inputs'), intra_cv, False)"
        >>> EngineerStep(dummy_f, stage="pre_cv").stringify()  # doctest: +ELLIPSIS
        "EngineerStep(dummy_f, ..., ('train_inputs', 'non_train_inputs'), pre_cv, False)"
        """
        return "{}({}, {}, {}, {}, {})".format(
            self.__class__.__name__,
            self.name,
            make_hash_sha256(self.f),
            self.params,
            self.stage,
            self.do_validate,
        )

    @classmethod
    def honorary_step_from_dict(cls, step_dict: dict, dimension: Categorical):
        """Get an `EngineerStep` from `dimension` that is equal to its dict form, `step_dict`

        Parameters
        ----------
        step_dict: Dict
            Dict of form saved in Experiment description files for `EngineerStep`. Expected to
            have following keys, with values of the given types:

            * "name": String
            * "f": String (SHA256 hash)
            * "params": List[str], or Tuple[str, ...]
            * "stage": String in {"pre_cv", "intra_cv"}
            * "do_validate": Boolean
        dimension: Categorical
            `Categorical` instance expected to contain the `EngineerStep` equivalent of `step_dict`
            in its categories

        Returns
        -------
        EngineerStep
            From `dimension.categories` if it is the `EngineerStep` equivalent of `step_dict`

        Raises
        ------
        ValueError
            If `dimension.categories` does not contain an `EngineerStep` matching `step_dict`"""
        for category in dimension.categories:
            if category == step_dict:
                return category
        raise ValueError("`step_dict` could not be found in `dimension`")

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.name)


class FeatureEngineer:
    def __init__(self, steps=None, do_validate=False, **datasets: DFDict):
        """Class to organize feature engineering step callables `steps` (:class:`EngineerStep`
        instances) and the datasets that the steps request and return.

        Parameters
        ----------
        steps: List, or None, default=None
            List of arbitrary length, containing any of the following values:

                1. :class:`EngineerStep` instance,
                2. Function to provide as input to :class:`EngineerStep`, or
                3. :class:`~hyperparameter_hunter.space.dimensions.Categorical`, with `categories`
                   comprising a selection of the previous two `steps` values (optimization only)

            The third value can only be used during optimization. The `feature_engineer` provided to
            :class:`~hyperparameter_hunter.experiments.CVExperiment`, for example, may only contain
            the first two values. To search a space optionally including an `EngineerStep`, use the
            `optional` kwarg of :class:`~hyperparameter_hunter.space.dimensions.Categorical`.

            See :class:`EngineerStep` for information on properly formatted `EngineerStep`
            functions. Additional engineering steps may be added via :meth:`add_step`
        do_validate: Boolean, or "strict", default=False
            ... Experimental...
            Whether to validate the datasets resulting from feature engineering steps. If True,
            hashes of the new datasets will be compared to those of the originals to ensure they
            were actually modified. Results will be logged. If `do_validate` = "strict", an
            exception will be raised if any anomalies are found, rather than logging a message. If
            `do_validate` = False, no validation will be performed
        **datasets: DFDict
            This is not expected to be provided on initialization and is offered primarily for
            debugging/testing. Mapping of datasets necessary to perform feature engineering steps

        See Also
        --------
        :class:`EngineerStep`
            For proper formatting of non-`Categorical` values of `steps`

        Notes
        -----
        If `steps` does include any instances of
        :class:`hyperparameter_hunter.space.dimensions.Categorical`, this `FeatureEngineer` instance
        will not be usable by Experiments. It can only be used by Optimization Protocols.
        Furthermore, the `FeatureEngineer` that the Optimization Protocol actually ends up using
        will not pass identity checks against the original `FeatureEngineer` that contained
        `Categorical` steps

        Examples
        --------
        >>> from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
        >>> # Define some engineer step functions to play with
        >>> def s_scale(train_inputs, non_train_inputs):
        ...     s = StandardScaler()
        ...     train_inputs[train_inputs.columns] = s.fit_transform(train_inputs.values)
        ...     non_train_inputs[train_inputs.columns] = s.transform(non_train_inputs.values)
        ...     return train_inputs, non_train_inputs
        >>> def mm_scale(train_inputs, non_train_inputs):
        ...     s = MinMaxScaler()
        ...     train_inputs[train_inputs.columns] = s.fit_transform(train_inputs.values)
        ...     non_train_inputs[train_inputs.columns] = s.transform(non_train_inputs.values)
        ...     return train_inputs, non_train_inputs
        >>> def q_transform(train_targets, non_train_targets):
        ...     t = QuantileTransformer(output_distribution="normal")
        ...     train_targets[train_targets.columns] = t.fit_transform(train_targets.values)
        ...     non_train_targets[train_targets.columns] = t.transform(non_train_targets.values)
        ...     return train_targets, non_train_targets, t
        >>> def sqr_sum(all_inputs):
        ...     all_inputs["square_sum"] = all_inputs.agg(
        ...         lambda row: np.sqrt(np.sum([np.square(_) for _ in row])), axis="columns"
        ...     )
        ...     return all_inputs

        *FeatureEngineer steps wrapped by `EngineerStep` == raw function steps - as long as the
        `EngineerStep` is using the default parameters*

        >>> # FeatureEngineer steps wrapped by `EngineerStep` == raw function steps
        >>> #   ... As long as the `EngineerStep` is using the default parameters
        >>> fe_0 = FeatureEngineer([sqr_sum, s_scale])
        >>> fe_1 = FeatureEngineer([EngineerStep(sqr_sum), EngineerStep(s_scale)])
        >>> fe_0.steps == fe_1.steps
        True
        >>> fe_2 = FeatureEngineer([sqr_sum, EngineerStep(s_scale), q_transform])

        *`Categorical` can be used during optimization and placed anywhere in `steps`. `Categorical`
        can also handle either `EngineerStep` categories or raw functions. Use the `optional` kwarg
        of `Categorical` to test some questionable steps*

        >>> fe_3 = FeatureEngineer([sqr_sum, Categorical([s_scale, mm_scale]), q_transform])
        >>> fe_4 = FeatureEngineer([Categorical([sqr_sum], optional=True), s_scale, q_transform])
        >>> fe_5 = FeatureEngineer([
        ...     Categorical([sqr_sum], optional=True),
        ...     Categorical([EngineerStep(s_scale), mm_scale]),
        ...     q_transform
        ... ])
        """
        self.steps = []
        self.do_validate = do_validate
        self.datasets = datasets or {}

        for step in steps or []:
            self.add_step(step)

    def __call__(self, stage: str, **datasets: DFDict):
        """Execute all feature engineering steps in :attr:`steps` for `stage`, with datasets
        `datasets` as inputs

        Parameters
        ----------
        stage: String in {"pre_cv", "intra_cv"}
             Feature engineering stage, specifying which :class:`EngineerStep` instances in
             :attr:`steps` should be executed
        datasets: DFDict
            Original dict of datasets, containing all datasets, some of which may be superfluous, or
            may require additional processing to resolve merged/coupled datasets"""
        if datasets:
            self.datasets = datasets

        for i, step in enumerate(self.steps):
            if step.stage == stage:
                self.datasets = step(**self.datasets)

    def __eq__(self, other: "FeatureEngineer"):
        return (
            isinstance(other, FeatureEngineer)
            and len(self.steps) == len(other.steps)
            and all(s_self == s_other for (s_self, s_other) in zip(self.steps, other.steps))
        )

    def inverse_transform(self, data):
        """Perform the inverse transformation for all engineer steps in :attr:`steps` in sequence
        on `data`

        Parameters
        ----------
        data: Array-like
            Data to inverse transform with any inversions present in :attr:`steps`

        Returns
        -------
        Array-like
            Result of sequentially calling inverse transformations in :attr:`steps` on `data`. If
            any step has :attr:`EngineerStep.inversion` = None, `data` is unmodified for that step,
            and proceeds to next engineer step inversion"""
        inverted_data = data

        # TODO: Make sure "pre_cv"-stage steps are inverted first, then "intra_cv"-stage
        for i, step in enumerate(self.steps):
            inverted_data = step.inverse_transform(inverted_data)

        return inverted_data

    @property
    def steps(self) -> List[EngineerStep]:
        """Feature engineering steps to execute in sequence on :meth:`FeatureEngineer.__call__`"""
        return self._steps

    @steps.setter
    def steps(self, value: list):
        self._steps = value

    def get_key_data(self) -> dict:
        """Produce a dict of critical attributes describing the :class:`FeatureEngineer` instance
        for use by key-making classes

        Returns
        -------
        Dict
            Important attributes describing this :class:`FeatureEngineer` instance"""
        return dict(
            steps=[_.get_key_data() if isinstance(_, EngineerStep) else _ for _ in self.steps],
            do_validate=self.do_validate,
            datasets=self.datasets,
        )

    def add_step(
        self,
        step: Union[Callable, EngineerStep, Categorical],
        stage: str = None,
        name: str = None,
        before: str = EMPTY_SENTINEL,
        after: str = EMPTY_SENTINEL,
        number: int = EMPTY_SENTINEL,
    ):
        """Add an engineering step to :attr:`steps` to be executed with the other contents of
        :attr:`steps` on :meth:`FeatureEngineer.__call__`

        Parameters
        ----------
        step: Callable, or `EngineerStep`, or `Categorical`
            If `EngineerStep` instance, will be added directly to :attr:`steps`. Otherwise, must be
            a feature engineering step callable that requests, modifies, and returns datasets, which
            will be used to instantiate a :class:`EngineerStep` to add to :attr:`steps`. If
            `Categorical`, `categories` should contain `EngineerStep` instances or callables
        stage: String in {"pre_cv", "intra_cv"}, or None, default=None
            Feature engineering stage during which the callable `step` will be executed
        name: String, or None, default=None
            Identifier for the transformation applied by this engineering step. If None and `step`
            is not an `EngineerStep`, will be inferred during :class:`EngineerStep` instantiation
        before: String, default=EMPTY_SENTINEL
            ... Experimental...
        after: String, default=EMPTY_SENTINEL
            ... Experimental...
        number: String, default=EMPTY_SENTINEL
            ... Experimental..."""
        if isinstance(step, Categorical):
            cat_params = step.get_params()
            cat_params["categories"] = [self._to_step(_) for _ in cat_params["categories"]]
            self._steps.append(Categorical(**cat_params))
        else:
            self._steps.append(self._to_step(step, stage=stage, name=name))

    def _to_step(self, step: Union[Callable, EngineerStep], stage=None, name=None) -> EngineerStep:
        """Ensure a candidate `step` is an `EngineerStep` instance, and return it

        Parameters
        ----------
        step: Callable, or `EngineerStep`
            If `EngineerStep` instance, will be added directly to :attr:`steps`. Otherwise, must be
            a feature engineering step callable that requests, modifies, and returns datasets, which
            will be used to instantiate a :class:`EngineerStep` to add to :attr:`steps`
        stage: String in {"pre_cv", "intra_cv"}, or None, default=None
            Feature engineering stage during which the callable `step` will be executed
        name: String, or None, default=None
            Identifier for the transformation applied by this engineering step. If None and `step`
            is not an `EngineerStep`, will be inferred during :class:`EngineerStep` instantiation

        Returns
        -------
        EngineerStep
            `step` if already an instance of `EngineerStep`. Else an `EngineerStep` initialized
            using `step`, `name`, and `stage`"""
        if isinstance(step, EngineerStep):
            return step
        elif step == RejectedOptional():
            return step  # Return as-is - OptimizationProtocol will take care of it
        else:
            return EngineerStep(step, name=name, stage=stage, do_validate=self.do_validate)


# FLAG: Tally number of columns "transformed" and "added" at each step and report


def get_engineering_step_stage(datasets: Tuple[str, ...]) -> str:
    """Determine the stage in which a feature engineering step that requests `datasets` as input
    should be executed

    Parameters
    ----------
    datasets: Tuple[str]
        Dataset names requested by a feature engineering step callable

    Returns
    -------
    stage: {"pre_cv", "intra_cv"}
        "pre_cv" if a step processing the given `datasets` should be executed in the
        pre-cross-validation stage. "intra_cv" if the step should be executed for each
        cross-validation split. If any of the elements in `datasets` is prefixed with "validation"
        or "non_train", `stage` will be "intra_cv". Otherwise, it will be "pre_cv"

    Notes
    -----
    Generally, feature engineering conducted in the "pre_cv" stage should regard each sample/row as
    independent entities. For example, steps like converting a string day of the week to one-hot
    encoded columns, or imputing missing values by replacement with -1 might be conducted "pre_cv",
    since they are unlikely to introduce an information leakage. Conversely, steps like
    scaling/normalization, whose results for the data in one row are affected by the data in other
    rows should be performed "intra_cv" in order to recalculate the final values of the datasets for
    each cross validation split and avoid information leakage

    Technically, the inference of `stage="intra_cv"` due to the existence of a "non_train"-prefixed
    value in `datasets` could unnecessarily force steps to be executed "intra_cv" if, for example,
    there is no validation data. However, this is safer than the alternative of executing these
    steps "pre_cv", in which validation data would be a subset of train data, probably introducing
    information leakage. A simple workaround for this is to explicitly provide :class:`EngineerStep`
    with the desired `stage` parameter to bypass this inference

    Examples
    --------
    >>> get_engineering_step_stage(("train_inputs", "validation_inputs", "holdout_inputs"))
    'intra_cv'
    >>> get_engineering_step_stage(("all_data"))
    'pre_cv'
    >>> get_engineering_step_stage(("all_inputs", "all_targets"))
    'pre_cv'
    >>> get_engineering_step_stage(("train_data", "non_train_data"))
    'intra_cv'"""
    if any(_.startswith("validation_") for _ in datasets):
        return "intra_cv"
    if any(_.startswith("non_train_") for _ in datasets):
        return "intra_cv"
    return "pre_cv"


class ParameterParser(ast.NodeVisitor):
    def __init__(self):
        """`ast.NodeVisitor` subclass that collects the arguments specified in the signature of a
        callable node, as well as the values returned by the callable, in the attributes `args` and
        `returns`, respectively"""
        self.args = []
        self.returns = []

    def visit_arg(self, node):
        with suppress(AttributeError):
            if isinstance(node.parent.parent, ast.FunctionDef):
                if isinstance(node.parent.parent.parent, ast.Module):
                    self.args.append(node.arg)
        self.generic_visit(node)

    def visit_Return(self, node):
        try:
            self.returns.append(node.value.id)
        except AttributeError:
            for element in node.value.elts:
                try:
                    self.returns.append(element.id)
                except AttributeError:  # Straight-up function probably, instead of variable name
                    self.returns.append(getattr(element, "attr", element.__class__.__name__))
        self.generic_visit(node)


def get_engineering_step_params(f: callable) -> Tuple[str]:
    """Verify that callable `f` requests valid input parameters, and returns a tuple of the same
    parameters, with the assumption that the parameters are modified by `f`

    Parameters
    ----------
    f: Callable
        Feature engineering step function that requests, modifies, and returns datasets

    Returns
    -------
    Tuple
        Argument/return value names declared by `f`

    Examples
    --------
    >>> def impute_negative_one(all_inputs):
    ...     all_inputs.fillna(-1, inplace=True)
    ...     return all_inputs
    >>> get_engineering_step_params(impute_negative_one)
    ('all_inputs',)
    >>> def standard_scale(train_inputs, non_train_inputs):
    ...     scaler = StandardScaler()
    ...     train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    ...     non_train_inputs[train_inputs.columns] = scaler.transform(non_train_inputs.values)
    ...     return train_inputs, non_train_inputs
    >>> get_engineering_step_params(standard_scale)
    ('train_inputs', 'non_train_inputs')
    >>> def error_invalid_dataset(train_inputs, foo):
    ...     return train_inputs, foo
    >>> get_engineering_step_params(error_invalid_dataset)
    Traceback (most recent call last):
        File "feature_engineering.py", line ?, in get_engineering_step_params
    ValueError: Invalid dataset name: 'foo'"""
    valid_datasets = MERGED_DATASET_NAMES + STANDARD_DATASET_NAMES
    source_code = getsource(f)
    tree = ast.parse(source_code)

    #################### Add Links to Nodes' Parents ####################
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    #################### Collect Parameters and Returns ####################
    parser = ParameterParser()
    parser.visit(tree)

    for name in parser.args:
        if name not in valid_datasets:
            raise ValueError(f"Invalid dataset name: {name!r}")
        if name.endswith("_data"):
            raise ValueError(
                f"Sorry, 'data'-suffixed parameters like {name!r} are not supported yet. "
                "Try using both the 'inputs' and 'targets' params for this dataset, instead!"
            )

    return tuple(parser.args)


def _hash_dataset(dataset: pd.DataFrame) -> dict:
    """Generate hashes for `dataset` at various levels of specificity

    Parameters
    ----------
    dataset: pandas.DataFrame
        DataFrame to be described with a dict of hashes

    Returns
    -------
    dict
        "dataset" (str): Hash of `dataset`, itself
        "column_names" (str): Hash of `dataset.columns`, capturing names, order, and add/drops
        "column_values" (dict): Keys are `dataset.columns`, and values are hashes for each column

    Examples
    --------
    >>> _hash_dataset(pd.DataFrame(dict(a=[0, 1], b=[2, 3])))  # doctest: +NORMALIZE_WHITESPACE
    {'dataset': 't0rdT14SDIH-CVm-dce1Hlsr2oM7q6pss_GpV3rJ6bw=',
     'column_names': 't2r52T-rdDqIDs75-83buoieqk0KyHEpRJMJAAzfzb4=',
     'column_values': {'a': 'buQ0yuUUbLN57tC6050g7yWrvAdk-NwGIEEWHJC88EY=',
                       'b': 'j9nBFZVu4ZEnsoaRYiI93DcrbV3A_hzcKdf0P5gS7g4='}}
    >>> _hash_dataset(pd.DataFrame(dict(x=[0, 1], b=[6, 7])))  # doctest: +NORMALIZE_WHITESPACE
    {'dataset': 'TNLSddRnWVfoytkhHrSNWXqVW2TV7cHKht8MMLWcbhY=',
     'column_names': '9l1vTGGIxfuA4rJZ-ePalM-9Q5D0BfLp5bogE0U-oYQ=',
     'column_values': {'x': 'l2dZ6AeGRuHH97J0qb8I1H-pwK-ubHqElDqFIuKAbIw=',
                       'b': 'uIvA32AuBuj9LTU652UQUBI0VH9UmF2ZJeL4NefiiLg='}}
    >>> _hash_dataset(None)
    {'dataset': None, 'column_names': None, 'column_values': None}"""
    if (not isinstance(dataset, pd.DataFrame)) and (dataset is None or dataset == 0):
        return dict(dataset=None, column_names=None, column_values=None)
    return dict(
        dataset=make_hash_sha256(dataset),
        column_names=make_hash_sha256(dataset.columns),
        column_values={_: make_hash_sha256(dataset[_]) for _ in dataset.columns},
    )


def hash_datasets(datasets: dict) -> dict:
    """Describe `datasets` with dicts of hashes for their values, column names, and column values

    Parameters
    ----------
    datasets: Dict
        Mapping of dataset names to `pandas.DataFrame` instances

    Returns
    -------
    hashes: Dict
        Mapping with same keys as `datasets`, whose values are dicts returned from
        :func:`_hash_dataset` that provide hashes for each DataFrame and its column names/values

    Examples
    --------
    >>> df_x = pd.DataFrame(dict(a=[0, 1], b=[2, 3], c=[4, 5]))
    >>> df_y = pd.DataFrame(dict(a=[0, 1], b=[6, 7], d=[8, 9]))
    >>> hash_datasets(dict(x=df_x, y=df_y)) == dict(x=_hash_dataset(df_x), y=_hash_dataset(df_y))
    True"""
    hashes = {k: _hash_dataset(v) for k, v in datasets.items()}
    return hashes


# def _compare_hash_(columns_a: dict, columns_b: dict):
#     """
#
#     Parameters
#     ----------
#     columns_a
#     columns_b
#
#     Returns
#     -------
#
#     """
#     columns_added = dict()
#     columns_dropped = dict()
#     columns_modified = dict()
#     columns_unchanged = dict()
#
#
# def compare_dataset_columns(datasets_a: dict, datasets_b: dict):
#     compare_column_hashes(..., ...)


# def step(order=None, before=None, after=None, returns="frame"):
#     """
#
#     Parameters
#     ----------
#     order: Integer, or None, default=None
#         ...
#     before: String, or None, default=None
#         ...
#     after: String, or None, default=None
#         ...
#     returns: {"frame", "cols"}, default="frame"
#         ...
#
#     Returns
#     -------
#
#     """
#     ...
