"""This module is still in an experimental stage and should not be assumed to be "reliable", or
"useful", or anything else that might be expected of a normal module"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.keys.hashing import make_hash_sha256
from hyperparameter_hunter.utils.boltons_utils import remap, default_visit, default_enter
from hyperparameter_hunter.utils.general_utils import subdict

##################################################
# Import Miscellaneous Assets
##################################################
from ast import NodeVisitor, parse
from inspect import getsource
import pandas as pd
from typing import List, Callable, Dict, Union

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
    def __init__(self, params: List[str], stage: str):
        """Characterize the relationships between the dataset names `params`

        Parameters
        ----------
        params: List[str]
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
        self.params: List[str] = params
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
    df_names = [_ for _ in df_names if dfs.get(_, None) is not None]
    try:
        merged_df = pd.concat([dfs[_] for _ in df_names], keys=df_names)
    except ValueError as _ex:
        raise ValueError(f"Merging {df_names} into {merge_to} does not produce DataFrame") from _ex
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


def validate_dataset_names(params: List[str], stage: str) -> List[str]:
    """Produce the names of merged datasets in `params` and verify there are no duplicate references
    to any datasets in `params`

    Parameters
    ----------
    params: List[str]
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
            err_str += "".join([f"\n   - {a_path}" for a_path in leaf_paths])
            err_str += "\nEach dataset may only be requested by a single param for each function"
            raise ValueError(err_str)

    return [_[0] if len(_) == 1 else _ for _ in report.merged_datasets]


class EngineerStep:
    def __init__(self, f: Callable, name=None, params=None, stage=None, do_validate=False):
        """:class:`FeatureEngineer` helper, compartmentalizing functions of singular engineer steps

        Parameters
        ----------
        f: Callable
            Feature engineering step function that requests, modifies, and returns datasets `params`
        name: String, or None, default=None
            Identifier for the transformation applied by this engineering step. If None,
            `f.__name__` will be used
        params: List[str], or None, default=None
            Dataset names requested by feature engineering step callable `f`. Must be a subset of
            {"train_data", "train_inputs", "train_targets", "validation_data", "validation_inputs",
            "validation_targets", "holdout_data", "holdout_inputs", "holdout_targets",
            "test_inputs", "all_data", "all_inputs", "all_targets", "non_train_data",
            "non_train_inputs", "non_train_targets"}. If None, will be inferred by parsing the
            abstract syntax tree of `f`
        stage: String in {"pre_cv", "intra_cv}, or None, default=None
            Feature engineering stage during which the callable `f` will be given the datasets
            `params` to modify and return
        do_validate: Boolean, or "strict", default=False
            ... Experimental...
            Whether to validate the datasets resulting from feature engineering steps. If True,
            hashes of the new datasets will be compared to those of the originals to ensure they
            were actually modified. Results will be logged. If `do_validate`="strict", an exception
            will be raised if any anomalies are found, rather than logging a message. If
            `do_validate`=False, no validation will be performed"""
        self._f = f
        self._name = name
        self._params = params
        self._stage = stage
        self.do_validate = do_validate

        self.merged_datasets = []
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
        self.original_hashes = hash_datasets(datasets)

        datasets_for_f = self.get_datasets_for_f(datasets)
        step_result = self.f(**datasets_for_f)
        step_result = (step_result,) if not isinstance(step_result, tuple) else step_result

        new_datasets = dict(zip(self.params, step_result))
        for dataset_name, dataset_value in new_datasets.items():
            if dataset_name in self.merged_datasets:
                new_datasets = dict(new_datasets, **split_merged_df(dataset_value))
        new_datasets = dict(datasets, **new_datasets)

        self.updated_hashes = hash_datasets(new_datasets)
        # TODO: Check `self.do_validate` here to decide whether to `compare_dataset_columns`
        return new_datasets

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
        self.merged_datasets: List[str] = validate_dataset_names(self.params, self.stage)
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
    def params(self) -> list:
        """Dataset names requested by feature engineering step callable :attr:`f`. See documentation
        in :meth:`EngineerStep.__init__` for more information/restrictions"""
        if self._params is None:
            self._params = get_engineering_step_params(self.f)
        return self._params

    @property
    def stage(self) -> str:
        """Feature engineering stage during which the `EngineerStep` will be executed"""
        if self._stage is None:
            self._stage = get_engineering_step_stage(self.params)
        return self._stage


class FeatureEngineer:
    def __init__(self, steps=None, do_validate=False, **datasets: DFDict):
        """Class to organize feature engineering step callables `steps` (:class:`EngineerStep`
        instances) and the datasets that the steps request and return.

        Parameters
        ----------
        steps: List, or None, default=None
            If not None, should be list containing any of the following: :class:`EngineerStep`
            instances, or callables used to instantiate :class:`EngineerStep`
        do_validate: Boolean, or "strict", default=False
            ... Experimental...
            Whether to validate the datasets resulting from feature engineering steps. If True,
            hashes of the new datasets will be compared to those of the originals to ensure they
            were actually modified. Results will be logged. If `do_validate`="strict", an exception
            will be raised if any anomalies are found, rather than logging a message. If
            `do_validate`=False, no validation will be performed
        **datasets: DFDict
            Mapping of datasets necessary to perform feature engineering steps. This is not expected
            to be provided on initialization and is offered primarily for debugging/testing"""
        self.steps = []
        for step in steps or []:
            self.add_step(step)

        self.do_validate = do_validate
        self.datasets = datasets or {}

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
            self.datasets = step(**self.datasets)

        # if stage == "pre_cv":
        #     ...  # TODO: Execute all steps in "pre_cv" stage
        # elif stage == "intra_cv":
        #     ...  # TODO: Execute all steps in "intra_cv" stage
        # else:
        #     raise ValueError("")

    @property
    def steps(self) -> List[EngineerStep]:
        """Feature engineering steps to execute in sequence on :meth:`FeatureEngineer.__call__"""
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
            steps=[_.get_key_data() for _ in self.steps],
            do_validate=self.do_validate,
            datasets=self.datasets,
        )

    def add_step(
        self,
        step: Union[Callable, EngineerStep],
        name: str = None,
        before: str = EMPTY_SENTINEL,
        after: str = EMPTY_SENTINEL,
        number: int = EMPTY_SENTINEL,
    ):
        """Add an engineering step to :attr:`steps` to be executed with the other contents of
        :attr:`steps` on :meth:`FeatureEngineer.__call__`

        Parameters
        ----------
        step: Callable, or `EngineerStep`
            If `EngineerStep` instance, will be added directly to :attr:`steps`. Otherwise, must be
            a feature engineering step callable that requests, modifies, and returns datasets, which
            will be used with `name` to instantiate a :class:`EngineerStep` to add to :attr:`steps`
        name: String, or None, default=None
            Identifier for the transformation applied by this engineering step. If None and `step`
            is not an `EngineerStep`, will be inferred during :class:`EngineerStep` instantiation
        before: String, default=EMPTY_SENTINEL
            ... Experimental...
        after: String, default=EMPTY_SENTINEL
            ... Experimental...
        number: String, default=EMPTY_SENTINEL
            ... Experimental..."""
        if isinstance(step, EngineerStep):
            self._steps.append(step)
        else:
            self._steps.append(EngineerStep(step, name))


# FLAG: Tally number of columns "transformed" and "added" at each step and report


def get_engineering_step_stage(datasets: List[str]) -> str:
    """Determine the stage in which a feature engineering step that requests `datasets` as input
    should be executed

    Parameters
    ----------
    datasets: List[str]
        Dataset names requested by a feature engineering step callable

    Returns
    -------
    stage: {"pre_cv", "intra_cv"}
        "pre_cv" if a step processing the given `datasets` should be executed in the
        pre-cross-validation stage. "intra_cv" if the step should be executed for each
        cross-validation split. Generally, feature engineering conducted in the "pre_cv" stage
        should regard each sample/row as independent entities. For example, steps like converting
        a string day of the week to one-hot encoded columns, or imputing missing values by replacing
        them with -1 might be conducted "pre_cv", since they are unlikely to introduce an
        information leakage. Conversely, steps like scaling/normalization, whose results for the
        data in one row are affected by the data in other rows should be performed "intra_cv" in
        order to recalculate the final values of the datasets for each cross validation split and
        avoid information leakage

    Examples
    --------
    >>> get_engineering_step_stage(["train_inputs", "validation_inputs", "holdout_inputs"])
    'intra_cv'
    >>> get_engineering_step_stage(["all_data"])
    'pre_cv'
    >>> get_engineering_step_stage(["all_inputs", "all_targets"])
    'pre_cv'
    >>> get_engineering_step_stage(["train_data", "non_train_data"])
    'intra_cv'
    """
    if all(_.startswith("all_") for _ in datasets):
        return "pre_cv"
    return "intra_cv"


class ParameterParser(NodeVisitor):
    def __init__(self):
        """`ast.NodeVisitor` subclass that collects the arguments specified in the signature of a
        callable node, as well as the values returned by the callable, in the attributes `args` and
        `returns`, respectively"""
        self.args = []
        self.returns = []

    def visit_arg(self, node):
        self.args.append(node.arg)
        self.generic_visit(node)

    def visit_Return(self, node):
        try:
            self.returns.append(node.value.id)
        except AttributeError:
            for element in node.value.elts:
                self.returns.append(element.id)
        self.generic_visit(node)


def get_engineering_step_params(f: callable) -> List[str]:
    """Verify that callable `f` requests valid input parameters, and returns a tuple of the same
    parameters, with the assumption that the parameters are modified by `f`

    Parameters
    ----------
    f: Callable
        Feature engineering step function that requests, modifies, and returns datasets

    Returns
    -------
    List
        Argument/return value names declared by `f`

    Examples
    --------
    >>> def impute_negative_one(all_data):
    ...     all_data.fillna(-1, inplace=True)
    ...     return all_data
    >>> get_engineering_step_params(impute_negative_one)
    ['all_data']
    >>> def standard_scale(train_inputs, non_train_inputs):
    ...     scaler = StandardScaler()
    ...     train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    ...     non_train_inputs[train_inputs.columns] = scaler.transform(non_train_inputs.values)
    ...     return train_inputs, non_train_inputs
    >>> get_engineering_step_params(standard_scale)
    ['train_inputs', 'non_train_inputs']
    >>> def error_mismatch(train_inputs, non_train_inputs):
    ...     return validation_inputs, holdout_inputs
    >>> get_engineering_step_params(error_mismatch)
    Traceback (most recent call last):
        File "feature_engineering.py", line ?, in get_engineering_step_params
    ValueError: Mismatched `f` inputs (['train_inputs', 'non_train_inputs']), and returns (['validation_inputs', 'holdout_inputs'])
    >>> def error_invalid_dataset(train_inputs, foo):
    ...     return train_inputs, foo
    >>> get_engineering_step_params(error_invalid_dataset)
    Traceback (most recent call last):
        File "feature_engineering.py", line ?, in get_engineering_step_params
    ValueError: Invalid dataset name in ['train_inputs', 'foo']"""
    valid_datasets = MERGED_DATASET_NAMES + STANDARD_DATASET_NAMES
    source_code = getsource(f)
    tree = parse(source_code)
    parser = ParameterParser()
    parser.visit(tree)

    if parser.args != parser.returns:
        raise ValueError(f"Mismatched `f` inputs ({parser.args}), and returns ({parser.returns})")
    elif any(_ not in valid_datasets for _ in parser.args):
        raise ValueError(f"Invalid dataset name in {parser.args}")
    return parser.args


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
    >>> assert _hash_dataset(pd.DataFrame(dict(a=[0, 1], b=[2, 3], c=[4, 5]))) == {
    ...     'dataset': 'UD0kfFLj7_eX4P5g02UWV-P04yuJkrsOcnS6yBa48Ps=',
    ...     'column_names': 'OUPCVME21ryrnjJtyZ1R-_rrr-wSMPxo9Gc1KxcdlhM=',
    ...     'column_values': {
    ...         'a': 'buQ0yuUUbLN57tC6050g7yWrvAdk-NwGIEEWHJC88EY=',
    ...         'b': 'j9nBFZVu4ZEnsoaRYiI93DcrbV3A_hzcKdf0P5gS7g4=',
    ...         'c': 'qO0pJn3TLhlsYj3nqliMBi8zds66JPsQ1uCJSFv9q9g=',
    ...     },
    ... }
    >>> assert _hash_dataset(pd.DataFrame(dict(a=[0, 1], b=[6, 7], d=[8, 9]))) == {
    ...     'dataset': '0jA8SnjKAbyG6tnwxwJ51Q8haeVcfMhBZ45ELuD2U6k=',
    ...     'column_names': 'G-xgYT0flyJV26HrfFYiMh_BiSkStKkh-Utqq94DZAM=',
    ...     'column_values': {
    ...         'a': 'buQ0yuUUbLN57tC6050g7yWrvAdk-NwGIEEWHJC88EY=',
    ...         'b': 'uIvA32AuBuj9LTU652UQUBI0VH9UmF2ZJeL4NefiiLg=',
    ...         'd': 'G_y3SLas04T-_ejL4AVACrDQM_uyT4HFxo1Ig1tF5Z8=',
    ...     },
    ... }
    >>> _hash_dataset(None)
    {'dataset': None, 'column_names': None, 'column_values': None}"""
    if dataset is None:
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
