"""This module is still in an experimental stage and should not be assumed to be "reliable", or
"useful", or anything else that might be expected of a normal module"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.keys.hashing import make_hash_sha256
from hyperparameter_hunter.utils.general_utils import subdict

##################################################
# Import Miscellaneous Assets
##################################################
from ast import NodeVisitor, parse
from inspect import getsource
import pandas as pd
from typing import List, Callable, Dict

##################################################
# Global Variables
##################################################
EMPTY_SENTINEL = type("EMPTY_SENTINEL", tuple(), {})
DFDict = Dict[str, pd.DataFrame]

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


def merge_dfs(merge_to: str, stage: str, dfs: DFDict) -> pd.DataFrame:
    """Construct a multi-indexed DataFrame containing the values of `dfs` deemed necessary by
    `merge_to` and `stage`. This is the opposite of `split_merged_df`

    Parameters
    ----------
    merge_to: String
        Type of `merged_df` to produce. Should be one of the following: {"all_data", "all_inputs",
        "all_targets", "non_train_data", "non_train_inputs", "non_train_targets"}
    stage: String in {"pre_cv", "intra_cv}
        Feature engineering stage for which `merged_df` is requested. The results produced with each
        option differ only in that a `merged_df` created with `stage="pre_cv"` will never contain
        "validation" data because it doesn't exist before cross-validation has begun. Conversely,
        a `merged_df` created with `stage="intra_cv"` will contain the appropriate "validation" data
        if it exists
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
        contents of `merged_df`, rendering `merged_df` invalid since the data is unavailable"""
    m_map = {
        ("all", "data", "intra_cv"): ["train", "validation", "holdout"],
        ("all", "data", "pre_cv"): ["train", "holdout"],
        ("all", "inputs", "intra_cv"): ["train", "validation", "holdout", "test"],
        ("all", "inputs", "pre_cv"): ["train", "holdout", "test"],
        ("all", "targets", "intra_cv"): ["train", "validation", "holdout"],
        ("all", "targets", "pre_cv"): ["train", "holdout"],
    }

    # Add "non_train"-prefixed entries to `m_map` based on "all"-prefixed entries
    for group in ["data", "inputs", "targets"]:
        m_map[("non_train", group, "intra_cv")] = m_map[("all", group, "intra_cv")][1:]
        m_map[("non_train", group, "pre_cv")] = m_map[("all", group, "pre_cv")][1:]

    target_group = tuple(merge_to.rsplit("_", 1))
    df_names = [f"{_}_{target_group[-1]}" for _ in m_map[target_group + (stage,)]]
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


class EngineerStep:
    def __init__(self, f: Callable, name=None, params=None, stage=None, do_validate=False):
        self._f = f
        self._name = name
        self._params = params
        self._stage = stage
        self.do_validate = do_validate

        self.original_hashes = dict()
        self.updated_hashes = dict()

    def __call__(self, **datasets) -> dict:
        """
        ...

        Parameters
        ----------
        **datasets: Dict
            ...

        Returns
        -------
        new_datasets: Dict
            ...

        """
        self.original_hashes = hash_datasets(datasets)
        step_result = self.f(**subdict(datasets, keep=self.params))
        new_datasets = dict(datasets, **dict(zip(self.params, step_result)))
        self.updated_hashes = hash_datasets(new_datasets)
        # TODO: Check `self.do_validate` here to decide whether to `compare_dataset_columns`
        return new_datasets

    def get_key_data(self) -> dict:
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
        return self._f

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = self.f.__name__
        return self._name

    @property
    def params(self) -> list:
        if self._params is None:
            self._params = get_engineering_step_params(self.f)
        return self._params

    @property
    def stage(self) -> str:
        if self._stage is None:
            self._stage = get_engineering_step_stage(self.params)
        return self._stage


class FeatureEngineer:
    def __init__(self, steps=None, do_validate=False, **datasets):
        """
        ...

        Parameters
        ----------
        steps: List, or None, default=None
            ...
        do_validate: Boolean, or "strict", default=False
            Whether to validate the datasets resulting from feature engineering steps. If True,
            hashes of the new datasets will be compared to those of the originals to ensure they
            were actually modified. Results will be logged. If `do_validate`="strict", an exception
            will be raised if any anomalies are found, rather than logging a message. If
            `do_validate`=False, no validation will be performed
        **datasets: Dict
            Mapping of datasets necessary to perform feature engineering steps. This is not expected
            to be provided on initialization and is offered primarily for debugging/testing
        """
        self._steps = steps or []
        self.do_validate = do_validate
        self.datasets = datasets or {}

    def __call__(self, stage, **datasets):
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

    # def do_step(self, step_number, **datasets):
    #     """Perform the specified step in the feature engineering workflow"""
    #     if datasets:
    #         self.datasets = datasets
    #     self.datasets = self.steps[step_number](**self.datasets)

    @property
    def steps(self) -> list:
        return self._steps

    @steps.setter
    def steps(self, value: list):
        self._steps = value

    def get_key_data(self) -> dict:
        return dict(
            steps=[_.get_key_data() for _ in self.steps],
            do_validate=self.do_validate,
            datasets=self.datasets,
        )

    def add_step(
        self,
        step: Callable,
        name: str = None,
        before: str = EMPTY_SENTINEL,
        after: str = EMPTY_SENTINEL,
        number: int = EMPTY_SENTINEL,
    ):
        self._steps.append(EngineerStep(step, name))

    ##################################################
    # Constructors
    ##################################################
    @classmethod
    def from_list(cls, steps):
        """Construct a `FeatureEngineer` instance using a list of engineering steps"""
        f = FeatureEngineer()
        f.steps = steps
        return f


# FLAG: Tally number of columns "transformed" and "added" at each step and report


def get_engineering_step_stage(datasets: List[str]):
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


def get_engineering_step_params(f):
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
