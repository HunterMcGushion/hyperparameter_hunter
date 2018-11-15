"""This module defines the Leaderboard classes that are saved to the
'HyperparameterHunterAssets/Leaderboards' subdirectory. It provides the ability to compare all
Experiment results at a glance

Related
-------
:mod:`hyperparameter_hunter.recorders`
    This module initiates the saving of Experiment entries to Leaderboards"""
##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod
import pandas as pd


class Leaderboard(metaclass=ABCMeta):
    def __init__(self, data=None):
        """The Leaderboard class is used for reading, updating, and saving leaderboard files within
        the 'HyperparameterHunterAssets/Leaderboards' subdirectory

        Parameters
        ----------
        data: pd.DataFrame, or None, default=None
            The starting state of the Leaderboard. If None, an empty DataFrame is used"""
        self.data = data if data is not None else pd.DataFrame()

    @classmethod
    def from_path(cls, path, assert_existence=False):
        """Initialize a Leaderboard from a .csv `path`

        Parameters
        ----------
        path: str
            The path of the file to read in as a DataFrame
        assert_existence: boolean, default=False
            If False, and :func:`pandas.read_csv` raises FileNotFoundError, the Leaderboard will be
            initialized with None. Else the exception is raised normally"""
        try:
            data = pd.read_csv(path, index_col=None)
        except FileNotFoundError:
            if assert_existence is True:
                raise
            data = None
        return cls(data=data)

    @abstractmethod
    def add_entry(self, experiment, **kwargs):
        """Add an entry row for `experiment` to :attr:`data`

        Parameters
        ----------
        experiment: :class:`experiments.BaseExperiment`
            An instance of a completed Experiment from which to construct a Leaderboard entry"""
        raise NotImplementedError()

    def save(self, path, **kwargs):
        """Save the Leaderboard instance

        Parameters
        ----------
        path: str
            The file to which the Leaderboard instance should be saved
        **kwargs: Dict
            Additional arguments to supply to :meth:`pandas.DataFrame.to_csv`"""
        self.data.to_csv(path_or_buf=path, index=False, **kwargs)

    def sort(self, by, ascending=False):
        """Sort the rows in :attr:`data` according to the values of a column

        Parameters
        ----------
        by: str, or list of str
            The column name(s) by which to sort the rows of :attr:`data`
        ascending: boolean, default=False
            The direction in which to sort the rows of :attr:`data`"""
        self.data.sort_values(by=by, axis=0, inplace=True, ascending=ascending)


class GlobalLeaderboard(Leaderboard):
    def add_entry(self, experiment, **kwargs):
        """Add an entry row to :attr:`Leaderboard.data` (pandas.DataFrame). This method also handles
        column conflicts to an extent

        Parameters
        ----------
        experiment: Instance of :class:`experiments.BaseExperiment` descendant
            An Experiment instance for which a leaderboard entry row should be added
        **kwargs: Dict
            Extra keyword arguments"""
        final_evaluations = experiment.last_evaluation_results
        entry_columns, entry_data = [], []
        # TODO: Resolve cases where `data` contains an aliased column for a metric, but the current experiment uses the
        # TODO: ... standard metric name. EX) 'oof_roc' vs 'oof_roc_auc_score' - They should be considered the same - Use alias
        evaluation_columns, evaluation_values = list(
            zip(*evaluations_to_columns(final_evaluations))
        )
        entry_columns.extend(evaluation_columns)
        entry_data.extend(evaluation_values)

        identifier_cols = [
            "experiment_id",
            "hyperparameter_key",
            "cross_experiment_key",
            "algorithm_name",
        ]
        entry_columns.extend(identifier_cols)
        for id_col in identifier_cols:
            val = getattr(experiment, id_col)
            if id_col in ["hyperparameter_key", "cross_experiment_key"]:
                val = val.key
            entry_data.append(val)

        entry_columns.append("experiment_#")
        entry_data.append(self.data.shape[0])
        identifier_cols.append("experiment_#")

        entry = pd.DataFrame(data=[entry_data], columns=entry_columns)

        self.data = self.data.append(entry, ignore_index=True)[
            combine_column_order(self.data, entry, both_cols=identifier_cols)
        ]


# class AlgorithmLeaderboard(Leaderboard):
#     pass


# class EnvironmentLeaderboard(Leaderboard):
#     pass


def evaluations_to_columns(evaluation):
    """Convert the results of :meth:`metrics.ScoringMixIn.evaluate` to a pd.DataFrame-ready format

    Parameters
    ----------
    evaluation: dict of OrderedDicts
        The result of consecutive calls to :meth:`metrics.ScoringMixIn.evaluate` for all given
        dataset types

    Returns
    -------
    column_metrics: list of pairs
        A pair for each data_type-metric combination, where the first item is the key, and the
        second is the metric value

    Examples
    --------
    >>> from collections import OrderedDict
    >>> evaluations_to_columns({
    ...     'in_fold': None,
    ...     'holdout': OrderedDict([('roc_auc_score', 0.9856), ('f1_score', 0.9768)]),
    ...     'oof': OrderedDict([('roc_auc_score', 0.9634)])
    ... })
    [['oof_roc_auc_score', 0.9634], ['holdout_roc_auc_score', 0.9856], ['holdout_f1_score', 0.9768]]
    """
    data_types = ["oof", "holdout", "in_fold"]
    column_metrics = []

    for data_type in data_types:
        if evaluation[data_type] is not None:
            for metric_key, metric_value in evaluation[data_type].items():
                column_metrics.append([f"{data_type}_{metric_key}", metric_value])

    return column_metrics


def combine_column_order(df_1, df_2, both_cols=None):
    """Determine the sort order for the combined columns of two DataFrames

    Parameters
    ----------
    df_1: pd.DataFrame
        The first DataFrame, whose columns will be sorted. Columns unique to `df_1` will be sorted
        before those of `df_2`
    df_2: pd.DataFrame
        The second DataFrame, whose columns will be sorted. Columns unique to `df_2` will be sorted
        after those of `df_1`
    both_cols: list, or None, default=None
        If list, the column names that should be common to both DataFrames and placed last in the
        sort order

    Returns
    -------
    combined_cols: list of strings
        The result of combining and sorting column names from `df_1`, and `df_2`

    Examples
    --------
    >>> df_1 = pd.DataFrame(columns=['A', 'B', 'C', 'Common_1', 'Common_2'])
    >>> df_2 = pd.DataFrame(columns=['A', 'D', 'E', 'Common_1', 'Common_2'])
    >>> combine_column_order(df_1, df_2, both_cols=['Common_1', 'Common_2'])
    ['A', 'B', 'C', 'D', 'E', 'Common_1', 'Common_2']
    >>> combine_column_order(df_1, df_2, both_cols=None)
    ['A', 'Common_1', 'Common_2', 'B', 'C', 'D', 'E']
    """
    both_cols = both_cols or []
    df_1_cols = [_ for _ in list(df_1.columns) if _ not in both_cols]
    df_2_cols = [_ for _ in list(df_2.columns) if _ not in both_cols]

    common_cols = [_ for _ in df_1_cols if _ in df_2_cols]
    unique_cols_1 = [_ for _ in df_1_cols if _ not in df_2_cols]
    unique_cols_2 = [_ for _ in df_2_cols if _ not in df_1_cols]

    combined_cols = common_cols + unique_cols_1 + unique_cols_2 + both_cols
    return combined_cols


if __name__ == "__main__":
    pass
