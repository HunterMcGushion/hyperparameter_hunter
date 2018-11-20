##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.library_helpers.keras_helper import keras_callback_to_dict
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.boltons_utils import remap, get_path
from hyperparameter_hunter.utils.optimization_utils import (
    get_ids_by,
    get_scored_params,
    filter_by_space,
    filter_by_guidelines,
)

##################################################
# Import Miscellaneous Assets
##################################################
from copy import deepcopy


def finder_selector(module_name):
    """Selects the appropriate ResultFinder class to use for `module_name`

    Parameters
    ----------
    module_name: String
        The module from whence the algorithm being used came

    Returns
    -------
    :class:`ResultFinder`, or one of its descendants"""
    if module_name.lower() == "keras":
        return KerasResultFinder
    else:
        return ResultFinder


class ResultFinder:
    def __init__(
        self,
        algorithm_name,
        module_name,
        cross_experiment_key,
        target_metric,
        hyperparameter_space,
        leaderboard_path,
        descriptions_dir,
        model_params,
        sort=None,  # TODO: Unfinished - To be used in `_get_scored_params`/`_get_ids`
    ):
        """Locate saved Experiments that are compatible with the given constraints

        Parameters
        ----------
        algorithm_name: String
            Name of the algorithm whose hyperparameters are being optimized
        module_name: String
            Name of the module from whence the algorithm being used came
        cross_experiment_key: String
            The cross_experiment_key produced by currently active :class:`environment.Environment`
        target_metric: Tuple
            A path denoting the metric to be used. The first value should be one of ['oof',
            'holdout', 'in_fold'], and the second value should be the name of a metric supplied in
            :attr:`environment.Environment.metrics_params`
        hyperparameter_space: :class:`space.Space`
            Hyperparameter search space constraints
        leaderboard_path: String
            Path to a leaderboard file, whose listed Experiments will be tested for compatibility
        descriptions_dir: String
            Path to a directory containing the description files of saved Experiments
        model_params: Dict
            Concrete hyperparameters for the model. Common keys include 'model_init_params', and
            'model_extra_params', both of which can be pointers to dicts of hyperparameters
        sort: "target_asc", "target_desc", "chronological", "reverse_chronological", int
            How to sort the experiment results that fit within the given constraints

            * "target_asc": Sort from experiments with the lowest value for `target_metric` to
              those with the greatest
            * "target_desc": Sort from experiments with the highest value for `target_metric` to
              those with the lowest
            * "chronological": Sort from oldest experiments to newest
            * "reverse_chronological": Sort from newest experiments to oldest
            * int: Random seed with which to shuffle experiments"""
        self.algorithm_name = algorithm_name
        self.module_name = module_name
        self.cross_experiment_key = cross_experiment_key
        self.target_metric = target_metric
        self.hyperparameter_space = hyperparameter_space
        self.leaderboard_path = leaderboard_path
        self.descriptions_dir = descriptions_dir
        self.model_params = model_params
        self.sort = sort  # TODO: Unfinished - To be used in `_get_scored_params`/`_get_ids`

        self.experiment_ids = []
        self.hyperparameters_and_scores = []
        self.similar_experiments = []

    def find(self):
        """Execute full result-finding workflow"""
        self._get_ids()
        G.debug_(f"Experiments matching cross-experiment key/algorithm: {len(self.experiment_ids)}")
        self._get_scored_params()
        self._filter_by_space()
        G.debug_(f"Experiments fitting in the given space: {len(self.hyperparameters_and_scores)}")

        if self.module_name == "keras":
            multi_targets = [("model_init_params", "compile_params", "optimizer")]
            if multi_targets[0] in self.hyperparameter_space.names():
                self._filter_by_guidelines_multi(multi_targets[0])
            else:
                self._filter_by_guidelines()
        else:
            self._filter_by_guidelines()
        G.debug_(f"Experiments matching current guidelines: {len(self.similar_experiments)}")

    def _get_ids(self):
        """Get ids of Experiments matching :attr:`algorithm_name` and :attr:`cross_experiment_key`"""
        # TODO: If `sort`-ing chronologically, can use the "experiment_#" column in leaderboard
        self.experiment_ids = get_ids_by(
            leaderboard_path=self.leaderboard_path,
            algorithm_name=self.algorithm_name,
            cross_experiment_key=self.cross_experiment_key,
            hyperparameter_key=None,
        )

    def _get_scored_params(self):
        """For all :attr:`experiment_ids`, add a tuple of the Experiment's hyperparameters, and its
        :attr:`target_metric` value"""
        for _id in self.experiment_ids:
            # TODO: Receive `description` from `get_scored_params` and extract whatever value is required by :attr:`sort`
            vals = get_scored_params(f"{self.descriptions_dir}/{_id}.json", self.target_metric)
            self.hyperparameters_and_scores.append(vals + (_id,))

    def _filter_by_space(self):
        """Remove any elements of :attr:`hyperparameters_and_scores` whose values are declared in
        :attr:`hyperparameter_space`, but do not fit within the space constraints"""
        self.hyperparameters_and_scores = filter_by_space(
            self.hyperparameters_and_scores, self.hyperparameter_space
        )

    def _filter_by_guidelines(self, model_params=None):
        """Remove any elements of :attr:`hyperparameters_and_scores` whose values are not declared
        in :attr:`hyperparameter_space` but are provided in :attr:`model_params` that do not match
        the values in :attr:`model_params`

        Parameters
        ----------
        model_params: Dict, default=:attr:`model_params`
            If not None, a dict of model parameters that closely resembles :attr:`model_params`"""
        self.similar_experiments.extend(
            filter_by_guidelines(
                self.hyperparameters_and_scores,
                self.hyperparameter_space,
                **(model_params or self.model_params),
            )
        )

    def _filter_by_guidelines_multi(self, location):
        """Helper to filter by guidelines when one of the guideline hyperparameters is directly
        affected by a hyperparameter that is given as a space choice

        Parameters
        ----------
        location: Tuple
            Location of the hyperparameter space choice that affects the acceptable guideline values
            of a particular hyperparameter. In other words, this is the path of a hyperparameter,
            which, if changed, would change the expected default value of another hyperparameter

        Notes
        -----
        This is used for Keras Experiments when the `optimizer` value in a model's `compile_params`
        is given as a hyperparameter space choice. Each possible value of `optimizer` prescribes
        different default values for the `optimizer_params` argument, so special measures need to be
        taken to ensure the correct Experiments are declared to fit within the constraints"""
        _model_params = deepcopy(self.model_params)

        if location == ("model_init_params", "compile_params", "optimizer"):
            from keras.optimizers import get as k_opt_get

            update_location = ("model_init_params", "compile_params", "optimizer_params")
            allowed_values = get_path(_model_params, location).bounds

            #################### Handle First Value (Dummy) ####################
            self._filter_by_guidelines()
            allowed_values = allowed_values[1:]

            #################### Handle Remaining Values ####################
            for allowed_val in allowed_values:
                updated_value = k_opt_get(allowed_val).get_config()

                def _visit(path, key, value):
                    """If `path` + `key` == `update_location`, return default for this choice. Else,
                    default_visit"""
                    if path + (key,) == update_location:
                        return (key, updated_value)
                    return (key, value)

                self._filter_by_guidelines(model_params=remap(_model_params, visit=_visit))

            self.similar_experiments = sorted(
                self.similar_experiments, key=lambda _: _[1], reverse=True
            )
        else:
            raise ValueError("Received unhandled location: {}".format(location))


class KerasResultFinder(ResultFinder):
    def __init__(
        self,
        algorithm_name,
        module_name,
        cross_experiment_key,
        target_metric,
        hyperparameter_space,
        leaderboard_path,
        descriptions_dir,
        model_params,
        sort=None,  # TODO: Unfinished - To be used in `_get_scored_params`/`_get_ids`
    ):
        """ResultFinder for locating saved Keras Experiments compatible with the given constraints

        Parameters
        ----------
        algorithm_name: String
            The name of the algorithm whose hyperparameters are being optimized
        module_name: String
            The name of the module from whence the algorithm being used came
        cross_experiment_key: String
            The cross_experiment_key produced by the current :class:`environment.Environment`
        target_metric: Tuple
            Path denoting the metric to be used. The first value should be one of ['oof',
            'holdout', 'in_fold'], and the second value should be the name of a metric supplied in
            :attr:`environment.Environment.metrics_params`
        hyperparameter_space: :class:`space.Space`
            Hyperparameter search space constraints
        leaderboard_path: String
            Path to a leaderboard file, whose listed Experiments will be tested for compatibility
        descriptions_dir: String
            Path to a directory containing the description files of saved Experiments
        model_params: Dict
            Concrete hyperparameters for the model. Common keys include 'model_init_params', and
            'model_extra_params', both of which can be pointers to dicts of hyperparameters
        sort: "target_asc", "target_desc", "chronological", "reverse_chronological", int
            How to sort the experiment results that fit within the given constraints

            * "target_asc": Sort from experiments with the lowest value for `target_metric` to
              those with the greatest
            * "target_desc": Sort from experiments with the highest value for `target_metric` to
              those with the lowest
            * "chronological": Sort from oldest experiments to newest
            * "reverse_chronological": Sort from newest experiments to oldest
            * int: Random seed with which to shuffle experiments"""
        super().__init__(
            algorithm_name=algorithm_name,
            module_name=module_name,
            cross_experiment_key=cross_experiment_key,
            target_metric=target_metric,
            hyperparameter_space=hyperparameter_space,
            leaderboard_path=leaderboard_path,
            descriptions_dir=descriptions_dir,
            model_params=model_params,
            sort=sort,  # TODO: Unfinished - To be used in `_get_scored_params`/`_get_ids`
        )

        from keras.callbacks import Callback as BaseKerasCallback

        # noinspection PyUnusedLocal
        def _visit(path, key, value):
            """If `value` is `BaseKerasCallback`, return dict representation. Else default_visit"""
            if isinstance(value, BaseKerasCallback):
                return (key, keras_callback_to_dict(value))
            return (key, value)

        self.model_params = remap(self.model_params, visit=_visit)

        try:
            del self.model_params["model_extra_params"]["params"]
        except KeyError:
            pass
