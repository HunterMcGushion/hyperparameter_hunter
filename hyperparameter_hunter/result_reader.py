##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.library_helpers.keras_helper import keras_callback_to_dict
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.boltons_utils import remap
from hyperparameter_hunter.utils.optimization_utils import get_ids_by, get_scored_params, filter_by_space, filter_by_guidelines


def finder_selector(module_name):
    """Selects the appropriate ResultFinder class to use for `module_name`

    Parameters
    ----------
    module_name: String
        The module from whence the algorithm being used came

    Returns
    -------
    :class:`ResultFinder`, or one of its descendants"""
    if module_name.lower() == 'keras':
        return KerasResultFinder
    else:
        return ResultFinder


class ResultFinder():
    def __init__(
            self, algorithm_name, cross_experiment_key, target_metric, hyperparameter_space, leaderboard_path, descriptions_dir,
            model_params,
    ):
        self.algorithm_name = algorithm_name
        self.cross_experiment_key = cross_experiment_key
        self.target_metric = target_metric
        self.hyperparameter_space = hyperparameter_space
        self.leaderboard_path = leaderboard_path
        self.descriptions_dir = descriptions_dir
        self.model_params = model_params

        self.experiment_ids = []
        self.hyperparameters_and_scores = []
        self.similar_experiments = []

    def find(self):
        self._get_ids()
        G.debug_(F'Experiments found with matching cross-experiment key and algorithm: {len(self.experiment_ids)}')
        self._get_scored_params()
        self._filter_by_space()
        G.debug_(F'Experiments whose hyperparameters fit in the currently defined space: {len(self.hyperparameters_and_scores)}')
        self._filter_by_guidelines()
        G.debug_(F'Experiments whose hyperparameters match the current guidelines: {len(self.hyperparameters_and_scores)}')

    def _get_ids(self):
        self.experiment_ids = get_ids_by(
            leaderboard_path=self.leaderboard_path, algorithm_name=self.algorithm_name,
            cross_experiment_key=self.cross_experiment_key, hyperparameter_key=None
        )

    def _get_scored_params(self):
        for experiment_id in self.experiment_ids:
            self.hyperparameters_and_scores.append(
                get_scored_params('{}/{}.json'.format(self.descriptions_dir, experiment_id), self.target_metric)
            )

    def _filter_by_space(self):
        self.hyperparameters_and_scores = filter_by_space(self.hyperparameters_and_scores, self.hyperparameter_space)

    def _filter_by_guidelines(self):
        self.hyperparameters_and_scores = filter_by_guidelines(
            self.hyperparameters_and_scores, self.hyperparameter_space, **self.model_params
        )


class KerasResultFinder(ResultFinder):
    def __init__(
            self, algorithm_name, cross_experiment_key, target_metric, hyperparameter_space, leaderboard_path, descriptions_dir,
            model_params,
    ):
        super().__init__(
            algorithm_name=algorithm_name, cross_experiment_key=cross_experiment_key, target_metric=target_metric,
            hyperparameter_space=hyperparameter_space, leaderboard_path=leaderboard_path, descriptions_dir=descriptions_dir,
            model_params=model_params,
        )

        from keras.callbacks import Callback as BaseKerasCallback

        # noinspection PyUnusedLocal
        def _visit(path, key, value):
            if isinstance(value, BaseKerasCallback):
                return (key, keras_callback_to_dict(value))
            return (key, value)

        self.model_params = remap(self.model_params, visit=_visit)

        try:
            del self.model_params['model_extra_params']['params']
        except KeyError:
            pass


def _execute():
    pass


if __name__ == '__main__':
    _execute()
