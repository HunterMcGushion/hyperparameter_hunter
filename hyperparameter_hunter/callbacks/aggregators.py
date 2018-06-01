##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseAggregatorCallback
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
from datetime import datetime
import numpy as np


class AggregatorTimes(BaseAggregatorCallback):
    def __init__(self):
        """Uncalled - See the 'Notes' section in the documentation of :class:`callbacks.bases.BaseCallback` for details"""
        self.stat_aggregates = dict()
        self._rep = None
        self._fold = None
        self._run = None
        super().__init__()

    def on_experiment_start(self):
        self.stat_aggregates.setdefault('times', dict(runs=[], folds=[], reps=[], total_elapsed=None, start=None, end=None))
        self.stat_aggregates['times']['start'] = str(datetime.now())
        self.stat_aggregates['times']['total_elapsed'] = datetime.now()
        super().on_experiment_start()

    def on_repetition_start(self):
        self.stat_aggregates['times']['reps'].append(datetime.now())
        super().on_repetition_start()

    def on_fold_start(self):
        self.stat_aggregates['times']['folds'].append(datetime.now())
        super().on_fold_start()

    def on_run_start(self):
        self.stat_aggregates['times']['runs'].append(datetime.now())
        super().on_run_start()

    def on_run_end(self):
        self.__to_elapsed('runs')
        super().on_run_end()

    def on_fold_end(self):
        self.__to_elapsed('folds')
        super().on_fold_end()

    def on_repetition_end(self):
        self.__to_elapsed('reps')
        super().on_repetition_end()

    def on_experiment_end(self):
        #################### Reshape Run/Fold Aggregates to be of Proper Dimensions ####################
        runs_shape, folds_shape = (self._rep + 1, self._fold + 1, self._run + 1), (self._rep + 1, self._fold + 1)
        self.stat_aggregates['times']['runs'] = np.reshape(self.stat_aggregates['times']['runs'], runs_shape).tolist()
        self.stat_aggregates['times']['folds'] = np.reshape(self.stat_aggregates['times']['folds'], folds_shape).tolist()

        self.stat_aggregates['times']['end'] = str(datetime.now())
        self.__to_elapsed('total_elapsed')
        super().on_experiment_end()

    def __to_elapsed(self, agg_key):
        start_val = self.stat_aggregates['times'][agg_key]
        if isinstance(start_val, list):
            self.stat_aggregates['times'][agg_key][-1] = (datetime.now() - start_val[-1]).total_seconds()
        else:
            self.stat_aggregates['times'][agg_key] = (datetime.now() - start_val).total_seconds()


class AggregatorEvaluations(BaseAggregatorCallback):
    def __init__(self):
        """Uncalled - See the 'Notes' section in the documentation of :class:`callbacks.bases.BaseCallback` for details"""
        self.stat_aggregates = dict()
        self.last_evaluation_results = dict(in_fold=None, oof=None, holdout=None)
        self._rep = None
        self._fold = None
        self._run = None
        super().__init__()

    def on_run_end(self):
        #################### Initialize Evaluations Aggregator ####################
        if len(self.stat_aggregates.setdefault('evaluations', {}).keys()) == 0:
            for dataset_key, metric_results in self.last_evaluation_results.items():
                if metric_results is not None:
                    self.stat_aggregates['evaluations'].update({'{}_{}'.format(dataset_key, metric_key): dict(
                        runs=[], folds=[], reps=[], final=None
                    ) for metric_key in metric_results.keys()})

        #################### Update Evaluations for Run ####################
        for agg_key, agg_val in self.stat_aggregates['evaluations'].items():
            agg_val['runs'].append(self.__loop_helper(agg_key))

        super().on_run_end()

    def on_fold_end(self):
        for agg_key, agg_val in self.stat_aggregates['evaluations'].items():
            agg_val['folds'].append(self.__loop_helper(agg_key))
        super().on_fold_end()

    def on_repetition_end(self):
        for agg_key, agg_val in self.stat_aggregates['evaluations'].items():
            agg_val['reps'].append(self.__loop_helper(agg_key))
        super().on_repetition_end()

    def on_experiment_end(self):
        for agg_key, agg_val in self.stat_aggregates['evaluations'].items():
            agg_val['final'] = self.__loop_helper(agg_key)

            #################### Reshape Run/Fold Aggregates to be of Proper Dimensions ####################
            agg_val['runs'] = np.reshape(agg_val['runs'], (self._rep + 1, self._fold + 1, self._run + 1)).tolist()
            agg_val['folds'] = np.reshape(agg_val['folds'], (self._rep + 1, self._fold + 1)).tolist()

        super().on_experiment_end()

    def __loop_helper(self, agg_key):
        for dataset_key, metric_results in self.last_evaluation_results.items():
            if metric_results is not None:
                for metric_key, metric_value in metric_results.items():
                    if agg_key == '{}_{}'.format(dataset_key, metric_key):
                        return metric_value
        return None


class AggregatorOOF(BaseAggregatorCallback): pass  # TODO: Record "full_oof_predictions"


class AggregatorHoldout(BaseAggregatorCallback): pass  # TODO: Record "full_holdout_predictions"


class AggregatorTest(BaseAggregatorCallback): pass  # TODO: Record "full_test_predictions"


class AggregatorLosses(BaseAggregatorCallback): pass


class AggregatorEpochsElapsed(BaseAggregatorCallback):
    # TODO: Make this class work with RepeatedCVExperiments - Currently only considers fold, run
    def __init__(self):
        """Uncalled - See the 'Notes' section in the documentation of :class:`callbacks.bases.BaseCallback` for details"""
        self.stat_aggregates = dict()
        self.model = None
        self._rep = None
        self._fold = None
        self._run = None
        super().__init__()

    def on_run_end(self):
        # G.log('AggregatorEpochsElapsed.on_run_end()')

        rep_key, fold_key = 'rep_{}'.format(self._rep), 'fold_{}'.format(self._fold)

        #################### Initialize stat_aggregates aggregator group ####################
        if self.model.epochs_elapsed is not None:
            self.stat_aggregates.setdefault('epochs_elapsed', OrderedDict())

            #################### Update run value ####################
            self.stat_aggregates['epochs_elapsed'].setdefault(fold_key, dict(
                run_values=[], simple_average=None
            ))['run_values'].append(self.model.epochs_elapsed)

        super().on_run_end()

    def on_fold_end(self):
        # G.log('AggregatorEpochsElapsed.on_fold_end()')

        rep_key, fold_key = 'rep_{}'.format(self._rep), 'fold_{}'.format(self._fold)

        #################### Simple Average of Fold's Runs ####################
        try:
            self.stat_aggregates['epochs_elapsed'][fold_key]['simple_average'] = np.average(
                self.stat_aggregates['epochs_elapsed'][fold_key]['run_values']
            )
        except KeyError:
            # self.stat_aggregates does not have 'epochs_elapsed' key - epochs never recorded in first place
            pass
        except TypeError:
            G.warn('\n'.join([
                'TypeError encountered when averaging stat_aggregates[{}][{}]:'.format('epochs_elapsed', fold_key),
                '\tValues: {}'.format(self.stat_aggregates['epochs_elapsed'][fold_key]['run_values']),
                '\tTypes: {}'.format([type(_) for _ in self.stat_aggregates['epochs_elapsed'][fold_key]['run_values']]),
                'If the above values are numbers and you want them averaged, fix me ASAP! If not, ignore me'
            ]))

        super().on_fold_end()


##################################################
# Old Callbacks
##################################################
class AggregatorEvaluationsToDicts(BaseAggregatorCallback):
    def __init__(self):
        """Uncalled - See the 'Notes' section in the documentation of :class:`callbacks.bases.BaseCallback` for details"""
        self.stat_aggregates = dict()
        self.last_evaluation_results = dict(in_fold=None, oof=None, holdout=None)
        self._fold = None
        self.experiment_params = None
        super().__init__()

    def on_experiment_start(self):
        self.stat_aggregates['evaluations'] = dict()
        super().on_experiment_start()

    def on_run_end(self):
        rep_key, fold_key = 'rep_{}'.format(self._rep), 'fold_{}'.format(self._fold)
        #################### Initialize stat_aggregates ####################
        if len(self.stat_aggregates['evaluations'].keys()) == 0:
            for data_type_key, data_type_values in self.last_evaluation_results.items():
                if data_type_values is None: continue

                for metric_key, metric_value in data_type_values.items():
                    agg_key = '{}_{}'.format(data_type_key, metric_key)
                    self.stat_aggregates['evaluations'][agg_key] = OrderedDict()

        #################### Update stat_aggregates ####################
        for agg_key, agg_val in self.stat_aggregates['evaluations'].items():
            value_to_append = None

            for data_type_key, data_type_values in self.last_evaluation_results.items():
                if data_type_values is None: continue

                for metric_key, metric_value in data_type_values.items():
                    if agg_key == '{}_{}'.format(data_type_key, metric_key):
                        value_to_append = metric_value
                        break  # Breaks both inner and outer loops due to below else, break
                else:
                    continue  # Continues outer loop when inner loop is not broken
                break  # Only executes if the inner loop is broken out of

            #################### Update Run Values ####################
            agg_val.setdefault(rep_key, OrderedDict())

            agg_val[rep_key].setdefault(fold_key, dict(
                run_values=[], simple_average=None
            ))['run_values'].append(value_to_append)

        super().on_run_end()

    def on_fold_end(self):
        rep_key, fold_key = 'rep_{}'.format(self._rep), 'fold_{}'.format(self._fold)

        for agg_key, agg_val in self.stat_aggregates['evaluations'].items():
            value_to_append = None

            for data_type_key, data_type_values in self.last_evaluation_results.items():
                if data_type_values is None: continue

                for metric_key, metric_value in data_type_values.items():
                    if agg_key == '{}_{}'.format(data_type_key, metric_key):
                        value_to_append = metric_value
                        break  # Breaks both inner and outer loops due to below else, break
                else:
                    continue  # Continues outer loop when inner loop is not broken
                break  # Only executes if the inner loop is broken out of

            agg_val[rep_key][fold_key]['actual_average'] = value_to_append

            #################### Simple Average of Fold's Runs ####################
            try:
                agg_val[rep_key][fold_key]['simple_average'] = np.average(agg_val[rep_key][fold_key]['run_values'])
            except TypeError:
                G.warn('\n'.join([
                    'TypeError when averaging stat_aggregates[{}][{}][{}]:'.format(agg_key, rep_key, fold_key),
                    '\tValues: {}'.format(agg_val[rep_key][fold_key]['run_values']),
                    '\tTypes: {}'.format([type(_) for _ in agg_val[rep_key][fold_key]['run_values']]),
                    'If the above values are numbers and you want them averaged, fix me ASAP! If not, ignore me'
                ]))

        super().on_fold_end()


def execute():
    pass


if __name__ == '__main__':
    execute()
