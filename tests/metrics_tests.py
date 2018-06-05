##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import metrics
from hyperparameter_hunter.utils.test_utils import exception_suite, format_suites, get_module

##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
import sys
from unittest import TestCase, TestSuite, TextTestRunner, TextTestResult

##################################################
# Import Learning Assets
##################################################
from sklearn.metrics import roc_auc_score


class EmptyClass(object):
    pass


class TestScoringMixInInitialization(TestCase):
    _metrics_map = dict(roc_auc_score=roc_auc_score)
    _in_fold, _oof, _holdout = 'all', 'all', 'all'
    empty_class, empty_func, empty_tuple = EmptyClass(), lambda _: _, tuple()

    valid_initialization_tests = dict(
        metrics_map=[
            (_metrics_map, _in_fold, _oof, _holdout),
            ({'1': roc_auc_score}, _in_fold, _oof, _holdout),
            (dict(my_roc_auc=roc_auc_score, roc_auc_score=None), _in_fold, _oof, _holdout),
            (dict(foo=roc_auc_score, roc_auc_score=None), _in_fold, _oof, _holdout),
            (dict(foo=roc_auc_score, roc_auc_score=None, foo_2='roc_auc_score'), _in_fold, _oof, _holdout),
            (['roc_auc_score'], _in_fold, _oof, _holdout),
            (['f1_score', 'accuracy_score', 'roc_auc_score'], _in_fold, _oof, _holdout),
        ],
        metrics_lists=[
            (_metrics_map, _in_fold, None, None),
            (_metrics_map, None, None, None),
            (_metrics_map, ['roc_auc_score'], _oof, _holdout),
            (['f1_score', 'accuracy_score', 'roc_auc_score'], ['f1_score'], ['accuracy_score'], ['roc_auc_score']),
            (['f1_score', 'accuracy_score', 'roc_auc_score'], ['f1_score'], _oof, _holdout),
            #################### Below cases will result in no metrics being calculated at all ####################
            (dict(), None, None, None),
            ([], None, None, None),
        ],
    )
    type_error_tests = dict(
        metrics_map=[
            ('foo', _in_fold, _oof, _holdout),
            (1, _in_fold, _oof, _holdout),
            (None, _in_fold, _oof, _holdout),
            # (['f1_score', 'accuracy_score', 'roc_auc_score'], _in_fold, _oof, _holdout),  # This correctly fails
            (empty_class, _in_fold, _oof, _holdout),
            (empty_func, _in_fold, _oof, _holdout),
            (empty_tuple, _in_fold, _oof, _holdout),
        ],
        metrics_map_key=[
            ({1: roc_auc_score}, _in_fold, _oof, _holdout),
            ({empty_class: roc_auc_score}, _in_fold, _oof, _holdout),
            ({empty_func: roc_auc_score}, _in_fold, _oof, _holdout),
            ({empty_tuple: roc_auc_score}, _in_fold, _oof, _holdout),
        ],
        metrics_map_value=[
            ({'roc_auc_score': 1}, _in_fold, _oof, _holdout),
            ({'roc_auc_score': 1.2}, _in_fold, _oof, _holdout),
            ({'roc_auc_score': ['a', 'b']}, _in_fold, _oof, _holdout),
            ({'roc_auc_score': dict(a=1, b=2)}, _in_fold, _oof, _holdout),
            ({'roc_auc_score': ('a', 'b')}, _in_fold, _oof, _holdout),
        ],
        metrics_lists=[
            (_metrics_map, 'foo', _oof, _holdout),
            (_metrics_map, _in_fold, 'foo', _holdout),
            (_metrics_map, _in_fold, _oof, 'foo'),
            (_metrics_map, empty_class, _oof, _holdout),
            (_metrics_map, empty_func, _oof, _holdout),
            (_metrics_map, ('a', 'b'), _oof, _holdout),
            (_metrics_map, 1, _oof, _holdout),
            (_metrics_map, 1.2, _oof, _holdout),
            (_metrics_map, 1.2, 'foo', empty_func),
        ],
        metrics_lists_values=[
            (_metrics_map, [1], _oof, _holdout),
            (_metrics_map, _in_fold, [1.2], _holdout),
            (_metrics_map, _in_fold, _oof, [empty_func]),
            (_metrics_map, [empty_class], _oof, _holdout),
            (_metrics_map, [empty_tuple], _oof, _holdout),
            (_metrics_map, [['roc_auc']], _oof, _holdout),
            (_metrics_map, [dict(a=1, b=2)], 1, 1),
            (_metrics_map, [None], _oof, _holdout),
        ]
    )
    attribute_error_tests = dict(
        not_in_sklearn_metrics=[
            (dict(roc_auc='foo'), _in_fold, _oof, _holdout),
            (dict(foo=None), _in_fold, _oof, _holdout),
            (['foo'], _in_fold, _oof, _holdout),
            (['roc_auc', 'foo'], _in_fold, _oof, _holdout),
        ]
    )
    key_error_tests = dict(
        not_in_metrics_map=[
            (_metrics_map, ['foo'], _oof, _holdout),
            (_metrics_map, _in_fold, ['foo'], _holdout),
            (_metrics_map, _in_fold, _oof, ['foo']),
            (_metrics_map, ['roc_auc', 'foo'], _oof, _holdout),
            (dict(), ['roc_auc'], _oof, _holdout),
            (dict(), _in_fold, ['roc_auc'], _holdout),
            ([], _in_fold, _oof, ['roc_auc']),
        ]
    )

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=0).run)
        self.suite = partial(exception_suite, metrics.ScoringMixIn)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name, target):
        cases, keys = format_suites(suite_group, group_name)
        cases = [dict(metrics_map=_[0], in_fold=_[1], oof=_[2], holdout=_[3]) for _ in cases]
        targets = [target] * len(cases)
        return cases, targets, keys

    def do_valid_initialization_tests(self):
        cases, targets, keys = self.prep(self.valid_initialization_tests, 'valid_initialization_{}_', None)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_type_error_tests(self):
        cases, targets, keys = self.prep(self.type_error_tests, 'type_error_{}_', TypeError)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_attribute_error_tests(self):
        cases, targets, keys = self.prep(self.attribute_error_tests, 'attribute_error_{}_', AttributeError)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_key_error_tests(self):
        cases, targets, keys = self.prep(self.key_error_tests, 'key_error_{}_', KeyError)
        self.run_suite(self.suite(cases, targets, keys, self.module))
