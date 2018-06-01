##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.object_frameworks.reporting_frameworks import AdvancedDisplayLayoutFrameworks
from hyperparameter_hunter.utils import framework_utils
from hyperparameter_hunter.utils.test_utils import truth_suite, format_suites, get_module

##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
from unittest import TestCase, TestSuite, TextTestRunner


class TestIsTyping(TestCase):
    valid_advanced_display_layout_tests = {
        'column_name': [
            #################### Strings ####################
            ('OOF Scores', AdvancedDisplayLayoutFrameworks.column_name),
            ('oof_roc', AdvancedDisplayLayoutFrameworks.column_name),
        ],
        'sub_column_name_entry': [
            #################### Strings ####################
            ('oof_roc', AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
            #################### Lists/tuples of strings ####################
            (['oof_roc', 'OOF ROC'], AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
            (('oof_roc',), AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
        ],
        'sub_column_names': [
            #################### Simple cases ####################
            (None, AdvancedDisplayLayoutFrameworks.sub_column_names),
            (['oof_roc', 'oof_f1'], AdvancedDisplayLayoutFrameworks.sub_column_names),
            (('oof_roc', 'oof_f1'), AdvancedDisplayLayoutFrameworks.sub_column_names),
            #################### Nested lists and tuples ####################
            ([['oof_roc'], ['oof_f1']], AdvancedDisplayLayoutFrameworks.sub_column_names),
            ([['oof_roc', 'OOF ROC'], ['oof_f1', 'OOF F1']], AdvancedDisplayLayoutFrameworks.sub_column_names),
            ((('oof_roc', 'OOF ROC'), ('oof_f1', 'OOF F1')), AdvancedDisplayLayoutFrameworks.sub_column_names),
            ((('oof_roc', 'OOF ROC'),), AdvancedDisplayLayoutFrameworks.sub_column_names),
            ((['oof_roc', 'OOF ROC'], ['oof_f1', 'OOF F1']), AdvancedDisplayLayoutFrameworks.sub_column_names),
            #################### Nested lists and tuples of varying lengths ####################
            ([['oof_roc', 'OOF ROC'], ['oof_f1']], AdvancedDisplayLayoutFrameworks.sub_column_names),
            #################### Mixing nested lists and tuples ####################
            ([['oof_roc', 'OOF ROC'], ('oof_f1', 'OOF F1')], AdvancedDisplayLayoutFrameworks.sub_column_names),
            ((['oof_roc', 'OOF ROC'], ('oof_f1', 'OOF F1')), AdvancedDisplayLayoutFrameworks.sub_column_names),
            ([('oof_roc', 'OOF ROC'), ['oof_f1']], AdvancedDisplayLayoutFrameworks.sub_column_names),
            ([('oof_roc', 'OOF ROC'), 'oof_f1'], AdvancedDisplayLayoutFrameworks.sub_column_names),
        ],
        'sub_column_min_sizes': [
            #################### Simple cases ####################
            (None, AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
            (3, AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
            # ({3, 4}, AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),  # This correctly fails
            #################### Lists/tuples of ints ####################
            ([3], AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
            ((3, 4), AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
        ]
    }
    invalid_advanced_display_layout_tests = {
        'column_name': [
            #################### Iterables ####################
            (['OOF Scores'], AdvancedDisplayLayoutFrameworks.column_name),
            (('oof_roc',), AdvancedDisplayLayoutFrameworks.column_name),
            ({'oof_roc'}, AdvancedDisplayLayoutFrameworks.column_name),
            #################### Non-strings ####################
            (None, AdvancedDisplayLayoutFrameworks.column_name),
            (1, AdvancedDisplayLayoutFrameworks.column_name),
        ],
        'sub_column_name_entry': [
            #################### Non-string values ####################
            (3.14, AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
            (None, AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
            #################### Non-list/tuple iterables ####################
            (dict(oof_roc=1), AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
            ({'oof_roc'}, AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
            ({3.14, 'oof_roc'}, AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
            #################### Lists/tuples containing non-string values ####################
            (['oof_roc', 1], AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
            (('oof_roc', None), AdvancedDisplayLayoutFrameworks.sub_column_name_entry),
        ],
        'sub_column_names': [
            #################### Non-list/tuple iterables ####################
            ({'oof_roc', 'oof_f1'}, AdvancedDisplayLayoutFrameworks.sub_column_names),
            ([{'oof_roc', 'oof_f1'}], AdvancedDisplayLayoutFrameworks.sub_column_names),
            (({'oof_roc', 'oof_f1'},), AdvancedDisplayLayoutFrameworks.sub_column_names),
            ([{'oof_roc': 'OOF ROC'}, {'oof_f1': 'OOF F1'}], AdvancedDisplayLayoutFrameworks.sub_column_names),
            #################### Valid iterables containing non-string values ####################
            (['oof_roc', None], AdvancedDisplayLayoutFrameworks.sub_column_names),
            ([('oof_roc', 'OOF ROC'), ['oof_f1', ['OOF F1']]], AdvancedDisplayLayoutFrameworks.sub_column_names),
            ([('oof_roc', 'OOF ROC'), ['oof_f1', None]], AdvancedDisplayLayoutFrameworks.sub_column_names),
            ([('oof_roc', 'OOF ROC'), ['oof_f1', 3.14]], AdvancedDisplayLayoutFrameworks.sub_column_names),
        ],
        'sub_column_min_sizes': [
            #################### Invalid values ####################
            ('foo', AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
            (3.14, AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
            #################### Non-list/tuple iterables ####################
            ({3, 4}, AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
            ({'oof_roc': 3}, AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
            #################### Valid iterables containing invalid values ####################
            (['foo'], AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
            ((3, None), AdvancedDisplayLayoutFrameworks.sub_column_min_sizes),
        ]
    }

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=0).run)
        self.suite = partial(truth_suite, framework_utils.is_typing)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name, target):
        cases, keys = format_suites(suite_group, group_name)
        cases = [dict(obj=_[0], framework=_[1]) for _ in cases]
        targets = [target] * len(cases)
        return cases, targets, keys

    ##################################################
    # Build Suites for Test Cases
    ##################################################
    def do_valid_advanced_display_layout_tests(self):
        cases, targets, keys = self.prep(self.valid_advanced_display_layout_tests, 'valid_advanced_display_layout_{}_', True)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_invalid_advanced_display_layout_tests(self):
        cases, targets, keys = self.prep(self.invalid_advanced_display_layout_tests, 'invalid_advanced_display_layout_{}_', False)
        self.run_suite(self.suite(cases, targets, keys, self.module))


class TestLikeFramework(TestCase):
    ##################################################
    # Declare Test Cases
    ##################################################
    valid_typing_instance_tests = TestIsTyping.valid_advanced_display_layout_tests
    invalid_typing_instance_tests = TestIsTyping.invalid_advanced_display_layout_tests
    valid_standard_type_tests = {
        'str': [
            ('foo', str),
            ("HELLO", str),
            ('{}'.format(1.34), str)
        ],
        'int': [
            (45, int),
        ],
        'float': [
            (3.14, float),
        ],
        'list': [
            ([], list),
            (['a', 'b', 'c'], list),
            ([3.14, 'b', 'c'], list),
            (list('a'), list),
        ],
        'tuple': [
            (tuple(), tuple),
            (('a', 'b', 'c'), tuple),
            ((3.14, 'b', 'c'), tuple),
            (tuple('a'), tuple),
        ],
        'dict': [
            (dict(a=10, b=20), dict),
            ({'a': 10, 'b': 20}, dict),
            ({_: None for _ in ['a', 'b']}, dict)
        ],
    }
    invalid_standard_type_tests = {
        'str': [
            ('foo', int),
            ('foo', float),
            ('foo', list),
            ('foo', tuple),
            ('foo', dict),
        ],
        'int': [
            (45, str),
            (45, float),
            (45, list),
            (45, tuple),
            (45, dict),
        ],
        'float': [
            (3.14, str),
            (3.14, int),
            (3.14, list),
            (3.14, tuple),
            (3.14, dict),
        ],
        'list': [
            (['a', 'b'], str),
            (['a', 'b'], int),
            (['a', 'b'], float),
            (['a', 'b'], tuple),
            (['a', 'b'], dict),
        ],
        'tuple': [
            (('a', 'b'), str),
            (('a', 'b'), int),
            (('a', 'b'), float),
            (('a', 'b'), list),
            (('a', 'b'), dict),
        ],
        'dict': [
            (dict(a=10, b=20), str),
            (dict(a=10, b=20), int),
            (dict(a=10, b=20), float),
            (dict(a=10, b=20), list),
            (dict(a=10, b=20), tuple),
        ],
    }
    valid_dict_framework_tests = {
        'simple_value': [
            (dict(foo=3.14, bar=15), dict(foo=float, bar=int)),
        ],
        'typing_instance_value': [
            (dict(
                column_name='OOF Scores',
                sub_column_names=[['oof_roc', 'ROC'], ['oof_f1', 'F1']],
                sub_column_min_sizes=[12, 12],
            ), AdvancedDisplayLayoutFrameworks.advanced_column_entry),
            (dict(
                column_name='OOF Scores',
                sub_column_names=['oof_roc', 'oof_f1'],
                sub_column_min_sizes=None,
            ), AdvancedDisplayLayoutFrameworks.advanced_column_entry),
        ],
        'dict_framework_value': [
            (
                dict(column_1=dict(column_name='OOF Scores',
                                   sub_column_names=[['oof_roc', 'ROC'], ['oof_f1', 'F1']],
                                   sub_column_min_sizes=[12, 12]),
                     column_2=dict(column_name='Holdout Scores',
                                   sub_column_names=[['holdout_roc', 'ROC'], ['holdout_f1', 'F1']],
                                   sub_column_min_sizes=[12, 12])),
                dict(column_1=AdvancedDisplayLayoutFrameworks.advanced_column_entry,
                     column_2=AdvancedDisplayLayoutFrameworks.advanced_column_entry)
            ),
        ],
    }
    invalid_dict_framework_tests = {
        'simple_value': [
            (dict(foo='I am 3.14 in a string', bar=15), dict(foo=float, bar=int)),
        ],
        'typing_instance_value': [
            (dict(
                column_name=list('OOF Scores'),
                sub_column_names=[['oof_roc', 'OOF ROC'], ['oof_f1', 'OOF F1']],
                sub_column_min_sizes=[12, 12],
            ), AdvancedDisplayLayoutFrameworks.advanced_column_entry),
            (dict(
                column_name='OOF Scores',
                sub_column_names=[3.14, 3.14],
                sub_column_min_sizes=[12, 12],
            ), AdvancedDisplayLayoutFrameworks.advanced_column_entry),
            (dict(
                column_name='OOF Scores',
                sub_column_names=['oof_roc', 'oof_f1']
            ), AdvancedDisplayLayoutFrameworks.advanced_column_entry),
        ],
        'dict_framework_value': [
            (
                dict(column_1=dict(column_name='OOF Scores',
                                   sub_column_names=[['oof_roc', 'ROC'], ['oof_f1', 'F1']],
                                   sub_column_min_sizes=[12, 12]),
                     column_2=dict(column_name='Holdout Scores',
                                   sub_column_names=[['holdout_roc', 'ROC'], ['holdout_f1', 'F1']],
                                   sub_column_min_sizes=['foo', 'bar'])),
                dict(column_1=AdvancedDisplayLayoutFrameworks.advanced_column_entry,
                     column_2=AdvancedDisplayLayoutFrameworks.advanced_column_entry)
            ),
        ],
    }

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=0).run)
        self.suite = partial(truth_suite, framework_utils.like_framework)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name, target):
        cases, keys = format_suites(suite_group, group_name)
        cases = [dict(obj=_[0], framework=_[1]) for _ in cases]
        targets = [target] * len(cases)
        return cases, targets, keys

    def do_valid_typing_instance_tests(self):
        cases, targets, keys = self.prep(self.valid_typing_instance_tests, 'valid_typing_instance_{}_', True)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_invalid_typing_instance_tests(self):
        cases, targets, keys = self.prep(self.invalid_typing_instance_tests, 'invalid_typing_instance_{}_', False)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_valid_standard_type_tests(self):
        cases, targets, keys = self.prep(self.valid_standard_type_tests, 'valid_standard_type_{}_', True)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_invalid_standard_type_tests(self):
        cases, targets, keys = self.prep(self.invalid_standard_type_tests, 'invalid_standard_type_{}_', False)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_valid_dict_framework_tests(self):
        cases, targets, keys = self.prep(self.valid_dict_framework_tests, 'valid_dict_framework_{}_', True)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_invalid_dict_framework_tests(self):
        cases, targets, keys = self.prep(self.invalid_dict_framework_tests, 'invalid_dict_framework_{}_', False)
        self.run_suite(self.suite(cases, targets, keys, self.module))
