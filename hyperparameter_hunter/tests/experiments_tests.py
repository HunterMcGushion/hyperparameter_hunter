##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.experiments import StandardCVExperiment, RepeatedCVExperiment
from hyperparameter_hunter.utils.test_utils import equals_suite, exception_suite, format_suites, get_module

##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
import numpy as np
import pandas as pd
from unittest import TestCase, TestSuite, TextTestRunner

##################################################
# Import Learning Assets
##################################################
# from sklearn.metrics import roc_auc_score


class TestCrossValidationExperimentInitialization(TestCase):
    valid_initialization_tests = {

    }

    type_error_tests = {

    }

    value_error_tests = {

    }

    # TODO: Other tests for errors thrown by initialization of StandardCVExperiment

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=2).run)
        self.suite = partial(exception_suite, StandardCVExperiment)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name, target):
        cases, keys = format_suites(suite_group, group_name)
        # FLAG: Add formatting of cases
        # cases = [dict(obj=_[0], framework=_[1]) for _ in cases]
        # FLAG: Add formatting of cases
        targets = [target] * len(cases)
        return cases, targets, keys

    ##################################################
    # Build Suites for Test Cases
    ##################################################
    def do_valid_initialization_tests(self):
        cases, targets, keys = self.prep(self.valid_initialization_tests, 'valid_initialization_{}_', None)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_type_error_tests(self):
        cases, targets, keys = self.prep(self.type_error_tests, 'type_error_{}_', False)
        self.run_suite(self.suite(cases, targets, keys, self.module))

    def do_value_error_tests(self):
        cases, targets, keys = self.prep(self.value_error_tests, 'value_error_{}_', False)
        self.run_suite(self.suite(cases, targets, keys, self.module))


class TestCrossValidationExperimentResult(TestCase):
    pass
