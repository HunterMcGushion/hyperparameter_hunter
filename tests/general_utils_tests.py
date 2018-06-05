##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils import general_utils
from hyperparameter_hunter.utils.test_utils import equals_suite

###############################################
# Import Miscellaneous Assets
###############################################
from functools import partial
# import pandas as pd
from unittest import TestCase, TestSuite, TextTestRunner


class TestStandardString(TestCase):
    equal_tests = [
        ['I am Hunter.', 'iamhunter', True],
        ['.. . 1', '1', True]
    ]

    unequal_tests = [
        ['I am Hunter.r', 'iamhunter', False],
        ['.. . 1', '12', False]
    ]

    def test_to_standard_string(self):
        for test in self.equal_tests:
            self.assertEqual(general_utils.to_standard_string(test[0]), test[1])

    def test_standard_equality(self):
        for test in self.equal_tests + self.unequal_tests:
            self.assertEqual(general_utils.standard_equality(test[0], test[1]), test[2])
