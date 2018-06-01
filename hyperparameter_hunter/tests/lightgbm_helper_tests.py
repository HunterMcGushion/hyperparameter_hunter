##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.library_helpers import lightgbm_helper

###############################################
# Import Miscellaneous Assets
###############################################
from unittest import TestCase


class TestLightgbmTestFunction(TestCase):
    def test_is_string(self):
        result = lightgbm_handler.lightgbm_test_function()
        self.assertTrue(isinstance(result, str))
