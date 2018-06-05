##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.tests import experiments_tests
from hyperparameter_hunter.tests import framework_utils_tests
from hyperparameter_hunter.tests import general_utils_tests
from hyperparameter_hunter.tests import key_handler_tests
from hyperparameter_hunter.tests import metrics_tests

# import cross_validation_wrapper_tests
# import framework_utils_tests
# import general_utils_tests
# import key_handler_tests
# import metrics_tests

# from . import cross_validation_wrapper_tests
# from . import framework_utils_tests
# from . import general_utils_tests
# from . import key_handler_tests
# from . import metrics_tests

##################################################
# Import Miscellaneous Assets
##################################################
import unittest

# TODO: This isn't working
loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(experiments_tests))
suite.addTests(loader.loadTestsFromModule(framework_utils_tests))
suite.addTests(loader.loadTestsFromModule(general_utils_tests))
suite.addTests(loader.loadTestsFromModule(key_handler_tests))
suite.addTests(loader.loadTestsFromModule(metrics_tests))

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)
