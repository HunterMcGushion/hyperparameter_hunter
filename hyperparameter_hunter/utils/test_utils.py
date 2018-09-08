"""This module contains utilities for building and executing unit tests, as well as for reporting
the results

Related
-------
:mod:`hyperparameter_hunter.tests`
    This module contains all unit tests for the library, and it uses
    :mod:`hyperparameter_hunter.utils.test_utils` throughout"""
##################################################
# Import Miscellaneous Assets
##################################################
import re
from unittest import TestCase, TestSuite


##################################################
# Test Suites
##################################################
def suite_builder(test_type, test_function, test_cases, targets, case_ids, override_module):
    """Handles creation of custom test suites of test_type

    Parameters
    ----------
    test_type: Descendant of BaseAssertionTest
        Any class that inherits BaseAssertionTest and overrides "execute_method_override" to set
        new methods to an assertion method defined in BaseAssertionTest
    test_function: Callable
        The function to be called on each set of inputs in test_cases. See "Notes" section below for
        usage details
    test_cases: List
        The case inputs to be tested. Should correspond to elements in targets, and case_ids
    targets: List
        The expected test results. Should correspond to elements in test_cases, and case_ids
    case_ids: List
        The string identifiers for each test case. Should correspond to elements in test_cases, and
        targets
    override_module: String, or None
        If string, this should be the module that is currently executing the tests. Otherwise, the
        file containing this docstring will be listed as the location of the test. override_module
        will be appended to any other module information

    Returns
    -------
    suite: TestSuite
        The populated test suite to be supplied to a unittest.TextTestRunner.run() call (or similar)

    Notes
    -----
    If test_function is a class method, or is set as an attribute of the class running the tests, it
    may need to be enclosed in functools.partial(). Should work fine if test_function is: a global
    variable pointing to a function, a class instantiation, or a partial"""
    #################### Validate Parameters ####################
    if not issubclass(test_type, BaseAssertionTest):
        raise TypeError(
            "test_type must inherit BaseAssertionTest. See hyperparameter_hunter.utils.test_utils for examples"
        )
    if not callable(test_function):
        raise TypeError(
            "test_function must be callable. Received: {}: {}".format(
                type(test_function), test_function
            )
        )

    names_vals = [("test_cases", test_cases), ("targets", targets), ("case_ids", case_ids)]
    for (arg_name, an_arg) in names_vals:
        if not isinstance(an_arg, (list, tuple)):
            raise TypeError(
                "{} must be of type list or tuple. Received {}".format(arg_name, type(an_arg))
            )
    if len(set([len(_[1]) for _ in names_vals])) != 1:
        raise ValueError(
            "Lengths of the following arguments must be equal: [{}]. Received lengths: {}, respectively".format(
                [_[0] for _ in names_vals], [len(_[1]) for _ in names_vals]
            )
        )

    #################### Build Test Suite ####################
    suite = TestSuite()

    suite.addTests(
        test_type(test_function, test_cases[i], case_ids[i], targets[i], override_module)
        for i in range(len(test_cases))
    )

    return suite


def equals_suite(test_function, test_cases, targets, case_ids, override_module):
    """A test suite for asserting object equality"""
    return suite_builder(
        TestAssertEquals, test_function, test_cases, targets, case_ids, override_module
    )


def truth_suite(test_function, test_cases, targets, case_ids, override_module):
    """A test suite for asserting boolean values"""
    return suite_builder(
        TestAssertTrue, test_function, test_cases, targets, case_ids, override_module
    )


def exception_suite(test_function, test_cases, targets, case_ids, override_module):
    """A test suite for ensuring that some Exception is raised"""
    return suite_builder(
        TestAssertException, test_function, test_cases, targets, case_ids, override_module
    )


##################################################
# Test Cases
##################################################
class BaseAssertionTest(TestCase):
    def __init__(
        self,
        test_function,
        test_input,
        override_method_name,
        target,
        override_module,
        *args,
        **kwargs
    ):
        self.test_function = test_function
        self.test_input = test_input
        self.target = target
        self.args = args
        self.kwargs = kwargs

        self.__class__.__module__ = do_module_override(override_module, self.__class__.__module__)
        self.execute_method_override(override_method_name)

        TestCase.__init__(self, methodName=override_method_name)

    def execute_method_override(self, override_method_name):
        setattr(self, override_method_name, self._placeholder_test)

    ##################################################
    # Assertion Methods
    ##################################################
    def _placeholder_test(self):
        pass

    def _is_equal(self):
        self.assertEqual(self.test_function(**self.test_input), self.target)

    def _is_true(self):  # noinspection PyCompatibility
        self.assertTrue(self.test_function(**self.test_input, **self.kwargs))

    def _is_false(self):  # noinspection PyCompatibility
        self.assertFalse(self.test_function(**self.test_input, **self.kwargs))

    def _raises(self):
        with self.assertRaises(self.target):
            self.test_function(**self.test_input)

    def _passes(self):
        try:
            self.test_function(**self.test_input)
        except Exception as _ex:
            self.fail("{} raised".format(_ex.__repr__()))


class TestAssertEquals(BaseAssertionTest):
    def execute_method_override(self, override_method_name):
        setattr(self, override_method_name, self._is_equal)


class TestAssertTrue(BaseAssertionTest):
    def execute_method_override(self, override_method_name):
        setattr(self, override_method_name, (self._is_true if self.target else self._is_false))


class TestAssertException(BaseAssertionTest):
    def execute_method_override(self, override_method_name):
        setattr(self, override_method_name, (self._raises if self.target else self._passes))


##################################################
# Utilities
##################################################
def get_module(filename, current_class):
    return "***{}.{}***".format(filename, type(current_class).__name__)


def do_module_override(override, current):
    if isinstance(override, str) and (not current.startswith(override)):
        current = re.sub(re.compile("\*\*\*[^*]+\*\*\*"), "", current)
        return "{}{}".format(override, current)

    return current


def format_case_id(method_name_prefix, test_case_index):
    return "{}{:0>2}".format(method_name_prefix, test_case_index)


def format_suites(test_suites, group_format, reindex=True):
    """...

    Parameters
    ----------
    test_suites: Dict
        Keys should be strings that identify the test suite. Values should be lists, containing the
        arguments for each test case
    group_format: Str
        A formatting string containing "{}" exactly once, where the supplied input will be a key in
        test_suites
    reindex: Boolean, default=True
        If True, case indexing restarts at 0 for each suite. Else, indexing progresses from 0 to
        the total number of cases - 1

    Returns
    -------
    all_cases: list
        All test case arguments; its basically a flattened version of test_suites.values()
    all_keys: list
        ID keys from test_suites for each element in all_cases, formatted according to group_format

    Examples
    --------
    >>> from hyperparameter_hunter.utils.test_utils import format_suites
    >>> suites = {'a': [[None], [None]], 'b': [[None], [None]]}
    >>> print(format_suites(suites, 'suites_{}_', reindex=True)[1])
    ... print(format_suites(suites, 'suites_{}_', reindex=False)[1])
    ['suites_a_00', 'suites_a_01', 'suites_b_00', 'suites_b_01']
    ['suites_a_00', 'suites_a_01', 'suites_b_02', 'suites_b_03']

    Notes
    -----
    Notice in the example above that suites contains two test suites ('a', and 'b'), each of which
    contains dummy data for two test cases. See that the result of the first print statement names
    suite b's first key "suites_b_00" because it restarts indexing for each suite. Conversely, the
    same result key from the second print statement is "suites_b_02" because indexing is never
    restarted. reindex=True can facilitate locating test cases, especially with a multiplicity of
    cases/suites"""
    group_format = group_format if group_format else "{}"
    all_cases, all_keys = [], []

    for suite_key, test_cases in test_suites.items():
        formatted_keys = [group_format.format(suite_key)] * len(test_cases)

        for i, k in enumerate(formatted_keys):
            all_keys.append(format_case_id(k, (i + (0 if reindex else len(all_cases)))))

        all_cases.extend(test_cases)

    return all_cases, all_keys
