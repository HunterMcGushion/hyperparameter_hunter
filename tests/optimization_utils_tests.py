##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils import optimization_utils
from hyperparameter_hunter.space import Real, Integer, Categorical
from hyperparameter_hunter.utils.test_utils import equals_suite, format_suites, get_module

##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
from unittest import TestCase, TextTestRunner


class TestGetIdsBy(TestCase):
    valid_tests = dict(
        standard=[
            (
                dict(
                    leaderboard_path="tests/file_resources/optimization_utils_tests/GlobalLeaderboard0.csv",
                    algorithm_name=None,
                    cross_experiment_key=None,
                    hyperparameter_key=None,
                    drop_duplicates=False,
                ),
                [
                    "3cb92c7a-35d7-4d2a-8332-d67b78287173",
                    "1d1627ac-1587-4dbd-8eb3-4611ae74505a",
                    "ebf7f7e5-9c59-44ac-ba3d-defefa6ab4ec",
                    "6f04724a-a111-410c-9f7a-0decdebe09ee",
                    "9fc4494f-215d-4bcd-88b6-563ef90b2ba0",
                    "6dc093f1-5e92-4131-ac5c-e9303be5fead",
                    "9fb299a5-ffdf-424a-8964-082f856e427f",
                    "7f61ca00-62bf-49bd-b880-9be92ef07af9",
                    "1f4ce7ca-1df6-49b3-acd8-a052e1e3eb87",
                    "8cc0a3a1-d99f-4bd0-bd9c-fed498904eb4",
                ],
            )
        ],
        algorithm_name=[
            (
                dict(
                    leaderboard_path="tests/file_resources/optimization_utils_tests/GlobalLeaderboard0.csv",
                    algorithm_name="XGBClassifier",
                    cross_experiment_key=None,
                    hyperparameter_key=None,
                    drop_duplicates=False,
                ),
                [
                    "1d1627ac-1587-4dbd-8eb3-4611ae74505a",
                    "9fc4494f-215d-4bcd-88b6-563ef90b2ba0",
                    "6dc093f1-5e92-4131-ac5c-e9303be5fead",
                ],
            ),
            (
                dict(
                    leaderboard_path="tests/file_resources/optimization_utils_tests/GlobalLeaderboard0.csv",
                    algorithm_name="not_a_real_algorithm",
                    cross_experiment_key=None,
                    hyperparameter_key=None,
                    drop_duplicates=False,
                ),
                [],
            ),
        ],
        cross_experiment_key=[
            (
                dict(
                    leaderboard_path="tests/file_resources/optimization_utils_tests/GlobalLeaderboard0.csv",
                    algorithm_name=None,
                    cross_experiment_key="gplzfJrx-5GVzmhkeY07T9-WW_DOJZFEjEqtzsDkKCM=",
                    hyperparameter_key=None,
                    drop_duplicates=False,
                ),
                [
                    "ebf7f7e5-9c59-44ac-ba3d-defefa6ab4ec",
                    "6f04724a-a111-410c-9f7a-0decdebe09ee",
                    "9fb299a5-ffdf-424a-8964-082f856e427f",
                    "8cc0a3a1-d99f-4bd0-bd9c-fed498904eb4",
                ],
            )
        ],
        hyperparameter_key=[
            (
                dict(
                    leaderboard_path="tests/file_resources/optimization_utils_tests/GlobalLeaderboard0.csv",
                    algorithm_name=None,
                    cross_experiment_key=None,
                    hyperparameter_key="eGjxwq35MmEiMHtJ0ANMqvxKLFpp4ZVKUKXNgXOiumQ=",
                    drop_duplicates=False,
                ),
                [
                    "1d1627ac-1587-4dbd-8eb3-4611ae74505a",
                    "9fc4494f-215d-4bcd-88b6-563ef90b2ba0",
                    "6dc093f1-5e92-4131-ac5c-e9303be5fead",
                ],
            )
        ],
        drop_duplicates=[
            (
                dict(
                    leaderboard_path="tests/file_resources/optimization_utils_tests/GlobalLeaderboard0.csv",
                    algorithm_name=None,
                    cross_experiment_key=None,
                    hyperparameter_key=None,
                    drop_duplicates=True,
                ),
                [
                    "3cb92c7a-35d7-4d2a-8332-d67b78287173",
                    "1d1627ac-1587-4dbd-8eb3-4611ae74505a",
                    "ebf7f7e5-9c59-44ac-ba3d-defefa6ab4ec",
                    "6f04724a-a111-410c-9f7a-0decdebe09ee",
                    "9fc4494f-215d-4bcd-88b6-563ef90b2ba0",
                    # '6dc093f1-5e92-4131-ac5c-e9303be5fead',
                    "9fb299a5-ffdf-424a-8964-082f856e427f",
                    "7f61ca00-62bf-49bd-b880-9be92ef07af9",
                    "1f4ce7ca-1df6-49b3-acd8-a052e1e3eb87",
                    "8cc0a3a1-d99f-4bd0-bd9c-fed498904eb4",
                ],
            )
        ],
        mixed=[
            (
                dict(
                    leaderboard_path="tests/file_resources/optimization_utils_tests/GlobalLeaderboard0.csv",
                    algorithm_name=None,
                    cross_experiment_key=None,
                    hyperparameter_key="eGjxwq35MmEiMHtJ0ANMqvxKLFpp4ZVKUKXNgXOiumQ=",
                    drop_duplicates=True,
                ),
                ["1d1627ac-1587-4dbd-8eb3-4611ae74505a", "9fc4494f-215d-4bcd-88b6-563ef90b2ba0"],
            ),
            (
                dict(
                    leaderboard_path="tests/file_resources/optimization_utils_tests/GlobalLeaderboard0.csv",
                    algorithm_name="RGFClassifier",
                    cross_experiment_key="3R-ZH7Yud_zsM7UIikeWXSU5kkRVCQRSI1NeFlrkp_U=",
                    hyperparameter_key=None,
                    drop_duplicates=False,
                ),
                ["7f61ca00-62bf-49bd-b880-9be92ef07af9"],
            ),
        ],
    )

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=2).run)
        self.suite = partial(equals_suite, optimization_utils.get_ids_by)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name):
        cases, keys = format_suites(suite_group, group_name)
        targets = [_[-1] for _ in cases]
        cases = [_[0] for _ in cases]
        return cases, targets, keys

    def do_valid_tests(self):
        cases, targets, keys = self.prep(self.valid_tests, "valid_{}_")
        self.run_suite(self.suite(cases, targets, keys, self.module))


class TestGetChoiceDimensions(TestCase):
    valid_tests = dict(
        simple=[
            (  # Mixed choices
                dict(
                    n_estimators=200,
                    max_depth=Integer(name="max_depth", low=2, high=20),
                    subsample=0.5,
                    learning_rate=Real(name="learning_rate", low=0.0001, high=0.5),
                ),
                [
                    (("max_depth",), Integer(name="max_depth", low=2, high=20)),
                    (("learning_rate",), Real(name="learning_rate", low=0.0001, high=0.5)),
                ],
            ),
            (dict(n_estimators=200, subsample=0.5), []),  # No choices
            (  # All choices
                dict(
                    max_depth=Integer(name="max_depth", low=2, high=20),
                    learning_rate=Real(name="learning_rate", low=0.0001, high=0.5),
                ),
                [
                    (("max_depth",), Integer(name="max_depth", low=2, high=20)),
                    (("learning_rate",), Real(name="learning_rate", low=0.0001, high=0.5)),
                ],
            ),
        ],
        complex=[
            (
                dict(
                    fit=dict(
                        eval_metric=Categorical(
                            ["auc", "rmse", "mae"], transform="onehot", name="eval_metric"
                        ),
                        early_stopping_rounds=5,
                    ),
                    predict=dict(ntree_limit=100),
                ),
                [
                    (
                        ("fit", "eval_metric"),
                        Categorical(["auc", "rmse", "mae"], transform="onehot", name="eval_metric"),
                    )
                ],
            ),
            (
                dict(
                    fit=dict(
                        eval_metric=Categorical(
                            ["auc", "rmse", "mae"], transform="onehot", name="eval_metric"
                        ),
                        early_stopping_rounds=5,
                    ),
                    predict=dict(
                        ntree_limit=100,
                        some_other_dict=dict(another_param=Real(0.1, 0.7, name="another_param")),
                    ),
                ),
                [
                    (
                        ("fit", "eval_metric"),
                        Categorical(["auc", "rmse", "mae"], transform="onehot", name="eval_metric"),
                    ),
                    (
                        ("predict", "some_other_dict", "another_param"),
                        Real(0.1, 0.7, name="another_param"),
                    ),
                ],
            ),
        ],
    )

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=2).run)
        self.suite = partial(equals_suite, optimization_utils.get_choice_dimensions)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name):
        cases, keys = format_suites(suite_group, group_name)
        targets = [_[-1] for _ in cases]
        cases = [dict(params=_[0]) for _ in cases]
        return cases, targets, keys

    def do_valid_tests(self):
        cases, targets, keys = self.prep(self.valid_tests, "valid_choice_dimensions_{}_")
        self.run_suite(self.suite(cases, targets, keys, self.module))
