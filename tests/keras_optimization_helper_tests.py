##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.library_helpers import keras_optimization_helper as k_opt_helper
from hyperparameter_hunter.space import Real, Categorical
from hyperparameter_hunter.utils.test_utils import (
    equals_suite,
    exception_suite,
    format_suites,
    get_module,
)

##################################################
# Import Miscellaneous Assets
##################################################
from collections import OrderedDict
from functools import partial
from unittest import TestCase, TextTestRunner

##################################################
# Test Build Function Strings
##################################################
_expected_params_0 = OrderedDict(
    [
        ("rate", "Real(0.0, 1.0)"),
        ("units", "Integer(256, 1024)"),
        ("kernel_initializer", "Categorical(['glorot_uniform', 'lecun_normal'])"),
        ("Activation", "Categorical(['relu', 'sigmoid'], transform='onehot')"),
        ("Dropout", "Real(low=0.0, high=1.0)"),
        ("loss", "Categorical(['categorical_crossentropy', 'binary_crossentropy'])"),
        ("optimizer", "Categorical(['rmsprop', 'adam', 'sgd'], transform='onehot')"),
    ]
)
_expected_params_1 = OrderedDict(
    [
        ("Dropout", "Real(0.0, 1.0)"),
        ("Dense", "Integer(256, 1024)"),
        ("Activation", "Categorical(['relu', 'sigmoid'], transform='onehot')"),
        ("Dropout_1", "Real(low=0.0, high=1.0)"),
        ("Dropout_2", "Categorical(categories=['three', 'four'])"),
        ("add", "Categorical([Dropout(0.5), Activation('linear')])"),
        ("loss", "Categorical(['categorical_crossentropy', 'binary_crossentropy'])"),
        ("optimizer", "Categorical(['rmsprop', 'adam', 'sgd'], transform='identity')"),
    ]
)


def _build_fn_source_0(stage):
    stage_vals = [(_v, "params[{!r}]".format(_k)) for _k, _v in _expected_params_0.items()]
    build_str = """def {7}:
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(rate={0}, seed=32))
    model.add(Dense(units={1}, kernel_initializer={2}))
    model.add(Activation({3}))
    model.add(Dropout({4}))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(
        loss={5},
        metrics=['accuracy'],
        optimizer={6}
    )

    return model"""

    if stage == "original":
        return build_str.format(*[_[0] for _ in stage_vals], "create_model(input_shape)")
    elif stage == "reusable":
        return build_str.format(
            *[_[1] for _ in stage_vals], "build_fn(input_shape=(10, ), params=None)"
        )


def _build_fn_source_1(stage):
    stage_vals = [(_v, "params[{!r}]".format(_k)) for _k, _v in _expected_params_1.items()]
    build_str = '''def {8}:
    """Keras model-building function that contains hyperparameter space declarations
    Parameters
    ----------
    input_shape: Int
        The shape of the input provided to the first layer of the model
    
    Returns
    -------
    model: Instance of :class:`keras.Sequential`
        A compiled Keras model"""
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({0}))
    model.add(Dense({1}))
    model.add(Activation({2}))
    model.add(Dropout({3}))
    
    if {4} == 'four':
        model.add(Dense(100))
        model.add({5})
        model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(
        loss={6}, 
        metrics=['accuracy'], 
        optimizer={7}
    )
    
    return model'''

    if stage == "original":
        return build_str.format(*[_[0] for _ in stage_vals], "create_model(input_shape)")
    elif stage == "reusable":
        return build_str.format(
            *[_[1] for _ in stage_vals], "build_fn(input_shape=(10, ), params=None)"
        )


simple_mlp_layers = [
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": {
            "activation": None,
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
            "kernel_regularizer": None,
            "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None,
            "bias_constraint": None,
        },
        "__hh_used_args": [100],
        "__hh_used_kwargs": {
            "kernel_initializer": "uniform",
            "input_shape": [30],
            "activation": "relu",
        },
    },
    {
        "class_name": "Dropout",
        "__hh_default_args": ["rate"],
        "__hh_default_kwargs": {"noise_shape": None, "seed": None},
        "__hh_used_args": [0.5],
        "__hh_used_kwargs": {},
    },
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": {
            "activation": None,
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
            "kernel_regularizer": None,
            "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None,
            "bias_constraint": None,
        },
        "__hh_used_args": [1],
        "__hh_used_kwargs": {"kernel_initializer": "uniform", "activation": "sigmoid"},
    },
]


class TestRewriteModelBuilder(TestCase):
    valid_tests = dict(
        simple=[
            (_build_fn_source_0("original"), (_build_fn_source_0("reusable"), _expected_params_0)),
            (_build_fn_source_1("original"), (_build_fn_source_1("reusable"), _expected_params_1)),
        ]
    )

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=2).run)
        self.suite = partial(equals_suite, k_opt_helper.rewrite_model_builder)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name):
        cases, keys = format_suites(suite_group, group_name)
        targets = [_[1] for _ in cases]
        cases = [{"build_fn_source": _[0]} for _ in cases]
        return cases, targets, keys

    def do_valid_rewrite_tests(self):
        cases, targets, keys = self.prep(self.valid_tests, "valid_rewrite_{}_")
        self.run_suite(self.suite(cases, targets, keys, self.module))


class TestFindSpaceFragments(TestCase):
    valid_fragment_tests = dict(
        simple=[
            (
                _build_fn_source_0("original"),
                [  # TODO: Update below to leverage `_expected_params_0`, plus index numbers
                    ["Real(0.0, 1.0)", "rate", 168],
                    ["Integer(256, 1024)", "units", 220],
                    ["Categorical(['glorot_uniform', 'lecun_normal'])", "kernel_initializer", 259],
                    ["Categorical(['relu', 'sigmoid'], transform='onehot')", "Activation", 334],
                    ["Real(low=0.0, high=1.0)", "Dropout", 411],
                    [
                        "Categorical(['categorical_crossentropy', 'binary_crossentropy'])",
                        "loss",
                        533,
                    ],
                    [
                        "Categorical(['rmsprop', 'adam', 'sgd'], transform='onehot')",
                        "optimizer",
                        647,
                    ],
                ],
            ),
            (
                _build_fn_source_1("original"),
                [  # TODO: Update below to leverage `_expected_params_1`, plus index numbers
                    ["Real(0.0, 1.0)", "Dropout", 478],
                    ["Integer(256, 1024)", "Dense", 515],
                    ["Categorical(['relu', 'sigmoid'], transform='onehot')", "Activation", 561],
                    ["Real(low=0.0, high=1.0)", "Dropout_1", 638],
                    [
                        "Categorical(categories=['three', 'four'])",
                        "Dropout_2",
                        676,
                    ],  # FLAG: This is weird, but probably ok, maybe
                    [
                        "Categorical([Dropout(0.5), Activation('linear')])",
                        "add",
                        777,
                    ],  # TODO: Layer selection not fully supported
                    [
                        "Categorical(['categorical_crossentropy', 'binary_crossentropy'])",
                        "loss",
                        970,
                    ],
                    [
                        "Categorical(['rmsprop', 'adam', 'sgd'], transform='identity')",
                        "optimizer",
                        1086,
                    ],
                ],
            ),
        ]
    )

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=2).run)
        self.suite = partial(equals_suite, k_opt_helper.find_space_fragments)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name):
        cases, keys = format_suites(suite_group, group_name)
        targets = [tuple(list(_) for _ in zip(*_case[1])) for _case in cases]
        cases = [{"string": _[0]} for _ in cases]
        return cases, targets, keys

    def do_valid_fragment_tests(self):
        cases, targets, keys = self.prep(self.valid_fragment_tests, "valid_fragment_{}_")
        self.run_suite(self.suite(cases, targets, keys, self.module))


class TestConsolidateLayers(TestCase):
    valid_layers_tests = dict(
        simple=[
            (
                {"layers": simple_mlp_layers, "class_name_key": True, "separate_args": True},
                [
                    {
                        "class_name": "Dense",
                        "arg_vals": {"units": 100},
                        "kwarg_vals": {
                            "activation": "relu",
                            "use_bias": True,
                            "kernel_initializer": "uniform",
                            "bias_initializer": "zeros",
                            "kernel_regularizer": None,
                            "bias_regularizer": None,
                            "activity_regularizer": None,
                            "kernel_constraint": None,
                            "bias_constraint": None,
                        },
                    },
                    {
                        "class_name": "Dropout",
                        "arg_vals": {"rate": 0.5},
                        "kwarg_vals": {"noise_shape": None, "seed": None},
                    },
                    {
                        "class_name": "Dense",
                        "arg_vals": {"units": 1},
                        "kwarg_vals": {
                            "activation": "sigmoid",
                            "use_bias": True,
                            "kernel_initializer": "uniform",
                            "bias_initializer": "zeros",
                            "kernel_regularizer": None,
                            "bias_regularizer": None,
                            "activity_regularizer": None,
                            "kernel_constraint": None,
                            "bias_constraint": None,
                        },
                    },
                ],
            )
        ],
        class_name_key_false=[
            (
                {"layers": simple_mlp_layers, "class_name_key": False, "separate_args": True},
                [
                    {
                        "Dense": {
                            "arg_vals": {"units": 100},
                            "kwarg_vals": {
                                "activation": "relu",
                                "use_bias": True,
                                "kernel_initializer": "uniform",
                                "bias_initializer": "zeros",
                                "kernel_regularizer": None,
                                "bias_regularizer": None,
                                "activity_regularizer": None,
                                "kernel_constraint": None,
                                "bias_constraint": None,
                            },
                        }
                    },
                    {
                        "Dropout": {
                            "arg_vals": {"rate": 0.5},
                            "kwarg_vals": {"noise_shape": None, "seed": None},
                        }
                    },
                    {
                        "Dense": {
                            "arg_vals": {"units": 1},
                            "kwarg_vals": {
                                "activation": "sigmoid",
                                "use_bias": True,
                                "kernel_initializer": "uniform",
                                "bias_initializer": "zeros",
                                "kernel_regularizer": None,
                                "bias_regularizer": None,
                                "activity_regularizer": None,
                                "kernel_constraint": None,
                                "bias_constraint": None,
                            },
                        }
                    },
                ],
            )
        ],
        separate_args_false=[
            (
                {"layers": simple_mlp_layers, "class_name_key": True, "separate_args": False},
                [
                    {
                        "class_name": "Dense",
                        "units": 100,
                        "activation": "relu",
                        "use_bias": True,
                        "kernel_initializer": "uniform",
                        "bias_initializer": "zeros",
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None,
                    },
                    {"class_name": "Dropout", "rate": 0.5, "noise_shape": None, "seed": None},
                    {
                        "class_name": "Dense",
                        "units": 1,
                        "activation": "sigmoid",
                        "use_bias": True,
                        "kernel_initializer": "uniform",
                        "bias_initializer": "zeros",
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None,
                    },
                ],
            )
        ],
        both_false=[
            (
                {"layers": simple_mlp_layers, "class_name_key": False, "separate_args": False},
                [
                    {
                        "Dense": {
                            "units": 100,
                            "activation": "relu",
                            "use_bias": True,
                            "kernel_initializer": "uniform",
                            "bias_initializer": "zeros",
                            "kernel_regularizer": None,
                            "bias_regularizer": None,
                            "activity_regularizer": None,
                            "kernel_constraint": None,
                            "bias_constraint": None,
                        }
                    },
                    {"Dropout": {"rate": 0.5, "noise_shape": None, "seed": None}},
                    {
                        "Dense": {
                            "units": 1,
                            "activation": "sigmoid",
                            "use_bias": True,
                            "kernel_initializer": "uniform",
                            "bias_initializer": "zeros",
                            "kernel_regularizer": None,
                            "bias_regularizer": None,
                            "activity_regularizer": None,
                            "kernel_constraint": None,
                            "bias_constraint": None,
                        }
                    },
                ],
            )
        ],
    )

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=2).run)
        self.suite = partial(equals_suite, k_opt_helper.consolidate_layers)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name):
        cases, keys = format_suites(suite_group, group_name)
        targets = [_[-1] for _ in cases]
        # cases = [{'string': _[0]} for _ in cases]
        cases = [_[0] for _ in cases]
        return cases, targets, keys

    def do_valid_fragment_tests(self):
        cases, targets, keys = self.prep(self.valid_layers_tests, "valid_layers_{}_")
        self.run_suite(self.suite(cases, targets, keys, self.module))


class TestMergeCompileParams(TestCase):
    valid_tests = dict(
        simple=[
            (
                # Input: `compile_params`
                {
                    "compile_kwargs": {},
                    "loss_function_names": ["binary_crossentropy"],
                    "loss_functions": ["<binary_crossentropy function>"],
                    "loss_weights": None,
                    "metrics": ["accuracy"],
                    "metrics_names": ["loss", "acc"],
                    "optimizer": "Adam",
                    "optimizer_params": {
                        "amsgrad": False,
                        "beta_1": 0.9,
                        "beta_2": 0.999,
                        "decay": 0.0,
                        "epsilon": 1e-07,
                        "lr": 0.001,
                    },
                    "sample_weight_mode": None,
                    "target_tensors": None,
                    "weighted_metrics": None,
                },
                # Input: `dummified_params`
                {("params", "optimizer"): Categorical(categories=("adam", "rmsprop"))},
                # Output: `merged_params`
                {
                    "compile_kwargs": {},
                    "loss_function_names": ["binary_crossentropy"],
                    "loss_functions": ["<binary_crossentropy function>"],
                    "loss_weights": None,
                    "metrics": ["accuracy"],
                    "metrics_names": ["loss", "acc"],
                    "optimizer": Categorical(categories=("adam", "rmsprop")),
                    "optimizer_params": {
                        "amsgrad": False,
                        "beta_1": 0.9,
                        "beta_2": 0.999,
                        "decay": 0.0,
                        "epsilon": 1e-07,
                        "lr": 0.001,
                    },
                    "sample_weight_mode": None,
                    "target_tensors": None,
                    "weighted_metrics": None,
                },
            ),
            (
                # Input: `compile_params`
                {
                    "compile_kwargs": {},
                    "loss_function_names": ["binary_crossentropy"],
                    "loss_functions": ["<binary_crossentropy function>"],
                    "loss_weights": None,
                    "metrics": ["accuracy"],
                    "metrics_names": ["loss", "acc"],
                    "optimizer": "Adam",
                    "optimizer_params": {
                        "amsgrad": False,
                        "beta_1": 0.9,
                        "beta_2": 0.999,
                        "decay": 0.0,
                        "epsilon": 1e-07,
                        "lr": 0.001,
                    },
                    "sample_weight_mode": None,
                    "target_tensors": None,
                    "weighted_metrics": None,
                },
                # Input: `dummified_params`
                {("params", "optimizer_params", "lr"): Real(0.0001, 0.1)},
                # Output: `merged_params`
                {
                    "compile_kwargs": {},
                    "loss_function_names": ["binary_crossentropy"],
                    "loss_functions": ["<binary_crossentropy function>"],
                    "loss_weights": None,
                    "metrics": ["accuracy"],
                    "metrics_names": ["loss", "acc"],
                    "optimizer": "Adam",
                    "optimizer_params": {
                        "amsgrad": False,
                        "beta_1": 0.9,
                        "beta_2": 0.999,
                        "decay": 0.0,
                        "epsilon": 1e-07,
                        "lr": Real(0.0001, 0.1),
                    },
                    "sample_weight_mode": None,
                    "target_tensors": None,
                    "weighted_metrics": None,
                },
            ),
        ]
    )

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=2).run)
        self.suite = partial(equals_suite, k_opt_helper.merge_compile_params)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name):
        cases, keys = format_suites(suite_group, group_name)
        targets = [_[2] for _ in cases]
        cases = [{"compile_params": _[0], "dummified_params": _[1]} for _ in cases]
        return cases, targets, keys

    def do_valid_tests(self):
        cases, targets, keys = self.prep(self.valid_tests, "valid_{}_")
        self.run_suite(self.suite(cases, targets, keys, self.module))


class TestCleanParenthesizedString(TestCase):
    valid_tests = dict(
        simple=[
            (
                "Categorical([Dropout(0.5), Activation('linear')]) ... I am some extra text",
                "Categorical([Dropout(0.5), Activation('linear')])",
            ),
            (
                "Dense(Integer(256, 1024)) ... I am some extra text",
                "Dense(Integer(256, 1024))",  # Notice, the beginning remains un-trimmed, despite not starting with `space` class
            ),
        ]
    )
    exception_tests = dict(
        value_error=[
            ("Categorical([Dropout(0.5), Activation('linear')]", ValueError),
            ("Dense(Integer(256, (1024))", ValueError),
        ]
    )

    def setUp(self):
        self.run_suite = partial(TextTestRunner(verbosity=2).run)
        self.suite_valid = partial(equals_suite, k_opt_helper.clean_parenthesized_string)
        self.suite_exception = partial(exception_suite, k_opt_helper.clean_parenthesized_string)
        self.module = get_module(__name__, self)

    @staticmethod
    def prep(suite_group, group_name):
        cases, keys = format_suites(suite_group, group_name)
        targets = [_[1] for _ in cases]
        cases = [{"string": _[0]} for _ in cases]
        return cases, targets, keys

    def do_valid_tests(self):
        cases, targets, keys = self.prep(self.valid_tests, "valid_{}_")
        self.run_suite(self.suite_valid(cases, targets, keys, self.module))

    def do_exception_tests(self):
        cases, targets, keys = self.prep(self.exception_tests, "exception_{}_")
        self.run_suite(self.suite_exception(cases, targets, keys, self.module))
