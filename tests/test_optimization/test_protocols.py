##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment
from hyperparameter_hunter import settings
from hyperparameter_hunter.i_o.exceptions import EnvironmentInactiveError, EnvironmentInvalidError
from hyperparameter_hunter.optimization.backends.skopt import protocols as hh_opt
from hyperparameter_hunter.utils.general_utils import flatten, subdict
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data

##################################################
# Import Miscellaneous Assets
##################################################
import pytest
from unittest import mock

##################################################
# Import Learning Assets
##################################################
from sklearn.linear_model import Ridge
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor
from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"

EST_OPT_PRO_PAIRS = dict(
    bo=dict(est=["GP", GaussianProcessRegressor()], opt=[hh_opt.BayesianOptPro]),
    gbrt=dict(
        est=["GBRT", GradientBoostingQuantileRegressor()],
        opt=[hh_opt.GradientBoostedRegressionTreeOptPro, hh_opt.GBRT],
    ),
    rf=dict(est=["RF", RandomForestRegressor()], opt=[hh_opt.RandomForestOptPro, hh_opt.RF]),
    et=dict(est=["ET", ExtraTreesRegressor()], opt=[hh_opt.ExtraTreesOptPro, hh_opt.ET]),
    dummy=dict(est=["DUMMY"], opt=[hh_opt.DummyOptPro]),
)

# Below is flat list of all "opt" values above. Aliases may appear as duplicated values
ALL_SK_OPT_PROS = flatten([_["opt"] for _ in EST_OPT_PRO_PAIRS.values()])


@pytest.fixture(scope="module", autouse=True)
def env_auto_module():
    return Environment(
        train_dataset=get_toy_classification_data(),
        results_path=assets_dir,
        metrics=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )


##################################################
# `BaseOptPro` Miscellaneous Tests
##################################################
@pytest.mark.parametrize("opt_pro", ALL_SK_OPT_PROS)
def test_go_before_forge_experiment_error(opt_pro):
    """Test that invoking :meth:`hyperparameter_hunter.optimization.protocol_core.BaseOptPro.go`
    before :meth:`hyperparameter_hunter.optimization.protocol_core.BaseOptPro.forge_experiment`
    raises ValueError"""
    opt = opt_pro()
    with pytest.raises(ValueError, match="Must invoke `forge_experiment` before starting .*"):
        opt.go()


##################################################
# `base_estimator` Tests
##################################################
#################### `base_estimator` Test Parametrization ####################
def est_except(skip: str) -> list:
    """Return flattened list of all "est" values in `EST_OPT_PRO_PAIRS` that are not members of
    the dict named by `skip`

    Parameters
    ----------
    skip: String
        Key in `EST_OPT_PRO_PAIRS`, declaring the `base_estimator` values to exclude from the result

    Returns
    -------
    List
        Flat list of `base_estimator` values in `EST_OPT_PRO_PAIRS`, less those specified by `skip`

    Examples
    --------
    >>> est_except("gbrt")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['GP', GaussianProcessRegressor(...),
     'RF', RandomForestRegressor(...),
     'ET', ExtraTreesRegressor(...),
     'DUMMY']
    """
    return flatten([_["est"] for _ in subdict(EST_OPT_PRO_PAIRS, drop=[skip]).values()])


def pytest_generate_tests(metafunc):
    """Generate parameter permutations for selected test functions:

    * :func:`test_valid_base_estimator`
    * :func:`test_invalid_base_estimator`

    Test function parametrization is performed using the module-level variable `EST_OPT_PRO_PAIRS`,
    which is a dict, with a key for each OptimizationProtocol. The value of each key is a dict,
    formatted thusly::
        "est": [<valid `base_estimator` for OptPro>, ...],
        "opt": [<OptPro class>, ...]

    Each value of `EST_OPT_PRO_PAIRS`, therefore, defines valid `base_estimator` values for the
    corresponding OptimizationProtocol class(es)

    Notes
    -----
    Although it's an official (and awesome) PyTest feature, `pytest_generate_tests` is not terribly
    well-known and seems a bit black-magic-y, so instead of leaving curious readers to do their own
    research, I humbly recommend the following PyTest examples:

    * `Basic "pytest_generate_tests" example
      <https://docs.pytest.org/en/latest/parametrize.html#basic-pytest-generate-tests-example>`_
    * `A quick port of "testscenarios"
      <https://docs.pytest.org/en/latest/example/parametrize.html#a-quick-port-of-testscenarios>`_
    """
    arg_names, arg_values, id_list = None, [], []

    # Only parametrize test functions explicitly named - Return other test functions unchanged
    if metafunc.function.__name__ == "test_valid_base_estimator":
        scenarios = EST_OPT_PRO_PAIRS
    elif metafunc.function.__name__ == "test_invalid_base_estimator":
        scenarios = {k: dict(est=est_except(k), opt=v["opt"]) for k, v in EST_OPT_PRO_PAIRS.items()}
    else:
        return

    # Parametrize functions using `scenarios`
    for id_prefix, scenario_dict in scenarios.items():
        arg_names = list(scenario_dict.keys())

        for est in scenario_dict["est"]:
            for opt in scenario_dict["opt"]:
                arg_values.append((est, opt))
                e = est if isinstance(est, str) else est.__class__.__name__
                id_list.append(f"{id_prefix}({e}, {opt.__name__})")

    metafunc.parametrize(arg_names, arg_values, ids=id_list, scope="function")


#################### Actual `base_estimator` Tests ####################
def test_valid_base_estimator(est, opt):
    """Test that an OptimizationProtocol does not complain when given a valid `base_estimator`.
    Also test that selected strings and Regressor instances are equally valid values. Parametrized
    via :func:`pytest_generate_tests`"""
    opt(base_estimator=est)


def test_invalid_base_estimator(est, opt):
    """Test that an OptimizationProtocol complains when given an invalid `base_estimator`.
    Parametrized via :func:`pytest_generate_tests`"""
    with pytest.raises(TypeError, match="Expected `base_estimator` in .*"):
        opt(base_estimator=est)


##################################################
# Deprecation Tests
##################################################
@pytest.mark.parametrize(
    "opt_pro",
    [
        hh_opt.BayesianOptimization,
        hh_opt.GradientBoostedRegressionTreeOptimization,
        hh_opt.RandomForestOptimization,
        hh_opt.ExtraTreesOptimization,
        hh_opt.DummySearch,
    ],
)
def test_opt_pro_deprecations(opt_pro):
    """Test that instantiating any OptPro with an outdated name raises a DeprecationWarning"""
    with pytest.deprecated_call():
        opt_pro()


@pytest.mark.parametrize("opt_pro", ALL_SK_OPT_PROS)
def test_opt_pro_n_random_starts_deprecation(opt_pro):
    """Check that instantiating any OptPro with `n_random_starts` raises a DeprecationWarning"""
    with pytest.deprecated_call():
        opt_pro(n_random_starts=10)


##################################################
# Deprecation Tests: `set_experiment_guidelines` -> `forge_experiment`
##################################################
@pytest.mark.parametrize("opt_pro", ALL_SK_OPT_PROS)
def test_opt_pro_set_experiment_guidelines_deprecation(opt_pro):
    """Check that invoking an OptPro's :meth:`set_experiment_guidelines` raises a
    DeprecationWarning"""
    opt = opt_pro()

    with pytest.deprecated_call():
        opt.set_experiment_guidelines(Ridge, {})


@pytest.mark.parametrize("opt_pro", ALL_SK_OPT_PROS)
@pytest.mark.parametrize(
    "forge_experiment_params",
    [
        dict(model_initializer=Ridge, model_init_params={}),
        dict(model_initializer=Ridge, model_init_params=dict(alpha=0.9, solver="svd")),
    ],
)
def test_opt_pro_set_experiment_guidelines_calls_forge_experiment(opt_pro, forge_experiment_params):
    """Check that invoking an OptPro's :meth:`set_experiment_guidelines` (although deprecated)
    invokes the new :meth:`forge_experiment` with the parameters originally given"""
    opt = opt_pro()
    mock_path = "hyperparameter_hunter.optimization.protocol_core.BaseOptPro.forge_experiment"

    with pytest.deprecated_call():
        with mock.patch(mock_path) as mock_forge_experiment:
            opt.set_experiment_guidelines(**forge_experiment_params)
            mock_forge_experiment.assert_called_once_with(**forge_experiment_params)


##################################################
# `BaseOptPro._validate_environment` Tests
##################################################
@pytest.mark.parametrize("opt_pro", ALL_SK_OPT_PROS)
def test_inactive_environment(monkeypatch, env_fixture_0, opt_pro):
    """Test that initializing an OptPro without an active `Environment` raises
    `EnvironmentInactiveError`"""
    # Currently have a valid `settings.G.Env` (`env_fixture_0`), so set it to None
    monkeypatch.setattr(settings.G, "Env", None)
    with pytest.raises(EnvironmentInactiveError):
        opt_pro()


@pytest.mark.parametrize("opt_pro", ALL_SK_OPT_PROS)
def test_invalid_environment(monkeypatch, env_fixture_0, opt_pro):
    """Test that initializing an OptPro when there is an active `Environment` -- but
    :attr:`hyperparameter_hunter.environment.Environment.current_task` is not None -- raises
    `EnvironmentInvalidError`"""
    # Currently have a valid `settings.G.Env` (`env_fixture_0`), so give it a fake `current_task`
    monkeypatch.setattr(settings.G.Env, "current_task", "some other task")
    with pytest.raises(EnvironmentInvalidError, match="Must finish current task before starting.*"):
        opt_pro()
