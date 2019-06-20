##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment
from hyperparameter_hunter import optimization as hh_opt
from hyperparameter_hunter.utils.general_utils import flatten, subdict

##################################################
# Import Miscellaneous Assets
##################################################
import pytest

##################################################
# Import Learning Assets
##################################################
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor
from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor


@pytest.fixture(scope="module", autouse=True)
def env():
    return Environment(None, metrics=["roc_auc_score"])


##################################################
# Test Parametrization
##################################################
scenario_pairs = dict(
    bo=dict(est=["GP", GaussianProcessRegressor()], opt=[hh_opt.BayesianOptPro]),
    gbrt=dict(
        est=["GBRT", GradientBoostingQuantileRegressor()],
        opt=[hh_opt.GradientBoostedRegressionTreeOptPro, hh_opt.GBRT],
    ),
    rf=dict(est=["RF", RandomForestRegressor()], opt=[hh_opt.RandomForestOptPro, hh_opt.RF]),
    et=dict(est=["ET", ExtraTreesRegressor()], opt=[hh_opt.ExtraTreesOptPro, hh_opt.ET]),
    dummy=dict(est=["DUMMY"], opt=[hh_opt.DummyOptPro]),
)


def est_except(skip: str) -> list:
    return flatten([_["est"] for _ in subdict(scenario_pairs, drop=[skip]).values()])


def pytest_generate_tests(metafunc):
    arg_names, arg_values, id_list = None, [], []

    if metafunc.function.__name__ == "test_valid":
        scenarios = scenario_pairs
    elif metafunc.function.__name__ == "test_invalid":
        scenarios = {k: dict(est=est_except(k), opt=v["opt"]) for k, v in scenario_pairs.items()}
    else:
        return

    for id_prefix, scenario_dict in scenarios.items():
        arg_names = list(scenario_dict.keys())

        for est in scenario_dict["est"]:
            for opt in scenario_dict["opt"]:
                arg_values.append((est, opt))
                e = est if isinstance(est, str) else est.__class__.__name__
                id_list.append(f"{id_prefix}({e}, {opt.__name__})")

    metafunc.parametrize(arg_names, arg_values, ids=id_list, scope="function")


##################################################
# Tests
##################################################
def test_valid(est, opt):
    opt(base_estimator=est)


def test_invalid(est, opt):
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
    with pytest.deprecated_call():
        opt_pro()
