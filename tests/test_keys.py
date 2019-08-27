##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import settings
from hyperparameter_hunter.io.exceptions import EnvironmentInactiveError, EnvironmentInvalidError
from hyperparameter_hunter.keys.makers import CrossExperimentKeyMaker
from hyperparameter_hunter.keys.hashing import make_hash_sha256


##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
import pandas as pd
import pytest


##################################################
# Dummy Objects for Testing
##################################################
def function_0(*args, **kwargs):
    return "foo"


def function_1(*args, **kwargs):
    return "bar"


def function_2(*args, **kwargs):
    # I am a comment
    return "bar"


def function_3(*args, **kwargs):
    # I am a comment
    # I am a second comment
    return "bar"


def function_4(*args, **kwargs):
    # I am a slightly altered comment
    # I am a slightly altered second comment
    return "bar"


partial_0 = partial(function_0)
partial_1 = partial(function_1)
partial_2 = partial(function_2)
partial_3 = partial(function_3)
partial_4 = partial(function_4)

lambda_0 = lambda _: _
lambda_1 = lambda _: "foo"
lambda_2 = lambda _: "bar"


class EmptyClass0(object):
    pass


params_0 = dict()
params_1 = dict(ignore_line_comments=True)
params_2 = dict(ignore_line_comments=False)
params_3 = dict(
    ignore_line_comments=True, ignore_name=True, ignore_first_line=True
)  # Only hash uncommented source
params_4 = dict(
    ignore_name=True, ignore_first_line=True, ignore_source_lines=True
)  # Ignore all, except module
params_5 = dict(ignore_module=True, ignore_name=True, ignore_source_lines=True)  # Ignore everything


def args_ids_for(scenarios):
    return dict(argvalues=scenarios, ids=[f"{_}" for _ in range(len(scenarios))])


##################################################
# make_hash_sha256 Scenarios
##################################################
scenarios_string = [
    ["", "b0nNvYDhuV1eZCfhUB_CF3kNruhwVfpbTnEGQoi93t4="],
    ["foo", "r2k9pBRi7CR1POUUA70e5bUAzWlB9i88l0Vt2P-n-VI="],
    ["bar", "nUgj2eanz5rkoSJjAtpkH54SokNETqbeoiqjuJTfuVo="],
    # ["foo2", "r2k9pBRi7CR1POUUA70e5bUAzWlB9i88l0Vt2P-n-VI="],  # FLAG: SHOULD FAIL
]
scenarios_number = [
    [1, "a4ayc_80_OGda4BO_1o_V0etpOqiLx1JwB5S3beHW0s="],
    [3.14, "Lv_xJhwl2U3WaY6hBH9cCnEHypiwpsJCfuZhQUNQAhU="],
    [-100.7, "XEczU3F1jILwx10pPCqNHdsQfFnE-YpbJtfYg_OkXTo="],
    # [-100.8, "XEczU3F1jILwx10pPCqNHdsQfFnE-YpbJtfYg_OkXTo="],  # FLAG: SHOULD FAIL
]
scenarios_tuple = [
    [tuple(), "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10="],  # Same as empty list, dict
    [("foo", "bar"), "nLa17RepW5vZ-h-Tmoj56p_xIznyxOK7HXJX-Y4XieE="],
    [("bar", "foo"), "5UmXFMC8LmyZJLnaImLH108nXTNQE4Ei4ZzmLsqxzCE="],
]
scenarios_list = [
    [list(), "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10="],  # Same as empty tuple, dict
    [["foo", "bar"], "nLa17RepW5vZ-h-Tmoj56p_xIznyxOK7HXJX-Y4XieE="],
    [["bar", "foo"], "5UmXFMC8LmyZJLnaImLH108nXTNQE4Ei4ZzmLsqxzCE="],
]
scenarios_dict = [
    [dict(), "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10="],  # Same as empty tuple, list
    [dict(foo=10, bar=20), "9HnwEsXdYmfufzs6He31LRWV03wnLltwkgHh-AKG3Ko="],
    [dict(bar=20, foo=10), "9HnwEsXdYmfufzs6He31LRWV03wnLltwkgHh-AKG3Ko="],
]
scenarios_dataframe = [
    [pd.DataFrame(), "VlRtKQmvYEB_GxROp1dL8MtMSK9y4YuqxsnaIDbfKog="],
    [
        pd.DataFrame(data=[[10, 15], [20, 25]], columns=["foo", "bar"]),
        "pemlOQr8mbAFePlwfhI1lPiEAl1TC-luqJUQVabt4mM=",
    ],
    [
        pd.DataFrame(data=[[10, 15], [20, 25]], columns=["bar", "foo"]),
        "t0WO1WL0egKoRGP-3aKwHlt0etRN7ZkuBys2_Ic23X0=",
    ],
    [
        pd.DataFrame(data=[[10, 15], [20, 9000]], columns=["foo", "bar"]),
        "O_OLSm0Nr1ytVAxqwDrkcm8wzF6du3vwEiJ9ODlfF0c=",
    ],
    [
        pd.DataFrame(data=[[10, 15], [20, 25]], columns=["foo", "bar"], index=[92, 14]),
        "X--QHC8mgcQvhqeZFRLFw3W_th2vCVJavzGajSvgYno=",
    ],
]
# FLAG: Below test cases are highly sensitive. Any changes to declarations above (including comments), ...
# FLAG: ... or to the module name can cause them to break. This behavior is intentional.
scenarios_lambda = [
    [lambda_0, "Fblv_UfLw4qIe3RMQrNU1vA9Z8gl-0Brn3fNpkLua5k="],
    [lambda_1, "at2AxzF9cO_GZyPzrgR0-zgAr3Mk-DWhmg4_-4ec0D8="],
    [lambda_2, "b7i31rBTGifMEltKdEmEiS6hD02fd1ydNq2xbzegpGA="],
]
scenarios_partial = [
    [partial_0, "DVq0axBwpUhxAritItk7KatUmVK3qFms0JC0UkV1pgQ="],
    [partial_1, "3rlttSo2Hp1xhipW2h1yENuGEyFLaa36SmMHCRsmu94="],
    [partial_2, "kf1RqoxejYXH0BwvkWhmchgPpV0-boHpGrU5VzRHZw0="],
    [partial_3, "3VtUUFGCaD9p051vuJpMheM3tjbaEbE_B-Wm-TOkycs="],
    [partial_4, "2RpsON6ciWZ_5bMQaOXYU6ZdLRePwuRD_e4UGdPVbS8="],
]
scenarios_function = [
    [function_0, "KiKTsFM_A3fPq03fd_8JfbxZ9fsDicQw8LJtMCWG5Oo=", params_0],  # 0.0 (0)
    [function_0, "KiKTsFM_A3fPq03fd_8JfbxZ9fsDicQw8LJtMCWG5Oo=", params_1],  # 0.1 (1)
    [function_0, "KiKTsFM_A3fPq03fd_8JfbxZ9fsDicQw8LJtMCWG5Oo=", params_2],  # 0.2 (2)
    [function_0, "SIpsp66nfrPgUTiyZ1YQqykOxciUs8m9pLxQninQcIo=", params_3],  # 0.3 (3)
    [function_0, "emsqFo8SSc8_ulrRPG2-x_7rWJZmqGQ5PmU5I3km6kY=", params_4],  # 0.4 (4)
    [function_0, "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", params_5],  # 0.5 (5)
    [function_1, "B2yVD1nYC4qBNqiOAgbuYnZ-lM0sfH6d_9lrWwMollE=", params_0],  # 1.0 (6)
    [function_1, "B2yVD1nYC4qBNqiOAgbuYnZ-lM0sfH6d_9lrWwMollE=", params_1],  # 1.1 (7)
    [function_1, "B2yVD1nYC4qBNqiOAgbuYnZ-lM0sfH6d_9lrWwMollE=", params_2],  # 1.2 (8)
    [function_1, "SnQQCiMvqBy0Sq-5CvxpvT-5h1qeowi_fLhELH4e35c=", params_3],  # 1.3 (9)
    [function_1, "emsqFo8SSc8_ulrRPG2-x_7rWJZmqGQ5PmU5I3km6kY=", params_4],  # 1.4 (10)
    [function_1, "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", params_5],  # 1.5 (11)
    [function_2, "sA01sIRaW4lrrtf8fz-fNVg5uJR5LElp5R9z8W38dNk=", params_0],  # 2.0 (12)
    [function_2, "sA01sIRaW4lrrtf8fz-fNVg5uJR5LElp5R9z8W38dNk=", params_1],  # 2.1 (13)
    [function_2, "Vl-CqpdOFSX_AAjPmhN23NJqE7DW6i2DuK5CNuT0zAU=", params_2],  # 2.2 (14)
    [function_2, "SnQQCiMvqBy0Sq-5CvxpvT-5h1qeowi_fLhELH4e35c=", params_3],  # 2.3 (15)
    [function_2, "emsqFo8SSc8_ulrRPG2-x_7rWJZmqGQ5PmU5I3km6kY=", params_4],  # 2.4 (16)
    [function_2, "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", params_5],  # 2.5 (17)
    [function_3, "Um_SX0jak8KrMcq1Lz7DIoYA0a5y8G5fhWfi9BBDXD4=", params_0],  # 3.0 (18)
    [function_3, "Um_SX0jak8KrMcq1Lz7DIoYA0a5y8G5fhWfi9BBDXD4=", params_1],  # 3.1 (19)
    [function_3, "PNtYek4MCHF35F1xcN8Bxe30ANM9tSEM7TxzLymvmZE=", params_2],  # 3.2 (20)
    [function_3, "SnQQCiMvqBy0Sq-5CvxpvT-5h1qeowi_fLhELH4e35c=", params_3],  # 3.3 (21)
    [function_3, "emsqFo8SSc8_ulrRPG2-x_7rWJZmqGQ5PmU5I3km6kY=", params_4],  # 3.4 (22)
    [function_3, "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", params_5],  # 3.5 (23)
    [function_4, "7MqD33qqcojZZ9ZVL26vDl9GUPLvzn5QxeV9ULwqYOk=", params_0],  # 4.0 (24)
    [function_4, "7MqD33qqcojZZ9ZVL26vDl9GUPLvzn5QxeV9ULwqYOk=", params_1],  # 4.1 (25)
    [function_4, "VBAiE1thD1KsI504xMDcW1mfNl6JCnhk1Ie8VhEEfCA=", params_2],  # 4.2 (26)
    [function_4, "SnQQCiMvqBy0Sq-5CvxpvT-5h1qeowi_fLhELH4e35c=", params_3],  # 4.3 (27)
    [function_4, "emsqFo8SSc8_ulrRPG2-x_7rWJZmqGQ5PmU5I3km6kY=", params_4],  # 4.4 (28)
    [function_4, "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", params_5],  # 4.5 (29)
    # Notes:
    # - "X.0" == "X.1" for X in <0, 1, 2, 3, 4> because default kwargs used
    # - "X.0" == "X.1" == "X.2" for X in <0, 1> because function_<0, 1> contain no comments
    # - <1, 2, 3, 4>.3 are identical because kwargs only hash uncommented body of source - function_0 had different body
    # - <0, 1, 2, 3, 4>.4 are identical because kwargs ignore everything, except module
    # - <0, 1, 2, 3, 4>.5 are identical because kwargs ignore everything
]


@pytest.mark.parametrize(["obj", "expected"], **args_ids_for(scenarios_string))
def test_make_hash_sha256_string(obj, expected):
    assert make_hash_sha256(obj) == expected


@pytest.mark.parametrize(["obj", "expected"], **args_ids_for(scenarios_number))
def test_make_hash_sha256_number(obj, expected):
    assert make_hash_sha256(obj) == expected


@pytest.mark.parametrize(["obj", "expected"], **args_ids_for(scenarios_tuple))
def test_make_hash_sha256_tuple(obj, expected):
    assert make_hash_sha256(obj) == expected


@pytest.mark.parametrize(["obj", "expected"], **args_ids_for(scenarios_list))
def test_make_hash_sha256_list(obj, expected):
    assert make_hash_sha256(obj) == expected


@pytest.mark.parametrize(["obj", "expected"], **args_ids_for(scenarios_dict))
def test_make_hash_sha256_dict(obj, expected):
    assert make_hash_sha256(obj) == expected


@pytest.mark.parametrize(["obj", "expected"], **args_ids_for(scenarios_dataframe))
def test_make_hash_sha256_dataframe(obj, expected):
    assert make_hash_sha256(obj) == expected


@pytest.mark.parametrize(["obj", "expected"], **args_ids_for(scenarios_lambda))
def test_make_hash_sha256_lambda(obj, expected):
    assert make_hash_sha256(obj) == expected


@pytest.mark.parametrize(["obj", "expected"], **args_ids_for(scenarios_partial))
def test_make_hash_sha256_partial(obj, expected):
    assert make_hash_sha256(obj) == expected


@pytest.mark.parametrize(["obj", "expected", "kwargs"], **args_ids_for(scenarios_function))
def test_make_hash_sha256_function(obj, expected, kwargs):
    assert make_hash_sha256(obj, **kwargs) == expected


##################################################
# KeyMaker Scenarios
##################################################
def test_inactive_environment(monkeypatch):
    monkeypatch.setattr(settings.G, "Env", None)
    with pytest.raises(EnvironmentInactiveError):
        CrossExperimentKeyMaker(dict(a="foo", b="bar"))


@pytest.mark.parametrize("missing_attr", ["result_paths", "cross_experiment_key"])
def test_invalid_environment(monkeypatch, env_fixture_0, missing_attr):
    monkeypatch.delattr(settings.G.Env, missing_attr)
    with pytest.raises(EnvironmentInvalidError):
        CrossExperimentKeyMaker(dict(a="foo", b="bar"))


# def pytest_generate_tests(metafunc):
#     id_list = []
#     arg_values = []
#
#     for scenarios in metafunc.cls.scenarios:
#         id_prefix = scenarios[0]
#
#         # print(f"ID_PREFIX:   {id_prefix}")
#
#         for i, scenario in enumerate(scenarios[1]):
#             # print(f"I, ID_PREFIX, SCENARIO:   {i},   {id_prefix},   {scenario}")
#             id_list.append("{}[{}]".format(id_prefix, i))
#
#             items = scenario.items()
#             arg_names = [_[0] for _ in items]
#             arg_values.append(([_[1] for _ in items]))
#
#     metafunc.parametrize(arg_names, arg_values, ids=id_list, scope="class")


# NOTE: BELOW WORKS FOR SEPARATED SCENARIOS
# def pytest_generate_tests(metafunc):
#     id_list = []
#     arg_values = []
#
#     for scenario in metafunc.cls.scenarios:
#         id_list.append(scenario[0])
#         items = scenario[1].items()
#         arg_names = [_[0] for _ in items]
#         arg_values.append(([_[1] for _ in items]))
#
#     metafunc.parametrize(arg_names, arg_values, ids=id_list, scope="class")
# NOTE: ABOVE WORKS FOR SEPARATED SCENARIOS

# FLAG: PREFERRED IMPLEMENTATION
# def pytest_generate_tests(metafunc):
#     arg_names, arg_values, id_list = None, [], []
#
#     if not metafunc.cls:
#         return
#
#     for id_prefix, scenarios in metafunc.cls.scenarios.items():
#         for i, scenario in enumerate(scenarios):
#             id_list.append("{}[{}]".format(id_prefix, i))
#
#             items = scenario.items()
#             arg_names = [_[0] for _ in items]
#             arg_values.append(([_[1] for _ in items]))
#
#     metafunc.parametrize(arg_names, arg_values, ids=id_list, scope="class")
# FLAG: PREFERRED IMPLEMENTATION

# scenarios_string = [
#     dict(obj="", expected="b0nNvYDhuV1eZCfhUB_CF3kNruhwVfpbTnEGQoi93t4=", kwargs={}),
#     dict(obj="foo", expected="r2k9pBRi7CR1POUUA70e5bUAzWlB9i88l0Vt2P-n-VI=", kwargs={}),
#     dict(obj="bar", expected="nUgj2eanz5rkoSJjAtpkH54SokNETqbeoiqjuJTfuVo=", kwargs={}),
#     # dict(obj="foo2", expected="r2k9pBRi7CR1POUUA70e5bUAzWlB9i88l0Vt2P-n-VI=", kwargs={}),  # FLAG: SHOULD FAIL
# ]
# scenarios_number = [
#     dict(obj=1, expected="a4ayc_80_OGda4BO_1o_V0etpOqiLx1JwB5S3beHW0s=", kwargs={}),
#     dict(obj=3.14, expected="Lv_xJhwl2U3WaY6hBH9cCnEHypiwpsJCfuZhQUNQAhU=", kwargs={}),
#     # dict(obj=-100.8, expected="XEczU3F1jILwx10pPCqNHdsQfFnE-YpbJtfYg_OkXTo=", kwargs={}),  # FLAG: SHOULD FAIL
#     dict(obj=-100.7, expected="XEczU3F1jILwx10pPCqNHdsQfFnE-YpbJtfYg_OkXTo=", kwargs={}),
# ]
# scenarios_tuple = [
#     dict(obj=tuple(), expected="LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", kwargs={}),  # Same as empty list, dict
#     dict(obj=("foo", "bar"), expected="nLa17RepW5vZ-h-Tmoj56p_xIznyxOK7HXJX-Y4XieE=", kwargs={}),
#     dict(obj=("bar", "foo"), expected="5UmXFMC8LmyZJLnaImLH108nXTNQE4Ei4ZzmLsqxzCE=", kwargs={}),
# ]
# scenarios_list = [
#     dict(obj=list(), expected="LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", kwargs={}),  # Same as empty tuple, dict
#     dict(obj=["foo", "bar"], expected="nLa17RepW5vZ-h-Tmoj56p_xIznyxOK7HXJX-Y4XieE=", kwargs={}),
#     dict(obj=["bar", "foo"], expected="5UmXFMC8LmyZJLnaImLH108nXTNQE4Ei4ZzmLsqxzCE=", kwargs={}),
# ]
# class TestHashing2(object):
#     scenarios = {
#         "string": scenarios_string,
#         "number": scenarios_number,
#         "tuple": scenarios_tuple,
#         "list": scenarios_list,
#         # "list": [
#         #     [list(), "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10="],  # Same as empty tuple, dict
#         #     [["foo", "bar"], "nLa17RepW5vZ-h-Tmoj56p_xIznyxOK7HXJX-Y4XieE="],
#         #     [["bar", "foo"], "5UmXFMC8LmyZJLnaImLH108nXTNQE4Ei4ZzmLsqxzCE="],
#         # ],
#         # "dict": [
#         #     [dict(), "LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10="],  # Same as empty tuple, list
#         #     [dict(foo=10, bar=20), "9HnwEsXdYmfufzs6He31LRWV03wnLltwkgHh-AKG3Ko="],
#         #     [dict(bar=20, foo=10), "9HnwEsXdYmfufzs6He31LRWV03wnLltwkgHh-AKG3Ko="],
#         # ],
#         # "dataframe": [
#         #     [pd.DataFrame(), "_EMr0fJ5U8Z5MetmSTxLIrBChcdGLN2RM1Nlh6H1HSg="],
#         #     [
#         #         pd.DataFrame(data=[[10, 15], [20, 25]], columns=["foo", "bar"]),
#         #         "CtUj-FraurT-ppSV5q-MN-FGiG4-NZw_WpmIssbqal8=",
#         #     ],
#         #     [
#         #         pd.DataFrame(data=[[10, 15], [20, 25]], columns=["bar", "foo"]),
#         #         "yFS_JgF__tIUVIvQta7C89zzSvkNNqAWQY8aLNxQeVk=",
#         #     ],
#         # ],
#         # # FLAG: The below test cases are highly sensitive. Any changes to their declarations above (including comments), ...
#         # # FLAG: ... or to the module name can cause them to break. This behavior is intentional.
#         # "lambda": [
#         #     [lambda_0, "jVqMujmZrfTQ_ghu45UEQCsiojoW1XA2-UxnzmVYqPw="],
#         #     [lambda_1, "bINmmRPfDw_w7gpvZmnBBbisU_2jjk7o6zsZiEfVXB0="],
#         #     [lambda_2, "ffdUtskHZM5a6sh6UZwuOah4_zvbGHVrLYngpDIpFIo="],
#         # ],
#         # "partial": [
#         #     [partial_0, "U8mKyyREb7AsazKq4NpqEx5S84NOkzJqO9MgLgVZMRI="],
#         #     [partial_1, "K8bcoIh5rY5sTKEw_iIAbsYTFVUJgI59BAUjBZLb8oY="],
#         #     [partial_2, "9hfqU_ok9W3Co9tmbrq9DndPallclyJp_9qHicKvnxA="],
#         #     [partial_3, "gc4vinFOVr-nBbI0MGT2DTKZWrwYBePQGNpP61E6t_Q="],
#         #     [partial_4, "AYb7OpGRaSYF2fmPKkhdB7KmL9VMr-ed0BvnE0fzo2Q="],
#         # ],
#
#         # "function": [
#         #     dict(obj=function_0, expected="Cb2TUYwjXfcjwIBvT-bBc6HAwAadhzuDwCTEtB-bMO0=", kwargs=params_0),  # 0.0 (0)
#         #     dict(obj=function_0, expected="Cb2TUYwjXfcjwIBvT-bBc6HAwAadhzuDwCTEtB-bMO0=", kwargs=params_1),  # 0.1 (1)
#         #     dict(obj=function_0, expected="Cb2TUYwjXfcjwIBvT-bBc6HAwAadhzuDwCTEtB-bMO0=", kwargs=params_2),  # 0.2 (2)
#         #     dict(obj=function_0, expected="WrN40EWtq4wfgOLe6gtHxa0iw96bwtgn9wS9BdGvMmQ=", kwargs=params_3),  # 0.3 (3)
#         #     dict(obj=function_0, expected="oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=", kwargs=params_4),  # 0.4 (4)
#         #     dict(obj=function_0, expected="LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", kwargs=params_5),  # 0.5 (5)
#         #     dict(obj=function_1, expected="SKhekwGS2XXI-w_Eh6hdH7Z9I9T_y2J7Eo1k7HVNEVs=", kwargs=params_0),  # 1.0 (6)
#         #     dict(obj=function_1, expected="SKhekwGS2XXI-w_Eh6hdH7Z9I9T_y2J7Eo1k7HVNEVs=", kwargs=params_1),  # 1.1 (7)
#         #     dict(obj=function_1, expected="SKhekwGS2XXI-w_Eh6hdH7Z9I9T_y2J7Eo1k7HVNEVs=", kwargs=params_2),  # 1.2 (8)
#         #     dict(obj=function_1, expected="U9866f0OuEbX77rPqwgU69cYnSpy-XIbbS9JlK3jHmc=", kwargs=params_3),  # 1.3 (9)
#         #     dict(obj=function_1, expected="oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=", kwargs=params_4),  # 1.4 (10)
#         #     dict(obj=function_1, expected="LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", kwargs=params_5),  # 1.5 (11)
#         #     dict(obj=function_2, expected="f-KKuNUimE0gvshlm6_lYAZWcXxe5-OtFHrMmBPUBIY=", kwargs=params_0),  # 2.0 (12)
#         #     dict(obj=function_2, expected="f-KKuNUimE0gvshlm6_lYAZWcXxe5-OtFHrMmBPUBIY=", kwargs=params_1),  # 2.1 (13)
#         #     dict(obj=function_2, expected="bKNrJcDuyvaFTVXaK8dVwobnmNF9Tr1jmH-DkzWwOBc=", kwargs=params_2),  # 2.2 (14)
#         #     dict(obj=function_2, expected="U9866f0OuEbX77rPqwgU69cYnSpy-XIbbS9JlK3jHmc=", kwargs=params_3),  # 2.3 (15)
#         #     dict(obj=function_2, expected="oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=", kwargs=params_4),  # 2.4 (16)
#         #     dict(obj=function_2, expected="LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", kwargs=params_5),  # 2.5 (17)
#         #     dict(obj=function_3, expected="6APbPpWnYCupflXiz2UDPGaXFsMNEOdqCGvBiGilDTw=", kwargs=params_0),  # 3.0 (18)
#         #     dict(obj=function_3, expected="6APbPpWnYCupflXiz2UDPGaXFsMNEOdqCGvBiGilDTw=", kwargs=params_1),  # 3.1 (19)
#         #     dict(obj=function_3, expected="0rVSc96v3go-p0FXtFgf8ICFN1NtczKyr3efqHGr0To=", kwargs=params_2),  # 3.2 (20)
#         #     dict(obj=function_3, expected="U9866f0OuEbX77rPqwgU69cYnSpy-XIbbS9JlK3jHmc=", kwargs=params_3),  # 3.3 (21)
#         #     dict(obj=function_3, expected="oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=", kwargs=params_4),  # 3.4 (22)
#         #     dict(obj=function_3, expected="LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", kwargs=params_5),  # 3.5 (23)
#         #     dict(obj=function_4, expected="cgefnjGlRiBTfcXR5spb6U2WJ3cKhr19ZEhRN5kAxoY=", kwargs=params_0),  # 4.0 (24)
#         #     dict(obj=function_4, expected="cgefnjGlRiBTfcXR5spb6U2WJ3cKhr19ZEhRN5kAxoY=", kwargs=params_1),  # 4.1 (25)
#         #     dict(obj=function_4, expected="fmNJN3jD5jC_BaBYd1gxzwsgVNqPoUH3rbQ3_UBNY40=", kwargs=params_2),  # 4.2 (26)
#         #     dict(obj=function_4, expected="U9866f0OuEbX77rPqwgU69cYnSpy-XIbbS9JlK3jHmc=", kwargs=params_3),  # 4.3 (27)
#         #     dict(obj=function_4, expected="oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=", kwargs=params_4),  # 4.4 (28)
#         #     dict(obj=function_4, expected="LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=", kwargs=params_5),  # 4.5 (29)
#         #     # Notes:
#         #     # - "X.0" == "X.1" for X in <0, 1, 2, 3, 4> because default kwargs used
#         #     # - "X.0" == "X.1" == "X.2" for X in <0, 1> because function_<0, 1> contain no comments
#         #     # - <1, 2, 3, 4>.3 are identical because kwargs only hash uncommented body of source - function_0 had different body
#         #     # - <0, 1, 2, 3, 4>.4 are identical because kwargs ignore everything, except module
#         #     # - <0, 1, 2, 3, 4>.5 are identical because kwargs ignore everything
#         # ],
#     }
#
#     def test_simple(self, obj, expected, kwargs):
#         assert make_hash_sha256(obj, **(kwargs or {})) == expected
