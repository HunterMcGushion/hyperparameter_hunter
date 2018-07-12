##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import key_handler
from hyperparameter_hunter.utils.test_utils import equals_suite, format_suites, get_module

##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
import pandas as pd
from unittest import TestCase, TestSuite, TextTestRunner

##################################################
# Import Learning Assets
##################################################
from sklearn.metrics import roc_auc_score


##################################################
# Dummy Objects for Testing
##################################################
def function_0(*args, **kwargs):
    return 'foo'


def function_1(*args, **kwargs):
    return 'bar'


def function_2(*args, **kwargs):
    # I am a comment
    return 'bar'


def function_3(*args, **kwargs):
    # I am a comment
    # I am a second comment
    return 'bar'


def function_4(*args, **kwargs):
    # I am a slightly altered comment
    # I am a slightly altered second comment
    return 'bar'


partial_0 = partial(function_0)
partial_1 = partial(function_1)
partial_2 = partial(function_2)
partial_3 = partial(function_3)
partial_4 = partial(function_4)

lambda_0 = lambda _: _
lambda_1 = lambda _: 'foo'
lambda_2 = lambda _: 'bar'


class EmptyClass0(object):
    pass


class TestHashing(TestCase):
    ##################################################
    # Declare Parameter Groups
    ##################################################
    params_0 = dict()
    params_1 = dict(ignore_line_comments=True)
    params_2 = dict(ignore_line_comments=False)
    params_3 = dict(ignore_line_comments=True, ignore_name=True, ignore_first_line=True)  # Only hash uncommented source
    params_4 = dict(ignore_name=True, ignore_first_line=True, ignore_source_lines=True)  # Ignore all, except module
    params_5 = dict(ignore_module=True, ignore_name=True, ignore_source_lines=True)  # Ignore everything

    simple_tests = {
        'string': [
            ['', 'b0nNvYDhuV1eZCfhUB_CF3kNruhwVfpbTnEGQoi93t4='],
            ['foo', 'r2k9pBRi7CR1POUUA70e5bUAzWlB9i88l0Vt2P-n-VI='],
            ['bar', 'nUgj2eanz5rkoSJjAtpkH54SokNETqbeoiqjuJTfuVo='],
            # ['foo2', 'r2k9pBRi7CR1POUUA70e5bUAzWlB9i88l0Vt2P-n-VI='],  # This correctly fails
        ],
        'number': [
            [1, 'a4ayc_80_OGda4BO_1o_V0etpOqiLx1JwB5S3beHW0s='],
            [3.14, 'Lv_xJhwl2U3WaY6hBH9cCnEHypiwpsJCfuZhQUNQAhU='],
            [-100.7, 'XEczU3F1jILwx10pPCqNHdsQfFnE-YpbJtfYg_OkXTo='],
        ],
        'tuple': [
            [tuple(), 'LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10='],  # Same as empty list, dict
            [('foo', 'bar'), 'nLa17RepW5vZ-h-Tmoj56p_xIznyxOK7HXJX-Y4XieE='],
            [('bar', 'foo'), '5UmXFMC8LmyZJLnaImLH108nXTNQE4Ei4ZzmLsqxzCE='],
        ],
        'list': [
            [list(), 'LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10='],  # Same as empty tuple, dict
            [['foo', 'bar'], 'nLa17RepW5vZ-h-Tmoj56p_xIznyxOK7HXJX-Y4XieE='],
            [['bar', 'foo'], '5UmXFMC8LmyZJLnaImLH108nXTNQE4Ei4ZzmLsqxzCE='],
        ],
        'dict': [
            [dict(), 'LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10='],  # Same as empty tuple, list
            [dict(foo=10, bar=20), '9HnwEsXdYmfufzs6He31LRWV03wnLltwkgHh-AKG3Ko='],
            [dict(bar=20, foo=10), '9HnwEsXdYmfufzs6He31LRWV03wnLltwkgHh-AKG3Ko='],
        ],
        'dataframe': [
            [pd.DataFrame(), '_EMr0fJ5U8Z5MetmSTxLIrBChcdGLN2RM1Nlh6H1HSg='],
            [pd.DataFrame(data=[[10, 15], [20, 25]], columns=['foo', 'bar']), 'CtUj-FraurT-ppSV5q-MN-FGiG4-NZw_WpmIssbqal8='],
            [pd.DataFrame(data=[[10, 15], [20, 25]], columns=['bar', 'foo']), 'yFS_JgF__tIUVIvQta7C89zzSvkNNqAWQY8aLNxQeVk='],
        ],
        # FLAG: The below test cases are highly sensitive. Any changes to their declarations above (including comments), ...
        # FLAG: ... or to the module name can cause them to break. This behavior is intentional.
        'lambda': [
            [lambda_0, 'jVqMujmZrfTQ_ghu45UEQCsiojoW1XA2-UxnzmVYqPw='],
            [lambda_1, 'vEXfVnK7iQLNc_dWGRLKmzq0UYeH202dT0YKF1a6Jtg='],
            [lambda_2, '4tXhp4qQ0s2_19igDvhF6so9Q0wbvxsOE8V7ZV2N2TQ='],
        ],
        'partial': [
            [partial_0, 'Lzy_FgITj6cPqmpO4Z4czbSXB86o3stIBGIRqJAXG4c='],
            [partial_1, 'WNE7m5W5pFlM1e0Ewnf9KpaffM5JVhCoJB6dEbeP7Y4='],
            [partial_2, 'OYNt0N4MSprG7II088xQQ2xidEaBZoV2IDNsZ2hpTN4='],
            [partial_3, 'xs-EaoYaXDUKF1u0GmFUJQH7YlIxnAK67VPtvN_f3GM='],
            [partial_4, 'oosun9ZXISNyKmzK9xx28pAK-YALEGkLwUID7sCrwNQ='],
        ],
        'function': [
            [function_0, 'rNXPNHNeXt75nRmtBwcgexq4r6_HSo8lHVT36RGLVVU=', params_0],  # 0.0 (0)
            [function_0, 'rNXPNHNeXt75nRmtBwcgexq4r6_HSo8lHVT36RGLVVU=', params_1],  # 0.1 (1)
            [function_0, 'rNXPNHNeXt75nRmtBwcgexq4r6_HSo8lHVT36RGLVVU=', params_2],  # 0.2 (2)
            [function_0, 'YxUI9Pc7QVIPC2JluLqpiTWH4_Q9vxzi13xKsgGXhEA=', params_3],  # 0.3 (3)
            [function_0, 'oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=', params_4],  # 0.4 (4)
            [function_0, 'LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=', params_5],  # 0.5 (5)

            [function_1, 'TcISnIxBYdmJLI7gFvAy1u16Iz-QSOz0CYH3A1dA8uI=', params_0],  # 1.0 (6)
            [function_1, 'TcISnIxBYdmJLI7gFvAy1u16Iz-QSOz0CYH3A1dA8uI=', params_1],  # 1.1 (7)
            [function_1, 'TcISnIxBYdmJLI7gFvAy1u16Iz-QSOz0CYH3A1dA8uI=', params_2],  # 1.2 (8)
            [function_1, '6hXudrmeOzenqKZj3Xduht3iy0ZWjDwmwlitHnGgzHc=', params_3],  # 1.3 (9)
            [function_1, 'oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=', params_4],  # 1.4 (10)
            [function_1, 'LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=', params_5],  # 1.5 (11)

            [function_2, 'aGpU6pHp_nMAVLR2IauGWFF1kQxwf1bDBASs_zloSsQ=', params_0],  # 2.0 (12)
            [function_2, 'aGpU6pHp_nMAVLR2IauGWFF1kQxwf1bDBASs_zloSsQ=', params_1],  # 2.1 (13)
            [function_2, 'kbdhcPeNmPc3p8mHCKYux3x3OZSC2YawePmur47s_x0=', params_2],  # 2.2 (14)
            [function_2, '6hXudrmeOzenqKZj3Xduht3iy0ZWjDwmwlitHnGgzHc=', params_3],  # 2.3 (15)
            [function_2, 'oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=', params_4],  # 2.4 (16)
            [function_2, 'LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=', params_5],  # 2.5 (17)

            [function_3, 'T6ey_hqPgB5CkxHI6PV3mK01kKIhMieuJBkJNJv9szM=', params_0],  # 3.0 (18)
            [function_3, 'T6ey_hqPgB5CkxHI6PV3mK01kKIhMieuJBkJNJv9szM=', params_1],  # 3.1 (19)
            [function_3, '2y6ZrKlE0v2xSGqUklnLhiYuimji3YdFHX-aq0XEVi4=', params_2],  # 3.2 (20)
            [function_3, '6hXudrmeOzenqKZj3Xduht3iy0ZWjDwmwlitHnGgzHc=', params_3],  # 3.3 (21)
            [function_3, 'oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=', params_4],  # 3.4 (22)
            [function_3, 'LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=', params_5],  # 3.5 (23)

            [function_4, 'q951DymktgCBsp_fihNrVUEjldwqGgrATIIv0XKAiRo=', params_0],  # 4.0 (24)
            [function_4, 'q951DymktgCBsp_fihNrVUEjldwqGgrATIIv0XKAiRo=', params_1],  # 4.1 (25)
            [function_4, 'LAM0_0uqb1LD1EnnTYdaAu5aQ5aFTft45-LH-tjKDFI=', params_2],  # 4.2 (26)
            [function_4, '6hXudrmeOzenqKZj3Xduht3iy0ZWjDwmwlitHnGgzHc=', params_3],  # 4.3 (27)
            [function_4, 'oW1isd367-zUz0DaBRiY9-sx0Y0NCYbvh3QSH4rZAlk=', params_4],  # 4.4 (28)
            [function_4, 'LjjneyLDFKRJ6R-v7ZKkOCasaqQDrmqKy2z1gjn7r10=', params_5],  # 4.5 (29)

            # Notes:
            # - "X.0" == "X.1" for X in <0, 1, 2, 3, 4> because default kwargs used
            # - "X.0" == "X.1" == "X.2" for X in <0, 1> because function_<0, 1> contain no comments
            # - <1, 2, 3, 4>.3 are identical because kwargs only hash uncommented body of source - function_0 had different body
            # - <0, 1, 2, 3, 4>.4 are identical because kwargs ignore everything, except module
            # - <0, 1, 2, 3, 4>.5 are identical because kwargs ignore everything
        ],
    }
    complex_tests = dict(

    )

    def setUp(self):
        self.test_runner = partial(TextTestRunner, verbosity=0)  # verbosity=2  # verbosity=0
        self.test_function = key_handler.make_hash_sha256

    ##################################################
    # Build Suites for Test Cases
    ##################################################
    def do_simple_tests(self):
        cases, keys = format_suites(self.simple_tests, group_format='simple_test_{}_')
        targets = [_[1] for _ in cases]
        cases = [dict(obj=_[0], **(_[-1] if len(_) > 2 else dict())) for _ in cases]

        self.test_runner().run(equals_suite(self.test_function, cases, targets, keys, get_module(__name__, self)))
