##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils.general_utils import to_standard_string, standard_equality

###############################################
# Import Miscellaneous Assets
###############################################
import pytest


###############################################
# `to_standard_string` Scenarios
###############################################
@pytest.mark.parametrize(
    ["string", "expected_string"], [["I am Hunter.", "iamhunter"], [".. . 1", "1"]], ids=["0", "1"]
)
def test_to_standard_string(string, expected_string):
    assert to_standard_string(string) == expected_string


###############################################
# `standard_equality` Scenarios
###############################################
@pytest.mark.parametrize(
    ["string_1", "string_2", "expected_equality"],
    [
        pytest.param("I am Hunter.", "iamhunter", True, id="true_0"),
        pytest.param(".. . 1", "1", True, id="true_1"),
        pytest.param("I am Hunter.r", "iamhunter", False, id="false_0"),
        pytest.param(".. . 1", "12", False, id="false_1"),
    ],
)
def test_standard_equality(string_1, string_2, expected_equality):
    assert standard_equality(string_1, string_2) is expected_equality
