##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.feature_engineering import merge_dfs, split_merged_df
from hyperparameter_hunter.feature_engineering import DatasetNameReport, validate_dataset_names
from hyperparameter_hunter.feature_engineering import get_engineering_step_params

##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd
import pytest


##################################################
# `merge_dfs` / `split_merged_df`
##################################################
train_data_0 = pd.DataFrame(dict(a=[0, 1, 2, 3], b=["a", "b", "c", "d"], t=[1, 1, 0, 0]))
validation_data_0 = pd.DataFrame(dict(a=[4, 5], b=["e", "f"], t=[1, 0]))
holdout_data_0 = pd.DataFrame(dict(a=[6, 7], b=["g", "h"], t=[0, 1]))
test_inputs_0 = pd.DataFrame(dict(a=[8, 9], b=["i", "j"]))

dfs_0 = dict(
    train_data=train_data_0,
    train_inputs=train_data_0.loc[:, ["a", "b"]],
    train_targets=train_data_0.loc[:, ["t"]],
    validation_data=validation_data_0,
    validation_inputs=validation_data_0.loc[:, ["a", "b"]],
    validation_targets=validation_data_0.loc[:, ["t"]],
    holdout_data=holdout_data_0,
    holdout_inputs=holdout_data_0.loc[:, ["a", "b"]],
    holdout_targets=holdout_data_0.loc[:, ["t"]],
    test_inputs=test_inputs_0,
)

dfs_1 = dict(dfs_0, **dict(holdout_data=None, holdout_inputs=None, holdout_targets=None))


def build_merged_df(df_names, group, original_dfs):
    keys = [f"{_}_{group}" for _ in df_names]
    return pd.concat([original_dfs[_] for _ in keys], keys=keys)


@pytest.mark.parametrize(
    ["merge_to", "stage", "expected_dfs", "expected_group", "original_dfs"],
    [
        #################### `dfs_0` ####################
        ("all_data", "intra_cv", ["train", "validation", "holdout"], "data", dfs_0),
        ("all_inputs", "intra_cv", ["train", "validation", "holdout", "test"], "inputs", dfs_0),
        ("all_targets", "intra_cv", ["train", "validation", "holdout"], "targets", dfs_0),
        ("all_data", "pre_cv", ["train", "holdout"], "data", dfs_0),
        ("all_inputs", "pre_cv", ["train", "holdout", "test"], "inputs", dfs_0),
        ("all_targets", "pre_cv", ["train", "holdout"], "targets", dfs_0),
        ("non_train_data", "intra_cv", ["validation", "holdout"], "data", dfs_0),
        ("non_train_inputs", "intra_cv", ["validation", "holdout", "test"], "inputs", dfs_0),
        ("non_train_targets", "intra_cv", ["validation", "holdout"], "targets", dfs_0),
        ("non_train_data", "pre_cv", ["holdout"], "data", dfs_0),
        ("non_train_inputs", "pre_cv", ["holdout", "test"], "inputs", dfs_0),
        ("non_train_targets", "pre_cv", ["holdout"], "targets", dfs_0),
        #################### `dfs_1` ####################
        ("all_data", "intra_cv", ["train", "validation", "holdout"], "data", dfs_1),
        ("all_inputs", "intra_cv", ["train", "validation", "holdout", "test"], "inputs", dfs_1),
        ("all_targets", "intra_cv", ["train", "validation", "holdout"], "targets", dfs_1),
        ("all_data", "pre_cv", ["train", "holdout"], "data", dfs_1),
        ("all_inputs", "pre_cv", ["train", "holdout", "test"], "inputs", dfs_1),
        ("all_targets", "pre_cv", ["train", "holdout"], "targets", dfs_1),
        ("non_train_data", "intra_cv", ["validation", "holdout"], "data", dfs_1),
        ("non_train_inputs", "intra_cv", ["validation", "holdout", "test"], "inputs", dfs_1),
        ("non_train_targets", "intra_cv", ["validation", "holdout"], "targets", dfs_1),
        ("non_train_inputs", "pre_cv", ["holdout", "test"], "inputs", dfs_1),
    ],
)
def test_merge_dfs_0(merge_to, stage, expected_dfs, expected_group, original_dfs):
    # Test building `actual_merged_df` via `merge_dfs`
    actual_merged_df = merge_dfs(merge_to, stage, original_dfs)
    expected_merged_df = build_merged_df(expected_dfs, expected_group, original_dfs)
    assert actual_merged_df.equals(expected_merged_df)
    # Test splitting `actual_merged_df` via `split_merged_df`
    actual_split_dfs = split_merged_df(actual_merged_df)
    expected_split_dfs_names = [f"{_}_{expected_group}" for _ in expected_dfs]
    expected_split_dfs = {_: original_dfs[_] for _ in expected_split_dfs_names}
    assert all(v.equals(expected_split_dfs[k]) for k, v in actual_split_dfs.items())


@pytest.mark.parametrize(
    ["merge_to", "stage", "expected_dfs", "expected_group", "original_dfs"],
    [
        ("non_train_data", "pre_cv", ["holdout"], "data", dfs_1),
        ("non_train_targets", "pre_cv", ["holdout"], "targets", dfs_1),
    ],
)
def test_merge_dfs_value_error(merge_to, stage, expected_dfs, expected_group, original_dfs):
    with pytest.raises(ValueError, match="Merging .* into .* does not produce DataFrame"):
        merge_dfs(merge_to, stage, original_dfs)


##################################################
# `DatasetNameReport` Scenarios
##################################################
#################### Convenience Aliases ####################
TR_D, TR_I, TR_T = ["train_data", "train_inputs", "train_targets"]
VA_D, VA_I, VA_T = ["validation_data", "validation_inputs", "validation_targets"]
HO_D, HO_I, HO_T = ["holdout_data", "holdout_inputs", "holdout_targets"]
TE_D, TE_I, TE_T = [None, "test_inputs", None]

A_D, A_I, A_T = ["all_data", "all_inputs", "all_targets"]
NT_D, NT_I, NT_T = ["non_train_data", "non_train_inputs", "non_train_targets"]

#################### `DatasetNameReport` Scenario #0 ####################
params_0 = (TR_D, NT_I)
stage_0 = "pre_cv"
merged_datasets_0 = [(NT_I,)]
coupled_datasets_0 = [(TR_D,)]
leaves_0 = {(NT_I, HO_I): HO_I, (NT_I, TE_I): TE_I, (TR_D, TR_I): TR_I, (TR_D, TR_T): TR_T}
descendants_0 = {TR_D: {TR_I: None, TR_T: None}, NT_I: {HO_I: None, TE_I: None}}

#################### `DatasetNameReport` Scenario #1 ####################
params_1 = (TR_D, NT_I)
stage_1 = "intra_cv"
merged_datasets_1 = [(NT_I,)]
coupled_datasets_1 = [(TR_D,)]
leaves_1 = {
    (NT_I, HO_I): HO_I,
    (NT_I, TE_I): TE_I,
    (NT_I, VA_I): VA_I,
    (TR_D, TR_I): TR_I,
    (TR_D, TR_T): TR_T,
}
descendants_1 = {TR_D: {TR_I: None, TR_T: None}, NT_I: {VA_I: None, HO_I: None, TE_I: None}}

#################### `DatasetNameReport` Scenario #2 ####################
params_2 = (NT_D, A_D)
stage_2 = "pre_cv"
merged_datasets_2 = [(NT_D,), (A_D,)]
coupled_datasets_2 = [(NT_D, HO_D), (A_D, TR_D), (A_D, HO_D)]
leaves_2 = {
    (A_D, HO_D, HO_I): HO_I,
    (A_D, HO_D, HO_T): HO_T,
    (A_D, TR_D, TR_I): TR_I,
    (A_D, TR_D, TR_T): TR_T,
    (NT_D, HO_D, HO_I): HO_I,
    (NT_D, HO_D, HO_T): HO_T,
}
descendants_2 = {
    NT_D: {HO_D: {HO_I: None, HO_T: None}},
    A_D: {TR_D: {TR_I: None, TR_T: None}, HO_D: {HO_I: None, HO_T: None}},
}

#################### `DatasetNameReport` Scenario #3 ####################
params_3 = (NT_D, A_D)
stage_3 = "intra_cv"
merged_datasets_3 = [(NT_D,), (A_D,)]
coupled_datasets_3 = [(NT_D, VA_D), (NT_D, HO_D), (A_D, TR_D), (A_D, VA_D), (A_D, HO_D)]
leaves_3 = {
    (A_D, HO_D, HO_I): HO_I,
    (A_D, HO_D, HO_T): HO_T,
    (A_D, TR_D, TR_I): TR_I,
    (A_D, TR_D, TR_T): TR_T,
    (A_D, VA_D, VA_I): VA_I,
    (A_D, VA_D, VA_T): VA_T,
    (NT_D, HO_D, HO_I): HO_I,
    (NT_D, HO_D, HO_T): HO_T,
    (NT_D, VA_D, VA_I): VA_I,
    (NT_D, VA_D, VA_T): VA_T,
}
descendants_3 = {
    NT_D: {VA_D: {VA_I: None, VA_T: None}, HO_D: {HO_I: None, HO_T: None}},
    A_D: {
        TR_D: {TR_I: None, TR_T: None},
        VA_D: {VA_I: None, VA_T: None},
        HO_D: {HO_I: None, HO_T: None},
    },
}

#################### `DatasetNameReport` Scenario #4 ####################
params_4 = (A_T, NT_D, TR_T)
stage_4 = "pre_cv"
merged_datasets_4 = [(A_T,), (NT_D,)]
coupled_datasets_4 = [(NT_D, HO_D)]
leaves_4 = {
    (A_T, HO_T): HO_T,
    (A_T, TR_T): TR_T,
    (NT_D, HO_D, HO_I): HO_I,
    (NT_D, HO_D, HO_T): HO_T,
    (TR_T,): TR_T,
}
descendants_4 = {A_T: {TR_T: None, HO_T: None}, NT_D: {HO_D: {HO_I: None, HO_T: None}}, TR_T: None}

#################### `DatasetNameReport` Scenario #5 ####################
params_5 = (A_T, NT_D, TR_T)
stage_5 = "intra_cv"
merged_datasets_5 = [(A_T,), (NT_D,)]
coupled_datasets_5 = [(NT_D, VA_D), (NT_D, HO_D)]
leaves_5 = {
    (A_T, HO_T): HO_T,
    (A_T, TR_T): TR_T,
    (A_T, VA_T): VA_T,
    (NT_D, HO_D, HO_I): HO_I,
    (NT_D, HO_D, HO_T): HO_T,
    (NT_D, VA_D, VA_I): VA_I,
    (NT_D, VA_D, VA_T): VA_T,
    (TR_T,): TR_T,
}
descendants_5 = {
    A_T: {TR_T: None, VA_T: None, HO_T: None},
    NT_D: {VA_D: {VA_I: None, VA_T: None}, HO_D: {HO_I: None, HO_T: None}},
    TR_T: None,
}

#################### `DatasetNameReport` Scenario #6 ####################
params_6 = (A_D, A_T, NT_D, TR_T)
stage_6 = "pre_cv"
merged_datasets_6 = [(A_D,), (A_T,), (NT_D,)]
coupled_datasets_6 = [(A_D, TR_D), (A_D, HO_D), (NT_D, HO_D)]
leaves_6 = {
    (A_D, HO_D, HO_I): HO_I,
    (A_D, HO_D, HO_T): HO_T,
    (A_D, TR_D, TR_I): TR_I,
    (A_D, TR_D, TR_T): TR_T,
    (A_T, HO_T): HO_T,
    (A_T, TR_T): TR_T,
    (NT_D, HO_D, HO_I): HO_I,
    (NT_D, HO_D, HO_T): HO_T,
    (TR_T,): TR_T,
}
descendants_6 = {
    A_D: {TR_D: {TR_I: None, TR_T: None}, HO_D: {HO_I: None, HO_T: None}},
    A_T: {TR_T: None, HO_T: None},
    NT_D: {HO_D: {HO_I: None, HO_T: None}},
    TR_T: None,
}

#################### `DatasetNameReport` Scenario #7 ####################
params_7 = (A_D, A_T, NT_D, TR_T)
stage_7 = "intra_cv"
merged_datasets_7 = [(A_D,), (A_T,), (NT_D,)]
coupled_datasets_7 = [(A_D, TR_D), (A_D, VA_D), (A_D, HO_D), (NT_D, VA_D), (NT_D, HO_D)]
leaves_7 = {
    (A_D, HO_D, HO_I): HO_I,
    (A_D, HO_D, HO_T): HO_T,
    (A_D, TR_D, TR_I): TR_I,
    (A_D, TR_D, TR_T): TR_T,
    (A_D, VA_D, VA_I): VA_I,
    (A_D, VA_D, VA_T): VA_T,
    (A_T, HO_T): HO_T,
    (A_T, TR_T): TR_T,
    (A_T, VA_T): VA_T,
    (NT_D, HO_D, HO_I): HO_I,
    (NT_D, HO_D, HO_T): HO_T,
    (NT_D, VA_D, VA_I): VA_I,
    (NT_D, VA_D, VA_T): VA_T,
    (TR_T,): TR_T,
}
descendants_7 = {
    A_D: {
        TR_D: {TR_I: None, TR_T: None},
        VA_D: {VA_I: None, VA_T: None},
        HO_D: {HO_I: None, HO_T: None},
    },
    A_T: {TR_T: None, VA_T: None, HO_T: None},
    NT_D: {VA_D: {VA_I: None, VA_T: None}, HO_D: {HO_I: None, HO_T: None}},
    TR_T: None,
}

#################### `DatasetNameReport` Scenario #8 ####################
params_8 = (A_D, A_T, NT_D, TR_T, NT_I)
stage_8 = "pre_cv"
merged_datasets_8 = [(A_D,), (A_T,), (NT_D,), (NT_I,)]
coupled_datasets_8 = [(A_D, TR_D), (A_D, HO_D), (NT_D, HO_D)]
leaves_8 = {
    (A_D, HO_D, HO_I): HO_I,
    (A_D, HO_D, HO_T): HO_T,
    (A_D, TR_D, TR_I): TR_I,
    (A_D, TR_D, TR_T): TR_T,
    (A_T, HO_T): HO_T,
    (A_T, TR_T): TR_T,
    (NT_D, HO_D, HO_I): HO_I,
    (NT_D, HO_D, HO_T): HO_T,
    (NT_I, HO_I): HO_I,
    (NT_I, TE_I): TE_I,
    (TR_T,): TR_T,
}
descendants_8 = {
    A_D: {TR_D: {TR_I: None, TR_T: None}, HO_D: {HO_I: None, HO_T: None}},
    A_T: {TR_T: None, HO_T: None},
    NT_D: {HO_D: {HO_I: None, HO_T: None}},
    TR_T: None,
    NT_I: {HO_I: None, TE_I: None},
}

#################### `DatasetNameReport` Scenario #9 ####################
params_9 = (A_D, A_T, NT_D, TR_T, NT_I)
stage_9 = "intra_cv"
merged_datasets_9 = [(A_D,), (A_T,), (NT_D,), (NT_I,)]
coupled_datasets_9 = [(A_D, TR_D), (A_D, VA_D), (A_D, HO_D), (NT_D, VA_D), (NT_D, HO_D)]
leaves_9 = {
    (A_D, HO_D, HO_I): HO_I,
    (A_D, HO_D, HO_T): HO_T,
    (A_D, TR_D, TR_I): TR_I,
    (A_D, TR_D, TR_T): TR_T,
    (A_D, VA_D, VA_I): VA_I,
    (A_D, VA_D, VA_T): VA_T,
    (A_T, HO_T): HO_T,
    (A_T, TR_T): TR_T,
    (A_T, VA_T): VA_T,
    (NT_D, HO_D, HO_I): HO_I,
    (NT_D, HO_D, HO_T): HO_T,
    (NT_D, VA_D, VA_I): VA_I,
    (NT_D, VA_D, VA_T): VA_T,
    (NT_I, HO_I): HO_I,
    (NT_I, TE_I): TE_I,
    (NT_I, VA_I): VA_I,
    (TR_T,): TR_T,
}
descendants_9 = {
    A_D: {
        TR_D: {TR_I: None, TR_T: None},
        VA_D: {VA_I: None, VA_T: None},
        HO_D: {HO_I: None, HO_T: None},
    },
    A_T: {TR_T: None, VA_T: None, HO_T: None},
    NT_D: {VA_D: {VA_I: None, VA_T: None}, HO_D: {HO_I: None, HO_T: None}},
    TR_T: None,
    NT_I: {VA_I: None, HO_I: None, TE_I: None},
}


#################### `DatasetNameReport` Test Execution ####################
@pytest.mark.parametrize(
    ["params", "stage", "merged_datasets", "coupled_datasets", "leaves", "descendants"],
    [
        (params_0, stage_0, merged_datasets_0, coupled_datasets_0, leaves_0, descendants_0),
        (params_1, stage_1, merged_datasets_1, coupled_datasets_1, leaves_1, descendants_1),
        (params_2, stage_2, merged_datasets_2, coupled_datasets_2, leaves_2, descendants_2),
        (params_3, stage_3, merged_datasets_3, coupled_datasets_3, leaves_3, descendants_3),
        (params_4, stage_4, merged_datasets_4, coupled_datasets_4, leaves_4, descendants_4),
        (params_5, stage_5, merged_datasets_5, coupled_datasets_5, leaves_5, descendants_5),
        (params_6, stage_6, merged_datasets_6, coupled_datasets_6, leaves_6, descendants_6),
        (params_7, stage_7, merged_datasets_7, coupled_datasets_7, leaves_7, descendants_7),
        (params_8, stage_8, merged_datasets_8, coupled_datasets_8, leaves_8, descendants_8),
        (params_9, stage_9, merged_datasets_9, coupled_datasets_9, leaves_9, descendants_9),
    ],
)
def test_dataset_name_report(params, stage, merged_datasets, coupled_datasets, leaves, descendants):
    report = DatasetNameReport(params, stage)
    assert report.merged_datasets == merged_datasets
    assert report.coupled_datasets == coupled_datasets
    assert report.leaves == leaves
    assert report.descendants == descendants


##################################################
# `validate_dataset_names` Scenarios
##################################################
@pytest.mark.parametrize(
    ["params", "stage", "expected"],
    [
        ((TR_I, HO_I, TE_I), "pre_cv", []),
        ((TR_I, VA_I, HO_I, TE_I), "intra_cv", []),
        ((TR_I, NT_I), "pre_cv", [NT_I]),
        ((TR_I, NT_I), "intra_cv", [NT_I]),
        ((A_I, A_T), "pre_cv", [A_I, A_T]),
        ((A_I, A_T), "intra_cv", [A_I, A_T]),
        ((TR_D, NT_I), "pre_cv", [NT_I]),
        ((TR_D, NT_I), "intra_cv", [NT_I]),
        ((TR_D, NT_D), "pre_cv", [NT_D]),
        ((TR_D, NT_D), "intra_cv", [NT_D]),
    ],
)
def test_validate_dataset_names(params, stage, expected):
    assert validate_dataset_names(params, stage) == expected


@pytest.mark.parametrize("params", [(A_D, A_T), (A_T, TR_T)])
@pytest.mark.parametrize("stage", ["pre_cv", "intra_cv"])
def test_validate_dataset_names_value_error(params, stage):
    with pytest.raises(ValueError, match="Requested params include duplicate references to .*"):
        validate_dataset_names(params, stage)


##################################################
# `get_engineering_step_params` Scenarios
##################################################
def _err_param_train_data(train_data):
    return train_data


def _err_param_validation_data(validation_data):
    return validation_data


def _err_param_holdout_data(holdout_data):
    return holdout_data


def _err_param_all_data(all_data):
    return all_data


def _err_param_non_train_data(non_train_data):
    return non_train_data


@pytest.mark.parametrize(
    "f",
    [
        _err_param_train_data,
        _err_param_validation_data,
        _err_param_holdout_data,
        _err_param_all_data,
        _err_param_non_train_data,
    ],
)
def test_get_engineering_step_params_value_error(f):
    with pytest.raises(ValueError, match="Sorry, 'data'-suffixed parameters like .*"):
        get_engineering_step_params(f)
