"""To execute this script, the competition data must be added to a "data" directory, located in the
same directory as the script. "data" should contain the competition's "test.csv" and "train.csv"
files, which can be found here: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

This is a HyperparameterHunter adaptation of a Kaggle kernel for the "Porto Seguro's Safe Driver
Prediction" competition (https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283).
The kernel's author is Olivier Grellier (https://www.kaggle.com/ogrellier), and he deserves
absolutely all the credit for his excellent work. This script is simply an adaptation of Olivier
Grellier's kernel to take advantage of HyperparameterHunter's advanced record-keeping and
hyperparameter optimization, among numerous other fantastical features.

Note that the docstring for :func:`target_encode` was written by Olivier Grellier, with some minor
modifications. This adaptation uses the Gini function created by Mohsin Hasan
(https://www.kaggle.com/tezdhar/faster-gini-calculation), rather then the function used by the
original kernel. Below is Olivier Grellier's original module-level docstring for his kernel:

This simple scripts demonstrates the use of xgboost eval results to get the best round for the
current fold and across folds. It also shows an upsampling method that limits cross-validation
overfitting"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, FeatureEngineer, EngineerStep

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
import pandas as pd

##################################################
# Import Learning Assets
##################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


##################################################
# Custom Metrics
##################################################
def gini_c(target, prediction):
    target = np.asarray(target)  # In case someone passes Series or list
    n = len(target)
    t_s = target[np.argsort(prediction)]
    t_c = t_s.cumsum()
    gini_sum = t_c.sum() / t_s.sum() - (n + 1) / 2.0
    return gini_sum / n


def gini_normalized_c(target, prediction):
    # Below changed because targets/predictions passed as 2D arrays to metric functions
    if prediction.shape[1] == 2:
        prediction = prediction[:, 1]  # If proba array contains both 0 and 1 classes, pick class 1
    if prediction.ndim == 2:  # Required for sklearn wrapper
        prediction = prediction[:, 0]
    if target.ndim == 2:  # Required for sklearn wrapper
        target = target[:, 0]
    return gini_c(target, prediction) / gini_c(target, target)


def gini_xgb(prediction, target):
    target = target.get_label()
    gini_score = gini_normalized_c(target, prediction)
    return [("gini", gini_score)]


##################################################
# Feature Engineering
##################################################
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(
    trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0
):
    """Smoothing is computed like in the following paper by Daniele Micci-Barreca (download link):
    http://helios.mm.di.uoa.gr/~rouvas/ssi/sigkdd/sigkdd.vol3.1/barreca.ps

    Parameters
    ----------
    trn_series: pd.Series
        Training categorical feature as a pd.Series
    tst_series: pd.Series
        Test categorical feature as a pd.Series
    target: pd.Series
        Target data as a pd.Series
    min_samples_leaf: Int
        Minimum samples to take category average into account
    smoothing: Int
        Smoothing effect to balance categorical average vs prior
    noise_level
        ...

    Returns
    -------
    ft_trn_series: pd.Series
        ...
    ft_tst_series: pd.Series
        ...
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data
    prior = target.mean()

    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    #################### Apply Averages to Train Series ####################
    # This section has been slightly modified from the original for clarity
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={"index": target.name, target.name: "average"}),
        on=trn_series.name,
        how="left",
    )
    ft_trn_series = ft_trn_series["average"]
    ft_trn_series = ft_trn_series.rename(trn_series.name + "_mean")
    ft_trn_series = ft_trn_series.fillna(prior)
    ft_trn_series.index = trn_series.index  # Restore index after `pd.merge`

    #################### Apply Averages to Test Series ####################
    # This section has been slightly modified from the original for clarity
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={"index": target.name, target.name: "average"}),
        on=tst_series.name,
        how="left",
    )
    ft_tst_series = ft_tst_series["average"]
    ft_tst_series = ft_tst_series.rename(trn_series.name + "_mean")
    ft_tst_series = ft_tst_series.fillna(prior)
    ft_tst_series.index = tst_series.index  # Restore index after `pd.merge`

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def feature_combinations(train_inputs, test_inputs, train_targets):
    combos = [("ps_reg_01", "ps_car_02_cat"), ("ps_reg_01", "ps_car_04_cat")]

    #################### Build Combination Features ####################
    for n_c, (f1, f2) in enumerate(combos):
        name1 = f1 + "_plus_" + f2

        train_inputs[name1] = (
            train_inputs[f1].apply(lambda x: str(x))
            + "_"
            + train_inputs[f2].apply(lambda x: str(x))
        )
        test_inputs[name1] = (
            test_inputs[f1].apply(lambda x: str(x)) + "_" + test_inputs[f2].apply(lambda x: str(x))
        )

        #################### Label Encode ####################
        lbl = LabelEncoder()
        lbl.fit(list(train_inputs[name1].values) + list(test_inputs[name1].values))
        train_inputs[name1] = lbl.transform(list(train_inputs[name1].values))
        test_inputs[name1] = lbl.transform(list(test_inputs[name1].values))

    #################### Target Encode ####################
    categorical_features = [f for f in train_inputs.columns if "_cat" in f]

    for f in categorical_features:
        train_inputs[f + "_avg"], test_inputs[f + "_avg"] = target_encode(
            trn_series=train_inputs[f],
            tst_series=test_inputs[f],
            target=train_targets["target"],
            min_samples_leaf=200,
            smoothing=10,
            noise_level=0,
        )

    return train_inputs, test_inputs, train_targets


def upsample_train_data(train_inputs, train_targets):
    # Get positive examples
    # NOTE: Differs from original kernel with addition of `["target"]` - Inputs always DataFrames
    pos = pd.Series(train_targets["target"] == 1)
    # Add positive examples
    train_inputs = pd.concat([train_inputs, train_inputs.loc[pos]], axis=0)
    train_targets = pd.concat([train_targets, train_targets.loc[pos]], axis=0)
    # Shuffle data
    idx = np.arange(len(train_inputs))
    np.random.shuffle(idx)
    train_inputs = train_inputs.iloc[idx]
    train_targets = train_targets.iloc[idx]
    return train_inputs, train_targets


##################################################
# HyperparameterHunter
##################################################
def execute():
    env = Environment(
        train_dataset="data/train.csv",
        test_dataset="data/test.csv",
        results_path="HyperparameterHunterAssets",
        target_column="target",
        metrics=dict(gini=gini_normalized_c),
        id_column="id",
        cv_type=StratifiedKFold,
        cv_params=dict(n_splits=5, shuffle=True, random_state=15),
        do_predict_proba=1,
        to_csv_params=dict(index=False),  # Drops index from final prediction files
    )

    exp = CVExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            n_estimators=200,
            max_depth=4,
            objective="binary:logistic",
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0,
            reg_lambda=1,
            nthread=2,
        ),
        model_extra_params=dict(
            eval_set=[
                (env.train_input, env.train_target),
                (env.validation_input, env.validation_target),
            ],
            eval_metric=gini_xgb,
            early_stopping_rounds=None,
            verbose=False,
        ),
        feature_engineer=FeatureEngineer(
            [feature_combinations, EngineerStep(upsample_train_data, stage="intra_cv")]
        ),
        feature_selector=[
            "ps_car_13",  # : 1571.65 / shadow  609.23
            "ps_reg_03",  # : 1408.42 / shadow  511.15
            "ps_ind_05_cat",  # : 1387.87 / shadow   84.72
            "ps_ind_03",  # : 1219.47 / shadow  230.55
            "ps_ind_15",  # :  922.18 / shadow  242.00
            "ps_reg_02",  # :  920.65 / shadow  267.50
            "ps_car_14",  # :  798.48 / shadow  549.58
            "ps_car_12",  # :  731.93 / shadow  293.62
            "ps_car_01_cat",  # :  698.07 / shadow  178.72
            "ps_car_07_cat",  # :  694.53 / shadow   36.35
            "ps_ind_17_bin",  # :  620.77 / shadow   23.15
            "ps_car_03_cat",  # :  611.73 / shadow   50.67
            "ps_reg_01",  # :  598.60 / shadow  178.57
            "ps_car_15",  # :  593.35 / shadow  226.43
            "ps_ind_01",  # :  547.32 / shadow  154.58
            "ps_ind_16_bin",  # :  475.37 / shadow   34.17
            "ps_ind_07_bin",  # :  435.28 / shadow   28.92
            "ps_car_06_cat",  # :  398.02 / shadow  212.43
            "ps_car_04_cat",  # :  376.87 / shadow   76.98
            "ps_ind_06_bin",  # :  370.97 / shadow   36.13
            "ps_car_09_cat",  # :  214.12 / shadow   81.38
            "ps_car_02_cat",  # :  203.03 / shadow   26.67
            "ps_ind_02_cat",  # :  189.47 / shadow   65.68
            "ps_car_11",  # :  173.28 / shadow   76.45
            "ps_car_05_cat",  # :  172.75 / shadow   62.92
            "ps_calc_09",  # :  169.13 / shadow  129.72
            "ps_calc_05",  # :  148.83 / shadow  120.68
            "ps_ind_08_bin",  # :  140.73 / shadow   27.63
            "ps_car_08_cat",  # :  120.87 / shadow   28.82
            "ps_ind_09_bin",  # :  113.92 / shadow   27.05
            "ps_ind_04_cat",  # :  107.27 / shadow   37.43
            "ps_ind_18_bin",  # :   77.42 / shadow   25.97
            "ps_ind_12_bin",  # :   39.67 / shadow   15.52
            "ps_ind_14",  # :   37.37 / shadow   16.65
            "ps_car_11_cat",  # Very nice spot from Tilii : https://www.kaggle.com/tilii7
        ],
    )


if __name__ == "__main__":
    execute()
