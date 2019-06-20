from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter import BayesianOptPro, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

#################### Using Multiple Metrics in Environments ####################
# This example will go over how to record multiple metrics with HyperparameterHunter, how to
# ... interpret the results, and how to switch between them for hyperparameter optimization.

# As with most examples, we will start with preparing our data.

#################### 1. Format DataFrame ####################
data = load_breast_cancer()
train_df = pd.DataFrame(data.data, columns=[_.replace(" ", "_") for _ in data.feature_names])
train_df["diagnosis"] = data.target

#################### 2. Set Up Environment ####################
# Now we'll set up our `Environment`. If you've gone through the other examples, everything below
# ... should be pretty standard, except for the `metrics`. In most examples, we give `metrics`
# ... a single metric to record, but what if we just can't choose? Answer: Give `Environment` a bunch
# ... of metrics in `metrics`! Notice that we provide the individual metrics in a few different
# ... formats accepted and documented by `Environment`.

# First, near the top, we import `f1_score` from `sklearn.metrics`. Continuing to our `metrics`...
# 1. We start with the string "roc_auc_score", identifying the `sklearn.metrics` callable, and we name it **"roc_auc"**
# 2. We add our imported `f1_score`, and name it **"f1"**
# 3. We customize `f1_score` to use the `average="micro"` kwarg, and we name it **"f1_micro"**, and
# 4. We customize `f1_score` again, using the `average="macro"` kwarg this time, and we name it **"f1_macro"**
env = Environment(
    train_dataset=train_df,
    results_path="HyperparameterHunterAssets",
    target_column="diagnosis",
    metrics=dict(
        roc_auc="roc_auc_score",
        f1=f1_score,
        f1_micro=lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
        f1_macro=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    ),
    cv_type="KFold",
    cv_params=dict(n_splits=10, shuffle=True, random_state=42),
    verbose=1,
)

# Now, any Experiments we execute will record all four of these metrics!

#################### 3. Perform Experiments ####################
experiment_0 = CVExperiment(
    model_initializer=LGBMClassifier,
    model_init_params=dict(
        boosting_type="gbdt", max_depth=-1, min_child_samples=5, subsample=0.5, verbose=-1
    ),
)
# As we can see above, the final report for `experiment_0` shows all four metrics, each with different values.

# You may be wondering what happens when we perform hyperparameter optimization. Which of our metrics will be optimized?
# An excellent question! The answer is, the first metric - unless we tell our optimizer otherwise. An example will better illustrate this.

#################### 4. Hyperparameter Optimization ####################
# We'll start by setting aside a `model_init_params` dict, so we can easily reuse them later.
# ... That's all - nothing sneaky going on there!
OPT_MODEL_INIT_PARAMS = dict(
    boosting_type=Categorical(["gbdt", "dart"]),
    num_leaves=Integer(15, 45),
    max_depth=-1,
    min_child_samples=5,
    subsample=Real(0.4, 0.7),
    verbose=-1,
)

optimizer_0 = BayesianOptPro(iterations=2, random_state=32)
optimizer_0.set_experiment_guidelines(LGBMClassifier, OPT_MODEL_INIT_PARAMS)
optimizer_0.go()

# Now, take note of the single saved experiment that was found by `optimizer_0`. It lists the
# ... experiment ID given to the `experiment_0` we performed above. Furthermore, `optimizer_0`, lists
# ... the value of `experiment_0` as 0.95858. Therefore, we know that `optimizer_0` is using "roc_auc"
# ... score as its `target_metric` to optimize, because that is the final "roc_auc" value reported by `experiment_0`.

#################### 5. Changing Target Metrics ####################
# Suppose we now want to perform additional rounds of `BayesianOptPro` using our "f1_micro"
# ... metric as the optimized `target_metric`, instead. We would need to start all over from scratch,
# ... right? WRONG! HyperparameterHunter recorded all four of the metrics we declared in `env` for
# ... all experiments executed during optimization, as well!

# Even better, telling HyperparameterHunter to switch `target_metric`s is easy! Here's how to do it:
optimizer_1 = BayesianOptPro(target_metric="f1_micro", iterations=2, random_state=32)
optimizer_1.set_experiment_guidelines(LGBMClassifier, OPT_MODEL_INIT_PARAMS)
optimizer_1.go()

# The only difference between the code for `optimizer_1` and the code for `optimizer_0` before is
# ... the addition of `target_metric="f1_micro"`.

# That's all we have to do! Notice that, once again, we see `experiment_0` at the top of the saved
# ... experiments being learned from, and now it shows a value of 0.96585. With a quick scroll upwards,
# ... we can verify that is the "f1_micro" score originally reported by `experiment_0`.

# We can also see two other saved experiments that were located, which are the two experiments
# ... produced by `optimizer_0`. Note that their values also differ from those reported by `optimizer_0`,
# ... because `target_metric="f1_micro"` now, instead of the inferred "roc_auc" default.

#################### 6. I Can't Make Up My Mind ####################
# What if we now decide that we actually want to optimize using our normal "f1" metric, instead of
# ... either "roc_auc" or "f1_micro"? Easy!
optimizer_2 = BayesianOptPro(target_metric="f1", iterations=2, random_state=32)
optimizer_2.set_experiment_guidelines(LGBMClassifier, OPT_MODEL_INIT_PARAMS)
optimizer_2.go()

# Just like that, `optimizer_2` is reporting our "f1" scores! Let's finish by optimizing with the last of our four metrics.
optimizer_3 = BayesianOptPro(target_metric="f1_macro", iterations=2, random_state=32)
optimizer_3.set_experiment_guidelines(LGBMClassifier, OPT_MODEL_INIT_PARAMS)
optimizer_3.go()

#################### 7. Bonus Exercises ####################
# If you've been reading the documentation as you should be, you may have noticed the `target_metric`
# ... argument of all children of `BaseOptPro` is usually a tuple. The `BayesianOptPro`
# ... class we used above is just one of the descendants of `BaseOptPro`, but we were
# ... passing `target_metric` values of strings.

# As the documentation notes, all `target_metric` values are cast to tuples, in which the first value
# ... identifies which dataset's evaluations should be used. The default behavior is to target the
# ... "oof", or out-of-fold, predictions' results. So, when we were using `target_metric="<string>"`
# ... in our examples above, our optimizer interpreted it as `target_metric=("oof", "<string>")`.

# This allows us to tell our optimizers to optimize metrics calculated using predictions on other
# ... datasets, like a holdout dataset. For example, had we initialized `Environment` with a
# ... `holdout_dataset`, our experiments would actually calculate 8 metrics instead of the 4 they
# ... currently do: 4 for our OOF predictions, and 4 for our holdout predictions. Then, if we wanted
# ... to optimize using holdout evaluations, we can use `target_metric=("holdout", <metric_name>)`.
