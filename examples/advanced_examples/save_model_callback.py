from hyperparameter_hunter import Environment, CVExperiment, lambda_callback
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
import numpy as np
from os import listdir, makedirs
from pprint import pprint as pp
from xgboost import XGBClassifier

##################################################
# Define Custom LambdaCallback
##################################################
# We'll make a simple Callback to save trained models, using our `lambda_callback` function
# Our first version will simply save every model after each training "run"
# This callback will be designed for XGBoost, but keep in mind that libraries have different ways
#   of saving their models, so you should tailor your callback to the library you're using


# This is just a temporary directory to hold our model files
makedirs("_saved_models", exist_ok=True)


# As with all HyperparameterHunter callback functions, we just need to declare the Experiment
#   attributes we want in the function signature, which will be automatically passed through

# For now, we'll use `_rep`, `_fold` and `_run` to name the files. Of course, we'll also use
#   `model`, which will pass the entire `Model` instance held by our Experiment
def save_model(_rep, _fold, _run, model):
    model.model.save_model(f"_saved_models/model_{_rep}_{_fold}_{_run}")
    print(f"Model saved for R{_rep} / f{_fold} / r{_run}")


# IMPORTANT: We're using `model.model.save_model` above because the `model` attribute of Experiments
#   is a HyperparameterHunter `Model` instance that has another `model` attribute. This nested
#   `model` is the actual model we're fitting (`XGBClassifier`).

# XGBoost classes define the `save_model` method to... well, save models...
# For other libraries, we would still use `model.model`, but we would replace `save_model` with
#   the method/function that library defines for model saving, or whatever tool you may want to
#   use to save your model (Pickle, Joblib, etc.)


cb_save_all = lambda_callback(on_run_end=save_model)
# Now we have our very own callback ready to go!

# There are two places we can tell HyperparameterHunter about our callback:
#   1. `Environment`, with the `experiment_callbacks` kwarg, or
#   2. `CVExperiment`, with the `callbacks` kwarg
# Using `Environment` will automatically tell all our Experiments about the callbacks, whereas
#   providing `callbacks` directly to `CVExperiment` restricts callbacks to just that Experiment

# We'll give our `callbacks` to `CVExperiment` because we want to try different callbacks later


##################################################
# Set Up Environment
##################################################
env = Environment(
    train_dataset=get_breast_cancer_data("target"),
    results_path="HyperparameterHunterAssets",
    metrics=["roc_auc_score"],
    cv_type="StratifiedKFold",
    cv_params=dict(n_splits=10, shuffle=True, random_state=32),
    runs=1,
    # experiment_callbacks=[cb_save_all],  # Do this to give callbacks to all Experiments
)

##################################################
# Experiment
##################################################
exp_0 = CVExperiment(
    model_initializer=XGBClassifier,
    model_init_params=dict(objective="reg:linear", max_depth=3, n_estimators=100, subsample=0.5),
    model_extra_params=dict(fit=dict(eval_set=[(env.validation_input, env.validation_target)])),
    callbacks=[cb_save_all],  # Here's our callback!
)

# Let's make sure our models actually got saved
print(listdir("_saved_models"))


# But who really wants to save all 10 models for the Experiment? Let's just save the best one!

# Maybe we're feeling a bit confused, though. Which Experiment attributes do we need for our next
#   callback? What Experiment attributes can we even use in the first place???

# Well there's a callback for that! In fact, there's a special parameter we can use to get ALL of
#   the Experiment's attributes: `kwargs`
def wtf(kwargs):
    for attr, val in sorted(kwargs.items()):
        print(f"   -   {attr:30}   -   {type(val)}")
        # We're just printing out the `type` of each value to avoid a wall of text

    # PRO TIP: It's much easier to view `kwargs` by using your IDE to set a breakpoint here

    # For now, we'll skip ahead by printing the very informative `stat_aggregates` attribute
    pp(kwargs["stat_aggregates"])


cb_wtf = lambda_callback(on_exp_end=wtf)
exp_1 = CVExperiment(model_initializer=XGBClassifier, callbacks=[cb_wtf])

# So what are we looking at? First, we have a long list of all the attributes in our Experiment,
#   with the type of each value. All of these attribute names are valid parameters to use in the
#   signature of our `lambda_callback` functions
# Below the list of attributes is the Experiment's `stat_aggregates`. This is a dict that records
#   various stats across all of the Experiment's time "divisions" (reps/folds/runs).
# Here, we can see that our `stat_aggregates` has two keys: "evaluations" and "times"
# Let's take a closer look at "evaluations", which keeps track of metrics for each division

pp(exp_1.stat_aggregates["evaluations"])

##################################################
# Save Best Models
##################################################
# Make another temporary directory to hold the models from our new callback
makedirs("_saved_models/best", exist_ok=True)


def save_best_model(_rep, _fold, _run, model, stat_aggregates):
    all_scores = stat_aggregates["evaluations"]["oof_roc_auc_score"]["runs"]
    # Get all scores by "run" that have been recorded.
    # NOTE: We need the "oof_" prefix in "oof_roc_auc_score" above to specify we want our
    #   out-of-fold scores as opposed to "holdout" scores

    current_score = all_scores[-1]
    # Get the last (most recent) score in the above `all_scores` list

    if current_score >= np.max(all_scores):
        # Check if our current score is greater than all previous scores
        # If you're measuring loss/error, change `np.max` to `np.min`

        print(f"NEW BEST:   {current_score}   @ R{_rep} / f{_fold} / r{_run}")
        model.model.save_model(f"_saved_models/best/model_{_rep}_{_fold}_{_run}_{current_score}")
        # If you want to overwrite your previous best model files, use the line below, instead
        # model.model.save_model(f"_saved_models/best/model")


cb_save_best = lambda_callback(on_run_end=save_best_model)
exp_2 = CVExperiment(model_initializer=XGBClassifier, callbacks=[cb_save_best])

# Check that our best models were saved
print(listdir("_saved_models/best"))
