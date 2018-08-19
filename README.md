HyperparameterHunter
====================

![HyperparameterHunter Overview](docs/media/overview.gif)

[![Documentation Status](https://readthedocs.org/projects/hyperparameter-hunter/badge/?version=latest)](https://hyperparameter-hunter.readthedocs.io/en/latest/?badge=latest)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=Q3EX3PQUV256G)

HyperparameterHunter provides wrappers for machine learning algorithms that
automatically save the testing conditions/hyperparameters, results, predictions, and
other data in a unified format. HyperparameterHunter simplifies the experimentation and hyperparameter
tuning process by allowing you to spend less time setting stuff up, and more time
doing the important stuff.

* **Source:** https://github.com/HunterMcGushion/hyperparameter_hunter
* **Documentation:** [https://hyperparameter-hunter.readthedocs.io](https://hyperparameter-hunter.readthedocs.io/en/latest/index.html)

Features
--------
* Truly informed hyperparameter optimization that automatically uses past Experiments
* Eliminate boilerplate code for cross-validation loops, predicting, and scoring
* Stop worrying about keeping track of hyperparameters, scores, or re-running the same Experiments

Getting Started
---------------

### 1) Environment:

Set up an Environment to organize Experiments and Optimization results.
<br>
Any Experiments or Optimization rounds we perform will use our active Environment.

```python
from hyperparameter_hunter import Environment, CrossValidationExperiment
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

env = Environment(
    train_dataset=df,  # Add holdout/test dataframes, too
    root_results_path='path/to/results/directory',  # Where your result files will go
    metrics_map=['roc_auc_score'],  # Callables, or strings referring to `sklearn.metrics`
    cross_validation_type=StratifiedKFold,  # Class, or string in `sklearn.model_selection`
    cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32)
)
```

### 2) Individual Experimentation:

Perform Experiments with your favorite libraries simply by providing model initializers and hyperparameters
<!-- Keras -->

<details>
<summary>Keras</summary>

```python
# Same format used by `keras.wrappers.scikit_learn`. Nothing new to learn
def build_fn(input_shape):  # `input_shape` calculated for you
    model = Sequential([
        Dense(100, kernel_initializer='uniform', input_shape=input_shape, activation='relu'),
        Dropout(0.5),
        Dense(1, kernel_initializer='uniform', activation='sigmoid')
    ])  # All layer arguments saved (whether explicit or Keras default) for future use
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

experiment = CrossValidationExperiment(
    model_initializer=KerasClassifier,
    model_init_params=build_fn,  # We interpret your build_fn to save hyperparameters in a useful, readable format
    model_extra_params=dict(
        callbacks=[ReduceLROnPlateau(patience=5)],  # Use Keras callbacks
        batch_size=32, epochs=10, verbose=0  # Fit/predict arguments
    )
)
```

</details>

<!-- SKLearn -->
<details>
<summary>SKLearn</summary>

```python
experiment = CrossValidationExperiment(
    model_initializer=LinearSVC,  # (Or any of the dozens of other SK-Learn algorithms)
    model_init_params=dict(penalty='l1', C=0.9)  # Default values used and recorded for kwargs not given
)
```
</details>
<!-- XGBoost -->
<details open>
<summary>XGBoost</summary>

```python
experiment = CrossValidationExperiment(
    model_initializer=XGBClassifier,
    model_init_params=dict(objective='reg:linear', max_depth=3, n_estimators=100, subsample=0.5)
)
```
</details>
<!-- LightGBM -->
<details>
<summary>LightGBM</summary>

```python
experiment = CrossValidationExperiment(
    model_initializer=LGBMClassifier,
    model_init_params=dict(boosting_type='gbdt', num_leaves=31, max_depth=-1, min_child_samples=5, subsample=0.5)
)
```
</details>
<!-- CatBoost -->
<details>
<summary>CatBoost</summary>

```python
experiment = CrossValidationExperiment(
    model_initializer=CatboostClassifier,
    model_init_params=dict(iterations=500, learning_rate=0.01, depth=7, allow_writing_files=False),
    model_extra_params=dict(fit=dict(verbose=True))  # Send kwargs to `fit` and other extra methods
)
```
</details>
<!-- RGF -->
<details>
<summary>RGF</summary>

```python
experiment = CrossValidationExperiment(
    model_initializer=RGFClassifier,
    model_init_params=dict(max_leaf=1000, algorithm='RGF', min_samples_leaf=10)
)
```
</details>

### 3) Hyperparameter Optimization:

Just like Experiments, but if you want to optimize a hyperparameter, use the classes imported below

```python
from hyperparameter_hunter import Real, Integer, Categorical
```

<!-- Keras -->
<details>
<summary>Keras</summary>

```python
def build_fn(input_shape):
    model = Sequential([
        Dense(Integer(50, 150), input_shape=input_shape, activation='relu'),
        Dropout(Real(0.2, 0.7)),
        Dense(1, activation=Categorical(['sigmoid', 'softmax']))
    ])
    model.compile(
        optimizer=Categorical(['adam', 'rmsprop', 'sgd', 'adadelta']),
        loss='binary_crossentropy', metrics=['accuracy']
    )
    return model

optimizer = RandomForestOptimization(iterations=7)
optimizer.set_experiment_guidelines(
    model_initializer=KerasClassifier,
    model_init_params=build_fn,
    model_extra_params=dict(
        callbacks=[ReduceLROnPlateau(patience=Integer(5, 10))],
        batch_size=Categorical([32, 64]),
        epochs=10, verbose=0
    )
)
optimizer.go()
```
</details>

<!-- SKLearn -->
<details>
<summary>SKLearn</summary>

```python
optimizer = DummySearch(iterations=42)
optimizer.set_experiment_guidelines(
    model_initializer=AdaBoostClassifier,  # (Or any of the dozens of other SKLearn algorithms)
    model_init_params=dict(
        n_estimators=Integer(75, 150),
        learning_rate=Real(0.8, 1.3),
        algorithm='SAMME.R'
    )
)
optimizer.go()
```
</details>
<!-- XGBoost -->
<details open>
<summary>XGBoost</summary>

```python
optimizer = BayesianOptimization(iterations=10)
optimizer.set_experiment_guidelines(
    model_initializer=XGBClassifier,
    model_init_params=dict(
        max_depth=Integer(low=2, high=20),
        learning_rate=Real(0.0001, 0.5),
        n_estimators=200,
        subsample=0.5,
        booster=Categorical(['gbtree', 'gblinear', 'dart']),
    )
)
optimizer.go()
```
</details>
<!-- LightGBM -->
<details>
<summary>LightGBM</summary>

```python
optimizer = BayesianOptimization(iterations=100)
optimizer.set_experiment_guidelines(
    model_initializer=LGBMClassifier,
    model_init_params=dict(
        boosting_type=Categorical(['gbdt', 'dart']),
        num_leaves=Integer(5, 20),
        max_depth=-1,
        min_child_samples=5,
        subsample=0.5
    )
)
optimizer.go()
```
</details>
<!-- CatBoost -->
<details>
<summary>CatBoost</summary>

```python
optimizer = GradientBoostedRegressionTreeOptimization(iterations=32)
optimizer.set_experiment_guidelines(
    model_initializer=CatBoostClassifier,
    model_init_params=dict(
        iterations=100,
        eval_metric=Categorical(['Logloss', 'Accuracy', 'AUC']),
        learning_rate=Real(low=0.0001, high=0.5),
        depth=Integer(4, 7),
        allow_writing_files=False
    )
)
optimizer.go()
```
</details>
<!-- RGF -->
<details>
<summary>RGF</summary>

```python
optimizer = ExtraTreesOptimization(iterations=10)
optimizer.set_experiment_guidelines(
    model_initializer=RGFClassifier,
    model_init_params=dict(
        max_leaf=1000,
        algorithm=Categorical(['RGF', 'RGF_Opt', 'RGF_Sib']),
        l2=Real(0.01, 0.3),
        normalize=Categorical([True, False]),
        learning_rate=Real(0.3, 0.7),
        loss=Categorical(['LS', 'Expo', 'Log', 'Abs'])
    )
)
optimizer.go()
```
</details>

Installation
------------

```
pip install hyperparameter-hunter
```

If you like being on the cutting-edge, and you want all the latest developments, run:

```
pip install git+https://github.com/HunterMcGushion/hyperparameter_hunter.git
```

Tested Libraries
----------------
* [Keras](https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/keras_example.py)
* [scikit-learn](https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/sklearn_example.py)
* [LightGBM](https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/lightgbm_example.py)
* [CatBoost](https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/catboost_example.py)
* [XGBoost](https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/simple_example.py)
* [rgf_python](https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/rgf_example.py)
* ... More on the way
