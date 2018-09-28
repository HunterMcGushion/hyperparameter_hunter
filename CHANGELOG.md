<a name="Unreleased"></a>
## [Unreleased]

### Features
* Added support for multiple `target_column` values. Previously, `target_column` was required to be a string naming a single 
target output column in the dataset. Now, `target_column` can also be a list of strings, enabling usage with multiple-output 
problems (for example, multi-class image classification)
    * Example using Keras with UCI's hand-written digits dataset:
    
    ```python
  from hyperparameter_hunter import Environment, CrossValidationExperiment
  import pandas as pd
  from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape
  from keras.models import Sequential
  from keras.wrappers.scikit_learn import KerasClassifier
  from sklearn.datasets import load_digits
    
    
  def prep_data(n_class=10):
      input_data, target_data = load_digits(n_class=n_class, return_X_y=True)
      train_df = pd.DataFrame(data=input_data, columns=["c_{:02d}".format(_) for _ in range(input_data.shape[1])])
      train_df["target"] = target_data
      train_df = pd.get_dummies(train_df, columns=["target"], prefix="target")
      return train_df
    
    
  def build_fn(input_shape=-1):
      model = Sequential([
          Reshape((8, 8, -1), input_shape=(64,)),
          Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu"),
          MaxPooling2D(pool_size=(2, 2)),
          Dropout(0.5),
          Flatten(),
          Dense(10, activation="softmax"),
      ])
      model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
      return model
    
  env = Environment(
      train_dataset=prep_data(),
      root_results_path="HyperparameterHunterAssets",
      metrics_map=["roc_auc_score"],
      target_column=[f"target_{_}" for _ in range(10)],
      cross_validation_type="StratifiedKFold",
      cross_validation_params=dict(n_splits=10, shuffle=True, random_state=True),
  )

  experiment = CrossValidationExperiment(
      model_initializer=KerasClassifier,
      model_init_params=build_fn,
      model_extra_params=dict(batch_size=32, epochs=10, verbose=0, shuffle=True),
  )
    ```
* Added callback recipes, which contains some commonly-used extra callbacks created using 
`hyperparameter_hunter.callbacks.bases.lambda_callback`
    * This serves not only to provide additional callback functionality like creating confusion 
    matrices, but also to create examples for how anyone can use `lambda_callback` to implement 
    their own custom functionality
    * This also contains the replacement for the broken `AggregatorEpochsElapsed` callback: 
    `aggregator_epochs_elapsed`
* Updated `hyperparameter_hunter.callbacks.bases.lambda_callback` to handle automatically 
aggregating values returned by "on_..." callable parameters
    * This new functionality is used in `callbacks.recipes.confusion_matrix_oof`; whereas, 
    `callbacks.recipes.confusion_matrix_holdout` continues to aggregate values using the original 
    method for comparison 


### Bug-Fixes
* Fixed bug requiring Keras to be installed even when not in use
* Fixed bug where OptimizationProtocols would not take into account saved result files when 
determining whether the hyperparameter search space had been exhausted
* Fixed bug where Hyperparameter Optimization headers were not properly underlined
* Fixed bug where `AggregatorEpochsElapsed` would not work with repeated cross validation schemes 
(#47) by converting it to a `lambda_callback` recipe in `hyperparameter_hunter.callbacks.recipes` 

### Changes
* Adopted [Black](https://github.com/ambv/black) code formatting
    * Breaks compatibility with result files created by previous HyperparameterHunter versions due 
    to docstring reformatting of default functions used by `cross_experiment_key`
* Miscellaneous formatting changes and code cleanup suggested by Black, Flake8, Codacy, and Code 
Climate
* Development-related changes, including minor TravisCI revisions, pre-commit hooks, and updated 
utility/documentation files
* `experiment_core` no longer applies a callback to record epochs elapsed for Keras NNs by default. 
For this functionality, use `callbacks.recipes.aggregator_epochs_elapsed`

<a name="1.0.2"></a>
## [1.0.2] (2018-08-26)

### Features
* Added `sentinels` module, which includes :class:`DatasetSentinel` that allows users to pass yet-undefined datasets as arguments
to Experiments or OptimizationProtocols
    * This functionality can be achieved by using the following new properties of :class:`environment.Environment`:
    [`train_input`, `train_target`, `validation_input`, `validation_target`, `holdout_input`, `holdout_target`]
    * Example usage:

    ```python
  from hyperparameter_hunter import Environment, CrossValidationExperiment
  from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
  from xgboost import XGBClassifier

  env = Environment(
      train_dataset=get_breast_cancer_data(target='target'),
      root_results_path='HyperparameterHunterAssets',
      metrics_map=['roc_auc_score'],
      cross_validation_type='StratifiedKFold',
      cross_validation_params=dict(n_splits=10, shuffle=True, random_state=32),
  )
    
  experiment = CrossValidationExperiment(
      model_initializer=XGBClassifier,
      model_init_params=dict(objective='reg:linear', max_depth=3, n_estimators=100, subsample=0.5),
      model_extra_params=dict(
          fit=dict(
              eval_set=[(env.train_input, env.train_target), (env.validation_input, env.validation_target)],
              early_stopping_rounds=5
          )
      )
  )
    ```

* Added ability to print `experiment_id` (or first n characters) during optimization rounds via the `show_experiment_id` kwarg in
:class:`hyperparameter_hunter.reporting.OptimizationReporter` (#42)
* Lots of other documentation additions, and improvements to example scripts

### Bug-Fixes
* Moved the temporary `build_fn` file created during Keras optimization, so there isn't a temporary file floating around in the
present working directory (#54)
* Fixed :meth:`models.XGBoostModel.fit` using `eval_set` by default with introduction of :class:`sentinels.DatasetSentinel`,
allowing users to define `eval_set` only if they want to (#22)

<a name="1.0.1"></a>
## [1.0.1] (2018-08-19)

### Bug-Fixes
* Fixed bug where `nbconvert`, and `nbformat` were required even when not using an iPython notebook

<a name="1.0.0"></a>
## [1.0.0] (2018-08-19)

### Features
* Simplified providing hyperparameter search dimensions during optimization
    * Old method of providing search dimensions:

        ```python
      from hyperparameter_hunter import BayesianOptimization, Real, Integer, Categorical

      optimizer = BayesianOptimization(
          iterations=100, read_experiments=True, dimensions=[
              Integer(name='max_depth', low=2, high=20),
              Real(name='learning_rate', low=0.0001, high=0.5),
              Categorical(name='booster', categories=['gbtree', 'gblinear', 'dart'])
          ]
      )
      optimizer.set_experiment_guidelines(
          model_initializer=XGBClassifier,
          model_init_params=dict(n_estimators=200, subsample=0.5, learning_rate=0.1)
      )
      optimizer.go()
        ```
    * New method:

        ```python
      from hyperparameter_hunter import BayesianOptimization, Real, Integer, Categorical

      optimizer = BayesianOptimization(iterations=100, read_experiments=True)
      optimizer.set_experiment_guidelines(
          model_initializer=XGBClassifier,
          model_init_params=dict(
              n_estimators=200, subsample=0.5,
              learning_rate=Real(0.0001, 0.5),
              max_depth=Integer(2, 20),
              booster=Categorical(['gbtree', 'gblinear', 'dart'])
          )
      )
      optimizer.go()
        ```
    * The `dimensions` kwarg is removed from the OptimizationProtocol classes, and hyperparameter search dimensions are now provided along with the concrete hyperarameters via `set_experiment_guidelines`. If a value is a descendant of `hyperparameter_hunter.space.Dimension`, it is automatically detected as a space to be searched and optimized
* Improved support for Keras hyperparameter optimization
    * Keras Experiment:

        ```python
      from hyperparameter_hunter import CrossValidationExperiment
      from keras import *

      def build_fn(input_shape):
          model = Sequential([
              Dense(100, kernel_initializer='uniform', input_shape=input_shape, activation='relu'),
              Dropout(0.5),
              Dense(1, kernel_initializer='uniform', activation='sigmoid')
          ])
          model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
          return model

      experiment = CrossValidationExperiment(
          model_initializer=KerasClassifier,
          model_init_params=build_fn,
          model_extra_params=dict(
              callbacks=[ReduceLROnPlateau(patience=5)],
              batch_size=32, epochs=10, verbose=0
          )
      )
        ```
    * Keras Optimization:

        ```python
      from hyperparameter_hunter import Real, Integer, Categorical, RandomForestOptimization
      from keras import *

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
* Lots of other new features and bug-fixes

<a name="0.0.1"></a>
## 0.0.1 (2018-06-14)

### Features
* Initial release


[Unreleased]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v0.0.1...v1.0.0