<a name="Unreleased"></a>
## [Unreleased]

### Features
* Enabled optimization of tuple values via [`Categorical`](https://hyperparameter-hunter.readthedocs.io/en/stable/source/hyperparameter_hunter.space.html#hyperparameter_hunter.space.dimensions.Categorical)
    * This can be used with Keras to search over different `kernel_size` values for `Conv2D` or 
      `pool_size` values for `MaxPooling2D`, for example:
    ```python
  Conv2D(64, kernel_size=Categorical([(2, 2), (3, 3), (4, 4)]), activation="relu")
  MaxPooling2D(pool_size=Categorical([(1, 1), (3, 3)]))
    ```
    
### Changes
* Removed the "Validated Environment ..." log messages made when initializing an Experiment/OptPro


<a name="3.0.0"></a>
## [3.0.0] (2019-08-06) Artemis
*This changelog entry combines the contents of all 3.0.0 pre-release entries*

*Artemis: Greek goddess of hunting*

This is the most significant release since the birth of HyperparameterHunter, adding not only 
feature engineering, but also feature optimization. The goal of feature engineering in 
HyperparameterHunter is to enable you to manipulate your data however you need to, without imposing 
restrictions on what's allowed - all while seamlessly keeping track of your feature engineering 
steps so they can be learned from and optimized. In that spirit, feature engineering steps are 
defined by your very own functions. That may sound a bit silly at first, but it affords maximal
freedom and customization, with only the minimal requirement that you tell your function what data 
you want from HyperparameterHunter, and you give it back when you're done playing with it. 

The best way to really understand feature engineering in HyperparameterHunter is to dive into some
code and check out the "examples/feature_engineering_examples" directory. In no time at all, you'll 
be ready to spread your wings by experimenting with the creative feature engineering steps only you 
can build. Let your faithful assistant, HyperparameterHunter, meticulously and lovingly record them 
for you, so you can optimize your custom feature functions just like normal hyperparameters.

You're a glorious peacock, and we just wanna let you fly.

### Features
* Feature engineering via `FeatureEngineer` and `EngineerStep`
    * This will be a "brief" summary of the new features. For more detail, see the aforementioned
      "examples/feature_engineering_examples" directory or the extensively documented 
      [`FeatureEngineer`](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.feature_engineering.FeatureEngineer) 
      and [`EngineerStep`](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.feature_engineering.EngineerStep)
      classes 
    * `FeatureEngineer` can be passed as the `feature_engineer` kwarg to either: 
        1. Instantiate a [`CVExperiment`](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.experiments.BaseExperiment), or
        2. Call the [`forge_experiment`](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.optimization.html#hyperparameter_hunter.optimization.protocol_core.BaseOptPro.forge_experiment)
           method of any Optimization Protocol
    * `FeatureEngineer` is just a container for `EngineerStep`s
        * Instantiate it with a simple list of `EngineerStep`s, or functions to construct `EngineerStep`s
    * Most important `EngineerStep` parameter is a function you define to perform your data transformation (whatever that is)
        * This function is often creatively referred to as a "step function"
    * Step function definitions have only two requirements:
        1. Name the data you want to transform in the signature's input parameters
            * 16 different parameter names, documented in 
              [`EngineerStep`'s `params` kwarg](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.feature_engineering.EngineerStep)
        2. Return the data when you're done with it
    * Step functions may be given directly to `FeatureEngineer`, or wrapped in an `EngineerStep` for greater customization
    * Here are just a few step functions you might want to make:
    
    ```python
  from hyperparameter_hunter import CVExperiment, FeatureEngineer, EngineerStep
  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import QuantileTransformer, StandardScaler
  from sklearn.impute import SimpleImputer

  def standard_scale(train_inputs, non_train_inputs):
      s = StandardScaler()
      train_inputs[train_inputs.columns] = s.fit_transform(train_inputs.values)
      non_train_inputs[train_inputs.columns] = s.transform(non_train_inputs.values)
      return train_inputs, non_train_inputs
  
  def quantile_transform(train_targets, non_train_targets):
      t = QuantileTransformer(output_distribution="normal")
      train_targets[train_targets.columns] = t.fit_transform(train_targets.values)
      non_train_targets[train_targets.columns] = t.transform(non_train_targets.values)
      return train_targets, non_train_targets, t
  
  def set_nan(all_inputs):
      cols = [1, 2, 3, 4, 5]
      all_inputs.iloc[:, cols] = all_inputs.iloc[:, cols].replace(0, np.NaN)
      return all_inputs
  
  def impute_negative_one(all_inputs):
      all_inputs.fillna(-1, inplace=True)
      return all_inputs
  
  def impute_mean(train_inputs, non_train_inputs):
      imputer = SimpleImputer()
      train_inputs[train_inputs.columns] = imputer.fit_transform(train_inputs.values)
      non_train_inputs[train_inputs.columns] = imputer.transform(non_train_inputs.values)
      return train_inputs, non_train_inputs
  
  def sqr_sum_feature(all_inputs):
      all_inputs["my_sqr_sum_feature"] = all_inputs.agg(
          lambda row: np.sqrt(np.sum([np.square(_) for _ in row])),
          axis="columns",
      )
      return all_inputs
    
  def upsample_train_data(train_inputs, train_targets):
      pos = pd.Series(train_targets["target"] == 1)
      train_inputs = pd.concat([train_inputs, train_inputs.loc[pos]], axis=0)
      train_targets = pd.concat([train_targets, train_targets.loc[pos]], axis=0)
      return train_inputs, train_targets
  
  # Any of the above can be wrapped by `EngineerStep`, or added directly to a `FeatureEngineer`'s `steps`
  # Below, assume we have already activated an `Environment`
  exp_0 = CVExperiment(
      model_initializer=..., 
      model_init_params={},
      feature_engineer=FeatureEngineer([
          set_nan,
          EngineerStep(standard_scale),
          quantile_transform,
          EngineerStep(upsample_train_data, stage="intra_cv"),
      ]),
  )
    ```

* Feature optimization
    * `Categorical` can be used to optimize feature engineering steps, either as `EngineerStep` 
      instances or raw functions of the form expected by `EngineerStep`
    * Just throw your `Categorical` in with the rest of your `FeatureEngineer.steps` 
    * Features can, of course, be optimized alongside standard model hyperparameters 
        
    ```python
  from hyperparameter_hunter import GBRT, Real, Integer, Categorical, FeatureEngineer, EngineerStep
  import numpy as np
  import pandas as pd
  from sklearn.linear_model import Ridge
  from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
  
  def standard_scale(train_inputs, non_train_inputs):
      s = StandardScaler()
      train_inputs[train_inputs.columns] = s.fit_transform(train_inputs.values)
      non_train_inputs[train_inputs.columns] = s.transform(non_train_inputs.values)
      return train_inputs, non_train_inputs
  
  def min_max_scale(train_inputs, non_train_inputs):
      s = MinMaxScaler()
      train_inputs[train_inputs.columns] = s.fit_transform(train_inputs.values)
      non_train_inputs[train_inputs.columns] = s.transform(non_train_inputs.values)
      return train_inputs, non_train_inputs
  
  # Pretend we already set up our `Environment` and we want to optimize the our scaler
  # We'll also throw in some standard hyperparameter optimization - This is HyperparameterHunter, after all
  optimizer_0 = GBRT()
  optimizer_0.forge_experiment(
      Ridge, 
      dict(alpha=Real(0.5, 1.0), max_iter=Integer(500, 2000), solver="svd"), 
      feature_engineer=FeatureEngineer([Categorical([standard_scale, min_max_scale])])
  )
  
  # Then we remembered we should probably transform our target, too
  # OH NO! After transforming our targets, we'll need to `inverse_transform` the predictions!
  # OH YES! HyperparameterHunter will gladly accept a fitted transformer as an extra return value, 
  #   and save it to call `inverse_transform` on predictions 
  def quantile_transform(train_targets, non_train_targets):
      t = QuantileTransformer(output_distribution="normal")
      train_targets[train_targets.columns] = t.fit_transform(train_targets.values)
      non_train_targets[train_targets.columns] = t.transform(non_train_targets.values)
      return train_targets, non_train_targets, t
  
  # We can also tell HyperparameterHunter to invert predictions using a callable, rather than a fitted transformer
  def log_transform(all_targets):
      all_targets = np.log1p(all_targets)
      return all_targets, np.expm1
  
  optimizer_1 = GBRT()
  optimizer_1.forge_experiment(
      Ridge, {}, feature_engineer=FeatureEngineer([
          Categorical([standard_scale, min_max_scale]),
          Categorical([quantile_transform, log_transform]),
      ])
  )
    ``` 
    
* [`Categorical.optional`](https://hyperparameter-hunter.readthedocs.io/en/stable/source/hyperparameter_hunter.space.html#hyperparameter_hunter.space.dimensions.Categorical)
    * As `Categorical` is the means of optimizing `EngineerStep`s in simple lists, it became 
      necessary to answer the question of whether that crazy new feature you've been cooking up in 
      the lab should even be included at all
    * So the `optional` kwarg was added to `Categorical` to appease the mad scientist in us all
    * If True (default=False), the search space will include not only the `categories` you explicitly
      provide, but also the omission of the current `EngineerStep` entirely
    * `optional` is only intended for use in optimizing `EngineerStep`s. Don't expect it to work elsewhere
    * Brief example:
    
    ```python
  from hyperparameter_hunter import DummySearch, Categorical, FeatureEngineer, EngineerStep
  from sklearn.linear_model import Ridge

  def standard_scale(train_inputs, non_train_inputs):
      """Pretend this function scales data using SKLearn's `StandardScaler`"""
      return train_inputs, non_train_inputs
  
  def min_max_scale(train_inputs, non_train_inputs):
      """Pretend this function scales data using SKLearn's `MinMaxScaler`"""
      return train_inputs, non_train_inputs
  
  # Pretend we already set up our `Environment` and we want to optimize the our scaler
  optimizer_0 = DummySearch()
  optimizer_0.forge_experiment(
      Ridge, {}, feature_engineer=FeatureEngineer([
          Categorical([standard_scale, min_max_scale])
      ])
  )
  # `optimizer_0` above will try each of our scaler functions, but what if we shouldn't use either?
  optimizer_1 = DummySearch()
  optimizer_1.forge_experiment(
      Ridge, {}, feature_engineer=FeatureEngineer([
          Categorical([standard_scale, min_max_scale], optional=True)
      ])
  )
  # `optimizer_1`, using `Categorical.optional`, will search the same points as `optimizer_0`, plus
  #   a `FeatureEngineer` where the step is skipped completely, which would be the equivalent of
  #   no `FeatureEngineer` at all in this example
    ``` 
* Enable OptPro's to identify `similar_experiments` when using a search space whose dimensions 
  include `Categorical.optional` `EngineerStep`s at indexes that may differ from those of the 
  candidate Experiments
* Add `callbacks` kwarg to `CVExperiment`, which enables providing `LambdaCallback`s for an 
  Experiment right when you're initializing it
    * Functions like the existing `experiment_callbacks` kwarg of `Environment`
* Improve `Metric.direction` inference to check the name of the `metric_function` for "error"/"loss"
  strings, after checking the `name` itself
    * This means that an `Environment.metrics` value of ``{"mae": "median_absolute_error"}`` will be
      correctly inferred to have `direction`="min", making it easier to use short aliases for those
      extra-long error metric names

### Bug-Fixes
* Fix bug causing descendants of `SKOptimizationProtocol` to break when given non-string `base_estimator`s
* Fix bug causing `ScoringMixIn` to incorrectly keep track of the metrics to record for different dataset types
* Fix bug preventing `get_clean_predictions` from working with multi-output datasets
* Fix incorrect leaderboard sorting when evaluations are tied (again)
* Fix bug causing metrics to be evaluated using the transformed targets/predictions, rather than the 
  inverted (original space) predictions, after performing target transformation via `EngineerStep`
    * Adds new `Environment` kwarg: `save_transformed_metrics`, which dictates whether metrics are 
      calculated using transformed targets/predictions (True), or inverted data (False)
    * Default value of `save_transformed_metrics` is chosen based on dtype of targets. See [#169](https://github.com/HunterMcGushion/hyperparameter_hunter/pull/169)
* Fix bug causing `BayesianOptPro` to break, or fail experiment matching, when using an 
  exclusively-`Categorical` search space. For details, see [#154](https://github.com/HunterMcGushion/hyperparameter_hunter/issues/154)
* Fix bug causing Informed Optimization Protocols to break after the tenth optimization round when 
  attempting to fit `optimizer` with `EngineerStep` dicts, rather than proper instances
    * This was caused by the `EngineerStep`s stored in saved experiment descriptions not being 
      reinitialized in order to be compatible with the current search space
    * See [PR #139](https://github.com/HunterMcGushion/hyperparameter_hunter/pull/139) or 
      "tests/integration_tests/feature_engineering/test_saved_engineer_step.py" for details
* Fix broken inverse target transformation of LightGBM predictions
    * See [PR #140](https://github.com/HunterMcGushion/hyperparameter_hunter/pull/140) for details
* Fix incorrect "source_script" recorded in `CVExperiment` description files when executed within
  an Optimization Protocol
* Fix bug causing :mod:`data.data_chunks` to be excluded from installation

### Changes
* Metrics are now always invoked with NumPy arrays
    * Note that this is unlike `EngineerStep` functions, which always receive Pandas DataFrames as
      input and should always return DataFrames
* `DatasetSentinel` functions in `Environment` now retrieve data transformed by feature engineering
* `data`
    * Rather than being haphazardly stored in an assortment of experiment attributes, datasets are 
      now managed by both :mod:`data` and the overhauled :mod:`callbacks.wranglers` module
    * Affects custom user callbacks that used experiment datasets. See the next section for details
* Add `warn_on_re_ask` kwarg to all OptPro initializers. If True (default=False), a warning will be 
  logged whenever the internal optimizer suggests a point that has already been evaluated--before 
  returning a new, random point to evaluate instead
* `model_init_params` kwarg of both `CVExperiment` and all OptPros is now optional. If not given, 
  it will be evaluated as the default initialization parameters to `model_initializer`
* Convert `space.py` file module to `space` directory module, containing `space_core` and 
  `dimensions`
    * `space.dimensions` is the new home of the dimension classes used to define hyperparameter 
      search spaces via :meth:`optimization.protocol_core.BaseOptPro.forge_experiment`:
      `Real`, `Integer`, and `Categorical`
    * `space.space_core` houses :class:`Space`, which is only used internally
* Convert `optimization.py` and `optimization_core.py` file modules to `optimization` directory 
  module, containing `protocol_core` and the `backends` directory
    * `optimization_core.py` has been moved to `optimization.protocol_core.py`
    * `optimization.backends` contains `skopt.engine` and `skopt.protocols`, the latter of which is
      the new location of the original `optimization.py` file
    * `optimization.backends.skopt.engine` is a partial vendorization of [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize)'s
      `Optimizer` class, which acts as the backend for :class:`optimization.protocol_core.SKOptPro`
        * For additional information on the partial vendorization of key Scikit-Optimize components,
          see [the `optimization.backends.skopt` README](https://github.com/HunterMcGushion/hyperparameter_hunter/tree/master/hyperparameter_hunter/optimization/backends/skopt).
          A copy of Scikit-Optimize's original LICENSE can also be found in `optimization.backends.skopt`

### Deprecations
* OptPros' `set_experiment_guidelines` method renamed to `forge_experiment`
    * `set_experiment_guidelines` will be removed in v3.2.0
* Optimization Protocols in :mod:`hyperparameter_hunter.optimization` renamed to use "OptPro"
    * This change affects the following optimization protocol classes:
        * `BayesianOptimization` -> `BayesianOptPro`
        * `GradientBoostedRegressionTreeOptimization` -> `GradientBoostedRegressionTreeOptPro`
            * `GBRT` alias unchanged
        * `RandomForestOptimization` -> `RandomForestOptPro`
            * `RF` alias unchanged
        * `ExtraTreesOptimization` -> `ExtraTreesOptPro`
            * `ET` alias unchanged
        * `DummySearch` -> `DummyOptPro`
    * This change also affects the base classes for optimization protocols defined in 
      :mod:`hyperparameter_hunter.optimization.protocol_core` that are not available in the package 
      namespace
    * The original names will continue to be available until their removal in v3.2.0
* `lambda_callback` kwargs dealing with "experiment" and "repetition" time steps have been shortened
    * These four kwargs have been changed to the following values:
        * `on_experiment_start` -> `on_exp_start`
        * `on_experiment_end` -> `on_exp_end`
        * `on_repetition_start` -> `on_rep_start`
        * `on_repetition_end` -> `on_rep_end`
    * In summary, "experiment" is shortened to "exp", and "repetition" is shortened to "rep"
    * The originals will continue to be available until their removal in v3.2.0
    * This deprecation will break any custom callbacks created by subclassing `BaseCallback` (which
      is not the officially supported method), rather than using `lambda_callback`
        * To fix such callbacks, simply rename the above methods

### Breaking Changes
* Any custom callbacks (`lambda_callback` or otherwise) that accessed the experiment’s datasets will 
need to be updated to access their new locations. The new syntax is described in detail in 
:mod:`data.data_core`, but the general idea is as follows:
	1. Experiments have four dataset attributes: `data_train`, `data_oof`, `data_holdout`, `data_test`
	2. Each dataset has three `data_chunks`: `input`, `target`, `prediction`
	3. Each data_chunk has six attributes. The first five pertain to the experiment division for 
	   which the data is collected: `d` (initial data), `run` , `fold` , `rep`, and `final`
    4. The sixth attribute of each data_chunk is `T` , which contains the transformed states of the 
       five attributes described in step 3
       * Transformations are applied by feature engineering
       * Inversions of those transformations (if applicable) are stored in the five normal data_chunk 
         attributes from step 3
* Ignore Pandas version during dataset hashing for more consistent `Environment` keys. See [#166](https://github.com/HunterMcGushion/hyperparameter_hunter/issues/166)


<a name="3.0.0beta1"></a>
## [3.0.0beta1] (2019-08-05)

### Features
* Enable OptPro's to identify `similar_experiments` when using a search space whose dimensions 
  include `Categorical.optional` `EngineerStep`s at indexes that may differ from those of the 
  candidate Experiments
* Add `callbacks` kwarg to `CVExperiment`, which enables providing `LambdaCallback`s for an 
  Experiment right when you're initializing it
    * Functions like the existing `experiment_callbacks` kwarg of `Environment`
* Improve `Metric.direction` inference to check the name of the `metric_function` for "error"/"loss"
  strings, after checking the `name` itself
    * This means that an `Environment.metrics` value of ``{"mae": "median_absolute_error"}`` will be
      correctly inferred to have `direction`="min", making it easier to use short aliases for those
      extra-long error metric names

### Bug Fixes
* Fix bug causing metrics to be evaluated using the transformed targets/predictions, rather than the 
  inverted (original space) predictions, after performing target transformation via `EngineerStep`
    * Adds new `Environment` kwarg: `save_transformed_metrics`, which dictates whether metrics are 
      calculated using transformed targets/predictions (True), or inverted data (False)
    * Default value of `save_transformed_metrics` is chosen based on dtype of targets. See [#169](https://github.com/HunterMcGushion/hyperparameter_hunter/pull/169)

### Changes
* Add `warn_on_re_ask` kwarg to all OptPro initializers. If True (default=False), a warning will be 
  logged whenever the internal optimizer suggests a point that has already been evaluated--before 
  returning a new, random point to evaluate instead

### Breaking Changes
* Ignore Pandas version during dataset hashing for more consistent `Environment` keys. See [#166](https://github.com/HunterMcGushion/hyperparameter_hunter/issues/166)

<a name="3.0.0beta0"></a>
## [3.0.0beta0] (2019-07-14)

### Bug Fixes
* Fix bug causing `BayesianOptPro` to break, or fail experiment matching, when using an 
  exclusively-`Categorical` search space. For details, see [#154](https://github.com/HunterMcGushion/hyperparameter_hunter/issues/154)

### Changes
* `model_init_params` kwarg of both `CVExperiment` and all OptPros is now optional. If not given, 
  it will be evaluated as the default initialization parameters to `model_initializer`
* Convert `space.py` file module to `space` directory module, containing `space_core` and 
  `dimensions`
    * `space.dimensions` is the new home of the dimension classes used to define hyperparameter 
      search spaces via :meth:`optimization.protocol_core.BaseOptPro.forge_experiment`:
      `Real`, `Integer`, and `Categorical`
    * `space.space_core` houses :class:`Space`, which is only used internally
* Convert `optimization.py` and `optimization_core.py` file modules to `optimization` directory 
  module, containing `protocol_core` and the `backends` directory
    * `optimization_core.py` has been moved to `optimization.protocol_core.py`
    * `optimization.backends` contains `skopt.engine` and `skopt.protocols`, the latter of which is
      the new location of the original `optimization.py` file
    * `optimization.backends.skopt.engine` is a partial vendorization of [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize)'s
      `Optimizer` class, which acts as the backend for :class:`optimization.protocol_core.SKOptPro`
        * For additional information on the partial vendorization of key Scikit-Optimize components,
          see [the `optimization.backends.skopt` README](https://github.com/HunterMcGushion/hyperparameter_hunter/tree/master/hyperparameter_hunter/optimization/backends/skopt).
          A copy of Scikit-Optimize's original LICENSE can also be found in `optimization.backends.skopt`

### Deprecations
* OptPros' `set_experiment_guidelines` method renamed to `forge_experiment`
    * `set_experiment_guidelines` will be removed in v3.2.0
* Optimization Protocols in :mod:`hyperparameter_hunter.optimization` renamed to use "OptPro"
    * This change affects the following optimization protocol classes:
        * `BayesianOptimization` -> `BayesianOptPro`
        * `GradientBoostedRegressionTreeOptimization` -> `GradientBoostedRegressionTreeOptPro`
            * `GBRT` alias unchanged
        * `RandomForestOptimization` -> `RandomForestOptPro`
            * `RF` alias unchanged
        * `ExtraTreesOptimization` -> `ExtraTreesOptPro`
            * `ET` alias unchanged
        * `DummySearch` -> `DummyOptPro`
    * This change also affects the base classes for optimization protocols defined in 
      :mod:`hyperparameter_hunter.optimization.protocol_core` that are not available in the package 
      namespace
    * The original names will continue to be available until their removal in v3.2.0
* `lambda_callback` kwargs dealing with "experiment" and "repetition" time steps have been shortened
    * These four kwargs have been changed to the following values:
        * `on_experiment_start` -> `on_exp_start`
        * `on_experiment_end` -> `on_exp_end`
        * `on_repetition_start` -> `on_rep_start`
        * `on_repetition_end` -> `on_rep_end`
    * In summary, "experiment" is shortened to "exp", and "repetition" is shortened to "rep"
    * The originals will continue to be available until their removal in v3.2.0
    * This deprecation will break any custom callbacks created by subclassing `BaseCallback` (which
      is not the officially supported method), rather than using `lambda_callback`
        * To fix such callbacks, simply rename the above methods


<a name="3.0.0alpha2"></a>
## [3.0.0alpha2] (2019-06-12)

### Bug Fixes
* Fix bug causing Informed Optimization Protocols to break after the tenth optimization round when 
  attempting to fit `optimizer` with `EngineerStep` dicts, rather than proper instances
    * This was caused by the `EngineerStep`s stored in saved experiment descriptions not being 
      reinitialized in order to be compatible with the current search space
    * See [PR #139](https://github.com/HunterMcGushion/hyperparameter_hunter/pull/139) or 
      "tests/integration_tests/feature_engineering/test_saved_engineer_step.py" for details
* Fix broken inverse target transformation of LightGBM predictions
    * See [PR #140](https://github.com/HunterMcGushion/hyperparameter_hunter/pull/140) for details
* Fix incorrect "source_script" recorded in `CVExperiment` description files when executed within
  an Optimization Protocol


<a name="3.0.0alpha1"></a>
## [3.0.0alpha1] (2019-06-07)

### Bug Fixes
* Fix bug causing :mod:`data.data_chunks` to be excluded from installation


<a name="3.0.0alpha0"></a>
## [3.0.0alpha0] (2019-06-07)
This is the most significant release since the birth of HyperparameterHunter, adding not only 
feature engineering, but also feature optimization. The goal of feature engineering in 
HyperparameterHunter is to enable you to manipulate your data however you need to, without imposing 
restrictions on what's allowed - all while seamlessly keeping track of your feature engineering 
steps so they can be learned from and optimized. In that spirit, feature engineering steps are 
defined by your very own functions. That may sound a bit silly at first, but it affords maximal
freedom and customization, with only the minimal requirement that you tell your function what data 
you want from HyperparameterHunter, and you give it back when you're done playing with it. 

The best way to really understand feature engineering in HyperparameterHunter is to dive into some
code and check out the "examples/feature_engineering_examples" directory. In no time at all, you'll 
be ready to spread your wings by experimenting with the creative feature engineering steps only you 
can build. Let your faithful assistant, HyperparameterHunter, meticulously and lovingly record them 
for you, so you can optimize your custom feature functions just like normal hyperparameters.

You're a glorious peacock, and we just wanna let you fly.

### Features
* Feature engineering via `FeatureEngineer` and `EngineerStep`
    * This will be a "brief" summary of the new features. For more detail, see the aforementioned
      "examples/feature_engineering_examples" directory or the extensively documented 
      [`FeatureEngineer`](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.feature_engineering.FeatureEngineer) 
      and [`EngineerStep`](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.feature_engineering.EngineerStep)
      classes 
    * `FeatureEngineer` can be passed as the `feature_engineer` kwarg to either: 
        1. Instantiate a [`CVExperiment`](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.experiments.BaseExperiment), or
        2. Call the [`set_experiment_guidelines`](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.optimization_core.BaseOptimizationProtocol.set_experiment_guidelines)
           method of any Optimization Protocol
    * `FeatureEngineer` is just a container for `EngineerStep`s
        * Instantiate it with a simple list of `EngineerStep`s, or functions to construct `EngineerStep`s
    * Most important `EngineerStep` parameter is a function you define to perform your data transformation (whatever that is)
        * This function is often creatively referred to as a "step function"
    * Step function definitions have only two requirements:
        1. Name the data you want to transform in the signature's input parameters
            * 16 different parameter names, documented in 
              [`EngineerStep`'s `params` kwarg](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.feature_engineering.EngineerStep)
        2. Return the data when you're done with it
    * Step functions may be given directly to `FeatureEngineer`, or wrapped in an `EngineerStep` for greater customization
    * Here are just a few step functions you might want to make:
    
    ```python
  from hyperparameter_hunter import CVExperiment, FeatureEngineer, EngineerStep
  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import QuantileTransformer, StandardScaler
  from sklearn.impute import SimpleImputer

  def standard_scale(train_inputs, non_train_inputs):
      s = StandardScaler()
      train_inputs[train_inputs.columns] = s.fit_transform(train_inputs.values)
      non_train_inputs[train_inputs.columns] = s.transform(non_train_inputs.values)
      return train_inputs, non_train_inputs
  
  def quantile_transform(train_targets, non_train_targets):
      t = QuantileTransformer(output_distribution="normal")
      train_targets[train_targets.columns] = t.fit_transform(train_targets.values)
      non_train_targets[train_targets.columns] = t.transform(non_train_targets.values)
      return train_targets, non_train_targets, t
  
  def set_nan(all_inputs):
      cols = [1, 2, 3, 4, 5]
      all_inputs.iloc[:, cols] = all_inputs.iloc[:, cols].replace(0, np.NaN)
      return all_inputs
  
  def impute_negative_one(all_inputs):
      all_inputs.fillna(-1, inplace=True)
      return all_inputs
  
  def impute_mean(train_inputs, non_train_inputs):
      imputer = SimpleImputer()
      train_inputs[train_inputs.columns] = imputer.fit_transform(train_inputs.values)
      non_train_inputs[train_inputs.columns] = imputer.transform(non_train_inputs.values)
      return train_inputs, non_train_inputs
  
  def sqr_sum_feature(all_inputs):
      all_inputs["my_sqr_sum_feature"] = all_inputs.agg(
          lambda row: np.sqrt(np.sum([np.square(_) for _ in row])),
          axis="columns",
      )
      return all_inputs
    
  def upsample_train_data(train_inputs, train_targets):
      pos = pd.Series(train_targets["target"] == 1)
      train_inputs = pd.concat([train_inputs, train_inputs.loc[pos]], axis=0)
      train_targets = pd.concat([train_targets, train_targets.loc[pos]], axis=0)
      return train_inputs, train_targets
  
  # Any of the above can be wrapped by `EngineerStep`, or added directly to a `FeatureEngineer`'s `steps`
  # Below, assume we have already activated an `Environment`
  exp_0 = CVExperiment(
      model_initializer=..., 
      model_init_params={},
      feature_engineer=FeatureEngineer([
          set_nan,
          EngineerStep(standard_scale),
          quantile_transform,
          EngineerStep(upsample_train_data, stage="intra_cv"),
      ]),
  )
    ```

* Feature optimization
    * `Categorical` can be used to optimize feature engineering steps, either as `EngineerStep` 
      instances or raw functions of the form expected by `EngineerStep`
    * Just throw your `Categorical` in with the rest of your `FeatureEngineer.steps` 
    * Features can, of course, be optimized alongside standard model hyperparameters 
        
    ```python
  from hyperparameter_hunter import GBRT, Real, Integer, Categorical, FeatureEngineer, EngineerStep
  import numpy as np
  import pandas as pd
  from sklearn.linear_model import Ridge
  from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
  
  def standard_scale(train_inputs, non_train_inputs):
      s = StandardScaler()
      train_inputs[train_inputs.columns] = s.fit_transform(train_inputs.values)
      non_train_inputs[train_inputs.columns] = s.transform(non_train_inputs.values)
      return train_inputs, non_train_inputs
  
  def min_max_scale(train_inputs, non_train_inputs):
      s = MinMaxScaler()
      train_inputs[train_inputs.columns] = s.fit_transform(train_inputs.values)
      non_train_inputs[train_inputs.columns] = s.transform(non_train_inputs.values)
      return train_inputs, non_train_inputs
  
  # Pretend we already set up our `Environment` and we want to optimize the our scaler
  # We'll also throw in some standard hyperparameter optimization - This is HyperparameterHunter, after all
  optimizer_0 = GBRT()
  optimizer_0.set_experiment_guidelines(
      Ridge, 
      dict(alpha=Real(0.5, 1.0), max_iter=Integer(500, 2000), solver="svd"), 
      feature_engineer=FeatureEngineer([Categorical([standard_scale, min_max_scale])])
  )
  
  # Then we remembered we should probably transform our target, too
  # OH NO! After transforming our targets, we'll need to `inverse_transform` the predictions!
  # OH YES! HyperparameterHunter will gladly accept a fitted transformer as an extra return value, 
  #   and save it to call `inverse_transform` on predictions 
  def quantile_transform(train_targets, non_train_targets):
      t = QuantileTransformer(output_distribution="normal")
      train_targets[train_targets.columns] = t.fit_transform(train_targets.values)
      non_train_targets[train_targets.columns] = t.transform(non_train_targets.values)
      return train_targets, non_train_targets, t
  
  # We can also tell HyperparameterHunter to invert predictions using a callable, rather than a fitted transformer
  def log_transform(all_targets):
      all_targets = np.log1p(all_targets)
      return all_targets, np.expm1
  
  optimizer_1 = GBRT()
  optimizer_1.set_experiment_guidelines(
      Ridge, {}, feature_engineer=FeatureEngineer([
          Categorical([standard_scale, min_max_scale]),
          Categorical([quantile_transform, log_transform]),
      ])
  )
    ``` 
    
* [`Categorical.optional`](https://hyperparameter-hunter.readthedocs.io/en/latest/source/hyperparameter_hunter.html#hyperparameter_hunter.space.Categorical)
    * As `Categorical` is the means of optimizing `EngineerStep`s in simple lists, it became 
      necessary to answer the question of whether that crazy new feature you've been cooking up in 
      the lab should even be included at all
    * So the `optional` kwarg was added to `Categorical` to appease the mad scientist in us all
    * If True (default=False), the search space will include not only the `categories` you explicitly
      provide, but also the omission of the current `EngineerStep` entirely
    * `optional` is only intended for use in optimizing `EngineerStep`s. Don't expect it to work elsewhere
    * Brief example:
    
    ```python
  from hyperparameter_hunter import DummySearch, Categorical, FeatureEngineer, EngineerStep
  from sklearn.linear_model import Ridge

  def standard_scale(train_inputs, non_train_inputs):
      """Pretend this function scales data using SKLearn's `StandardScaler`"""
      return train_inputs, non_train_inputs
  
  def min_max_scale(train_inputs, non_train_inputs):
      """Pretend this function scales data using SKLearn's `MinMaxScaler`"""
      return train_inputs, non_train_inputs
  
  # Pretend we already set up our `Environment` and we want to optimize the our scaler
  optimizer_0 = DummySearch()
  optimizer_0.set_experiment_guidelines(
      Ridge, {}, feature_engineer=FeatureEngineer([
          Categorical([standard_scale, min_max_scale])
      ])
  )
  # `optimizer_0` above will try each of our scaler functions, but what if we shouldn't use either?
  optimizer_1 = DummySearch()
  optimizer_1.set_experiment_guidelines(
      Ridge, {}, feature_engineer=FeatureEngineer([
          Categorical([standard_scale, min_max_scale], optional=True)
      ])
  )
  # `optimizer_1`, using `Categorical.optional`, will search the same points as `optimizer_0`, plus
  #   a `FeatureEngineer` where the step is skipped completely, which would be the equivalent of
  #   no `FeatureEngineer` at all in this example
    ``` 

### Bug-Fixes
* Fix bug causing descendants of `SKOptimizationProtocol` to break when given non-string `base_estimator`s
* Fix bug causing `ScoringMixIn` to incorrectly keep track of the metrics to record for different dataset types
* Fix bug preventing `get_clean_predictions` from working with multi-output datasets
* Fix incorrect leaderboard sorting when evaluations are tied (again)

### Changes
* Metrics are now always invoked with NumPy arrays
    * Note that this is unlike `EngineerStep` functions, which always receive Pandas DataFrames as
      input and should always return DataFrames
* `DatasetSentinel` functions in `Environment` now retrieve data transformed by feature engineering
* `data`
    * Rather than being haphazardly stored in an assortment of experiment attributes, datasets are 
      now managed by both :mod:`data` and the overhauled :mod:`callbacks.wranglers` module
    * Affects custom user callbacks that used experiment datasets. See the next section for details

### Breaking Changes
* Any custom callbacks (`lambda_callback` or otherwise) that accessed the experiment’s datasets will 
need to be updated to access their new locations. The new syntax is described in detail in 
:mod:`data.data_core`, but the general idea is as follows:
	1. Experiments have four dataset attributes: `data_train`, `data_oof`, `data_holdout`, `data_test`
	2. Each dataset has three `data_chunks`: `input`, `target`, `prediction`
	3. Each data_chunk has six attributes. The first five pertain to the experiment division for 
	   which the data is collected: `d` (initial data), `run` , `fold` , `rep`, and `final`
    4. The sixth attribute of each data_chunk is `T` , which contains the transformed states of the 
       five attributes described in step 3
       * Transformations are applied by feature engineering
       * Inversions of those transformations (if applicable) are stored in the five normal data_chunk 
         attributes from step 3


<a name="2.2.0"></a>
## [2.2.0] (2019-02-10)

### Features
* Enhanced support for Keras `initializers`
    * In addition to providing strings to the various "...initializer" parameters of Keras layers 
    (like `Dense`'s `kernel_initializer`), you can now use the callables in `keras.initializers`, too
    * This means that all of the following will work in Keras `build_fn`s:
        * `Dense(10, kernel_initializer="orthogonal")` (original string-form)
        * `Dense(10, kernel_initializer=orthogonal)` (after `from keras.initializers import orthogonal`)
        * `Dense(10, kernel_initializer=orthogonal(gain=0.5))` 
    * You can even optimize callable initializers and their parameters:
        * `Dense(10, kernel_initializer=orthogonal(gain=Real(0.3, 0.7)))`
        * `Dense(10, kernel_initializer=Categorical(["orthogonal", "lecun_normal"))`

### Bug-Fixes
* Fix bug causing cross-validation to break occasionally if `n_splits=2`
* Fix bug causing optimization to break if only optimizing `model_extra_params` (not `build_fn`) in Keras

### Changes
* Shortened the preferred names of some `Environment` parameters:
    * `cross_validation_type` -> `cv_type`
    * `cross_validation_params` -> `cv_params`
    * `metrics_map` -> `metrics`
    * `reporting_handler_params` -> `reporting_params`
    * `root_results_path` -> `results_path`
* The original parameter names can still be used as aliases. See note in "Breaking Changes" section

### Breaking Changes
* To ensure compatibility with `Environment` keys created in earlier versions of HyperparameterHunter,
continue using the original names for the parameters mentioned above
    * Using the new (preferred) names will produce different `Environment` keys, which will cause 
    `Experiment`s to not be identified as valid learning material for optimization even though they 
    used the same parameter values, just with different names


<a name="2.1.1"></a>
## [2.1.1] (2019-01-15)

### Bug-Fixes
* Fix bug caused by yaml import when not using `recorders.YAMLDescriptionRecorder`


<a name="2.1.0"></a>
## [2.1.0] (2019-01-15)

### Features
* Add `experiment_recorders` kwarg to `Environment` that allows for providing custom Experiment 
result file-recording classes
    * The only syntax changes for this new feature occur in `Environment` initialization:
    
    ```python
  from hyperparameter_hunter import Environment
  from hyperparameter_hunter.recorders import YAMLDescriptionRecorder

  env = Environment(
      train_dataset=None,  # Placeholder value
      root_results_path="HyperparameterHunterAssets",
      # ... Standard `Environment` kwargs ...
      experiment_recorders=[
          (YAMLDescriptionRecorder, "Experiments/YAMLDescriptions"),
      ],
  )

  # ... Normal Experiment/Optimization execution
    ```
    
    * Each tuple in the `experiment_recorders` list is expected to contain the following:
        1. a new custom recorder class that descends from `recorders.BaseRecorder`, followed by
        2. a string path that is relative to the `Environment.root_results_path` kwarg and specifies 
        the location at which new result files should be saved 
    
    * A dedicated example for this feature has been added in "examples/advanced_examples/recorder_example.py"
* Update `Environment` verbosity settings by converting the `verbose` parameter from a boolean to an 
integer from 0-4 (inclusive). Enables greater control of logging frequency and level of detail. See 
the `environment.Environment` documentation for details on what is logged at each level
* Allow blacklisting the general heartbeat file by providing "current_heartbeat" as a value in 
`Environment.file_blacklist`. Doing this will also blacklist Experiment heartbeat files automatically

### Bug-Fixes
* Fix bug when comparing identical dataset sentinels used in a `CVExperiment`, followed 
by use in `BaseOptimizationProtocol.set_experiment_guidelines`
* Fix bug causing HyperparameterHunter warning messages to not be displayed
* Fix bug where the incorrect best experiment would be printed in the hyperparameter optimization 
summary when using a minimizing `target_metric`
* Fix bug where providing `Environment` with `root_results_path=None` would break key-making

### Changes
* Shortened name of `CrossValidationExperiment` to `CVExperiment`. `CrossValidationExperiment` will still
be available as a deprecated copy of `CVExperiment` until v2.3.0, but `CVExperiment` is preferred
* Update sorting of GlobalLeaderboard entries to take into account only the target metric column
and the "experiment_#" columns
    * This produces more predictable orders that don't rely on UUIDs/hashes and preserve historicity

### Breaking Changes
* Hyperparameter keys are not compatible with those created using previous versions due to updated
defaults for core Experiment parameters
    * This is in order to improve proper matching to saved Experiment results, especially when using
    "non-essential"/extra hyperparameters such as `verbose`
    * The following parameters of `experiments.BaseExperiment.__init__` will now be set to the 
    corresponding value by default if `None`:
        * `model_extra_params`: {}
        * `feature_selector`: []
        * `preprocessing_pipeline`: {}
        * `preprocessing_params`: {}
    * These changes are also reflected in `optimization_core.BaseOptimizationProtocol.set_experiment_guidelines`,
    and `utils.optimization_utils.filter_by_guidelines`


<a name="2.0.1"></a>
## [2.0.1] (2018-11-25)

### Changes
* KeyAttributeLookup entries are now saved by full hyperparameter paths, rather than simple keys for 
greater clarity (#75)
* Changed behavior of the `do_predict_proba` parameter of `environment.Environment` when `True`
    * All other behavior remains unchanged. However, instead of behaving identically to `do_predict_proba=0`,
    `do_predict_proba=True` will now use all predicted probability columns for the final predictions
* Deprecated classes `experiments.RepeatedCVExperiment` and `experiments.StandardCVExperiment`. The 
goals of both of these classes are accomplished by the preferred `experiments.CrossValidationExperiment`
class. The two aforementioned deprecated classes are scheduled for removal in v2.1.0. All uses of the 
deprecated classes should be replaced with `experiments.CrossValidationExperiment`


<a name="2.0.0"></a>
## [2.0.0] (2018-11-16)

### Breaking Changes
* The updates to `metrics_map` described below mean that the `cross_experiment_key`s produced by 
`environment.Environment` will be different from those produced by previous versions of 
HyperparameterHunter
    * This means that `OptimizationProtocol`s will not recognize saved experiment results from 
    previous versions as being compatible for learning

### Features
* Made the `metrics_map` parameter of `environment.Environment` more customizable and compatible with
measures of error/loss. Original `metrics_map` functionality/formats are unbroken
    * `metrics_map`s are automatically converted to dicts of `metrics.Metric` instances, which receive
    three parameters: `name`, `metric_function`, and `direction` (new)
        * `name` and `metric_function` mimic the original functionality of the `metrics_map`
        * `direction` can be one of the following three strings: "infer" (default), "max", "min"
            * "max" should be used for metrics in which greater values are preferred, like accuracy;
            whereas, "min" should be used for measures of loss/error, where lower values are better
            * "infer" will set `direction` to "min" if the metric's `name` contains one of the 
            following strings: \["error", "loss"\]. Otherwise, `direction` will be "max"
                * This means that for metrics names that do not contain the aforementioned strings
                but are measures of error/loss (such as "mae" for "mean_absolute_error"), `direction` 
                should be explicitly set to "min"
    * `environment.Environment` can receive `metrics_map` in many different formats, which are 
    documented in `environment.Environment` and `metrics.format_metrics_map`
* The `do_predict_proba` parameter of `environment.Environment` (and consequently `models.Model`) is
now allowed to be an int, as well as a bool. If `do_predict_proba` is an int, the `predict_proba` 
method is called, and the int specifies the index of the column in the model's probability 
predictions whose values should be passed on as the final predictions. Original behavior when 
passing a boolean is unaffected. See `Environment` documentation for usage notes and warnings about 
providing truthy or falsey values for the `do_predict_proba` parameter 

### Bug-Fixes
* Fixed bug where `OptimizationProtocol`s would optimize in the wrong direction when `target_metric`
was a measure of error/loss
    * This is fixed by the new `metrics_map` formatting feature listed above
* Fixed bug causing `OptimizationProtocol`s to fail to recognize similar experiments when 
`sentinels.DatasetSentinel`s were provided as experiment guidelines (#88)
* Fixed bug in which the logging for individual Experiments performed inside an `OptimizationProtocol` was not properly 
silenced if execution of the `OptimizationProtocol` took place immediately after executing a `CrossValidationExperiment` 
(#74)
    * Individual experiment logging is now only visible inside an `OptimizationProtocol` if `BaseOptimizationProtocol` 
    is initialized with `verbose=2`, as intended 

### Changes
* Deprecated `optimization_core.UninformedOptimizationProtocol`. This class was never finished, and is no longer necessary. 
It is scheduled for removal in v1.2.0, and the classes that descended from it have been removed
* Renamed `optimization_core.InformedOptimizationProtocol` to `SKOptimizationProtocol`, and added 
an `InformedOptimizationProtocol` stub with a deprecation warning
* Renamed `exception_handler` module (which was only used internally) to `exceptions`
* Added aliases for the particularly long optimization protocol classes defined in `optimization`:
    * `GradientBoostedRegressionTreeOptimization`, or `GBRT`,
    * `RandomForestOptimization`, or `RF`,
    * `ExtraTreesOptimization`, or `ET` 


<a name="1.1.0"></a>
## [1.1.0] (2018-10-4)

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
* Fixed bug where `holdout_dataset` was not properly recognized as a `DataFrame` (#78)
* Fixed bug where CatBoost was given both `silent` and `verbose` kwargs (#80)

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
    * The `dimensions` kwarg is removed from the OptimizationProtocol classes, and hyperparameter search dimensions are now provided along with the concrete hyperparameters via `set_experiment_guidelines`. If a value is a descendant of `hyperparameter_hunter.space.Dimension`, it is automatically detected as a space to be searched and optimized
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


[Unreleased]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v3.0.0beta1...v3.0.0
[3.0.0beta1]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v3.0.0beta0...v3.0.0beta1
[3.0.0beta0]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v3.0.0alpha2...v3.0.0beta0
[3.0.0alpha2]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v3.0.0alpha1...v3.0.0alpha2
[3.0.0alpha1]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v3.0.0alpha0...v3.0.0alpha1
[3.0.0alpha0]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v2.2.0...v3.0.0alpha0
[2.2.0]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v2.1.1...v2.2.0
[2.1.1]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/HunterMcGushion/hyperparameter_hunter/compare/v0.0.1...v1.0.0