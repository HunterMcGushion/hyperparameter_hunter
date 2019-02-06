HyperparameterHunter Library Compatibility
******************************************
This section lists libraries that have been tested with HyperparameterHunter and briefly outlines some works in progress.

Tested and Compatible
---------------------
* `CatBoost <https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/catboost_examples>`__
* `Keras <https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/keras_examples>`__
* `LightGBM <https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/lightgbm_examples>`__
* `Scikit-Learn <https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/sklearn_examples>`__
* `XGBoost <https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/xgboost_examples>`__
* `rgf_python <https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/rgf_examples>`__

Support On the Way
------------------
* PyTorch/Skorch
* TensorFlow
* Boruta
* Imbalanced-Learn

Not Yet Compatible
------------------
* TPOT

    * After admittedly minimal testing, problems arose due to the fact that TPOT implements its own cross-validation scheme
    * This resulted in (probably unexpected) nested cross validation, and extremely long execution times

Notes
-----
* If you don't see the one of your favorite libraries listed above, and you want to do something about that, let us know!
* See HyperparameterHunter's **'examples/'** directory for help on getting started with compatible libraries
* Improved support for hyperparameter tuning with Keras is on the way!

