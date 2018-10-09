# HyperparameterHunter Examples

For the most part, examples belong to one of two categories: 
1. special HyperparameterHunter functionality, or
2. using a particular machine learning library (experimentation or optimization)

## Special Functionality Examples
| Script                                                        | Focus                                                                                    |
|---------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [do_full_save](do_full_save_example.py)                       | Specify when Experiments are saved according to custom constraints like score thresholds |
| [environment_params_path](environment_params_path_example.py) | Use `environment_params.json` file to store default Environment parameters               |
| [functionality](functionality_example.py)                     | Walkthrough demonstrating what HyperparameterHunter does, and why you should care        |
| [holdout_test_datasets](holdout_test_datasets_example.py)     | Provide holdout and test datasets for predicting and scoring                             |
| [lambda_callback](lambda_callback_example.py)                 | Make custom callbacks executed during Experiments to add your own functionality          |

## Library Examples
|                     CatBoost                     |                                  Keras                                  |                     LightGBM                     |                     RGF                     |                SKLearn               |                   XGBoost                  |
|:------------------------------------------------:|:-----------------------------------------------------------------------:|:------------------------------------------------:|:-------------------------------------------:|:------------------------------------:|:------------------------------------------:|
|       [Experiment](lib_catboost_example.py)      |                    [Experiment](lib_keras_example.py)                   |       [Experiment](lib_lightgbm_example.py)      |       [Experiment](lib_rgf_example.py)      | [Experiment](lib_sklearn_example.py) |    [Experiment](lib_xgboost_example.py)    |
| [Optimize](lib_catboost_optimization_example.py) |              [Optimize](lib_keras_optimization_example.py)              | [Optimize](lib_lightgbm_optimization_example.py) | [Optimize](lib_rgf_optimization_example.py) |                                      | [Optimize](simple_optimization_example.py) |
|                                                  |    [Image Classification](lib_keras_image_classification_example.py)    |                                                  |                                             |                                      |    [Other](simple_experiment_example.py)   |
|                                                  | [Multi-Class Classification](lib_keras_multi_classification_example.py) |                                                  |                                             |                                      |                                            |
