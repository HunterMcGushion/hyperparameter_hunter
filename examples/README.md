# HyperparameterHunter Examples

1. [Beginner Examples](#beginner-examples): Simple examples to get started with HyperparameterHunter
2. [Library Examples](#library-examples): Examples using particular machine learning libraries
3. [Advanced Examples](#advanced-examples): Examples demonstrating advanced HyperparameterHunter functionality

----

## Beginner Examples

<a name="beginner-examples"/>

* [Get Started (Thorough)](extended_example.ipynb)
* [Get Started (Fast)](functionality_example.py)
* [Simple Experiment](simple_experiment_example.py)
* [Simple Hyperparameter Optimization](simple_optimization_example.py)

## Library Examples

<a name="library-examples"/>

<table>
    <tr>
        <th nowrap>Library (A-Z)</th>
        <th nowrap>Example</th>
        <th nowrap>Link</th>
        <th nowrap>Description</th>
    </tr>
    <tr>
        <th rowspan=2 nowrap>CatBoost</th>
        <td nowrap>Classification</td>
        <td nowrap>
            <a href="catboost_examples/classification.ipynb">NB</a>
            /
            <a href="catboost_examples/classification.py">Script</a>
        </td>
        <td>Large dataset (SKLearn's Forest Cover Types)</td>
    </tr>
    <tr>
        <td nowrap>Regression</td>
        <td nowrap>
            <a href="catboost_examples/regression.ipynb">NB</a>
            /
            <a href="catboost_examples/regression.py">Script</a>
        </td>
        <td>Model extra params (eval_set) - Environment data sentinels</td>    
    </tr>
    <tr>
        <th rowspan=4 nowrap>Keras</th>
        <td nowrap>Simple Experiment</td>
        <td nowrap>
            <a href="keras_examples/experiment_example.py">Script</a>
        </td>
        <td>Easy classification Experiment, including Keras callbacks</td>
    </tr>
    <tr>
        <td nowrap>Optimization</td>
        <td nowrap>
            <a href="keras_examples/optimization_example.py">Script</a>
        </td>
        <td>Building on last example with hyperparameter optimization</td>    
    </tr>
    <tr>
        <td nowrap>Multi-Classification</td>
        <td nowrap>
            <a href="keras_examples/multi_classification_example.py">Script</a>
        </td>
        <td>Hand-written digits (OHE) - Reshape, Convolutional layers</td>
    </tr>
    <tr>
        <td nowrap>Image Classification</td>
        <td nowrap>
            <a href="keras_examples/image_classification_example.py">Script</a>
        </td>
        <td>Hand-written digits dataset (label encoded)</td>    
    </tr>
    <tr>
        <th rowspan=2 nowrap>LightGBM</th>
        <td nowrap>Classification</td>
        <td nowrap>
            <a href="lightgbm_examples/classification.ipynb">NB</a>
            /
            <a href="lightgbm_examples/classification.py">Script</a>
        </td>
        <td>Custom metrics - Forest Cover Types</td>
    </tr>
    <tr>
        <td nowrap>Regression</td>
        <td nowrap>
            <a href="lightgbm_examples/regression.ipynb">NB</a>
            /
            <a href="lightgbm_examples/regression.py">Script</a>
        </td>
        <td>Named metrics - Boston Housing Prices dataset</td>    
    </tr>
    <tr>
        <th rowspan=2 nowrap>RGF</th>
        <td nowrap>Classification</td>
        <td nowrap>
            <a href="rgf_examples/classification.ipynb">NB</a>
            /
            <a href="rgf_examples/classification.py">Script</a>
        </td>
        <td>Repeated cross-validation schemes</td>
    </tr>
    <tr>
        <td nowrap>Regression</td>
        <td nowrap>
            <a href="rgf_examples/regression.ipynb">NB</a>
            /
            <a href="rgf_examples/regression.py">Script</a>
        </td>
        <td>Multiple-run-averaging in cross-validation</td>    
    </tr>
    <tr>
        <th rowspan=1 nowrap>SKLearn</th>
        <td nowrap>Classification</td>
        <td nowrap>
            <a href="sklearn_examples/classification.ipynb">NB</a>
            /
            <a href="sklearn_examples/classification.py">Script</a>
        </td>
        <td>Consecutive Experiments with different models</td>
    </tr>
    <tr>
        <th rowspan=2 nowrap>XGBoost</th>
        <td nowrap>Classification</td>
        <td nowrap>
            <a href="xgboost_examples/classification.ipynb">NB</a>
            /
            <a href="xgboost_examples/classification.py">Script</a>
        </td>
        <td>Model extra params (eval_set) - Environment data sentinels</td>
    </tr>
    <tr>
        <td nowrap>Regression</td>
        <td nowrap>
            <a href="xgboost_examples/regression.ipynb">NB</a>
            /
            <a href="xgboost_examples/regression.py">Script</a>
        </td>
        <td>Optimizing model extra params</td>
    </tr>
</table>

## Advanced Examples

<a name="advanced-examples"/>

<table>
    <tr>
        <th nowrap>Functionality (A-Z)</th>
        <th nowrap>Link</th>
        <th nowrap>Description</th>
    </tr>
    <tr>
        <th nowrap>do_full_save</th>
        <td nowrap>
            <a href="advanced_examples/do_full_save_example.py">Script</a>
        </td>
        <td>Specify when Experiments are saved according to custom constraints like score thresholds</td>
    </tr>
    <tr>
        <th nowrap>environment_params_path</th>
        <td nowrap>
            <a href="advanced_examples/environment_params_path_example.py">Script</a>
        </td>
        <td>Use `environment_params.json` file to store default Environment parameters</td>
    </tr>
    <tr>
        <th nowrap>holdout/test_datasets</th>
        <td nowrap>
            <a href="advanced_examples/holdout_test_datasets_example.py">Script</a>
        </td>
        <td>Provide holdout and test datasets for predicting and scoring</td>
    </tr>
    <tr>
        <th nowrap>lambda_callback</th>
        <td nowrap>
            <a href="advanced_examples/lambda_callback_example.py">Script</a>
        </td>
        <td>Make custom callbacks executed during Experiments to add your own functionality</td>
    </tr>
    <tr>
        <th nowrap>multiple metrics</th>
        <td nowrap>
            <a href="advanced_examples/multi_metric_example.ipynb">NB</a>
            /
            <a href="advanced_examples/multi_metric_example.py">Script</a>
        </td>
        <td>Recording multiple metrics and how to use each for hyperparameter optimization</td>
    </tr>
</table>