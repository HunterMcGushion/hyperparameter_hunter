The contents of this module are based on utilities from the excellent [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize)
library. Therefore, any code contained in this module originally from SKOpt is licensed under
the "New BSD License" (3-clause BSD license), which can be found in this module, [here](LICENSE). 

Any additions or modifications not originating from Scikit-Optimize fall under 
HyperparameterHunter's license, [here](../../../../LICENSE).

All assets originally from Scikit-Optimize have been significantly modified, refactored and/or 
documented. Below are all the files in the HyperparameterHunter source code/tests that are based on 
Scikit-Optimize code:
1. "hyperparameter_hunter/optimization/backends/skopt/engine.py" (based on "scikit-optimize/skopt/optimizer/optimizer.py")
2. "hyperparameter_hunter/space/dimensions.py" and "space_core.py" (based on "scikit-optimize/skopt/space/space.py")
3. "tests/test_optimization/test_backends/test_skopt/test_engine.py" (based on "scikit-optimize/tests/test_optimizer.py")
4. "tests/test_space/test_skopt_space.py" (based on "scikit-optimize/tests/test_space.py")

Thank you to the Scikit-Optimize developers for their fantastic work and for allowing others to 
build on that work.