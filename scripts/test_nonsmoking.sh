#!/bin/bash

##################################################
# Set Environment Path
##################################################
MY_ENV="env"

if ! [[ $# -eq 0 ]]; then
    MY_ENV=$1
fi

##################################################
# Activate Virtual Environment
##################################################
source ./${MY_ENV}/bin/activate

##################################################
# Run Tests
##################################################
pytest \
    --doctest-modules hyperparameter_hunter \
    --cov=hyperparameter_hunter \
    --disable-pytest-warnings \
    --cov-report html \
    --cov-report term \
    tests/ \
    --durations=10 \
    --ignore tests/test_workflows
