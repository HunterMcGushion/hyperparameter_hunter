#!/bin/bash

##################################################
# Set Environment Path
##################################################
MY_ENV="env"

if ! [[ $# -eq 0 ]]; then
    MY_ENV=$1
fi

##################################################
# Define Try/Catch Functions
##################################################
function try_setup() {
    source ./${MY_ENV}/bin/activate

    pip install catboost
    pip install lightgbm==2.1.0
    pip install rgf_python
    pip install xgboost

    pip install --ignore-installed --upgrade https://github.com/lakshayg/tensorflow-build/releases/download/tf1.11.0-macos-mojave-py2.7-py3.7/tensorflow-1.11.0-cp37-cp37m-macosx_10_13_x86_64.whl
    pip install keras
}

function log_error() {
    echo "\033[0;31m ERROR: Received non-existent environment path: $MY_ENV \033[0m"
    exit 1
}

##################################################
# Execute Try/Catch
##################################################
(
    try_setup
) || (
    log_error
)
