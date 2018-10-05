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
function try_dev() {
    source ./${MY_ENV}/bin/activate
    pip install --upgrade pip
    python setup.py develop
    pip install hyperparameter_hunter[dev]
    pre-commit install
}

function log_error() {
    echo "\033[0;31m ERROR: Received non-existent environment path: $MY_ENV \033[0m"
    exit 1
}

function after_success() {
    echo "run \`source <$MY_ENV>/bin/activate\` to develop hyperparameter_hunter"
}

##################################################
# Execute Try/Catch
##################################################
(
    (
        try_dev
    ) || (
        log_error
    )
) && (
    after_success
)
