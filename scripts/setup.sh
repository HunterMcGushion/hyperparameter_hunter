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
    pip install --upgrade pip
    pip install -U black isort nose twine
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
