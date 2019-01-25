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
# Run `docstr-coverage`
##################################################
docstr-coverage \
    hyperparameter_hunter \
    --skipmagic \
    --skipclassdef \
    --exclude ".*__init__.py|.*boltons_utils.py|.*__temp_model_builder.py"
