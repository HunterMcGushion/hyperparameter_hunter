#!/bin/bash

##################################################
# Set Paths
##################################################
MY_ENV="env"
MY_PYTHON="python3"  # ;)
# MY_PYTHON="/Library/Frameworks/Python.framework/Versions/3.6/bin/python3"

if ! [[ $# -eq 0 ]]; then
    MY_ENV=$1

    if [[ $# -eq 2 ]]; then
        MY_PYTHON=$2
    fi
fi

function get_python_version() {
    local version=`${MY_PYTHON} <<- HEREDOC
		import sys
		version = sys.version_info[:]
		print("OH SHIT" if version < (3, 6) else "all good")
	HEREDOC`

    if [[ "${version}" = "OH SHIT" ]]; then
        return 1
    else
        return 0
    fi
}

##################################################
# Check Python Version
##################################################
version=$(get_python_version)$?

if [[ ${version} -eq 1 ]]; then
    echo "ERROR: Python 3.6 or greater is required. Supply a path to a Python3.6 executable via:"
    echo "1) The \`PYTHON=<python_path>\` option if using \`make\`, or"
    echo "2) The second command-line argument if executing \`env.sh\` directly"
    exit 1
elif [[ ${version} -eq 0 ]]; then
    echo "Found at least Python 3.6"
else
    echo "Found something else: ${version}"
fi

##################################################
# Create Virtual Environment
##################################################
${MY_PYTHON} -m venv ${MY_ENV}

echo "run \`source <$MY_ENV>/bin/activate\` to use virtualenv"

