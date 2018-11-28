#!/bin/bash

##################################################
# Determine Version
##################################################
MY_VERSION=`cat hyperparameter_hunter/VERSION`

echo "Have the necessary release files (VERSION, CHANGELOG) been updated and \`git add\`-ed?"
read -p "Press 'y' to continue, or 'n' to abort... " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    ##################################################
    # Start Release Process
    ##################################################
    # Assume at least the necessary release files (VERSION, CHANGELOG.md) have been `git add`-ed
    git commit -m "Upload v$MY_VERSION to PyPI"
    git tag "v$MY_VERSION"

    ##################################################
    # Upload to PyPI
    ##################################################
    python setup.py sdist bdist_wheel
    twine upload dist/*

    ##################################################
    # Upload to Repository
    ##################################################
    git push origin HEAD
    git push origin --tags

    echo "Please finalize release on GitHub with supporting PyPI files"
else
    echo "Aborting release"
fi

