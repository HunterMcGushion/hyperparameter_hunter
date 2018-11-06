from setuptools import setup, find_packages


def get_version():
    """Get the current version number for the library

    Returns
    -------
    String
        Of the form "<major>.<minor>.<micro>", in which "major", "minor" and "micro" are numbers"""
    with open("hyperparameter_hunter/VERSION") as f:
        return f.read().strip()


def readme():
    """Get the content of the library's readme file

    Returns
    -------
    String
        The content of the "README.md" file located in the project's root directory"""
    with open("README.md") as f:
        return f.read()


setup(
    name="hyperparameter_hunter",
    version=get_version(),
    description=(
        "Easy hyperparameter optimization and automatic result "
        "saving across machine learning algorithms and libraries"
    ),
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=(
        "hyperparameter tuning optimization machine learning "
        "artificial intelligence neural network keras "
        "scikit-learn xgboost catboost lightgbm rgf"
    ),
    url="https://github.com/HunterMcGushion/hyperparameter_hunter",
    author="Hunter McGushion",
    author_email="hunter@mcgushion.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "dill",
        "nbconvert",
        "nbformat",
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-optimize",
        "scipy",
        "simplejson",
    ],
    extras_require={
        "dev": ["pre-commit"],
        "docs": ["numpydoc", "hyperparameter-hunter", "keras"],
        "travis": ["nose", "black", "hyperparameter-hunter", "keras", "tensorflow"],
    },
    include_package_data=True,
    zip_safe=False,
    test_suite="nose.collector",
    tests_require=["nose"],
    classifiers=(
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Utilities",
        "Topic :: Desktop Environment :: File Managers",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ),
)
