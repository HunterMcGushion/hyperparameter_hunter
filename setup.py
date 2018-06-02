from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='hyperparameter_hunter',
    version='0.0.1',
    description='Easy hyperparameter optimization and automatic result saving across machine learning algorithms and libraries',
    long_description=readme(),
    keywords='hyperparameter tuning optimization machine learning artificial intelligence neural network keras scikit-learn xgboost catboost lightgbm rgf',
    url='https://github.com/HunterMcGushion/hyperparameter_hunter',
    author='Hunter McGushion',
    author_email='hunter@mcgushion.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        # 'h5py',  # ?
        'keras',  # New
        'numpy',
        'pandas',
        'scikit-learn',
        'scikit-optimize',
        'scipy',
        'simplejson',
        'dill',  # New
        'tensorflow',  # New
        # 'xgboost',  # ?
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
)
