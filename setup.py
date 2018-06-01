from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='hyperparameter_hunter',
    version='0.1',
    description='Easy hyperparameter optimization and automatic result saving across machine learning algorithms and libraries',
    long_description=readme(),
    classifiers=[],
    keywords='hyperparameter tuning optimization machine learning artificial intelligence neural network keras scikit-learn xgboost catboost lightgbm rgf',
    url='',
    author='Hunter McGushion',
    author_email='hunter@mcgushion.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        # 'h5py',  # ?
        # 'keras',  # ?
        'numpy',
        'pandas',
        'scikit-learn',
        'scikit-optimize',
        'scipy',
        'simplejson',
        # 'tensorflow',  # ?
        # 'xgboost',  # ?
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose']
)
