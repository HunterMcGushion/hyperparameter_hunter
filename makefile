env:
	python3 -m venv env
	@echo 'run `source env/bin/activate` to use virtualenv'

setup: env
	# source env/bin/activate && pip3 install -U black isort twine
	source env/bin/activate
	pip3 install -U black isort nose twine

dev: setup
	# source env/bin/activate && python3 setup.py develop && pip install hyperparameter_hunter[dev]
	source env/bin/activate
	python3 setup.py develop
	pip install hyperparameter_hunter[dev]
	pre-commit install
	@echo 'run `source env/bin/activate` to develop hyperparameter_hunter'

release: lint test clean
	python3 setup.py sdist bdist_wheel
	twine upload dist/*

format:
	black hyperparameter_hunter setup.py
	isort

lint:
	black --check hyperparameter_hunter setup.py

test:
	source env/bin/activate
	nosetests

clean:
	rm -rf build dist *.egg-info

distclean: clean
	rm -rf env