
PROJECT_NAME := "pytmod"
PROJECT_DIR := `pwd`
VERSION := ```python3 -c "import toml;print(toml.load('pyproject.toml')['project']['version'])"```
URL := ```python3 -c "import toml;print(toml.load('pyproject.toml')['project']['urls']['Repository'])"```
BRANCH := `git branch --show-current`
LINT_FLAGS := "E501,F401,F403,F405,W503,E402,E203"


# List all recipes
list:
  just --list


# Install all
install-dev:
	pip install -e .[doc,test,dev]

# Install locally
install:
	pip install .



# Clean generated files
clean:
	cd doc && make clean
	rm -rf *.whl htmlcov builddir .pytest_cache coverage.xml .coverage
	rm -rf wheelhouse build


# Build documentation
doc:
    cd doc && make html

# Build documentation and watch
autodoc:
	cd doc && just autobuild

# Push to github
gh:
	@echo "Pushing to github..."
	git add -A
	@read -p "Enter commit message: " MSG; \
	git commit -a -m "$MSG"
	git push origin main

# Clean, reformat and push to github
save: format gh

# Format with black
format:
    black .

# Lint with flake8
lint:
	@flake8 --exit-zero --ignore={{LINT_FLAGS}} {{PROJECT_NAME}} 
	
# Lint using flake8
lint-extra:
	@flake8 --exit-zero --ignore={{LINT_FLAGS}} {{PROJECT_NAME}}  test/ examples/ --exclude "dev*"

# Check for duplicated code
dup:
	@pylint --exit-zero -f colorized --disable=all --enable=similarities {{PROJECT_NAME}}


# Clean test coverage reports
cleantest:
	@rm -rf .coverage* htmlcov coverage.xml



# Run the test suite
test:
	pytest tests \
	--cov={{PROJECT_NAME}}/ --cov-report term --cov-report html --cov-report xml  \
	--durations=0

#  Update header
header:
	cd dev && python update_header.py

# Show html documentation in the default browser
show:
    cd doc && make -s show

docker-build:
    docker build . -t pytmod
