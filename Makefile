ROOTDIR = $(shell pwd)

# This nifty perl one-liner collects all comments headed by the double "#" symbols next to each target and recycles them as comments
.PHONY: help
help: ## Print this help message
	@perl -nle'print $& if m{^[/a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'


.PHONY: lint
lint: ## Run ruff linting
	@ruff check --fix refl1d/ tests/ setup.py

.PHONY: format
format: ## Run ruff formatting
	@ruff format refl1d/ tests/ setup.py

.PHONY: clean
clean: ## Delete some cruft from builds/testing/etc.
	rm -f `find . -type f -name '*.py[co]'`
	rm -rf `find . -name __pycache__ -o -name "*.egg-info"` \
		`find . -name 'output-*'` \
		.coverage build dist docs/_build \
		.pytest_cache \
		.ruff_cache 
