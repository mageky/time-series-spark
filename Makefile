help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build dir
	rm -rf ./build
	rm -rf ./.pytest_cache
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

build: clean ## Build project
	mkdir -p ./build/dist
	cp ./src/*spark_driver.py ./build/dist
	cd ./src && zip -x *spark_driver.py -x \*__pycache__\* -r ../build/dist/app.zip .

test: build ## Run tests
	pytest tests/unit
