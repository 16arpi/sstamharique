#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = stt_amh
PYTHON_VERSION = 3
PYTHON_INTERPRETER = python$(PYTHON_VERSION)
PROJECT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

EXTERNAL_DATASET_FILES := mms.zip
EXTERNAL_DATASETS := $(patsubst %,data/external/%,$(EXTERNAL_DATASET_FILES))

PROCESSED_DATASET_FILES := mms/dataset.csv
PROCESSED_DATASETS := $(patsubst %,data/processed/%,$(PROCESSED_DATASET_FILES))

MODEL_FILES := amh-pretrained.tar.gz
MODELS := $(patsubst %,models/%,$(MODEL_FILES))
MODELS_CACHE := models/pretrained

DATASETS := $(EXTERNAL_DATASETS) $(PROCESSED_DATASETS)

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	aws s3 sync s3://tal-m2-amh/data/ \
		data/
	aws s3 sync s3://tal-m2-amh/models/ \
		models/


## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync data/ \
		s3://tal-m2-amh/data/
	aws s3 sync models/ \
		s3://tal-m2-amh/models/
	for dataset in $(EXTERNAL_DATASETS) ; do \
		aws s3api put-object-acl --bucket tal-m2-amh --acl public-read --key $${dataset} ; \
	done
	for model in $(MODELS) ; do \
		aws s3api put-object-acl --bucket tal-m2-amh --acl public-read --key $${model} ; \
	done


## Download experiment data w/o S3
$(EXTERNAL_DATASETS):
	mkdir -p data/{external,raw,interim,processed}
	wget "https://tal-m2-amh.s3.gra.io.cloud.ovh.net/$@" -O "$@"

$(PROCESSED_DATASETS):
	for dataset in $(EXTERNAL_DATASETS) ; do \
		[[ "$${dataset}" == *.zip ]] && unzip "$${dataset}" -d data/processed/ ; \
	done

$(MODELS):
	mkdir -p models
	wget "https://tal-m2-amh.s3.gra.io.cloud.ovh.net/$@" -O "$@"

$(MODELS_CACHE):
	for model in $(MODELS) ; do \
		[[ "$${model}" == *.tar.gz ]] && tar xvf "$${model}" -C models/ ; \
	done

.PHONY: download_data
download_data: $(DATASETS) $(MODELS) $(MODELS_CACHE)

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION) --system-site-packages
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) stt_amh/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
