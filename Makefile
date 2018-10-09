# Makefile for self-adaptive-programming
# Author: Huan LI <zixia@zixia.net> git.io/zixia

SOURCE_GLOB=$(wildcard bin/*.py src/*.py tests/*.py)

.PHONY: all
all : clean lint

.PHONY: clean
clean:
	echo "TODO: clean what?"

.PHONY: lint
lint: pylint pycodestyle flake8 mypy

.PHONY: pylint
pylint:
	pylint $(SOURCE_GLOB)

.PHONY: pycodestyle
pycodestyle:
	pycodestyle --statistics --count $(SOURCE_GLOB)

.PHONY: flake8
flake8:
	flake8 $(SOURCE_GLOB)

.PHONY: mypy
mypy:
	MYPYPATH=stubs/ mypy \
		$(SOURCE_GLOB)

.PHONY: download
download:
	./scripts/download.sh

.PHONY: dataset
dataset:
	python3 ./scripts/generate-dataset.py > data/dataset.txt

.PHONY: docker
docker:
	./scripts/docker.sh

.PHONY: install
install:
	pip3 install -r requirements.txt

.PHONY: pytest
pytest:
	PYTHONPATH=src/ pytest src/ tests/

.PHONY: test
test: check-version pytest

.PHONY: check-version
check-version:
	./scripts/check_version.py

code:
	# code src/	# vscode need to use src as root dir
	PYTHONPATH=src/ code .

.PHONY: preprocess
preprocess:
	PYTHONPATH=src/ python3 src/preprocess.py

.PHONY: train
train:
	PYTHONPATH=src/ python3 src/train.py

.PHONY: chitchat
chitchat:
	PYTHONPATH=src/ python3 src/chitchat.py
