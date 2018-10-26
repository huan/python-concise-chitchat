# Makefile for self-adaptive-programming
# Author: Huan LI <zixia@zixia.net> git.io/zixia

SOURCE_GLOB=$(wildcard bin/*.py chit-chat/*.py tests/*.py)

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
	python3 ./scripts/generate-dataset.py

.PHONY: docker
docker:
	./scripts/docker.sh

.PHONY: install
install:
	pip3 install -r requirements.txt

.PHONY: pytest
pytest:
	PYTHONPATH=. pytest chit-chat/ tests/

.PHONY: test
test: check-version pytest

.PHONY: check-version
check-version:
	./scripts/check_version.py

code:
	# vscode need to know where the modules are by setting PYTHONPATH
	PYTHONPATH=. code .

.PHONY: train
train:
	PYTHONPATH=. python3 bin/train.py

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir=./data/tensorboard/

.PHONY: chat
chat:
	PYTHONPATH=. python3 bin/chat.py
