# Makefile for self-adaptive-programming
# Author: Huan LI <zixia@zixia.net> git.io/zixia

SOURCE_GLOB=$(wildcard bin/*.py chit-chat/*.py tests/*.py)

.PHONY: all
all : clean lint

.PHONY: clean
clean:
	rm -fr data/board data/save

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
	PYTHONPATH=. python3 ./bin/generate-dataset.py

.PHONY: docker
docker:
	./scripts/docker.sh

.PHONY: install
install:
	pip3 install -r requirements.txt

.PHONY: pytest
pytest:
	PYTHONPATH=. pytest chit_chat/ tests/

.PHONY: test
test: pytest

code:
	# vscode need to know where the modules are by setting PYTHONPATH
	PYTHONPATH=. code .

.PHONY: train
train:
	PYTHONPATH=. python3 bin/train.py

.PHONY: board
board:
	tensorboard --logdir=./data/board/

.PHONY: chat
chat:
	PYTHONPATH=. python3 bin/chat.py
