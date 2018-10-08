#!/usr/bin/env bash
set -e

if [[ ! "$NV_GPU" ]]; then
  NV_GPU=0
fi

if [[ ! "$DOCKER_CMD" ]]; then
  DOCKER_CMD=docker
  IMAGE_NAME=tensorflow/tensorflow:latest-py3

  if [[ $(which nvidia-docker) ]]; then
    DOCKER_CMD=nvidia-docker
    IMAGE_NAME=tensorflow/tensorflow:latest-gpu-py3
  fi
fi

if [[ ! "$CONTAINER_NAME" ]]; then
  CONTAINER_NAME="$(id -nu)"
fi

cat <<_MSG_

Starting Docker Container ...
========================

NV_GPU=$NV_GPU
DOCKER_CMD=$DOCKER_CMD
IMAGE_NAME=$IMAGE_NAME
CONTAINER_NAME=$CONTAINER_NAME

------------------------
_MSG_

$DOCKER_CMD run \
		-t -i \
		--name "$CONTAINER_NAME" \
		--rm \
		--mount type=bind,source="$(pwd)",target=/notebooks \
		"$IMAGE_NAME" \
		/bin/bash
