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
    # IMAGE_NAME=tensorflow/tensorflow:latest-gpu-py3
    IMAGE_NAME=tensorflow/tensorflow:nightly-devel-gpu-py3
  fi

  if [[ "$DOCKER_PULL" ]]; then
    docker pull ${IMAGE_NAME}
  fi
fi

if [[ ! "$CONTAINER_NAME" ]]; then
  CONTAINER_NAME="$(id -nu)"-concise-chit-chat
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

DOCKER_CONTAINER_ID=$(docker ps -q -a -f name="$CONTAINER_NAME")

if [[ ! "$DOCKER_CONTAINER_ID" ]]; then
  echo "Creating new docker container: ${CONTAINER_NAME} ..."
  $DOCKER_CMD run \
      -t -i \
      --name "$CONTAINER_NAME" \
      --mount type=bind,source="$(pwd)",target=/notebooks \
      -p 6006:6006 \
      -p 8888:8888 \
      "$IMAGE_NAME" \
      /bin/bash
else
  echo "Resuming exiting docker container: ${CONTAINER_NAME}, press [Enter] to continue ..."
  $DOCKER_CMD start "${CONTAINER_NAME}"
  $DOCKER_CMD attach "${CONTAINER_NAME}"
fi
