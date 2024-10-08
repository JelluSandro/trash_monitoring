#!/bin/bash

set -e

rest=$@

IMAGE=fast_api_server_trash_monitoring:latest

CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${IMAGE} 2> /dev/null)
if [[ "${CONTAINER_ID}" ]]; then
    docker run --name garbage --shm-size=4g --gpus all --restart='always' \
      -v `pwd`:/scratch \
      -p 9131:9131 \
      --user $(id -u):$(id -g) --workdir=/scratch -e HOME=/scratch $IMAGE $@
else
    echo "Unknown container image: ${IMAGE}"
    exit 1
fi