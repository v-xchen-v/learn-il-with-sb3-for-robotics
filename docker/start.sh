#!/bin/bash

# Name of the container
CONTAINER_NAME=sb3_il_container

# Image name
IMAGE_NAME=sb3_il

# Optional: Mount host code or data directory if needed
# e.g., -v /path/to/data:/workspace/data

docker run --gpus all -itd \
  --name $CONTAINER_NAME \
  --rm \
#   --shm-size=16g \
  -v $(pwd):/workspace \
  $IMAGE_NAME

echo "✅ Container '$CONTAINER_NAME' started in background."