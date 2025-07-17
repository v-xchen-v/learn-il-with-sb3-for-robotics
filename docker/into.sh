#!/bin/bash

CONTAINER_NAME=sb3_il_container

# Check if container is running
if docker ps -f name=$CONTAINER_NAME --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
    echo "🔁 Attaching to container '$CONTAINER_NAME'..."
    echo "🔧 MUJOCO_GL mode: ${MUJOCO_GL}"
    docker exec -it $CONTAINER_NAME bash
else
    echo "❌ Container '$CONTAINER_NAME' is not running."
    echo "👉 Run './start.sh' first to start it."
fi