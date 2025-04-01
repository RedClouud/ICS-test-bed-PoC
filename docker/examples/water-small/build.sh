#!/bin/bash

# This script is only usable on Linux with Docker installed.

cd "$(dirname "$0")"

# Rebuild the Docker image for server
echo "Rebuilding Docker image..."
mv Dockerfile-TRIST Dockerfile
docker build -t redclouud/trist:latest .
mv Dockerfile Dockerfile-TRIST
echo "Docker image has been rebuilt."

# Deletes hanging (previous) image(s)
echo "Cleaning up..."
docker image prune --filter dangling=true -f
