#!/bin/bash

# This script is only usable on Linux with Docker installed.

cd "$(dirname "$0")"

# Rebuild the Docker image for server
echo "Rebuilding server Docker image..."
mv Dockerfile-SWaT Dockerfile
docker build -t swat:latest .
mv Dockerfile Dockerfile-SWaT
echo "Server Docker image has been rebuilt."

# Deletes hanging (previous) image(s)
echo "Cleaning up..."
docker image prune --filter dangling=true -f
