#!/bin/bash

# This script is only usable on Linux with Docker installed.

cd "$(dirname "$0")"

server_image_name=flwr-server:latest
client_image_name=flwr-client:latest

### DO NOT EDIT BELOW THIS LINE ###

# Download the CIFAR-10 dataset
python3 -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

# Rebuild the Docker image for server
echo "Rebuilding server Docker image..."
mv Dockerfile-server Dockerfile
docker build -t $server_image_name .
mv Dockerfile Dockerfile-server
echo "Server Docker image has been rebuilt."

# Rebuild the Docker image for client
echo "Rebuilding client Docker image..."
mv Dockerfile-client Dockerfile
docker build -t $client_image_name .
mv Dockerfile Dockerfile-client
echo "Client Docker image has been rebuilt."

# Deletes hanging (previous) image(s)
echo "Cleaning up..."
docker image prune --filter dangling=true -f
