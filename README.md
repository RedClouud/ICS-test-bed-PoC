# Federated Learning implemented into Docker

This example demonstrates a simple federated learning (FL) environment implemented into Docker. The deep learning model is a multilayer perceptron (MLP) to simplify training.

## Prerequisites

This example was developed in Windows, however, it \*_should_ work in both Windows and Linux.

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [python3](https://www.python.org/downloads/)

If you would like to use your compatable nvidia GPU for computing, install the [nvidia toolkit](https://developer.nvidia.com/cuda-downloads) on your Linux distribution.

Enjoy!

## How to

1. Execute build.sh: ensures that the required Docker images are built

   ```bash
   $ bash build.sh
   ```

2. Execute the docker-compose file: builds the FL environment and begins learning

   ```bash
   $ cd learn
   $ docker-compose up -d
   ```

You should now have three containers running:

- One container acting as the FL server
- Two containers acting as the FL clients

---

## TODO

- [ ] Upload the Docker images so that they are available without needing to build
