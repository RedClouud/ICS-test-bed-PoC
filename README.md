## About TRIST
TRIST is an open-source Cyber Physical System (CPS) Threat Research and Intelligence Sharing Testbed.
TRIST aims to make CPS simulation and dataset development easy by providing the following features:
- CPS simulation
- CPS dataset development and testing
- CPS attack simulation
- Full data analytics tools

It uses multiple technolgies to achieve this including [ELK](https://www.elastic.co/), [Docker](https://www.docker.com/), [Flower](https://flower.dev/), and more.

TRIST is part inspired by [MiniCPS](https://minicps.readthedocs.io/), a CPS simulation tool built on top of [Mininet](http://mininet.org/). However, TRIST takes this one step further by implementing any CPS into portable and instantly deployable Docker containers. This means no installtion, configuration, or setup is requried. Simply download the Docker image and press go. It is then as simple as accessing the TRIST Dashboard via a web browser to access all functionality.

We hope you enjoy using TRIST. 

---

# Federated Learning implemented into Docker

This example demonstrates a simple federated learning (FL) environment implemented into Docker. The deep learning model is a multilayer perceptron (MLP) to simplify training.

## Prerequisites

This example was developed in Windows, however, it _should_ work in both Windows and Linux.

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
