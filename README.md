# TRIST
The Threat Research and Intelligence Sharing Testbed aims to make CPS threat research easy. Using docker makes testing easily deployable and adaptable to your needs. 

## About

Contains the resources required to simulate federeated learning implemented with docker.
You will require wsl and docker desktop to run this example.

Enjoy!

## How to

1. Execute run.sh

    ``` bash
    $ bash run.sh
    ```

2. Watch as your images are prepared and clients are started!

---

Pro tip: use the flag -s (or --skip-build) to skip building!

Pro'er tip: DON'T do this if you see an error :)

### run.sh

Creates docker containers for server and clients connected to a network

    ``` bash
    bash run.sh [num of desired clients]
    ```

## Prerequisites


### Windows

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Linux

- [python3](https://www.python.org/downloads/)
