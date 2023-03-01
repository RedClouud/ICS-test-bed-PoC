# CPS testbed

## About
Contains the resources required to simulate federeated learning implemented with docker (windows) or simply terminals (linux)... Enjoy!

## buildrun.bat
Intended for use on Windows
Creates docker containers for server and clients connected to a network

.\build_run.bat [num of desired clients]

## run.sh
Intended for use on Linux
Creates three terminals; one for a server and two for two clients

bash run.sh

## Prerequisites
### Windows
- Docker Desktop

### Linux
- python3
- flwr
- torch
- tqdm
