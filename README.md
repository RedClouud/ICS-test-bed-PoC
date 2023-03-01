# CPS testbed

## About
Contains the resources required to simulate federeated learning implemented with docker... Enjoy!

## buildrun.bat
Intended for use on Windows
Creates docker containers for server and clients connected to a network

.\build_run.bat [num of desired clients]

## run.sh
Intended for use on Linux
Creates three terminals; one for a server and two for two clients

bash run.sh

## Prerequisites
- (nvidia-docker)[https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqazlKNzBtSk96dkdrWnNuLXNwdnBJRVFyWnRPZ3xBQ3Jtc0trMUwwMzFuYjNfWGRERUxjVVlXSTNRaWV4b2M1UlhXS1A0VjF6cUVtdnk3ZmlmYUpWdDFCZEVVdzh0d3Q2dkJIZW1sc3ZtMGxGdEg4MGdNQkQxNmtyamtOM3pIa1VtT3hzejFjanFCYzFpbEltX2szRQ&q=https%3A%2F%2Fgithub.com%2FNVIDIA%2Fnvidia-docker&v=-Y4T71UDcMY]

### Windows
- Docker Desktop

### Linux
- python3
- flwr
- torch
- tqdm
