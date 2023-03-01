@echo off

:: # This script is only usable on Windows 10 with Docker Desktop installed.

:: Environment parameters
set /A client_amount=2
set client_name=flwr-client
set server_name=flwr_server
set network_name=flwr-network
set client_image_name=flwr-client:1.0
set server_image_name=flwr-server:1.0

:: Rebuild the Docker image for server
echo Rebuilding Docker image...
ren Dockerfile-server Dockerfile
docker build -t %server_image_name% .
ren Dockerfile Dockerfile-server 
echo Docker image has been rebuilt.

:: Rebuild the Docker image for client
echo Rebuilding Docker image...
ren Dockerfile-client Dockerfile
docker build -t %client_image_name% .
ren Dockerfile Dockerfile-client 
echo Docker image has been rebuilt.

:: Deletes containers if they exist 
docker rm flwr-server

: Make loop
docker rm flwr-client1
docker rm flwr-client2

:: Creates containers and connects them to the fl_network
docker create --name flwr-server --network flwr-network flwr-server:1.0

:: Make loop
docker create --name flwr-client1 --network flwr-network flwr-client:1.0
docker create --name flwr-client2 --network flwr-network flwr-client:1.0

echo All FL nodes have been created and connected to the fl_network.

echo Cleaning up...
docker image prune --filter dangling=true -f

:: Start the server
docker start flwr-server

:: Allow the server to start
timeout /t 5 >nul

:: Start the clients
docker start flwr-client1
docker start flwr-client2
