@echo off

:: # This script is only usable on Windows 10 with Docker Desktop installed.

:: Environment configuration
set /A client_amount=5
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

:: Deletes all containers if they exist
for /L %%i in (1,1,%client_amount%) do docker rm %client_name%-%%i

:: Creates containers and connects them to the fl_network
docker create --name flwr-server --network flwr-network flwr-server:1.0

:: Create all clients
for /L %%i in (1,1,%client_amount%) do docker create --name %client_name%-%%i --network flwr-network flwr-client:1.0

echo All FL nodes have been created and connected to the fl_network.

echo Cleaning up...
docker image prune --filter dangling=true -f

:: Start the server
docker start flwr-server
timeout /t 5 >nul

:: Start all clients
for /L %%i in (1,1,%client_amount%) do docker start %client_name%-%%i
