@echo off

:: # This script is only usable on Windows 10 with Docker Desktop installed.

:: Environment configuration
set /A client_amount=%1
set client_name=flwr-client
set server_name=flwr-server
set network_name=flwr-network
set client_image_name=flwr-client:1.0
set server_image_name=flwr-server:1.0

:: DO NOT CHANGE ANYTHING BELOW THIS LINE

:: Rebuild the Docker image for server
echo Rebuilding server Docker image...
ren Dockerfile-server Dockerfile
docker build -t %server_image_name% .
ren Dockerfile Dockerfile-server 
echo Server Docker image has been rebuilt.

:: Rebuild the Docker image for client
echo Rebuilding client Docker image...
ren Dockerfile-client Dockerfile
docker build -t %client_image_name% .
ren Dockerfile Dockerfile-client 
echo Client Docker image has been rebuilt.

:: Deletes exisitng containers 
docker rm %server_name%
for /L %%i in (1,1,%client_amount%) do docker rm %client_name%-%%i

:: Creates the network if it doesn't exist
docker network create %network_name%

:: Creates containers and connects them to the fl_network
docker create --name %server_name% --network %network_name% %server_name%:1.0
for /L %%i in (1,1,%client_amount%) do docker create --name %client_name%-%%i --network %network_name% flwr-client:1.0

echo All FL nodes have been created and connected to the fl_network.

:: Deletes hanging (previous) image(s)
echo Cleaning up...
docker image prune --filter dangling=true -f

:: Starts the containers
docker start %server_name%
timeout /t 5 >nul
for /L %%i in (1,1,%client_amount%) do docker start %client_name%-%%i
