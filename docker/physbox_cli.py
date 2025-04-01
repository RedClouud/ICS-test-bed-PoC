import control_node as cn

chosenMenuOption = "-1"

client = cn.initDocker()
if client == None:
    print("Error: could not access Docker daemon")
    print("Possible fix: ensure that the Docker daemon is running!")
    exit(1)

if __name__ == "__main__":
    # 6 is exit
    while chosenMenuOption != "7":
        print(
            """
        -------------------
        --- PhysBox CLI ---
        -------------------

        1. Create client
        2. Create server
        3. Connect containers
        4. Ping containers

        5. Remove a container
        6. Remove all objects
        7. Exit
            """
        )

        chosenMenuOption = input("Input: ")
        print(f"You have selected option {chosenMenuOption}\n")

        if chosenMenuOption == "1":
            print("Starting client...")
            container = cn.startClient()
            if container == None:
                print("Error: could not start container")
            print(f"Started client with name {container.name}.")

        elif chosenMenuOption == "2":
            print("Starting server...")
            container = cn.startServer()
            if container == None:
                print("Error: could not start container")
            print(f"Started server with name {container.name}.")

        elif chosenMenuOption == "3":
            print(
                "Please specify an existing network name or specify the name of a new one..."
            )
            print("Exisitng networks:")
            exisitngNetworks = client.networks.list()
            for network in exisitngNetworks:
                print(f"{network.name}, {network.short_id}")
            networkName = input("Network name: ")
            try:
                networkObj = client.networks.get(networkName)
            except:
                networkObj = cn.createNetwork(networkName)
                print(f"Created {networkName}")

            print("Current containers:")
            for container in client.containers.list():
                print(f"{container.name}, {container.short_id}")
            clientName = input(f"Input name or id to connect to {networkObj.name}: ")
            clientObj = client.containers.get(clientName)

            isSuccess = cn.connectContainer(clientObj, networkObj)
            if isSuccess:
                print(f"Successfully connected {clientObj.name} to {networkObj.name}")
            else:
                print(f"Error: could not connect {clientName} to {networkObj.name}")

        elif chosenMenuOption == "4":
            print("Current containers:")
            for container in client.containers.list():
                print(f"{container.name}, {container.short_id}")
            clientName = input("Input client name or id to ping FROM: ")
            serverName = input("Input server name or ip to ping TO: ")

            clientObj = client.containers.get(clientName)

            print("Pinging containers, please wait...\n")
            pingOutput = cn.pingContainers(clientObj, serverName)
            print(pingOutput)
            _ = input("Press enter to continue...")
            _ = None  # clear user input

        elif chosenMenuOption == "5":
            print("Current containers:")
            for container in client.containers.list():
                print(f"{container.name}, {container.short_id}")
            containerName = input("Input container name or id to delete: ")
            containerObj = client.containers.get(containerName)
            isSuccess = cn.removeContainer(container)
            if isSuccess:
                print(f"Successfully removed {containerObj.name}")
            else:
                print("Error: could not remove container")

        elif chosenMenuOption == "6":
            print("Removing all objects")
            print("This may take a while...")
            nDeletedContainers, nDeletedNetworks = cn.removeAll()

            print(
                f"Successfully removed {nDeletedContainers} containers and {nDeletedNetworks} networks."
            )

        elif chosenMenuOption == "7":
            nRunningContainers = len(client.containers.list())
            if nRunningContainers > 0:
                print(f"NOTE: you still have {nRunningContainers} containers running.")

            print("Exiting...")
            exit(0)

        else:
            print("Please choose a valid option!")
