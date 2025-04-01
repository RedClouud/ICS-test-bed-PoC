import docker
import random
import string


def initDocker():
    try:
        client = docker.from_env()
        return client
    except:
        return None


def addTag():
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(5))
    return f"-{result_str}"


def startClient():
    # TODO: Ask for name, if none is given give tag
    return client.containers.run(
        "bfirsh/reticulate-splines", name=f"client{addTag()}", detach=True
    )


def startServer():
    # TODO: Ask for name, if none is given give tag
    return client.containers.run(
        "bfirsh/reticulate-splines", name=f"server{addTag()}", detach=True
    )


def createNetwork(name):
    network = client.networks.create(name)
    createdNetworks.append(network)
    return network


def connectContainer(client, network):
    network.connect(client)
    return True


def pingContainers(client, server):
    pingResult = client.exec_run(f"ping {server} -c5")

    # wxwc_run returns output from command as unformatted bytes in second index
    return pingResult[1].decode("utf-8")


def removeContainer(container):
    container.stop()
    container.remove()
    return True


def removeAll():
    # Remove containers
    for container in client.containers.list():
        container.stop()
    nDeletedContainers = client.containers.prune()  # Returns list of deleted containers

    # Remove networks
    nDeletedNetworks = client.networks.prune()  # Returns list of deleted networks

    # The first entry for each dict is the list of deleted objects
    return len(list(nDeletedContainers.values())[0]), len(
        list(nDeletedNetworks.values())[0]
    )


client = initDocker()
createdNetworks = []
