### there is a more readable version of this in the guide (this version is just optimised) ###

# Import the stuff you need
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
# telling what processor to use (where it is and where it is processed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class MLP(nn.Module):  # This model is specifically tailored for a dataset (CIFAR-10)
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    # Defines neural network layers which are tailored to learn from the dataset
    def __init__(self):
        super().__init__()
        # define layers of the neural network
        self.layers = nn.Sequential( # Sequential causes the data to be fed through the layers in the order written
            nn.Flatten(), # flattens the data (turns it into a 1D array)
            # here, we have the data represented in a way that the computer can understand (a 1D array with 64 values)
            # following are three dense layers
            nn.Linear (32 * 32 * 3, 64), # 32 * 32 (x and y of image) * 3 (RGB) = 3072. 64 is the number of neurons in the layer
            # at this stage, we have taken the data from the image and turned it into a 1D array with 64 values using the
            # Flatten() function previousy.
            # This 1D array is given to the first layer. As the input aray has 3072 different elements, 
            # this is used as the value of the first argument.
            # The second argument is where the input data is going to be stored in. In this example, we will
            # store the data in 64 different neurons. 64 is chosen because it is a power of 2, which is useful for
            # computers to process. you can use whatever amount you would like.
            # So now that we have stored the data in our 64 neurons, we can perform an operation on it to learn
            # about it's characteristics (how everything relates to eachother)
            nn.ReLU(), # this is a ReLU function which is a way ti actvate neurons
            # it is not important to understand how it works, rather what goes in and what goes out
            # what goes in is neurons which have data in them (this was done with nn.Linear())
            # what goes out is the computer's interpretation of how everything links between the neurons. in more technical
            # terms, the computer will change parameters and weights to simulate learning
            nn.Linear(64, 32), # this is our "hidden" layer. no idea, dont need to have an idea. just look at what goes
            # in and what goes out
            # in: reading 64 neurons
            # out: reading the values of those neurons (and weights?) in 32 neurons
            # again, 32 is a power of 2, which is useful for computers to process. no other reason
            nn.ReLU(), # do some more thinking/learing
            nn.Linear(32, 10) # this is our output layer
            # the computer has learnt from the data through two layers of neurons. in other words, it has 
            # only learnt a small amount of details about the data
            # in: the amount of neurons in the previous layer and their values
            # out: the amount of classes we want to classify the data into. in this case it is 10 different types
            # of vehicles and animals
        )

        # each of these lines defines a layer of the neural network
        # each layer of a neural network will learn specific things and about parts of the
        # dataset
        # e.g. one layer will learn that a dog has a tail, another layer will learn that a
        # dog has a nose
        # a deeper layer will then understand the characteristics of what makes up a tail
        # and what makes up a nose

        # this is where deep learning is useful: because you are able to characterise the data
        # into subcaegories (multilabel classification)
        # in this case, for example, you would be able to not only say that the image is a dog
        # but also that is is a german shepard

    # defines how it "progresses" (what parameters are going to change and how)
    def forward(self, x):
        return self.layers(x)
        # how we react to input data.

    # backward is already defined in nn.Module, so you don't need to define it
    # what backward does is it calculates the loss and then adjusts the parameters of the
    # neural network to make it more accurate by reducing the loss (the difference between
    # the actual value and the predicted value)
    # backward is also known as optimisation: it optimises the parameters of the neural network
    # by reducing the step size (the amount that the parameters are changed by) until the
    # loss is minimised

    # gradient decent is the process of finding the minimum of a function, in other words,
    # finding the best parameters for the neural network


def train(net, trainloader, epochs):  # trains the model to classify the data
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()  # loss calculation function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):  # one epoch is one pass through the dataset
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            # this is required because otherwise the gradients from the previous image will
            # be added to the gradients of the current image which will make learning
            # impossible

            loss = criterion(net(images.to(device)), labels.to(device))
            # calculates the current loss when using the parameters of the neural network
            # by comparing the images to the labels

            loss.backward()
            # calculates the gradient of the loss function (the gradient is the slope of the
            # loss function)

            optimizer.step()  # <-- this is where the learning happens!
            # adjust the parameters of the NN by using the previous loss to calculate the
            # gradient of the loss function (the gradient is the slope of the loss function)
            # the gradient is then used to calculate the step size (the amount that the
            # parameters are changed by)


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        # goes through the test set and compares the predicted values to the actual values
        for images, labels in tqdm(testloader):
            outputs = net(images.to(device))  # predicts values
            labels = labels.to(device)  # loads the actual values
            loss += criterion(outputs, labels).item()
            # calculates the difference betweenthe predicted and actual values (how correct
            # the model is)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # normalise is used so that the data isn't all over the place (this can have a problem
    # with deep learning when learning the characteristics of a dataset)

    # train set and dataset are split from the same dataset
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    testloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testset = DataLoader(testset)
    return testloader, testset
    # The first thing that is returned is the training set, used to train the model
    # (the model looks at the characteristics, makes a predciton, and then compares
    # it to the actual value. From the loss and accuracy, it will then adjust the
    # parameters of the model to make it more accurate)

    # notice that we shuffle the training set. This is to make sure that he model doesn't
    # learn the characteristics of the dataset in a specific order (e.g. if the dataset
    # is ordered by the type of animal, the model will learn the characteristics of
    # the first animal and then the second animal, and so on. This is not good because
    # the model will not be able to generalise to other animals. Shuffling the dataset
    # will make sure that the model learns the characteristics of the dataset in a random
    # order, so that it can generalise to other animals)
    # for example, if the dataset is ordered by lion, monkey, giraffe then the model will
    # remember that order. "If lion, then monkey", "if monkey, then giraffe", etc.
    # in other words, shuffling the dataset ensures that the model is learning just the
    # characteristics of the dataset, not the order of the dataset


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = MLP().to(device)
print(net)
trainloader, testloader = load_data()
print("trainloader", trainloader.dataset)
print("testloader", testloader.dataset)

# Define Flower client (nothing to do with PyTorch/Deep learning, just federated learning)


class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        print("get_parameters")
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        print("set_parameters")
        params_dict = zip(net.state_dict().keys(), parameters)
        print("params_dict")
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        print("state_dict")
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("fit")
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("evaluate")
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="172.18.0.2:8080",
    client=FlowerClient(),
)