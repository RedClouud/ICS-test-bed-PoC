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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # telling what processor to use (where it is and where it is processed)


class Net(nn.Module): # This model is specifically tailored for a dataset (CIFAR-10)
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    # Defines naural network layers which are tailored to learn from the dataset
    def __init__(self) -> None:
        super(Net, self).__init__()
        # example tailored variable below...
        # Conv2d
        # 3 is the number of channels (RGB), 6 is the number of filters,
        # 5 is the size of the filter
        self.conv1 = nn.Conv2d(3, 6, 5) 
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    # backward is already defined in nn.Module, so you don't need to define it
    # what backward does is it calculates the loss and then adjusts the parameters of the 
    # neural network to make it more accurate by reducing the loss (the difference between
    # the actual value and the predicted value)
    # backward is also known as optimisation: it optimises the parameters of the neural network
    # by reducing the step size (the amount that the parameters are changed by) until the
    # loss is minimised

    # gradient decent is the process of finding the minimum of a function, in other words, 
    # finding the best parameters for the neural network


def train(net, trainloader, epochs): # trains the model to classify the data
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss() # loss calculation function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs): # one epoch is one pass through the dataset
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            # this is required because otherwise the gradients from the previous image will
            # be added to the gradients of the current image which will make learning
            # impossible
            
            loss = criterion(net(images.to(DEVICE)), labels.to(DEVICE))
            # calculates the current loss when using the parameters of the neural network
            # by comparing the images to the labels

            loss.backward()
            # calculates the gradient of the loss function (the gradient is the slope of the
            # loss function)

            optimizer.step() # <-- this is where the learning happens!
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
            outputs = net(images.to(DEVICE)) # predicts values
            labels = labels.to(DEVICE) # loads the actual values
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
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
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
net = Net().to(DEVICE)
trainloader, testloader = load_data()

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
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
