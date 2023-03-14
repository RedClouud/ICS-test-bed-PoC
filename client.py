### A half client, half tutorial (for me) ###

# Import the stuff you need
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import numpy as np
import math


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
# telling what processor to use (where it is and where it is processed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_count = torch.cuda.device_count()
print(f"Using {device_count} {device} device(s)")


class MLP(nn.Module):  # This model is specifically tailored for a dataset (CIFAR-10)
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    # Defines neural network layers which are tailored to learn from the dataset
    def __init__(self):
        super().__init__()
        # define layers of the neural network
        self.layers = nn.Sequential( # Sequential causes the data to be fed through the layers in the order written
            nn.Flatten(), # flattens the data (turns it into a 1D array)
            nn.Linear (32 * 32 * 3, 64), # 32 * 32 (x and y of image) * 3 (RGB) = 3072. 64 is the number of neurons in the layer
            nn.ReLU(), # this is a ReLU function which is a way ti actvate neurons
            nn.Linear(64, 32), # this is our "hidden" layer. no idea, dont need to have an idea. just look at what goes
            nn.ReLU(), # do some more thinking/learing
            nn.Linear(32, 10) # this is our output layer
        )

    # defines how it "progresses" (what parameters are going to change and how)
    def forward(self, x):
        return self.layers(x) # how we react to input data.


def train(net, trainloader, epochs):  # trains the model to classify the data
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()  # loss calculation function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):  # one epoch is one pass through the dataset
        currentloss = 0.0

        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()  # <-- this is where the learning happens!
            

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        # goes through the test set and compares the predicted values to the actual values
        for images, labels in tqdm(testloader):
            outputs = net(images.to(device))  # predicts values
            labels = labels.to(device)  # loads the actual values
            loss += criterion(outputs, labels).item()# calculates the difference betweenthe predicted and actual values (how correct the model is)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

# Imports the KSL-KDD dataset
class KSLKDD(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('test.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, :-1]).float()
        self.y = torch.from_numpy(xy[:, [-1]]).float()
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # get a sample
        return (self.x[index], self.y[index])

    def __len__(self):
        # return the length of the dataset
        return self.n_samples

def load_data():
    """Load CIFAR-10 (training and test set)."""
    # Load NSL-KDD dataset
    # About the dataset: https://www.unb.ca/cic/datasets/nsl.html

    dataset = KSLKDD() # creates an instance of the dataset
    dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True, num_workers=2)
    return dataloader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Iterate through dataset
data = load_data() # loads the dataset
num_epochs = 2 # the amount of times we want to go over the data
total_samples = len(data) # the total number of samples
n_iterations = math.ceil(total_samples / 5) # this is calculated from the total number of samples (which 

print(f"Samples: {total_samples}, Iterations: {n_iterations}")

for epoch in range(num_epochs): # iterate over the data [epoch] times
    for i, sample in enumerate(data): # for each sample in the batch...
        features, labels = sample
        if (i+1) % 50 == 0: # print the progress every 50 iterations
            print(f"epoch: {epoch+1}/{num_epochs}, iteration: {(i+1)/5}/{n_iterations}, features: {features.shape}")

# Load model and data (simple CNN, CIFAR-10)
net = MLP().to(device)
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
    server_address="172.18.0.2:8080",
    client=FlowerClient(),
)
