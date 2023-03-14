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
        currentloss = 0.0

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

            # Prints current loss every 500 mini-batches
            

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

# Imports the KSL-KDD dataset
class KSLKDD(Dataset):
    # i need create a custom to be used when importing the dataset
    # this class needs an init, len, and getitem function
    # the init will be used to initialise the directory containing the csv's 
    # len is fairly simple as it simply returns the length of samples (e.g. records/rows)
    # getitem is used to get the data from the csv's depending on the index
    #   it will be used to get a single sample from the dataset to be used in training and testing
    #   it will return informaion about the data (features and class)

    def __init__(self):
        # data loading
        # xy = np.loadtxt('./data/NSL-KDD/KDDTrain+_20Percent.arff', delimiter=',', dtype=np.float32, skiprows=44)
        xy = np.loadtxt('test.csv', delimiter=',', dtype=np.float32, skiprows=1)
        # note: here, you may want to use pandas instead if you are looking to alter specific columns
        # location of data
        # delimiter (csv so comma)
        # data type of dataset (am confused because there are multiple datatypes per sample)
        self.x = torch.from_numpy(xy[:, :-1]).float()
        self.y = torch.from_numpy(xy[:, [-1]]).float()
        # :, all samples
        # :-1 all columns apart from end (features)
        # [-1] last column (class)
        # torch.from_numpy() converts the numpy array to a tensor
        self.n_samples = xy.shape[0]
        # xy.shape[0] first dimention is the number of samples

        # you should define the transform somewhere in here (have done so with torch.from_numpy)

    def __getitem__(self, index):
        # you should apply the transform somewhere in here
        # although isnt it more effeicient to tranform the entire dataset once rather than one at a time?
        # it is, but if you want to change a sample depending on some condition then this is where it happens
        #   e.g. i have some strings which i want to convert to numbers, do that here
        #   again, couldn't i just do that in the init by going through each sample and changing it before
        #   converting it to a tensor?
        #   answer: you could, but this is actually more effeicient because it is performing the same
        #   operation on each sample, like you would do in init, but only for samples you need

        # get a sample
        return (self.x[index], self.y[index]) # must return tensor, numpy, etc.

    def __len__(self):
        # return the length of the dataset
        return self.n_samples

def load_data():
    """Load CIFAR-10 (training and test set)."""
    # trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # normalise is used so that the data isn't all over the place (this can have a problem
    # with deep learning when learning the characteristics of a dataset)

    # train set and dataset are split from the same dataset
    
    # trainset = CIFAR10("./data", train=True, download=True, transform=ToTensor())
    # testset = CIFAR10("./data", train=False, download=True, transform=ToTensor())
    # trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
    # testloader = DataLoader(testset)
    # return trainloader, testloader
    
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

    # Load NSL-KDD dataset
    # About the dataset: https://www.unb.ca/cic/datasets/nsl.html

    dataset = KSLKDD() # creates an instance of the dataset
    dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True, num_workers=2)
    return dataloader
    # loads the data
    # dataset=dataset, the dataset to be loaded
    # batch_size=4, the number to device the samples by (e.g. there will be 4 batches)
    # shuffle=True, shuffles the order of samples
    # num_workers=2, will help speed up the loading
    # if we wanted, we could simply call __getitem__ manually when feeding data to the model
    # but using a datalodader is far easier
    
    # note about batches: if you have multiple gpus, you can split the batch across the gpus
    #  e.g. if you have 4 gpus and a batch size of 4, each gpu will get a batch of 1
    #  so if you ever have multiple gpus, make sure that your batch size is a multiple of the number of gpus

    # dataiter = iter(dataloader) # allows your to iterate through the dataset in batches
    # data = dataiter._next_data() # loads a single batch
    # features, labels = data
    # print(data) # print a single batch (features and labels)
    # print(data[0]) # print the features of the batch (format is [features, labels])
    # # features = data[0] and labels = data[1]
    # print(data[0][1]) # print the second sample of in the batch of features
    # print(data[0][1][2]) # print the third feature of the second sample in the batch of features
    # # data[features][samples][a feature of one of the samples]
    # data = dataiter._next_data()
    # print(data) # prints the next batch
    # # etc...


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Iterate through dataset
data = load_data() # loads the features and samples into data (data will now be a representation of
# test.csv as a tensor)
num_epochs = 2 # the amount of times we want to go over the data
total_samples = len(data) # teh total number of samples
# lem(data) will use the __len__ function in the dataset class to return the number of samples
n_iterations = math.ceil(total_samples / 5) # this is calculated from the total number of samples (which 
# is calculated in the __len__ function in the dataset class) divided by the batch size (which is 4)
# reminder: an iteration is the number of passes per epoch. each pass involves batch_size number of samples
# so if there are 100 samples and the batch size is 4, then there will be 25 iterations (100/4=25)
print(f"Samples: {total_samples}, Iterations: {n_iterations}")

for epoch in range(num_epochs): # iterate over the data [epoch] times
    for i, sample in enumerate(data): # for each sample in the batch...
        # i: index/counter of current sample
        # sample: contains the features and labels of sample (data[features. labels])
        # enumerate(data) returns (counter/index, element) - it simple makes each element of "data"
        # to be iteraterated over using a for loop
        # another way of doing it would be to (everything simplified as much as possible):
        # for epoch in range(num_epochs):
        #     for i in range(total_samples): # for each sample
        #         for sample in data[i]:
        #             features = sample[0]
        #             labels = sample[1]
        #             if (i+1) % 50 == 0:
        #                 print(f"epoch: {epoch+1}/{num_epochs}, step (sample): {i+1}/{total_samples}")
        #             do stuff
        features, labels = sample
        if (i+1) % 50 == 0:
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
