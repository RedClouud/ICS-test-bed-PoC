import torch
import numpy as np



def tensors(data):

    x_data = torch.tensor(data)

    np_array = np.array(data)
    np_data = torch.tensor(np_array)

    x_ones = torch.ones_like(x_data) # retains the properties of x_data
    x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data


    print(x_data)
    print(np_data)
    print(x_ones)
    print(x_rand)
    print("--------------------")

    return 0

data = [[1, 2], [3, 4]]
tensors(data)

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
tensors(data)
