import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class LinearConstraint(nn.Module):
    """
    A linear inequality constraint layer.
    A_np, n_np are numpy ndarrays that define the constraint set A_np.dot(x) <= n_np.

    """
    def __init__(self, b_np):
        super().__init__()

        b_np = np.float32(b_np)
        # dim is the amount of cells(the dimension of power=k)
        dim = b_np.shape[1]
        self.hidden1 = torch.nn.Linear((2*dim * dim), 180)
        # weights and biases are initialized by the Xavier method
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.constant_(self.hidden1.bias, 0)

        self.hidden4 = torch.nn.Linear(180, 90)
        torch.nn.init.xavier_uniform_(self.hidden4.weight)
        torch.nn.init.constant_(self.hidden4.bias, 0)

        self.output = torch.nn.Linear(90, pow(2, dim))
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.constant_(self.output.bias, 0)

    def forward(self, x, z, h):

        """
            input x is the matrix A in (5a), extra input z is the extreme points produced by 'cdd' module
            (double description method), extra input h is the indicator vector in order to tell the true
            number of extreme points.

        """
        x = (x.clone().detach()).float()
        x = x.view(x.shape[0], -1)

        x = f.leaky_relu(self.hidden1(x))
        # print(x.grad)
        x = f.leaky_relu(self.hidden4(x))
        x = f.leaky_relu(self.output(x))

        # map this input to coefficients lambda of the V-parameterization
        x = torch.abs(x)
        ra = torch.sum((x * h), dim=1)
        lamda = torch.divide(x * h, ra.view(-1, 1))
        lamda = lamda.view(x.shape[0], -1, 1)
        # matrix multiplication
        x = lamda * z
        x = torch.sum(x, dim=1)
        return x







