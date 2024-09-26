## This module stores all the activations used in this Neural Networks package.

## Import necessary packages.
import numpy as np
import cupy as cp

##########################################################################################################################################################################
## Neural Network activation functions created using NumPy.
## Matrix operations are compiled on CPU.
##########################################################################################################################################################################

## A class representing the Sigmoid activation function.
class Sigmoid():
    '''
    A class that contains the Sigmoid activation function as well as its derivative on CPU.

    Attributes:
        None

    Methods:
        activate (self, x): Applies the Sigmoid activation function on input.
        activateGrad (self, x): Takes the derivative of the Sigmoid function on given input.

    Usage:
        Used as an activation function for a neuron in a Neural Network.
    '''
    ## Initializes the Sigmoid activation function on CPU.
    def __init__(self):
        pass

    ## Applies the Sigmoid function to given input.
    def activate(self, x):
        '''
        Activates a neuron by applying Sigmoid function to input.

        Args:
            x (np.array): The input passed through the neuron.

        Returns:
            x_prime (np.array): The output after the activation function.
        '''
        ## Applies Sigmoid activation function.
        x_prime = 1 / (1 + np.exp(-x))
        ## Returns output of Sigmoid function.
        return x_prime
    
    ## Applies the gradient of the Sigmoid function to given input.
    def activateGrad(self, x):
        '''
        Returns the output of the gradient of the Sigmoid function.

        Args:
            x (np.array): The input passed through the neuron during backpropagation.

        Returns:
            x_prime (np.array): The outputs of the gradient of the Sigmoid function.
        '''
        ## The output of the Sigmoid function.
        sig_out = self.activate(x)
        ## The output of the gradient of the Sigmoid function.
        x_prime = sig_out * (1 - sig_out)
        ## Returns the output of the gradient of the Sigmoid function.
        return x_prime

##########################################################################################################################################################################
## Neural Network activation functions created using CuPy.
## Matrix operations are compiled on GPU.
##########################################################################################################################################################################

## A class representing the Sigmoid activation function compiled on GPU.
class Sigmoid_GPU():
    '''
    A class that contains the Sigmoid activation function as well as its derivative on GPU.

    Attributes:
        None

    Methods:
        activate (self, x): Applies the Sigmoid activation function on input.
        activateGrad (self, x): Takes the derivative of the Sigmoid function on given input.

    Usage:
        Used as an activation function for a neuron in a Neural Network compiled on GPU.
    '''
    ## Initializes the Sigmoid activation function on GPU.
    def __init__(self):
        pass

    ## Applies the Sigmoid function to given input using CuPy.
    def activate(self, x):
        '''
        Activates a neuron by applying Sigmoid function to input on GPU.

        Args:
            x (np.array): The input passed through the neuron.

        Returns:
            x_prime (np.array): The output after the activation function.
        '''
        x_prime = 1 / (1 + cp.exp(-x))
        return x_prime
    
    ## Applies the gradient of the Sigmoid function to given input using CuPy.
    def activateGrad(self, x):
        '''
        Returns the output of the gradient of the Sigmoid function compiled on GPU.

        Args:
            x (np.array): The input passed through the neuron during backpropagation.

        Returns:
            x_prime (np.array): The outputs of the gradient of the Sigmoid function.
        '''
        sig_out = self.activate(x)
        x_prime = sig_out * (1 - sig_out)
        return x_prime
