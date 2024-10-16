## This module stores all the layers for neural network models.

## Import necessary packages.
import numpy as np
import cupy as cp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Neural_Networks.utils as utils

##########################################################################################################################################################################
## Neural Network layers created using NumPy.
## Matrix operations are compiled on CPU.
##########################################################################################################################################################################

## A class representing the fully connected layer compiled on CPU.
class Fully_Connected():
    '''
    A class that acts as the fully connected layer with forward and backward propagation on CPU.

    Attributes:
        batch_size (int): The number of batches in the input.
        in_neurons (int): An integer representing the shape of the input.
        out_neurons (int): An integer representing the shape of the output.
        activation (activations): The activation function use for each neuron.
        test_weights (np.array): Custom weights that the user can assign for testing.
        test_bias (np.array): Custom biases that the user can assign for testing.

    Methods:
        forward_propagate (self, input): Performs a forward propagation/pass given an input. 
        backward_propagate (self, out_delta): Performs a backward propagation/pass given the 
                                              error of the succeeding layer.

    Usage:
        Used for building the dense portion of a neural network.
    '''
    ## Initializes the Fully Connected layer instance.
    def __init__(self, batch_size, in_neurons, out_neurons, activation, test_weights = None, test_bias = None):
        '''
        Initializes the fully connected layer on CPU.

        Args:
            batch_size (int): The number of batches in the input.
            in_neurons (int): An integer representing the number of input neurons.
            out_neurons (int): An integer representing the number of output neurons.
            activation (activations): The activation function use for each neuron.
            test_weights (np.array): Custom weights that the user can assign for testing.
            test_bias (np.array): Custom biases that the user can assign for testing.

        Returns:
            None
        '''
        ## Initializes the input and output shape, as well as the learning rate.
        self.inputs = None
        self.in_shape = (batch_size, in_neurons)
        self.out_shape = out_neurons

        ## Assigns the activation function.
        self.activation = activation

        ## Initializes the outputs for this layer.
        self.x = np.zeros(shape = (self.in_shape[0], self.out_shape))
        
        ## If user inputs test weights and biases, assigns them as private variables.
        if test_weights is None and test_bias is None:
            self.weights = None
            self.bias = None 
        else:
            self.weights = test_weights
            self.bias = test_bias

    ## Performs the forward propagation/pass in a fully connected layer on CPU.
    def forward_propagate(self, input):
        '''
        Executes the forward pass for this layer using CPU.

        Args:
            input (np.array): An array containing features.

        Returns:
            self.x (np.array): The output from this layer.
        '''
        ## Assigns the given input to private variable.
        self.input = input

        ## If weights is None, initialize it using Glorot Uniform Weight Initializer.
        if self.weights is None:
            ## The number of features in the input.
            n_in = self.in_shape[1]
            ## The number of features in the output.
            n_out = self.out_shape
            ## Initialized weights using Glorot Uniform Weight Initializer.
            self.weights = utils.Glorot_Uniform(n_in, n_out)
        
        ## If the bias is None, initialize it by assigning an array of zeros.
        if self.bias is None:
            self.bias = np.zeros(shape = self.out_shape)

        ## Multipy all the input features to their respective weights and add
        ## the bias.
        self.x = self.input.dot(self.weights) + self.bias

        ## Use the activation function to get the final outputs for this layer.
        output = self.activation.activate(self.x)

        ## Return the outputs for this layer.
        return output
    
    ## Performs the backward propagation/pass in a fully connected layer on CPU
    def backward_propagate(self, out_delta, optimizer, lr):
        '''
        Executes the backward pass for this layer using CPU.

        Args:
            out_delta (np.array): The dot product of the gradient of the succeeding layer and the weights between this layer and the preceding layer.
            optimizer (optimizers): The optimization function used for training.
            lr (float): The learning rate used for training.

        Returns:
            in_delta (np.array): The dot product of the gradient of this layer and the weights between the previous layer and this layer.
        '''
        ## Perform backward propagation using defined optimizer function.
        in_delta, w_prime, b_prime = optimizer.optimizeGrad(x = self.x, 
                                                            input = self.input, 
                                                            weights = self.weights, 
                                                            bias = self.bias,
                                                            activation = self.activation,
                                                            out_delta = out_delta, 
                                                            lr = lr)

        ## Update current weights and biases.
        self.weights = w_prime
        self.bias = b_prime

        ## Return the error for this layer.
        return in_delta
    
##########################################################################################################################################################################
## Neural Network layers created using CuPy.
## Matrix operations are compiled on GPU.
##########################################################################################################################################################################

## A class representing the fully connected layer compiled on GPU.
class Fully_Connected_GPU():
    '''
    A class that acts as the fully connected layer with forward and backward propagation compiled on GPU.

    Attributes:
        batch_size (int): The number of batches in the input.
        in_neurons (int): An integer representing the shape of the input.
        out_neurons (int): An integer representing the shape of the output.
        activation (activations): The activation function use for each neuron.
        optimizer (optimizers): The optimization function used for training.
        lr (float): The learning rate used for training.
        test_weights (cp.array): Custom weights that the user can assign for testing.
        test_bias (cp.array): Custom biases that the user can assign for testing.

    Methods:
        forward_propagate (self, input): Performs a forward propagation/pass given an input. 
        backward_propagate (self, out_delta): Performs a backward propagation/pass given the 
                                              error of the succeeding layer.

    Usage:
        Used for building the dense portion of a neural network.
    '''
    ## Initializes the Fully Connected layer instance on GPU.
    def __init__(self, batch_size, in_neurons, out_neurons, activation, optimizer, lr, test_weights = None, test_bias = None):
        '''
        Initializes the fully connected layer run on GPU.

        Args:
            batch_size (int): The number of batches in the input.
            in_neurons (int): An integer representing the number of input neurons.
            out_neurons (int): An integer representing the number of output neurons.
            activation (activations): The activation function use for each neuron.
            optimizer (optimizers): The optimization function used for training.
            lr (float): The learning rate used for training.
            test_weights (cp.array): Custom weights that the user can assign for testing.
            test_bias (cp.array): Custom biases that the user can assign for testing.

        Returns:
            None
        '''
        ## Initializes the input and output shape, as well as the learning rate.
        self.inputs = None
        self.in_shape = (batch_size, in_neurons)
        self.out_shape = out_neurons
        self.lr = lr

        ## Assigns the activation function.
        self.activation = activation

        ## Initializes the optimizer function.
        self.optimizer = optimizer(self.activation, self.lr)

        ## Initializes the outputs for this layer on the GPU.
        self.x = cp.zeros(shape = (self.in_shape[0], self.out_shape))
        
        ## If user inputs test weights and biases, assigns them as private variables.
        if test_weights is None and test_bias is None:
            self.weights = None
            self.bias = None 
        ## If not, assign None to self.weights and self.bias.
        else:
            self.weights = test_weights
            self.bias = test_bias

    ## Performs the forward propagation/pass in a fully connected layer on GPU.
    def forward_propagate(self, input):
        '''
        Executes the forward pass for this layer using GPU.

        Args:
            input (cp.array): An array containing features.

        Returns:
            self.x (cp.array): The output from this layer.
        '''
        ## Assigns the given input to private variable.
        self.input = input

        ## If weights is None, initialize it using Glorot Uniform Weight Initializer.
        if self.weights is None:
            ## The number of features in the input.
            n_in = self.in_shape[1]
            ## The number of features in the output.
            n_out = self.out_shape
            ## Initialized weights using Glorot Uniform Weight Initializer.
            self.weights = utils.Glorot_Uniform_GPU(n_in, n_out)
        
        ## If the bias is None, initialize it by assigning an array of zeros.
        if self.bias is None:
            self.bias = cp.zeros(shape = self.out_shape)

        ## Multipy all the input features to their respective weights and add
        ## the bias.
        self.x = self.input.dot(self.weights) + self.bias

        ## Use the activation function to get the final outputs for this layer.
        output = self.activation.activate(self.x)

        ## Return the outputs for this layer.
        return output
    
    ## Performs the backward propagation/pass in a fully connected layer on GPU
    def backward_propagate(self, out_delta):
        '''
        Executes the backward pass for this layer using GPU.

        Args:
            out_delta (cp.array): The dot product of the gradient of the succeeding layer and the weights between this layer and the preceding layer.

        Returns:
            in_delta (cp.array): The dot product of the gradient of this layer and the weights between the previous layer and this layer.
        '''
        ## Perform backward propagation using defined optimizer function.
        in_delta, w_prime, b_prime = self.optimizer.optimizeGrad(self.x, self.input, self.weights, self.bias, out_delta)

        ## Update current weights and biases.
        self.weights = w_prime
        self.bias = b_prime

        ## Return the error for this layer.
        return in_delta
