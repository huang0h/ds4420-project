## This module stores all the optimizers used for the Neural Networks pacakge.

## Import the necessary packages.
import numpy as np
import cupy as cp

##########################################################################################################################################################################
## Neural Network optimzer functions created using NumPy.
## Matrix operations are compiled on CPU.
##########################################################################################################################################################################

## A class that represents the Stochastic Gradient Descent optimzer.
class SGD():
    '''
    A class that contains the SGD optimzer used for training the model.

    Attributes:
        activation (activations): The activation function used in the fully connected layer.
        lr (float): The learning rate for optimizer.

    Methods:
        optimzeGrad (self, x, input, weights, bias, out_delta): Performs the backward pass using SGD optmization method.

    Usage:
        Used to train a fully connected layer through Stochastic Gradient Descent.
    '''
    ## Initializes the SGD optimizer on CPU.
    def __init__(self, activation, lr):
        '''
        Initializes the parameters for SGD optimzer.

        Args:
            activation (activations): The activation used on the fully connected layers.
            lr (float): The learning rate of the optimzer.

        Returns:
            None
        '''
        self.activation = activation
        self.lr = lr

    ## Performs the optimization process.
    def optimzeGrad(self, x, input, weights, bias, out_delta):
        '''
        Performs SGD optimization, updating the weights and biases as well as find the error for the previous layer.

        Args:
            x (np.array): The output from this layer.
            input (np.array): The input for this layer.
            weights (np.array): The array representing the weights.
            bias (np.array): The array representing the biases.
            out_delta (np.array): The array representing the gradient of the succeeding layer.

        Returns:
            in_delta (np.array): An array representing the gradient of this layer.
            w_prime (np.array): The updated weights.
            b_prime (np.array): The updated biases.
        '''
        ## The batch size used in this layer.
        batch_size = input.shape[0]

        ## Find the gradient of the activation function and multiply with error of the succeeding layer.
        nabla_activation = self.activation.activateGrad(x)
        delta = out_delta * nabla_activation

        ## Calucate the average change in weights over all the batches.
        nabla_w = input.T.dot(delta)
        nabla_w = nabla_w / batch_size

        ## Find the average chaneg in biases over all the batches.
        nabla_b = np.sum(delta, axis = 0) / batch_size

        ## Compute the updated weights and biases.
        w_prime = weights - (self.lr * nabla_w)
        b_prime = bias - (self.lr * nabla_b)

        ## Calculate the error of this layer.
        in_delta = delta.dot(weights.T)

        ## Return the error of this layer, updated weights, and updated biases.
        return in_delta, w_prime, b_prime
    
##########################################################################################################################################################################
## Neural Network optimzer functions created using CuPy.
## Matrix operations are compiled on GPU.
##########################################################################################################################################################################

## A class that represents the Stochastic Gradient Descent optimzer compiled on GPU.
class SGD_GPU():
    '''
    A class that contains the SGD optimzer used for training the model compiled on GPU.

    Attributes:
        activation (activations): The activation function used in the fully connected layer.
        lr (float): The learning rate for optimizer.

    Methods:
        optimzeGrad (self, x, input, weights, bias, out_delta): Performs the backward pass using SGD optmization method.

    Usage:
        Used to train a fully connected layer through Stochastic Gradient Descent on GPU.
    '''
    ## Initializes the SGD optimizer on GPU.
    def __init__(self, activation, lr):
        '''
        Initializes the parameters for SGD optimzer.

        Args:
            activation (activations): The activation used on the fully connected layers.
            lr (float): The learning rate of the optimzer.

        Returns:
            None
        '''
        self.activation = activation
        self.lr = lr

    ## Performs the optimization process on GPU.
    def optimzeGrad(self, x, input, weights, bias, out_delta):
        '''
        Performs SGD optimization, updating the weights and biases as well as find the error for the previous layer. Compiles on GPU.

        Args:
            x (cp.array): The output from this layer.
            input (cp.array): The input for this layer.
            weights (cp.array): The array representing the weights.
            bias (cp.array): The array representing the biases.
            out_delta (cp.array): The array representing the gradient of the succeeding layer.

        Returns:
            in_delta (cp.array): An array representing the gradient of this layer.
            w_prime (cp.array): The updated weights.
            b_prime (cp.array): The updated biases.
        '''
        ## The batch size used in this layer.
        batch_size = input.shape[0]

        ## Find the gradient of the activation function and multiply with error of the succeeding layer.
        nabla_activation = self.activation.activateGrad(x)
        delta = out_delta * nabla_activation

        ## Calucate the average change in weights over all the batches.
        nabla_w = input.T.dot(delta)
        nabla_w = nabla_w / batch_size

        ## Find the average chaneg in biases over all the batches.
        nabla_b = cp.sum(delta, axis = 0) / batch_size

        ## Compute the updated weights and biases.
        w_prime = weights - (self.lr * nabla_w)
        b_prime = bias - (self.lr * nabla_b)

        ## Calculate the error of this layer.
        in_delta = delta.dot(weights.T)

        ## Return the error of this layer, updated weights, and updated biases.
        return in_delta, w_prime, b_prime
