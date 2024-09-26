## This module stores all the utility functions needed in the neural network package.

## Import necessary packages.
import numpy as np
import cupy as cp

##########################################################################################################################################################################
## Neural Network utility functions created using NumPy.
## Matrix operations are compiled on CPU.
##########################################################################################################################################################################

## Initializes the weights in Fully Connected using Glorot function on CPU.
def Glorot_Uniform(n_in, n_out):
    '''
    Applies the Glorot Uniform Weight Initializer using CPU.

    Args:
        n_in (int): The number of input neurons.
        n_out (int): The number of output neurons.

    Returns:
        ini_weights (np.array): An array containing all the initialized weights.
    '''
    ## Determine the bounds from the number of input and output neurons.
    bound = np.sqrt(6 / (n_in + n_out))
    ## Initialize the weights using Glorot Uniform Initializer.
    ini_weights = np.random.uniform(-bound, bound, (n_in, n_out))

    ## Return initialized weights.
    return ini_weights

##########################################################################################################################################################################
## Neural Network utility functions created using CuPy.
## Matrix operations are compiled on GPU.
##########################################################################################################################################################################

## Initializes the weights in Fully Connected using Glorot function on GPU.
def Glorot_Uniform_GPU(n_in, n_out):
    '''
    Applies the Glorot Uniform Weight Initializer using GPU.

    Args:
        n_in (int): The number of input neurons.
        n_out (int): The number of output neurons.

    Returns:
        ini_weights (cp.array): An array containing all the initialized weights.
    '''
    ## Determine the bounds from the number of input and output neurons.
    bound = cp.sqrt(6 / (n_in + n_out))
    ## Initialize the weights using Glorot Uniform Initializer.
    ini_weights = cp.random.uniform(-bound, bound, (n_in, n_out))

    ## Return initialized weights.
    return ini_weights