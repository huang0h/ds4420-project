## This module stores all the loss functions for neural network models.

## Import necessary packages.
import numpy as np
import cupy as cp

##########################################################################################################################################################################
## Neural Network loss functions created using NumPy.
## Matrix operations are compiled on CPU.
##########################################################################################################################################################################

## A class that represents the Mean Absolute Error compiled on the CPU.
class MAE():
    '''
    A class that computes the Mean Absolute Error between the true value and the predicted value.

    Attributes:
        None

    Methods:
        calculateLoss (self, actual_value, predictions): Calculates the error between the true value and the model's predictions.
        calculateLossGrad (self, actual_value, predictions): Calculates the gradient of the error between the true value and the model's predictions.

    Usage:
        A method to calculate the loss between the model and the actual values.
    '''
    ## Initializes the MAE function on the CPU.
    def __init__(self):
        '''
        Initializes the parameters for MAE function.

        Args:
            None

        Returns:
            None
        '''
        pass

    ## Calculates the loss between the true value and the model's predictions on CPU.
    def calculateLoss(self, actual_value, predictions):
        '''
        Returns the average loss between the actual values and the model's predictions.

        Args:
            actual_value (np.array): An array containing all the true values.
            predictions (np.array): An array containing all the predictions from the model.

        Returns:
            mae (np.array): An array storing all the Mean Absolute Errors for each batch.
        '''
        ## Calculate the Mean Absolute Error between the actual value and the model's predictions.
        mae = np.mean(np.abs(actual_value - predictions), axis = 1)
        ## Return the MAE value.
        return mae
    
    ## Computes the gradient of the MAE loss function.
    def calculateLossGrad(self, actual_value, predictions):
        '''
        Returns the gradient of the loss function compiled on CPU.

        Args:
            actual_value (np.array): An array containing all the true values.
            predictions (np.array): An array containing all the predictions from the model.

        Returns:
            nabla_mae (np.array): An array storing all the gradient error values for each batch.
        '''
        ## Find the number of elements in the actual value vector.
        num_elements = actual_value.shape[0]
        ## Calculate the gradient of the MAE function.
        nabla_mae = (-1 / num_elements) * (np.sign(actual_value - predictions))
        ## Return the gradient of MAE.
        return nabla_mae
    
##########################################################################################################################################################################
## Neural Network loss functions created using CuPy.
## Matrix operations are compiled on GPU.
##########################################################################################################################################################################

## A class that represents the Mean Absolute Error compiled on the GPU.
class MAE_GPU():
    '''
    A class that computes the Mean Absolute Error between the true value and the predicted value on GPU.

    Attributes:
        None

    Methods:
        calculateLoss (self, actual_value, predictions): Calculates the error between the true value and the model's predictions on GPU.
        calculateLossGrad (self, actual_value, predictions): Calculates the gradient of the error between the true value and the model's predictions on GPU.

    Usage:
        A method to calculate the loss between the model and the actual values on GPU.
    '''
    ## Initializes the MAE function on the GPU.
    def __init__(self):
        '''
        Initializes the parameters for MAE function on GPU.

        Args:
            None

        Returns:
            None
        '''
        pass

    ## Calculates the loss between the true value and the model's predictions on GPU.
    def calculateLoss(self, actual_value, predictions):
        '''
        Returns the average loss between the actual values and the model's predictions on GPU.

        Args:
            actual_value (cp.array): An array containing all the true values.
            predictions (cp.array): An array containing all the predictions from the model.

        Returns:
            mae (cp.array): An array storing all the Mean Absolute Errors for each batch.
        '''
        ## Calculate the Mean Absolute Error between the actual value and the model's predictions.
        mae = cp.mean(cp.abs(actual_value - predictions), axis = 1)
        ## Return the MAE value.
        return mae
    
    ## Computes the gradient of the MAE loss function on GPU.
    def calculateLossGrad(self, actual_value, predictions):
        '''
        Returns the gradient of the loss function compiled on GPU.

        Args:
            actual_value (cp.array): An array containing all the true values.
            predictions (cp.array): An array containing all the predictions from the model.

        Returns:
            nabla_mae (cp.array): An array storing all the gradient error values for each batch.
        '''
        ## Find the number of elements in the actual value vector.
        num_elements = actual_value.shape[0]
        ## Calculate the gradient of the MAE function.
        nabla_mae = (-1 / num_elements) * (cp.sign(actual_value - predictions))
        ## Return the gradient of MAE.
        return nabla_mae
