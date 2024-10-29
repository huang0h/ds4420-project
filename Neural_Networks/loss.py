## This module stores all the loss functions for neural network models.

## Import necessary packages.
import numpy as np
import cupy as cp

##########################################################################################################################################################################
## Neural Network loss functions created using NumPy.
## Matrix operations are compiled on CPU.
##########################################################################################################################################################################

## A class that represents the Mean Squared Error compiled on the CPU.
class MSE():
    '''
    A class that computes the Mean Squared Error between the true value and the predicted value.

    Attributes:
        None

    Methods:
        calculateLoss (self, actual_value, predictions): Calculates the error between the true value and the model's predictions.
        calculateLossGrad (self, actual_value, predictions): Calculates the gradient of the error between the true value and the model's predictions.

    Usage:
        A method to calculate the loss between the model and the actual values.
    '''
    ## Initializes the MSE function on the CPU.
    def __init__(self):
        '''
        Initializes the parameters for MSE function.

        Args:
            None

        Returns:
            None
        '''
        pass

    ## Calculates the loss between the true value and the model's predictions using MSE on CPU.
    def calculateLoss(self, actual_value, predictions):
        '''
        Returns the average MSE loss between the actual values and the model's predictions.

        Args:
            actual_value (np.array): An array containing all the true values.
            predictions (np.array): An array containing all the predictions from the model.

        Returns:
            mse (np.array): An array storing all the Mean Squared Errors for each batch.
        '''
        ## Calculate the Mean Squared Error between the actual value and the model's predictions.
        mse_batch = np.mean((actual_value - predictions)**2, axis = 1)
        mse = np.mean(mse_batch)
        ## Return the MSE value.
        return mse
    
    ## Computes the gradient of the MSE loss function.
    def calculateLossGrad(self, actual_value, predictions):
        '''
        Returns the gradient of the MSE loss function compiled on CPU.

        Args:
            actual_value (np.array): An array containing all the true values.
            predictions (np.array): An array containing all the predictions from the model.
            
        Returns:
            delta (np.array): An array storing all the gradient error values for each batch w.r.t. the activation function.
        '''
        ## Find the number of elements in the actual value vector.
        num_elements = actual_value.shape[1]
        ## Calculate the gradient of the MAE function.
        nabla_mse = (2 / num_elements) * (predictions - actual_value)

        ## Return the gradient of MSE w.r.t. the activation function of the output layer.
        return nabla_mse

## A class that represents the Binary Cross Entropy Loss Function compiled on CPU.
class Binary_Cross_Entropy():
    '''
    A class that computes the Binary Cross Entropy Loss between the true value and the predicted value.

    Attributes:
        None

    Methods:
        calculateLoss (self, actual_value, predictions): Calculates the error between the true value and the model's predictions.
        calculateLossGrad (self, actual_value, predictions): Calculates the gradient of the error between the true value and the model's predictions.

    Usage:
        A method to calculate the loss between the model and the actual values.
    '''
    ## Initializes the BCE function on the CPU.
    def __init__(self):
        '''
        Initializes the parameters for BCE function.

        Args:
            None

        Returns:
            None
        '''
        pass

    ## Calculates the loss function between the true value and the model's predictions using BCE compiled on CPU.
    def calculateLoss(self, actual_value, predictions):
        '''
        Returns the average BCE loss function compiled on CPU.

        Args:
            actual_value (np.array): An array containing all the true values.
            predictions (np.array): An array containing all the predictions from the model.

        Returns:
            bce (np.array): An array storing all the Binary Cross Entropy for each batch.
        '''
        ## Adjusted predictions such that no values are 0 (has issues with log).
        pred_adj = predictions + 1e-100
        ## Calculate the BCE loss on each batch.
        bce_batch = -1 * np.mean((actual_value * np.log(pred_adj)) + ((1 - actual_value) * np.log(1 - pred_adj)), axis = 1)
        ## Averages the BCE loss over batches.
        bce = np.mean(bce_batch)
        ## Returns average BCE loss.
        return bce
    
    ## Computes the gradient of the BCE loss function.
    def calculateLossGrad(self, actual_value, predictions):
        '''
        Returns the gradient of the BCE loss function compiled on CPU.

        Args:
            actual_value (np.array): An array containing all the true values.
            predictions (np.array): An array containing all the predictions from the model.
            
        Returns:
            nabla_bce (np.array): An array storing all the gradient error values for each batch w.r.t. the activation function.
        '''
        ## Calculates the number of classes in the output.
        num_elements = actual_value.shape[1]
        ## Adjusts the predictions so that it's not 0 (has issue with log).
        pred_adj = predictions + 1e-100
        ## Computes the gradient of the BCE loss function w.r.t. the activation function.
        nabla_bce = (-1 / num_elements) * ((actual_value / pred_adj) - ((1 - actual_value) / (1 - pred_adj)))
        ## Returns the gradient of the BCE loss function.
        return nabla_bce
        
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
