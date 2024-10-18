## This module stores all the evaluation metrics for neural networks.

## Import necessary packages.
import numpy as np
import cupy as cp

##########################################################################################################################################################################
## Neural Network metric functions created using NumPy.
## Matrix operations are compiled on CPU.
##########################################################################################################################################################################

## A class that represents the Categorical Accuracy compiled on the CPU.
class Categorical_Accuracy:
    '''
    A class that computes the Categorical Accuracy between the true value and the predicted value. Counts how many
    predictions were correct over the total observations.

    Attributes:
        None

    Methods:
        calculateEqualBatch(self, actual_value, predictions): Calculates the number of right predictions for each batch.
        calculateAccuracy(self, tot_right, tot_obs): Finds the ratio between the number of right predictions and total
                                                     observations.

    Usage:
        A method to evaluate the performance of a model.
    '''
    ## Initializes the Categorical Accuracy function on the CPU.
    def __init__(self):
        '''
        Initializes the parameters for Categorical Accuracy function.

        Args:
            None

        Returns:
            None
        '''
        pass

    ## Calculates the number of right predictions for a batch.
    def calculateEqualBatch(self, actual_value, predictions):
        '''
        Finds the total number of right predictions for one batch.

        Args:
            actual_value (np.array): An array that stores the true labels, hot-encoded.
            predictions (np.array): An array that stores the predictions from the model.

        Returns:
            batch_acc (int): The total number of right predictions for this batch.
        '''
        ## Finds the index of the maximum value for the true labels and the predictions.
        ## The maximum value would denote the class.
        true_indices = np.argmax(actual_value, axis = 1)
        pred_indices = np.argmax(predictions, axis = 1)

        ## Counts all the number of indices in the predictions array that are the same 
        ## as the indices of the true labels array.
        batch_acc = np.sum(true_indices == pred_indices)

        ## Returns the total number of correct predictions.
        return batch_acc

    ## Returns the ratio of total right predictions over the total observations.
    def calculateAccuracy(self, tot_right, tot_obs):
        '''
        Finds the ratio between the total number of right predictions over total 
        dataset.

        Args:
            tot_right (int): The total number of all correct predictions.
            tot_obs (int): The total number of elements in a dataset.

        Returns:
            tot_acc (float): The ratio between the total right predictions and total observations.
        '''
        ## Calculates the ratio between correct predictions and total observations.
        tot_acc = tot_right / tot_obs

        ## Returns the final accuracy ratio.
        return tot_acc
