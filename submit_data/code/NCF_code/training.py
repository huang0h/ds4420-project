## This module stores all the functions that are necessary for training the content-based models.

## Import necessary packages.
import pandas as pd
import random
from tqdm import tqdm

## A class representing a Negative Sampler.
class NegativeSampler:
    '''
    Randomly chooses a certain number of negative interactions of users and items.

    Attributes:
        user (int): An integer that represents the ID of a given user.
        n_samples (int): The number of negative samples to choose.
        user_positive_item_pool (Set): A set of all positive interactions for each unique user.
        item_pool (Set): A set of all unique item IDs.

    Methods:
        sample (self): A function that returns a random list of items the user had negative interactions with.

    Usage:
        Used for randomly generating negative samples for training. Is able to handle sparse matrix data
        by avoiding bias.
    '''
    ## Initializes the NegativeSampler class.
    def __init__(self, user, n_samples, user_positive_item_pool, item_pool):
        '''
        Initializes the class by the given parameters.

        Args:
            user (int): An integer that represents the ID of a given user.
            n_samples (int): The number of negative samples to choose.
            user_positive_item_pool (Set): A set of all positive interactions for each unique user.
            item_pool (Set): A set of all unique item IDs.

        Returns:
            None
        '''
        ## Assigns the user to a private variable.
        self.user = user
        ## Assigns the number of negative samples as private variable.
        self.n_samples = n_samples
        ## Stores the positive interaction set.
        self.user_positive_item_pool = set(user_positive_item_pool)
        ## Stores all the unique item IDs.
        self.item_pool = set(item_pool)
        ## Finds the negative item IDs by subtracting the positive item IDs from the 
        ## total item IDs.
        self.user_negative_item_pool = self.item_pool - self.user_positive_item_pool

    ## Using the user_negative_item_pool, randomly selects n number of negative samples.
    def sample(self):
        '''
        Randomly chooses n number of items that the user had negative interactions with.

        Args:
            None

        Returns:
            (list): A list of randomly chosen negatively interacted item IDs.
        '''
        ## If the number of negative samples is greater than the length of the negative ID pool, 
        ## return the all the negative samples.
        if len(self.user_negative_item_pool) <= self.n_samples:
            return list(self.user_negative_item_pool)
        ## If the number of negative samples is less than the length of the negative ID pool, 
        ## randomly select the item IDs. 
        else:
            return random.sample(self.user_negative_item_pool, k = self.n_samples) 

## Randomly selects negative samples per training epoch.  
def GenerateTrainingData(train_df, user_positive_itemsets, item_pool, epochs, n_neg_multiplier = 4):
    '''
    For each epoch, randomly select negatively interacted items for each user.

    Args:
        train_df (DataFrame): The training DataFrame containing all the interactions of users and items.
        user_positive_itemsets (Set): A set containing all the positive interactions for each user.
        item_pool (Set): A set containing all the unique item IDs.
        epochs (int): The number of training epochs.
        n_neg_multiplier (int): The multiple of negative IDs to choose. If there were 5 positive interactions 
                                for a given user, choose 20 negative samples.

    Returns:
        inputs (np.array): An array containing the user ID and item ID.
        labels (np.array): An array storing the actual interaction score of the user and item combination.
    '''
    ## For each given epoch, randomly sample negative samples from set.
    for epoch in range(epochs):
        shuffle_buffer = []

        ## Iterate through users and generate training samples.
        for user in tqdm(train_df['user_id'].unique(), desc = "Generating Negative Samples"):
            user_positive_item_pool = user_positive_itemsets[user]
            
            ## Generate negative samples for each user.
            n_samples = n_neg_multiplier * len(user_positive_item_pool)
            sampler = NegativeSampler(user, n_samples, user_positive_item_pool, item_pool)
            negative_samples = sampler.sample()
            
            ## Combine positive and negative samples. 
            combined_samples = [(user, pos_item, 1) for pos_item in user_positive_item_pool] 
            combined_samples += [(user, neg_item, 0) for neg_item in negative_samples]

            shuffle_buffer.extend(combined_samples)

        ## Shuffle the buffer and return the training data for this epoch
        random.shuffle(shuffle_buffer)
        epoch_data = pd.DataFrame(shuffle_buffer, columns=['user_id', 'song_id', 'liked'])
        
        # Convert the dataset into inputs arrays and labels arrays.
        inputs = epoch_data[['user_id', 'song_id']].to_numpy()
        labels = epoch_data['liked'].to_numpy()

        ## Yield the inputs and labels so that they are only returned when called upon.
        yield inputs, labels