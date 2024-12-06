## This module stores the General Matrix Factorization model class.

## Import necessary packages.
import tensorflow as tf

## This class represent the General Matrix Factorization Model.
class GMFModel(tf.keras.Model):
    '''
    A class that represents the Matrix Factorization Collaborative Filtering model.

    Attributes:
        num_users (int): The number of unique users.
        num_items (int): The number of unique items.
        num_latent (int): The number of latent features for each user and item.

    Methods:
        call (self, input): Passes the inputs through the model's layers.

    Usage:
        This class is made for the most standard collaborative filtering model.
    '''
    ## Initializes the model.
    def __init__(self, num_users, num_items, num_latent):
        '''
        Initializes the embedding layers as well as the output function.

        Args: 
            num_users (int): The number of unique users.
            num_items (int): The number of unique items.
            num_latent (int): The number of latent features for each user and item.

        Returns:
            None
        '''
        super().__init__()
        ## Create an embedding layer for user-features.
        self.embed_users = tf.keras.layers.Embedding(num_users, num_latent)
        ## Create an embedding layer for item-features.
        self.embed_items = tf.keras.layers.Embedding(num_items, num_latent)
        ## Create output layer.
        self.calc_out = tf.keras.layers.Dense(1, activation = 'sigmoid')

    ## Passes the input through the model.
    def call(self, input):
        ''' 
        Applies the GMF model to the input.

        Args:
            input (np.array): An array of tuples containing user-item interactions.

        Returns:
            pred (np.array): The predicted interaction score.
        '''
        ## Extract the user and item variables from the input.
        users, items = input

        ## Create the latent features for each unique user and item.
        user_embed = self.embed_users(users)
        item_embed = self.embed_items(items)

        ## Apply the dot product between the two embedding layers for the 
        ## entire batch.
        interaction = tf.reduce_sum(user_embed * item_embed, axis=1) 
        interaction = tf.expand_dims(interaction, axis=-1)

        ## Calculate the predicted interaction score using sigmoid activation
        ## function.
        pred = self.calc_out(interaction)

        ## Return the predicted interaction score.
        return pred
