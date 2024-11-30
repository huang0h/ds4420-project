## This module stores the Neural Collaborative Filtering model class.

## Import necessary packages.
import tensorflow as tf

## This class represent the Neural Collaborative Filtering Model.
class NCFModel(tf.keras.Model):
    '''
    A class that represents the Neural Collaborative Filtering model.

    Attributes:
        num_users (int): The number of unique users.
        num_items (int): The number of unique items.
        num_latent (int): The number of latent features for each user and item.

    Methods:
        call (self, input): Passes the inputs through the model's layers.

    Usage:
        This class is made for the neural network collaborative filtering model.
    '''
    ## Initializes the model.
    def __init__(self, num_users, num_items, num_latent):
        '''
        Initializes the embedding layers and the MLP layers.

        Args: 
            num_users (int): The number of unique users.
            num_items (int): The number of unique items.
            num_latent (int): The number of latent features for each user and item.

        Returns:
            None
        '''
        super().__init__()
        ## Create the embeddings for each unique user and item.
        self.embed_users = tf.keras.layers.Embedding(num_users, num_latent)
        self.embed_items = tf.keras.layers.Embedding(num_items, num_latent)

        ## Creates the MLP layers for computing the interaction between the latent
        ## features.
        self.mlp_layer1 = tf.keras.layers.Dense(num_latent * 4, activation = 'leaky_relu')
        self.mlp_layer2 = tf.keras.layers.Dense(num_latent * 2, activation = 'leaky_relu')
        self.mlp_layer3 = tf.keras.layers.Dense(num_latent, activation = 'leaky_relu')

        ## Creates the output layer.
        self.out_layer = tf.keras.layers.Dense(1, activation = 'sigmoid')

    ## Passes the input throughout all the layers of the model.
    def call(self, input):
        ''' 
        Applies the NCF model to the input.

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

        ## Concatenate the embeddings into a 1D vector.
        ncf_input = tf.concat([user_embed, item_embed], axis = -1)

        ## Pass the concatenated vector through all the layers of the NCF model.
        x = self.mlp_layer1(ncf_input)
        x = self.mlp_layer2(x)
        x = self.mlp_layer3(x)
        pred = self.out_layer(x)

        ## Return the predicted output.
        return pred