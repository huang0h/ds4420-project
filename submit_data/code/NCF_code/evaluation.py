## This module stores all the evaluation functions for measuring the performance of a trained model.

## Import necessary packages.
import tensorflow as tf
import numpy as np
from tqdm import tqdm

## Generate random negative samples given the interactions in a test data.
def GenerateNegativeSamples(test_data, user_positive_itemsets, item_pool, n_negatives):
    """
    Generate negative samples for each user in the test data.

    Args:
        test_data (List): List of tuples representing user and item interactions.
        item_pool (List): List of all possible items.
        n_negatives (int): Number of negative samples per positive interaction.

    Returns:
        negative_samples (Dictionary): A Dictionary containing all the negative interactions for a given 
                                       user.
    """
    ## Initialize the negative samples dictionary.
    negative_samples = {}
    
    with tf.device('/GPU:0'):
        ## For each user in the test data, select random negative interactions.
        for user, test_item in tqdm(test_data, desc = f"Generating {n_negatives} Negative Test Samples"):
            train_item = user_positive_itemsets[user]
            neg_items = list(np.random.choice(
                list(set(item_pool) - {test_item} - train_item),
                size = n_negatives,
                replace = False
            ))
            negative_samples[user] = neg_items

    ## Return the dictionary containing the negative interactions of each test user.
    return negative_samples

## Computes the Hit Ratio and NDCG for a given model at certain parameters.
def EvaluateModel(model, test_data, user_positive_itemsets, item_pool, num_negatives, top_k):
    """
    Evaluate the Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) for the test data.

    Args:
        model (keras.Model): Trained collaborative filtering models.
        test_data (List): List of tuples representing all user-item interactions.
        user_positive_itemsets (Dictionary): A Dictionary storing all the positive interactions between user and item.
        item_pool (List): List of all unique item IDs in the dataset.
        top_k: Number of top recommendations to consider for HR and NDCG.

    Returns:
        hit_ratio (float): The average probability that an item the user likes is within the top k recommendation list.
        avg_ndcg (float): The average NDCG score for all the items in the test data.
    """
    ## Obtain the randomly generated negative samples for each user in the test data.
    negative_samples = GenerateNegativeSamples(test_data, user_positive_itemsets, item_pool, num_negatives)

    ## Initialize the number of hits.
    hits = 0
    total_users = len(test_data)

    ## Initialize the total NDCG.
    total_ndcg = 0
    total_users = len(test_data)

    with tf.device('/GPU:0'):
        ## For each user in the test data, evaluate the following metrics.
        for user, test_item in tqdm(test_data, desc="Applying Model"):
            neg_items = negative_samples[user]
            
            items_to_score = [test_item] + neg_items
            users = [user] * len(items_to_score)
            
            ## Use the model to predict the interaction score given an user and an item.
            scores = model.predict([np.array(users), np.array(items_to_score)], batch_size = 64, verbose = 0)
            
            ## Rank items based on predicted scores.
            ranked_indices = np.argsort(-scores.flatten())
            
            ## Check if the liked item is within the top K recommendation.
            is_hit = 1 if 0 in ranked_indices[:top_k] else 0
            hits += is_hit
            
            ## Find the index of the liked item and compute Discounted Cumulative Gain (DCG).
            test_item_rank = np.where(ranked_indices == 0)[0][0] + 1  
            dcg = 1 / np.log2(test_item_rank + 1) if test_item_rank <= top_k else 0

            ## Compute the Ideal Discounted Cumulative Gain (IDCG)
            idcg = 1 / np.log2(1 + 1) 

            ## Divide the DCG by IDCG to obtain the NDCG and add it to the total_ndcg variable.
            ndcg = dcg / idcg if idcg > 0 else 0
            total_ndcg += ndcg

    ## Calculate the average Hit Ratio for the test data.
    hit_ratio = hits / total_users if total_users > 0 else 0

    ## Calculate the average NDCG for the test data.
    avg_ndcg = total_ndcg / total_users if total_users > 0 else 0

    ## Return both the average Hit Ratio and average NDCG value.
    return hit_ratio, avg_ndcg