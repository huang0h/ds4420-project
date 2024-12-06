## This module stores all the preprocessing functions.

## Import necessary packages.
from tqdm import tqdm

## Leaves one of the interactions out for testing.
def LeaveOneOut(df):
    '''
    Randomly select one interaction and leaves it out for evaluation.

    Args:
        interaction_df (DataFrame): A DataFrame storing all the interactions of users and songs.

    Returns:
        train_df (DataFrame): A DataFrame containing all the training interactions.
        test_df (DataFrame): A DataFrame containing all the testing interactions.
    '''
    ## Extract only the valid interactions.
    positive_interactions = df[df['liked'] == 1]

    ## Randomly shuffles the positive interactions.
    positive_interactions = positive_interactions.sample(frac = 1, random_state = 42)

    ## Chooses a random user with a positive interaction.
    positive_interactions['rank'] = positive_interactions.groupby('user_id').cumcount() + 1
    random_test_set = positive_interactions.groupby('user_id').sample(n = 1, random_state = 42)
    test_users_items = set(zip(random_test_set['user_id'], random_test_set['song_id']))

    ## Excludes the selected interactions from the training set.
    train_set = df[~df.set_index(['user_id', 'song_id']).index.isin(test_users_items)].copy()
    return train_set, random_test_set

## Creates new numerical ID values for each user and item.
def MapUserItemID(df):
    '''
    Replaces the string ID with numerical ID for each unique user and item.

    Args:
        df (DataFrame): A DataFrame containing all the interactions between user and items.

    Returns:
        new_df (DataFrame): A new DataFrame with the numerical IDs in place.
    '''
    ## Create new_df DataFrame.
    new_df = df.copy()

    ## For each user and item, create new numerical ID values.
    user_mapping = {user_id: idx for idx, user_id in enumerate(new_df['user'].unique())}
    item_mapping = {item_id: idx for idx, item_id in enumerate(new_df['track'].unique())}

    ## Map the individual users and items to their new numerical IDs.
    new_df['user_id'] = new_df['user'].map(user_mapping)
    new_df['song_id'] = new_df['track'].map(item_mapping)

    ## Return the new DataFrame with the numerical ID values.
    return new_df

## Given a DataFrame, return a set with the positive interactions for each user and all 
## unique item IDs.
def CreatePositiveInteractions(df):
    '''
    For each user, store the items they had positive interactions with as a set, as well as
    create a set storing all unique item IDs.

    Args:
        df (DataFrame): A DataFrame storing all the interactions between users and items.

    Returns:
        user_positive_itemsets (Set): A set containing all the positvely interacted item for each user.
        item_pool (Set): The set of all unique item IDs.
    '''
    ## Define the set for storing all positive interactions of each unique users.
    user_positive_itemsets = {}

    ## For each unique user, add their respective positively interacted item ID.
    for user in tqdm(df['user_id'].unique(), desc = "Processing: "):
        user_positive_itemsets[user] = set(df[df['user_id'] == user]['song_id'].unique())

    ## Create the item_pool set, storing all unique item IDs.
    item_pool = set(df['song_id'].unique())

    ## Return the newly developed sets.
    return user_positive_itemsets, item_pool
