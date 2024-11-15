import pandas as pd
import numpy as np

def load_data(filename='../rawdata/piki_dataset.csv', verbose=True) -> tuple[pd.DataFrame, dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """
    Loads data from the Piki dataset into a dataframe for further processing.
    This maps Piki user and song IDs into 0-indexed IDs, which allows them to also nicely function as array indices for faster processing

    Args:
        filename (str, optional): The CSV file for the Piki dataset. Defaults to '../rawdata/piki_dataset.csv'.
        verbose (bool, optional): Toggle for informational printing. Defaults to True.

    Returns:
        tuple[pd.DataFrame, dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
        
        - the processed dataframe with Piki user and song IDs mapped to a new 0-indexed ID
        - A mapping of Piki user IDs to internal user IDs
        - The inverse of the above: a mapping from internal user ID to Piki user ID
        - A mapping of Piki song IDs to internal song IDs
        - The inverse of the above: a mapping from internal song ID to Piki song ID
        
    """
    df = pd.read_csv(filename)
    df = df.drop(columns=['Unnamed: 0', 'treatment_group', 'personalized', 'spotify_popularity'])
    
    # Initialize mappings
    user_id_mapping, song_id_mapping = {}, {}
    user_id_count = song_id_count = 0
    
    n_rows = df.shape[0]
    
    # For speed, we set the new user/song IDs in these arrays, which we'll copy to the dataframe at the end.
    new_user_id = np.zeros(n_rows, dtype=int)
    new_song_id = np.zeros(n_rows, dtype=int)
    
    for i, row in enumerate(df.itertuples()):
        if verbose and i % 1000 == 0:
            print(f'Processing row {i}/{n_rows}...', end='\r')
        
        row_user_id = getattr(row, 'user_id')
        if row_user_id not in user_id_mapping:
            user_id_mapping[row_user_id] = user_id_count
            user_id_count += 1
        new_user_id[i] = user_id_mapping[row_user_id]
        
        row_song_id = getattr(row, 'song_id')
        if row_song_id not in song_id_mapping:
            song_id_mapping[row_song_id] = song_id_count
            song_id_count += 1
        new_song_id[i] = song_id_mapping[row_song_id]
    
    # Copy the entire array to the column at once
    df['user_id'] = new_user_id
    df['song_id'] = new_song_id
    
    inverse_user_mapping = { value: key for key, value in user_id_mapping.items() }
    inverse_song_mapping = { value: key for key, value in song_id_mapping.items() }
    
    return df, user_id_mapping, inverse_user_mapping, song_id_mapping, inverse_song_mapping
  
def initialize_latent_features(user_size: int, song_size: int, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates n unintialized latent features for each user and song, returned as 2D arrays

    Args:
        user_size (int): the number of users to generate features for
        song_size (int): the number of songs to generate features for
        n_features (int): the number of latent features to generate

    Returns:
        tuple[np.ndarray, np.ndarray]: 
        
        - User latent features, as a user_size x n_features array
        - Song latent features, as a song_size x n_features array
    """
    return np.random.rand(user_size, n_features), np.random.rand(song_size, n_features)
  
def train_test_split(df: pd.DataFrame, n_users: int, verbose=False):
    """
    Generates training and testing data for collaborative filtering.

    Args:
        df (pd.DataFrame): The dataframe with user_id, song_id, timestamp, and liked columns
        n_users (int): The number of unique users present in the dataframe
        verbose (bool, optional): Toggle verbose printing. Defaults to False.

    Returns:
        (list[dict], list[tuple]): training and testing data, respectively.
        
        Traning data is in the form:
        ```
        {
            'liked': list[tuple[string, int]],
            'disliked': list[tuple[string, int]],
            'liked_count': int,
            'disliked_count': int
        }
        ```
        
        where liked/disliked is a list user ratings, in the form of tuples like ('2019-06-19 09:22:38', 0).
        The first element is the rating timestamp, and the second element is the song ID being rated.
        
        Test data is in the form: 
        list[tuple[string, int]]
        
        
        with length n_users
        User i's test data is at test_data[i] - i.e. each index of the test data list corresponds to that user's test data.
        Each element is in the same tuple form described above
    """
    user_ratings = np.array([{ 'liked': [], 'disliked': [], 'liked_count': 0, 'disliked_count': 0 } for _ in range(n_users)])
    
    n_rows = df.shape[0]
    
    # Construct user song ratings
    for i, row in enumerate(df.itertuples()):
        if verbose and i % 1000 == 0:
            print(f'Processing row {i}/{n_rows}...', end='\r')
        
        row_user = getattr(row, 'user_id')
        row_user_rating = user_ratings[row_user]
        
        if getattr(row, 'liked') == 0:
            row_user_rating['disliked'].append((getattr(row, 'timestamp'), getattr(row, 'song_id')))
            row_user_rating['disliked_count'] += 1
        else:
            row_user_rating['liked'].append((getattr(row, 'timestamp'), getattr(row, 'song_id')))
            row_user_rating['liked_count'] += 1

    # Create an array of structured tuples
    test_data = np.empty(n_users, dtype=object)
    
    for i, rating in enumerate(user_ratings):
        # No test data possible if the user has not liked any songs
        if rating['liked_count'] == 0:
            test_data[i] = None
        else:
            user_liked = rating['liked']
            # Sort by timestamp, ascending: latest liked is last
            user_liked.sort(key=lambda t: t[0])
            
            # Use the last liked song as test data, and remove from training
            test_data[i] = user_liked.pop()
            rating['liked_count'] -= 1
        
    return user_ratings, test_data

def generate_train_epoch(train_data: np.ndarray, k: int, strict=False, seed=None):
    """
    Generates an epoch of training data by randomly sampling liked and disliked songs for each user.
    If data cannot be generated for a user, the value for said user will be None.

    Args:
        train_data (np.ndarray): the training data to sample from
        k (int): the number of disliked songs to generate per liked song
        strict (bool, optional): If strict is True, then training data for a user will be None if they do not have 
                                 at least k disliked songs for each liked song. Defaults to False.
        seed (_type_, optional): Seed to standardize RNG. Defaults to None.

    Returns:
        np.ndarray: An epoch of randomly sampled data.
        
        Each element of the array is None, if data could not be generated for that user, or a 2-element array of int arrays.
        The first of these are the song IDs liked by the user.
        The second is the randomly sampled song IDs disliked by the user. 
    """
    rand = np.random.default_rng(seed)
    epoch_data = np.empty(train_data.shape[0], dtype=object)
    
    for i, user_data in enumerate(train_data):
        liked_count, disliked_count = user_data['liked_count'], user_data['disliked_count']
        
        # Cannot produce data for this user if:
        # 1. user has no disliked songs
        # 2. strict mode is on, and the number of disliked songs < samples to get (k * liked count)
        if (disliked_count == 0) or (strict and liked_count * k > disliked_count):
            epoch_data[i] = None
        else:
            liked_ids = [int(t[1]) for t in user_data['liked']]
            disliked_ids = [int(t[1]) for t in rand.choice(user_data['disliked'], min(disliked_count, k * liked_count), replace=False)]
            
            epoch_data[i] = [liked_ids, disliked_ids]
            
    return epoch_data
