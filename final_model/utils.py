import json
import numpy as np

class HybridCF:
    def __init__(self, source: str):
        USER_SIM_FILE = 'user_similarity.npy'
        SUBSET_INDEX_MAPPING_FILE = 'subset_index_mapping.json'
        USER_NEIGHBORHOODS_FILE = 'user_neighborhoods.json'
    
        # You'll want to create a folder called `saved_models` in this folder,
        # then put the model feature files in that folder.
        # They should have the names of the files above.
        dirname = f'saved_models/{source}'
        sims = np.load(f'{dirname}/{USER_SIM_FILE}')
        
        with open(f'{dirname}/{SUBSET_INDEX_MAPPING_FILE}', 'r') as f:
            index_map = json.load(f)
        
        with open(f'{dirname}/{USER_NEIGHBORHOODS_FILE}', 'r') as f:
            neighborhoods = json.load(f)
            
        # From unzipping rawdata/msd_subset_audio_features_normalized_data.zip
        with open('../rawdata/msd_subset_audio_features_normalized.json', 'r') as f:
            all_track_features = json.load(f)
            
        # From unzipping rawdata/msd_user_setlists_data.zip
        with open('../rawdata/msd_user_setlists.json', 'r') as f:
            user_setlists = json.load(f)
            
        # From unzipping rawdata/msd_average_user_profiles_data.zip
        with open('../rawdata/msd_average_user_profiles.json', 'r') as f:
            average_user_profiles = json.load(f)
            
        self.user_similarity = sims
        self.subset_index_mapping = index_map
        self.user_neighborhoods = neighborhoods
        
        self.all_track_features = all_track_features
        self.user_setlists = user_setlists
        self.average_user_profiles = average_user_profiles
        
    # Define a user's rating of a track as the number of times they listened to it
    # If they haven't listened to it, define rating as the similarity (i.e. inverse distance) between the user's average profile and the track's features
    def __user_rating(self, user_id, track_id):
        if track_id in self.user_setlists[user_id]:
            return self.user_setlists[user_id][track_id]
        else:
            user_avg_features = self.average_user_profiles[user_id]
            track_features = self.all_track_features[track_id]
            
            # Cosine similarity between features
            distance = np.linalg.norm(np.array(list(user_avg_features.values())) - np.array(list(track_features.values())))
            
            return 1 / (1 + distance)   

    def predict_rating(self, user_id: str, track_id: str):
        user_index = self.subset_index_mapping[user_id]
        neighborhood = self.user_neighborhoods[user_id]
        
        neighborhood_ratings = [self.__user_rating(neighbor, track_id) for neighbor in neighborhood]
        
        similarities = [self.user_similarity[user_index][self.subset_index_mapping[neighbor]] for neighbor in neighborhood]
        
        num = np.dot(similarities, neighborhood_ratings)
        den = np.abs(similarities).sum()
        
        if num == 0 or den == 0:
            return 0

        return num / den