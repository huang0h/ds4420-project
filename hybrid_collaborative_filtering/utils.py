import numpy as np

def generate_user_profile(taste_profile_raw, merged_features_processed, lyric_size: int=0):
  average_user_profiles = {}
  user_setlists = {}

  for user_id, tracklist in taste_profile_raw.items():
      initial = {
          'duration': 0,
          'key': 0,
          'key_confidence': 0,
          'mode': 0,
          'mode_confidence': 0,
          'loudness': 0,
          'tempo': 0,
          'time_signature': 0,
          'time_signature_confidence': 0,
      }
      if lyric_size > 0:
        initial['lyrics'] = np.zeros(lyric_size)
      
      user_setlists[user_id] = {}
      
      # Generate weighted average of features
      total_track_counts = 0
      
      for track in tracklist:
          track_id, count = track['track'], track['count']
          if track_id not in merged_features_processed:
              raise 'AAAAA'
          
          total_track_counts += count
          user_setlists[user_id][track_id] = count
          
          track_features = merged_features_processed[track_id]
          for feature, value in track_features.items():
              initial[feature] += value * count
              
      for feature in initial:
          initial[feature] /= total_track_counts
      
      average_user_profiles[user_id] = initial

  return average_user_profiles, user_setlists