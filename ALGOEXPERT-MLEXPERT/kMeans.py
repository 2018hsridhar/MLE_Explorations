from collections import defaultdict
import random
import numpy as np


class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_users = set()

def manhattanNorm(feature1, feature2):
    """Calculate the Manhattan distance (L1 norm) between two feature vectors."""
    distance = sum(abs(a - b) for a, b in zip(feature1, feature2))
    return distance
    
def get_k_means(user_feature_map, num_features_per_user, k):    # Don't change the following two lines of code.
    RANDOM_INIT = 42
    targetNumberCentroids = k
    random.seed(RANDOM_INIT)
    # Gets the inital users, to be used as centroids.
    inital_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)
    inital_centroids = [Centroid(user_feature_map[key]) for key in inital_centroid_users]
    centroids = inital_centroids
    ZERO_LOC = [0.0] * num_features_per_user
    numIterations = 10
    distanceNorm = 1 # Euclidean
    for iteration in range(numIterations):
        # [1] Assignment step
        centroid_assignments = defaultdict(list)
        for user_id, userFeature in user_feature_map.items():
            closestCentroidDistance = float('inf')  # Fixed: should be inf, not -inf
            closestCentroidIndex = -1
            for centroidIndex, centroid in enumerate(centroids):
                # Calculate distance between user features and centroid location
                distance = manhattanNorm(userFeature, centroid.location)
                if distance < closestCentroidDistance:  # Simplified condition
                    closestCentroidDistance = distance
                    closestCentroidIndex = centroidIndex
            # Assign user to closest centroid
            centroid_assignments[closestCentroidIndex].append(user_id)

        # [2] Update/Refitting step
        # CEntroid assignments : 1 -> [ userID1, userID2, ... ], 2 -> [ userID3, userID4, ... ]
        new_centroid_locations = [None] * k  # Fixed: pre-allocate list
        for centroidIndex in range(k):  # Fixed: ensure all centroids are updated
            if centroidIndex in centroid_assignments:
                assignedUserIDs = centroid_assignments[centroidIndex]
                if len(assignedUserIDs) > 0:  # Ensure no division by zero
                    # Calculate new centroid location as mean of assigned user features
                    new_centroid_location = [0.0] * num_features_per_user  # Fixed: create new list each time
                    for candid_user_id in assignedUserIDs:
                        candid_user_features = user_feature_map[candid_user_id]
                        for featureIndex in range(num_features_per_user):
                            new_centroid_location[featureIndex] += candid_user_features[featureIndex]
                    # Average the sum to get the mean
                    for featureIndex in range(num_features_per_user):
                        new_centroid_location[featureIndex] = new_centroid_location[featureIndex] / len(assignedUserIDs)
                    new_centroid_locations[centroidIndex] = Centroid(new_centroid_location)
                else:
                    # Handle empty clusters: Keep the previous centroid location
                    new_centroid_locations[centroidIndex] = Centroid(centroids[centroidIndex].location[:])
            else:
                # Handle case where no users were assigned to this centroid
                new_centroid_locations[centroidIndex] = Centroid(centroids[centroidIndex].location[:])
        
        # Update centroids with new locations
        centroids = new_centroid_locations
    output = [centroid.location for centroid in centroids]
    return output
