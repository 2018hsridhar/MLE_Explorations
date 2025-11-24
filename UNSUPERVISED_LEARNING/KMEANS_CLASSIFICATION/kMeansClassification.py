'''
K-Means Clustering Algorithm Implementation

K-means is an unsupervised machine learning algorithm used 
for clustering data points into k distinct clusters based on feature similarity. 
The algorithm iteratively assigns data points to the nearest centroid and 
updates the centroids until convergence.

What to expect :
- Definition of a Centroid class to represent cluster centroids.
- A function get_k_means to initialize centroids and prepare for clustering.
- Per each iteration, show assignment of colors based on closest centroid.

Trends To Note :
- Random initialization of centroids for reproducibility.
- Use of data structures to manage user-feature mappings and centroid assignments.

Parameters :
- user_feature_map: A dictionary mapping user IDs to their feature vectors ( 4d ) 
- num_features_per_user: The number of features associated with each user.
- k: The number of desired clusters.
- epochs : Number of iterations for the K-means algorithm to refine centroids.
- distanceNorm : The norm used to calculate distances between points and centroids (e.g., Euclidean norm).

Hyperparameters:
- RANDOM_INIT: A fixed seed for random number generation to ensure consistent results across runs.

References : 
- https://en.wikipedia.org/wiki/K-means_clustering

'''
from collections import defaultdict
import random


class Centroid:
    def __init__(self, location):
        self.location = location

def create_random_user_map(num_users=100, num_features_per_user=4, rand_lower=0, rand_upper=100):
    print(f"Creating random user-feature map with {num_users} users, each having {num_features_per_user} features.")
    user_feature_map = {}
    for user_id in range(num_users):
        features = [random.uniform(rand_lower, rand_upper) for _ in range(num_features_per_user)]
        user_feature_map[user_id] = features
    return user_feature_map, num_features_per_user

def l2Norm(feature1, feature2, norm=2):
    """Calculate the L2 norm (Euclidean distance) between two feature vectors."""
    if len(feature1) != len(feature2):
        raise ValueError("Feature vectors must be of the same length.")
    distance = sum((a - b) ** norm for a, b in zip(feature1, feature2)) ** (1/norm)
    return distance

def get_k_means(user_feature_map, num_features_per_user, targetNumberCentroids, numIterations=100, distanceNorm=2):
    # Don't change the following two lines of code.
    RANDOM_INIT = 42
    k = targetNumberCentroids
    random.seed(RANDOM_INIT)
    # Gets the inital users, to be used as centroids.
    userKeys = sorted(list(user_feature_map.keys()))
    kKeySample = random.sample(userKeys, k)
    inital_centroids = [Centroid(user_feature_map[key]) for key in kKeySample]
    centroids = inital_centroids
    ZERO_LOC = [0.0] * num_features_per_user
    for iteration in range(numIterations):
        print(f"\n--- K-Means Iteration {iteration + 1} / {numIterations} ---")
        # [1] Assignment step
        centroid_assignments = defaultdict(list)
        for user_id, userFeature in user_feature_map.items():
            closestCentroidDistance= float('-inf')
            closestCentroidIndex = -1
            for centroidIndex, centroid in enumerate(centroids):
                # Calculate distance between user features and centroid location
                distance = l2Norm(userFeature, centroid.location, distanceNorm)
                if closestCentroidIndex == -1 or distance < closestCentroidDistance:
                    closestCentroidDistance = distance
                    closestCentroidIndex = centroidIndex
            # Assign user to closest centroid
            centroid_assignments[closestCentroidIndex].append(user_id)


        # [2] Update/Refitting step
        # CEntroid assignments : 1 -> [ userID1, userID2, ... ], 2 -> [ userID3, userID4, ... ]
        new_centroid_locations = [Centroid(ZERO_LOC) for key in kKeySample]
        for centroidIndex, assignedUserIDs in centroid_assignments.items():
            # Calculate new centroid location as mean of assigned user features
            new_centroid_location = ZERO_LOC
            for candid_user_id in assignedUserIDs:
                candid_user_features = user_feature_map[candid_user_id]
                for featureIndex in range(num_features_per_user):
                    new_centroid_location[featureIndex] += candid_user_features[featureIndex]
            # Average the sum to get the mean
            for featureIndex in range(num_features_per_user):
                new_centroid_location[featureIndex] /= len(assignedUserIDs)
            new_centroid_locations[centroidIndex] = Centroid(new_centroid_location)
        # Update centroids with new locations
        for centroidIndex, new_centroid_location in enumerate(new_centroid_locations):
            centroids[centroidIndex].location = new_centroid_locations[centroidIndex].location
    return centroids

def main():
    print(f"\n{'='*20} K-Means Clustering Algorithm {'='*20}\n")
    # Create a random user-feature map for demonstration
    user_feature_map, num_features_per_user = create_random_user_map(
        num_users=100, num_features_per_user=4, rand_lower=0, rand_upper=100)
    # print(f"User Feature Map: {user_feature_map}\n")
    # # Define number of target clusters
    numTargetClusters = 3
    
    # # Get initial centroids
    centroids = get_k_means(user_feature_map, num_features_per_user, numTargetClusters, numIterations=10)
    
    # # Print centroid locations
    for idx, centroid in enumerate(centroids):
        print(f"Centroid {idx}: Location {centroid.location}")

main()