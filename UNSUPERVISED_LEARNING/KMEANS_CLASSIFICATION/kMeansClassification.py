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
import random


class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_users = set()

def create_random_user_map(num_users=100, num_features_per_user=4, rand_lower=0, rand_upper=100):
    print(f"Creating random user-feature map with {num_users} users, each having {num_features_per_user} features.")
    user_feature_map = {}
    for user_id in range(num_users):
        features = [random.uniform(rand_lower, rand_upper) for _ in range(num_features_per_user)]
        user_feature_map[user_id] = features
    return user_feature_map, num_features_per_user

def get_k_means(user_feature_map, num_features_per_user, k, numIterations=100, distanceNorm=2):
    # Don't change the following two lines of code.
    RANDOM_INIT = 42
    random.seed(RANDOM_INIT)
    # Gets the inital users, to be used as centroids.
    inital_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)
    # [1] Assignment step
    for iteration in range(numIterations):
        print(f"\n--- K-Means Iteration {iteration + 1} / {numIterations} ---")
        


    # Write your code here.
    pass

def main():
    print(f"\n{'='*20} K-Means Clustering Algorithm {'='*20}\n")
    # Create a random user-feature map for demonstration
    user_feature_map, num_features_per_user = create_random_user_map(
        num_users=100, num_features_per_user=4, rand_lower=0, rand_upper=100)
    # print(f"User Feature Map: {user_feature_map}\n")
    # # Define number of target clusters
    numTargetClusters = 3
    
    # # Get initial centroids
    centroids = get_k_means(user_feature_map, num_features_per_user, numTargetClusters, numIterations=100)
    
    # # Print centroid locations
    # for idx, centroid in enumerate(centroids):
    #     print(f"Centroid {idx}: Location {centroid.location}")

main()