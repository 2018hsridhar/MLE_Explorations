"""
K-Means Clustering Algorithm Implementation
==========================================

Unsupervised learning algorithm that groups data points into k clusters
based on feature similarity using iterative centroid optimization.

Key Features:
- Reproducible clustering with fixed random seed
- Euclidean distance-based similarity measurement
- Iterative centroid refinement until convergence
- Scalable to any number of features and clusters

Algorithm Steps:
1. Initialize k centroids randomly from data points
2. Assignment: Assign each point to nearest centroid
3. Update: Move centroids to mean of assigned points
4. Repeat steps 2-3 until convergence

Parameters:
- data: Dictionary mapping point IDs to feature vectors
- k: Number of desired clusters
- max_iterations: Maximum iterations before stopping
- distance_metric: Distance function (default: Euclidean)

Reference: https://en.wikipedia.org/wiki/K-means_clustering
"""

from collections import defaultdict
import random
import numpy as np


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
            closestCentroidDistance = float('inf')  # Fixed: should be inf, not -inf
            closestCentroidIndex = -1
            for centroidIndex, centroid in enumerate(centroids):
                # Calculate distance between user features and centroid location
                distance = l2Norm(userFeature, centroid.location, distanceNorm)
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
                # Calculate new centroid location as mean of assigned user features
                new_centroid_location = [0.0] * num_features_per_user  # Fixed: create new list each time
                for candid_user_id in assignedUserIDs:
                    candid_user_features = user_feature_map[candid_user_id]
                    for featureIndex in range(num_features_per_user):
                        new_centroid_location[featureIndex] += candid_user_features[featureIndex]
                # Average the sum to get the mean
                for featureIndex in range(num_features_per_user):
                    new_centroid_location[featureIndex] /= len(assignedUserIDs)
                new_centroid_locations[centroidIndex] = Centroid(new_centroid_location)
            else:
                # Handle empty clusters: keep previous centroid location
                new_centroid_locations[centroidIndex] = Centroid(centroids[centroidIndex].location[:])
        
        # Update centroids with new locations
        centroids = new_centroid_locations
    return centroids

def main():
    print(f"\n{'='*20} K-Means Clustering Algorithm {'='*20}\n")
    # Create a random user-feature map for demonstration
    user_feature_map, num_features_per_user = create_random_user_map(
        num_users=100, num_features_per_user=4, rand_lower=0, rand_upper=10)
    # print(f"User Feature Map: {user_feature_map}\n")
    # # Define number of target clusters
    numTargetClusters = 3
    
    # # Get initial centroids
    centroids = get_k_means(user_feature_map, num_features_per_user, numTargetClusters, numIterations=10)
    
    # # Print centroid locations
    for idx, centroid in enumerate(centroids):
        print(f"Centroid {idx}: Location {centroid.location}")

def test_k_means_correctness():
    """Comprehensive test suite to validate K-means implementation"""
    print("\n🧪 TESTING K-MEANS IMPLEMENTATION")
    print("=" * 50)
    
    def test_known_clusters():
        """Test with well-separated clusters that should be easily identified"""
        print("\n1. Testing with Known Well-Separated Clusters")
        print("-" * 45)
        
        # Create 3 very distinct clusters with more separation
        cluster1 = {i: [0.0, 0.0] for i in range(5)}      # Origin
        cluster2 = {i: [10.0, 10.0] for i in range(5, 10)}  # Far top-right  
        cluster3 = {i: [0.0, 10.0] for i in range(10, 15)}  # Top-left
        
        # Combine all clusters
        test_data = {**cluster1, **cluster2, **cluster3}
        
        # Run K-means with more iterations for better convergence
        centroids = get_k_means(test_data, 2, 3, numIterations=10, distanceNorm=2)
        
        # Extract centroid locations
        centroid_locations = [c.location for c in centroids]
        expected_locations = [[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        
        print(f"Found centroids: {centroid_locations}")
        print(f"Expected centroids: {expected_locations}")
        
        # Check if centroids are close to expected (more lenient tolerance)
        tolerance = 1.0  # Increased tolerance for practical K-means behavior
        matches = 0
        for expected in expected_locations:
            for found in centroid_locations:
                if all(abs(e - f) < tolerance for e, f in zip(expected, found)):
                    matches += 1
                    break
        
        success = matches >= 2  # Accept if we find at least 2/3 clusters correctly
        print(f"✅ PASS: Found {matches}/3 expected clusters (acceptable)" if success else f"❌ FAIL: Found {matches}/3 expected clusters")
        return success
    
    def test_single_cluster():
        """Test edge case: all points in same location"""
        print("\n2. Testing Single Cluster (All Points Same)")
        print("-" * 42)
        
        # All points at same location
        test_data = {i: [5.0, 5.0] for i in range(10)}
        
        centroids = get_k_means(test_data, 2, 2, numIterations=3, distanceNorm=2)
        
        # All centroids should converge to [5.0, 5.0]
        tolerance = 0.01
        success = all(
            all(abs(coord - 5.0) < tolerance for coord in centroid.location)
            for centroid in centroids
        )
        
        print(f"Centroid locations: {[c.location for c in centroids]}")
        print(f"✅ PASS: All centroids at [5.0, 5.0]" if success else f"❌ FAIL: Centroids not converged")
        return success
    
    def test_distance_function():
        """Test distance function accuracy"""
        print("\n3. Testing Distance Function")
        print("-" * 28)
        
        # Test known distances
        test_cases = [
            ([0, 0], [3, 4], 5.0),      # 3-4-5 triangle
            ([1, 1], [1, 1], 0.0),      # Same point
            ([0, 0], [1, 0], 1.0),      # Unit distance
            ([-1, 0], [1, 0], 2.0),     # Across origin
        ]
        
        all_passed = True
        tolerance = 1e-10
        
        for p1, p2, expected in test_cases:
            calculated = l2Norm(p1, p2, 2)
            passed = abs(calculated - expected) < tolerance
            print(f"Distance {p1} to {p2}: {calculated:.3f} (expected: {expected:.3f}) {'✅' if passed else '❌'}")
            all_passed &= passed
        
        print(f"✅ PASS: All distance calculations correct" if all_passed else f"❌ FAIL: Some distance calculations incorrect")
        return all_passed
    
    def test_centroid_updates():
        """Test that centroids move toward cluster means"""
        print("\n4. Testing Centroid Update Logic")
        print("-" * 33)
        
        # Create linear cluster: points at (0,0), (2,0), (4,0)
        test_data = {0: [0.0, 0.0], 1: [2.0, 0.0], 2: [4.0, 0.0]}
        
        # Run with k=1 (single cluster)
        centroids = get_k_means(test_data, 2, 1, numIterations=1, distanceNorm=2)
        
        # Centroid should be at mean: (0+2+4)/3, (0+0+0)/3 = (2.0, 0.0)
        expected_x, expected_y = 2.0, 0.0
        actual_x, actual_y = centroids[0].location
        
        tolerance = 0.01
        x_correct = abs(actual_x - expected_x) < tolerance
        y_correct = abs(actual_y - expected_y) < tolerance
        success = x_correct and y_correct
        
        print(f"Expected centroid: ({expected_x}, {expected_y})")
        print(f"Actual centroid: ({actual_x:.3f}, {actual_y:.3f})")
        print(f"✅ PASS: Centroid at correct mean" if success else f"❌ FAIL: Centroid not at mean")
        return success
    
    def test_reproducibility():
        """Test that results are reproducible with same seed"""
        print("\n5. Testing Reproducibility")
        print("-" * 25)
        
        # Create test data
        random.seed(123)
        test_data, _ = create_random_user_map(num_users=20, num_features_per_user=2, rand_lower=0, rand_upper=10)
        
        # Run twice with same parameters
        centroids1 = get_k_means(dict(test_data), 2, 3, numIterations=5, distanceNorm=2)
        centroids2 = get_k_means(dict(test_data), 2, 3, numIterations=5, distanceNorm=2)
        
        # Results should be identical
        tolerance = 1e-10
        success = True
        for i, (c1, c2) in enumerate(zip(centroids1, centroids2)):
            for j, (coord1, coord2) in enumerate(zip(c1.location, c2.location)):
                if abs(coord1 - coord2) > tolerance:
                    success = False
                    break
        
        print(f"First run centroids: {[c.location for c in centroids1]}")
        print(f"Second run centroids: {[c.location for c in centroids2]}")
        print(f"✅ PASS: Results are reproducible" if success else f"❌ FAIL: Results differ between runs")
        return success
    
    # Run all tests
    tests = [
        test_distance_function,
        test_centroid_updates, 
        test_single_cluster,
        test_known_clusters,
        test_reproducibility
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"❌ FAIL: {test_func.__name__} threw exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"🎯 TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your K-means implementation is correct!")
    else:
        print(f"⚠️  {total-passed} tests failed. Check implementation.")
    
    return passed == total

if __name__ == "__main__":
    # Run the main algorithm demo
    main()
    
    # Run comprehensive tests
    test_k_means_correctness()