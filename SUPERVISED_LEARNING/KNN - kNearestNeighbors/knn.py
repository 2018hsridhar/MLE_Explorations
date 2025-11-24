'''
k-Nearest Neighbors (kNN) implementation for binary classification.
Uses Euclidean distance (L2 norm) to find nearest neighbors.

Knn is a simple, interpretable algorithm that classifies data points 
based on the labels of their closest neighbors in the feature space.
It is a supervised learning algorithm commonly used for classification tasks, 
either binary classification ( in this case) or multi-class classification.

Note that kNN is a lazy learner, meaning it does not learn a model during training.
Which is why we only need to store the examples and their labels.

Being a lazy learner makes kNN fast to train but potentially slow to predict
because it needs to compute distances to all training examples for each prediction :-( !

aNN - Approximate Nearest Neighbors - faster but less accurate version of kNN
is often used for large datasets and a trade-off between speed and accuracy
preferred in real-time applications and large-scale systems.

'''
import math
import heapq
import numpy as np

# Euclidean distance (L2 norm)
def l2Norm(listOne, listTwo):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(listOne, listTwo)))

# Find k nearest neighbors using a max-heap
def find_k_nearest_neighbors(examples, features, k):
    maxHeap = []  # store (-distance, pid) to simulate a max-heap
    for pid, pidInfo in examples.items():
        pidFeatures = pidInfo["features"]
        curL2Dist = l2Norm(pidFeatures, features)
        
        if len(maxHeap) < k:
            heapq.heappush(maxHeap, (-curL2Dist, pid))
        else:
            # largest distance is at the top because we negated distances
            largestDist = -maxHeap[0][0]
            if curL2Dist < largestDist:
                heapq.heappop(maxHeap)
                heapq.heappush(maxHeap, (-curL2Dist, pid))
    
    # Extract just the pid values
    kNearestPid = [pid for (_, pid) in maxHeap]
    return kNearestPid

# Predict binary label based on k nearest neighbors
def predict_label(examples, features, k, label_key="is_intrusive"):
    myknn = find_k_nearest_neighbors(examples, features, k)
    # the top K nearest neighbors for the given feature `features` 
    # iterates over al examples given
    
    labelZeroCount = 0
    labelOneCount = 0
    for nearestNeighborPID in myknn:
        label = examples[nearestNeighborPID][label_key]
        if label == 0:
            labelZeroCount += 1
        else:
            labelOneCount += 1
            
    label = 0
    if(labelOneCount > labelZeroCount):
        label= 1
    return label

# Example usage:
# examples = {
#     "p1": {"features": [1.0, 2.0], "is_intrusive": 0},
#     "p2": {"features": [2.0, 3.0], "is_intrusive": 1},
#     "p3": {"features": [1.5, 2.5], "is_intrusive": 0}
# }
# features = [1.2, 2.1]
# k = 2
# print(predict_label(examples, features, k))
