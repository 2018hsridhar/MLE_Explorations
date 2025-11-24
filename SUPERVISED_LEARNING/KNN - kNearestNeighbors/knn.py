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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

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

# def print_k_nearest_neighbors(examples, features, k):
#     myknn = find_k_nearest_neighbors(examples, features, k)
#     print(f"The {k} nearest neighbors are:")
#     for neighborPID in myknn:
#         neighborInfo = examples[neighborPID]
#         print(f"PID: {neighborPID}, Features: {neighborInfo['features']}, Label: {neighborInfo['is_intrusive']}")

class KNNVisualizer:
    """
    Interactive KNN visualizer with user controls
    """
    
    def __init__(self, examples, test_features, k=3):
        self.examples = examples
        self.test_features = test_features
        self.current_feature_index = 0
        self.k = k
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Create buttons
        self.setup_buttons()
        
        # Initial plot
        self.update_plot()
        
    def setup_buttons(self):
        """Set up interactive buttons"""
        # Button positions
        ax_prev = plt.axes([0.2, 0.05, 0.15, 0.075])
        ax_next = plt.axes([0.4, 0.05, 0.15, 0.075])
        ax_k_down = plt.axes([0.6, 0.05, 0.1, 0.075])
        ax_k_up = plt.axes([0.75, 0.05, 0.1, 0.075])
        
        # Create buttons
        self.btn_prev = Button(ax_prev, 'Previous Feature')
        self.btn_next = Button(ax_next, 'Next Feature')
        self.btn_k_down = Button(ax_k_down, 'k-1')
        self.btn_k_up = Button(ax_k_up, 'k+1')
        
        # Connect button callbacks
        self.btn_prev.on_clicked(self.prev_feature)
        self.btn_next.on_clicked(self.next_feature)
        self.btn_k_down.on_clicked(self.decrease_k)
        self.btn_k_up.on_clicked(self.increase_k)
        
    def prev_feature(self, event):
        """Go to previous test feature"""
        if self.current_feature_index > 0:
            self.current_feature_index -= 1
            self.update_plot()
            
    def next_feature(self, event):
        """Go to next test feature"""
        if self.current_feature_index < len(self.test_features) - 1:
            self.current_feature_index += 1
            self.update_plot()
            
    def decrease_k(self, event):
        """Decrease k value"""
        if self.k > 1:
            self.k -= 1
            self.update_plot()
            
    def increase_k(self, event):
        """Increase k value"""
        if self.k < len(self.examples):
            self.k += 1
            self.update_plot()
    
    def update_plot(self):
        """Update the visualization"""
        self.ax.clear()
        
        current_feature = self.test_features[self.current_feature_index]
        
        # Find k nearest neighbors
        knn_pids = find_k_nearest_neighbors(self.examples, current_feature, self.k)
        predicted_label = predict_label(self.examples, current_feature, self.k)
        
        # Plot training examples as circles (black)
        for pid, info in self.examples.items():
            x, y = info["features"]
            label = info["is_intrusive"]
            
            # Color based on whether it's a nearest neighbor
            if pid in knn_pids:
                color = 'green'  # K-nearest neighbors in green
                size = 120
                alpha = 0.8
            else:
                color = 'black'  # Other examples in black
                size = 80
                alpha = 0.5
                
            # Shape based on label (circle for all training examples)
            if label == 0:
                marker = 'o'  # Circle for non-intrusive
            else:
                marker = 'o'  # Circle for intrusive (but with different edge)
                
            self.ax.scatter(x, y, c=color, s=size, marker=marker, 
                          alpha=alpha, edgecolors='black', linewidth=2,
                          label=f'Intrusive={label}' if pid == list(self.examples.keys())[0] else "")
        
        # Plot current test feature as red square
        test_x, test_y = current_feature
        self.ax.scatter(test_x, test_y, c='red', s=200, marker='s', 
                       alpha=0.9, edgecolors='black', linewidth=3,
                       label='Test Point')
        
        # Draw circles showing distances to k-nearest neighbors
        for pid in knn_pids:
            neighbor_features = self.examples[pid]["features"]
            distance = l2Norm(current_feature, neighbor_features)
            circle = plt.Circle(current_feature, distance, fill=False, 
                              color='green', alpha=0.3, linestyle='--')
            self.ax.add_patch(circle)
        
        # Annotations and labels
        self.ax.set_title(f'KNN Visualization (k={self.k})\n'
                         f'Test Feature: {current_feature} | '
                         f'Predicted Label: {predicted_label} | '
                         f'Feature {self.current_feature_index + 1}/{len(self.test_features)}',
                         fontsize=14)
        
        self.ax.set_xlabel('Feature 1')
        self.ax.set_ylabel('Feature 2')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Add text annotations for nearest neighbors
        info_text = f"K-Nearest Neighbors (k={self.k}):\n"
        for i, pid in enumerate(knn_pids):
            neighbor_info = self.examples[pid]
            distance = l2Norm(current_feature, neighbor_info["features"])
            info_text += f"{pid}: {neighbor_info['features']} (dist: {distance:.2f}, label: {neighbor_info['is_intrusive']})\n"
        
        # Add text box
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set axis limits with some padding
        all_x = [info["features"][0] for info in self.examples.values()] + [test_x]
        all_y = [info["features"][1] for info in self.examples.values()] + [test_y]
        self.ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
        self.ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
        
        plt.draw()

def visualize_knn(examples, test_features, k=3):
    """
    Create interactive KNN visualization
    
    Args:
        examples: Dictionary of training examples
        test_features: List of test features to visualize
        k: Initial k value
    """
    print("🎨 KNN INTERACTIVE VISUALIZATION")
    print("=" * 40)
    print("Controls:")
    print("- Previous/Next Feature: Navigate through test points")
    print("- k-1/k+1: Adjust number of nearest neighbors")
    print("- Training examples: Black circles")
    print("- Test point: Red square") 
    print("- K-nearest neighbors: Green circles")
    print("- Distance circles: Dashed green lines")
    print("=" * 40)
    
    visualizer = KNNVisualizer(examples, test_features, k)
    plt.show()
    
    return visualizer

def main():
    # Example usage
    examples = {
        "p1": {"features": [1.0, 2.0], "is_intrusive": 0},
        "p2": {"features": [2.0, 3.0], "is_intrusive": 1},
        "p3": {"features": [1.5, 2.5], "is_intrusive": 0},
        "p4": {"features": [5.0, 5.0], "is_intrusive": 1},
        "p5": {"features": [3.0, 3.5], "is_intrusive": 0},
        "p6": {"features": [4.0, 4.0], "is_intrusive": 1},
        "p7": {"features": [3.5, 2.0], "is_intrusive": 0},
        "p8": {"features": [2.5, 1.0], "is_intrusive": 1},
        "p9": {"features": [6.0, 5.5], "is_intrusive": 1},
        "p10": {"features": [7.0, 8.0], "is_intrusive": 0},
    }
    
    # Test features to visualize (squares - red)
    test_features = [
        [2.0, 3.0],   # Near p2
        [3.5, 3.0],   # In the middle
        [6.5, 6.0],   # Near p9
        [1.0, 1.5],   # Near p1
        [4.5, 2.5],   # Between clusters
    ]
    
    print("🔍 Testing KNN Algorithm with Visualization")
    print("=" * 50)
    print("\nStarting interactive visualization...")
    
    # Start interactive visualization
    visualizer = visualize_knn(examples, test_features, k=3)

if __name__ == "__main__":
    main()
