"""
Single Neuron Binary Classification: AND & OR Gates
==================================================

Train a single neuron to learn weights and biases 
for binary classification of inputs from circuit logical gates : AND and OR
Focus on binary classification because it's a classic example
and easy to visualize.

Key Concepts:
- Single perceptron (neuron) with sigmoid activation
- Binary classification (0 or 1 output)
- Gradient descent learning
- Logical gate patterns

Single neurons can learn linearly separable patterns
but not complex ones like XOR since it's not linearly separable.
We need multiple neurons/layers for complex patterns like XOR.
    AND gate: both inputs must be 1"
    OR gate: at least one input must be 1" 
Weights = feature importance"
Bias  = shifter of the decision boundary"

Experimentable Parameters:
- Learning rate: Adjust to see impact on training convergence speed/stability.
- Number of epochs: More epochs may improve learning but risk overfitting.
- Weight initialization: Different seeds or distributions can affect training.
- Bias initialization: Starting bias can shift decision boundary initially.
- TTT : Time-to-train affects convergence and performance.

Why shift decision boundary with bias?
- Bias allows the neuron to adjust the threshold for classification.
- Helps in fitting data that is not centered around the origin ( 0,0 ) 
- Enables better separation of classes by moving the decision boundary.
- Since data is fully linearly separable, bias helps find optimal line.

Output trends to observe:
- Loss decrease over number of training epochs
- Accuracy improvement over number of training epochs
- Decision boundary shifts visualization

What would break trends?
- Very high learning rates causing divergence
- Insufficient epochs leading to underfitting

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

class SingleNeuron:
    """
    A simple single neuron implementation for binary classification
    Perfect for learning AND/OR gates and basic logical operations
    """
    
    def __init__(self, input_size):
        """
        Initialize neuron with random weights
        
        Args:
            input_size: Number of input features
        """
        RAND_SEED_STARTER = 42
        np.random.seed(RAND_SEED_STARTER)
        # Initialize weights: one for each input + one bias
        # Weights initialized with small random values in range ~N(0, 0.5) vs N(0,1) because 
        # smaller variance helps in stable and faster convergence during training

        numWeights = input_size + 1
        rangeLower = 0
        rangeUpper = 0.5

        # Bias term included as the last weight
        self.weights = np.random.normal(rangeLower, rangeUpper, numWeights)

        # Learing_rate for weight updates
        # Controls the step size during gradient descent
        # Smaller values = slower but more stable learning
        # Larger values = faster but risk overshooting minima
        self.learning_rate = 0.1

        # History : track loss and accuracy over epochs
        # Useful for monitoring training progress and diagnosing issues
        self.history = {'loss': [], 'accuracy': []}
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip to prevent overflow in exp for large magnitude inputs
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def linear(self, x):
        """Linear activation function"""
        return x
    
    def forward(self, X):
        """
        Forward pass through the neuron
        
        Args:
            X: Input features (can be single sample or batch)
        
        Returns:
            Predicted probabilities / outputs based on activation function
        """
        # Add bias term (column of ones)
        if X.ndim == 1:
            X_bias = np.append(X, 1)
        else:
            X_bias = np.column_stack([X, np.ones(X.shape[0])])
        
        # Compute weighted sum
        z = np.dot(X_bias, self.weights)
        
        # Apply sigmoid activation
        a = self.sigmoid(z)
        return a
    
    def compute_binary_cross_entropy_loss(self, y_true, y_pred):
        """
        BCE = - (y*log(y_hat) + (1-y)*log(1-y_hat))
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
        Returns:
            Binary cross-entropy loss value
        """
        # Prevent log(0) by clipping predictions
        # What is clip? Clipping limits the values in an array to a specified range.
        # Here, it ensures predicted probabilities are never exactly 0 or 1 to avoid log(0) errors.
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        bceLoss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return bceLoss
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the single neuron using gradient descent
        Why gradient descent?
        Gradient descent is an optimization algorithm used to 
        minimize the loss function by iteratively moving towards the steepest descent direction.
        
        Args:
            X: Input features
            y: Target labels (0 or 1)
            epochs: Number of training iterations
            verbose: Print training progress
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        print(f"Training on {n_samples} samples with {n_features} features for {epochs} epochs.")
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_binary_cross_entropy_loss(y, y_pred)
            
            # Compute accuracy
            predictions = (y_pred >= 0.5).astype(int)
            accuracy = accuracy_score(y, predictions)
            
            # Store history
            self.history['loss'].append(loss)
            self.history['accuracy'].append(accuracy)
            
            # Compute gradients
            # Add bias to X for gradient computation
            X_bias = np.column_stack([X, np.ones(n_samples)])
            
            # Gradient of loss w.r.t. weights
            error = y_pred - y
            gradients = np.dot(X_bias.T, error) / n_samples
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            # Print progress
            if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        
        print(f"\nFinal weights: {self.weights}")
        print(f"Final bias: {self.weights[-1]}")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.forward(X)
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        return self.forward(X)

def train_and_gate():
    """
    Train a single neuron to learn the AND gate
    """
    print("🔗 TRAINING AND GATE")
    
    # AND gate truth table
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND logic: only 1,1 -> 1
    
    numSamples, numFeatures = X.shape
    print("AND Truth Table:")
    print("A | B | Output")
    print("-" * 15)
    for i in range(numSamples):
        print(f"{X[i][0]} | {X[i][1]} |   {y[i]}")
    
    # Create and train neuron
    neuron = SingleNeuron(input_size=numFeatures)
    neuron.train(X, y, epochs=1000)
    
    # Test the trained neuron
    print("\nAND Gate Test Results:")
    print("A | B | Expected | Predicted | Probability")
    
    for i in range(numSamples):
        pred_proba = neuron.predict_proba(X[i])
        pred_class = neuron.predict(X[i])
        print(f"{X[i][0]} | {X[i][1]} |    {y[i]}     |     {pred_class}     |   {pred_proba:.4f}")
    
    # Final accuracy
    predictions = neuron.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"\nFinal AND Gate Accuracy: {accuracy:.4f}")
    
    return neuron, X, y

# def train_or_gate():
#     """
#     Train a single neuron to learn the OR gate
#     """
#     print("\n🔀 TRAINING OR GATE")
#     print("=" * 30)
    
#     # OR gate truth table
#     X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     y = np.array([0, 1, 1, 1])  # OR logic: any 1 -> 1
    
#     print("OR Truth Table:")
#     print("A | B | Output")
#     print("-" * 15)
#     for i in range(len(X)):
#         print(f"{X[i][0]} | {X[i][1]} |   {y[i]}")
    
#     # Create and train neuron
#     neuron = SingleNeuron(input_size=2)
#     neuron.train(X, y, epochs=1000)
    
#     # Test the trained neuron
#     print("\nOR Gate Test Results:")
#     print("A | B | Expected | Predicted | Probability")
#     print("-" * 45)
    
#     for i in range(len(X)):
#         pred_proba = neuron.predict_proba(X[i])
#         pred_class = neuron.predict(X[i])
#         print(f"{X[i][0]} | {X[i][1]} |    {y[i]}     |     {pred_class}     |   {pred_proba:.4f}")
    
#     # Final accuracy
#     predictions = neuron.predict(X)
#     accuracy = accuracy_score(y, predictions)
#     print(f"\nFinal OR Gate Accuracy: {accuracy:.4f}")
    
#     return neuron, X, y

def visualize_training_progress(neuron, gate_name):
    """
    Visualize how the neuron learns over time
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(neuron.history['loss'])
    ax1.set_title(f'{gate_name} Gate - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(neuron.history['accuracy'])
    ax2.set_title(f'{gate_name} Gate - Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# def visualize_decision_boundary(neuron, X, y, gate_name):
#     """
#     Visualize the decision boundary learned by the neuron
#     """
#     # Create a mesh for the decision boundary
#     h = 0.01
#     x_min, x_max = -0.5, 1.5
#     y_min, y_max = -0.5, 1.5
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))
    
#     # Get predictions for the entire mesh
#     mesh_points = np.c_[xx.ravel(), yy.ravel()]
#     Z = neuron.predict_proba(mesh_points)
#     Z = Z.reshape(xx.shape)
    
#     # Create the plot
#     plt.figure(figsize=(10, 8))
    
#     # Plot decision boundary as contour
#     contour = plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
#     plt.colorbar(contour, label='Prediction Probability')
    
#     # Plot the decision line at 0.5 probability
#     plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, linestyles='--')
    
#     # Plot the actual data points
#     colors = ['red', 'blue']
#     labels = ['Output 0', 'Output 1']
#     for class_value in [0, 1]:
#         mask = y == class_value
#         plt.scatter(X[mask, 0], X[mask, 1], 
#                    c=colors[class_value], 
#                    label=labels[class_value], 
#                    s=200, 
#                    edgecolor='black',
#                    linewidth=2)
    
#     # Add labels and annotations
#     plt.xlabel('Input A')
#     plt.ylabel('Input B')
#     plt.title(f'{gate_name} Gate - Decision Boundary')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Annotate each point with coordinates and prediction
#     for i, (x_val, y_val) in enumerate(X):
#         pred = neuron.predict_proba(X[i])
#         plt.annotate(f'({x_val},{int(X[i][1])})\nP={pred:.3f}', 
#                     xy=(x_val, X[i][1]), 
#                     xytext=(x_val+0.1, X[i][1]+0.1),
#                     fontsize=10,
#                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
#     plt.tight_layout()
#     plt.show()

# def compare_gates():
#     """
#     Compare AND and OR gates side by side
#     """
#     print("\n🔄 COMPARING AND vs OR GATES")
#     print("=" * 40)
    
#     # Train both gates
#     and_neuron, X, y_and = train_and_gate()
#     or_neuron, _, y_or = train_or_gate()
    
#     # Create comparison visualization
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
#     # AND gate training progress
#     axes[0,0].plot(and_neuron.history['loss'])
#     axes[0,0].set_title('AND Gate - Training Loss')
#     axes[0,0].set_xlabel('Epoch')
#     axes[0,0].set_ylabel('Loss')
#     axes[0,0].grid(True, alpha=0.3)
    
#     # OR gate training progress
#     axes[0,1].plot(or_neuron.history['loss'])
#     axes[0,1].set_title('OR Gate - Training Loss')
#     axes[0,1].set_xlabel('Epoch')
#     axes[0,1].set_ylabel('Loss')
#     axes[0,1].grid(True, alpha=0.3)
    
#     # Decision boundaries comparison
#     h = 0.1
#     x_min, x_max = -0.2, 1.2
#     y_min, y_max = -0.2, 1.2
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))
#     mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
#     # AND gate decision boundary
#     Z_and = and_neuron.predict_proba(mesh_points).reshape(xx.shape)
#     axes[1,0].contourf(xx, yy, Z_and, levels=20, alpha=0.6, cmap='RdYlBu')
#     axes[1,0].contour(xx, yy, Z_and, levels=[0.5], colors='black', linewidths=2)
    
#     for class_value in [0, 1]:
#         mask = y_and == class_value
#         axes[1,0].scatter(X[mask, 0], X[mask, 1], 
#                          c=['red', 'blue'][class_value], 
#                          s=100, edgecolor='black')
#     axes[1,0].set_title('AND Gate Decision Boundary')
#     axes[1,0].set_xlabel('Input A')
#     axes[1,0].set_ylabel('Input B')
#     axes[1,0].grid(True, alpha=0.3)
    
#     # OR gate decision boundary  
#     Z_or = or_neuron.predict_proba(mesh_points).reshape(xx.shape)
#     axes[1,1].contourf(xx, yy, Z_or, levels=20, alpha=0.6, cmap='RdYlBu')
#     axes[1,1].contour(xx, yy, Z_or, levels=[0.5], colors='black', linewidths=2)
    
#     for class_value in [0, 1]:
#         mask = y_or == class_value
#         axes[1,1].scatter(X[mask, 0], X[mask, 1], 
#                          c=['red', 'blue'][class_value], 
#                          s=100, edgecolor='black')
#     axes[1,1].set_title('OR Gate Decision Boundary')
#     axes[1,1].set_xlabel('Input A')
#     axes[1,1].set_ylabel('Input B')
#     axes[1,1].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print weight comparison
#     print(f"\nWeight Comparison:")
#     print(f"AND Gate - Weight A: {and_neuron.weights[0]:.4f}, Weight B: {and_neuron.weights[1]:.4f}, Bias: {and_neuron.weights[2]:.4f}")
#     print(f"OR Gate  - Weight A: {or_neuron.weights[0]:.4f}, Weight B: {or_neuron.weights[1]:.4f}, Bias: {or_neuron.weights[2]:.4f}")

# def demonstrate_xor_limitation():
#     """
#     Show why XOR cannot be learned by a single neuron
#     """
#     print("\n❌ ATTEMPTING XOR GATE (WILL FAIL)")
#     print("=" * 40)
    
#     # XOR gate truth table
#     X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     y = np.array([0, 1, 1, 0])  # XOR: different inputs -> 1
    
#     print("XOR Truth Table:")
#     print("A | B | Output")
#     print("-" * 15)
#     for i in range(len(X)):
#         print(f"{X[i][0]} | {X[i][1]} |   {y[i]}")
    
#     # Try to train neuron on XOR (will fail)
#     neuron = SingleNeuron(input_size=2)
#     neuron.train(X, y, epochs=2000)
    
#     # Test results
#     print("\nXOR Gate Test Results (Expected to Fail):")
#     print("A | B | Expected | Predicted | Probability")
#     print("-" * 45)
    
#     for i in range(len(X)):
#         pred_proba = neuron.predict_proba(X[i])
#         pred_class = neuron.predict(X[i])
#         print(f"{X[i][0]} | {X[i][1]} |    {y[i]}     |     {pred_class}     |   {pred_proba:.4f}")
    
#     predictions = neuron.predict(X)
#     accuracy = accuracy_score(y, predictions)
#     print(f"\nXOR Gate Accuracy: {accuracy:.4f} (Should be ~0.5 - random guessing)")
#     print("📝 XOR is NOT linearly separable - needs multiple neurons!")
    
#     return neuron

def main():
    """
    Run all logical gate examples
    """
    print("SINGLE NEURON LOGICAL GATE TRAINING")
    print("Learning AND and OR gates with a single perceptron")
    
    # Train AND gate
    and_neuron, X_and, y_and = train_and_gate()
    print(f"\n📊 Visualizing AND gate training...")
    visualize_training_progress(and_neuron, 'AND')
    # visualize_decision_boundary(and_neuron, X_and, y_and, 'AND')
    
    # # Train OR gate  
    # or_neuron, X_or, y_or = train_or_gate()
    # print(f"\n📊 Visualizing OR gate training...")
    # visualize_training_progress(or_neuron, 'OR')
    # visualize_decision_boundary(or_neuron, X_or, y_or, 'OR')
    
    # # Compare both gates
    # compare_gates()
    
    # # Show XOR limitation
    # demonstrate_xor_limitation()
    
    # print("\n✅ ALL DEMONSTRATIONS COMPLETED!")


if __name__ == "__main__":
    main()