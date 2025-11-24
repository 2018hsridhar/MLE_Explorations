"""
Minimal FCNN (Fully Connected Neural Network) on XOR Problem
============================================================

WHY XOR?
The XOR problem is the "Hello World" of neural networks because:
- It's NOT linearly separable (can't solve with single perceptron)
- Requires at least 1 hidden layer
- Proves neural networks can learn non-linear decision boundaries
- Simple enough to understand every step

XOR Truth Table:
Input1 | Input2 | Output
-------|--------|-------
  0    |   0    |   0
  0    |   1    |   1
  1    |   0    |   1
  1    |   1    |   0

Architecture: FCNN without Conv2D, MaxPool, or Flatten
Input(2) → Dense(2, relu) → Dense(1, sigmoid)

This is the FOUNDATION before CNNs!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from UTILS.CentralizedLogger import get_logger
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
import warnings
warnings.filterwarnings('ignore')


class MinimalFCNN:
    """Minimal Fully Connected Neural Network for XOR"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.logger = get_logger()
        
    def create_xor_data(self):
        """Create XOR dataset"""
        print("\n📊 XOR DATASET")
        print("=" * 50)
        
        # XOR truth table
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], dtype=np.float32)
        
        y = np.array([0, 1, 1, 0], dtype=np.float32)
        
        print("XOR Truth Table:")
        print("Input1 | Input2 | Output")
        print("-------|--------|-------")
        for i in range(len(X)):
            print(f"  {int(X[i,0])}    |   {int(X[i,1])}    |   {int(y[i])}")
        
        return X, y
    
    def build_fcnn(self):
        """
        Build FCNN - NO Conv2D, NO MaxPool, NO Flatten!
        Just Dense layers (fully connected)
        """
        print("\n🏗️  BUILDING FCNN")
        print("=" * 50)
        
        model = models.Sequential([
            # Input layer: 2 features (x1, x2)
            layers.Dense(2, activation='relu', input_shape=(2,), name='hidden'),
            
            # Output layer: 1 output (0 or 1)
            layers.Dense(1, activation='sigmoid', name='output')
        ], name='minimal_fcnn')
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        print("\n🔍 ARCHITECTURE BREAKDOWN:")
        print(f"  Layer 1 (Dense): 2 inputs → 2 neurons (ReLU)")
        print(f"  Layer 2 (Dense): 2 neurons → 1 output (Sigmoid)")
        print(f"  Total params: (2×2 + 2) + (2×1 + 1) = 9")
        
        self.model = model
        return model
    
    def train(self, X, y, epochs=1000, verbose=False):
        """Train FCNN on XOR"""
        print(f"\n🚀 TRAINING FCNN")
        print("=" * 50)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            verbose=0  # Silent training
        )
        
        self.history = history
        
        # Print progress every 200 epochs
        print("\nTraining Progress:")
        for i in range(0, epochs, 200):
            if i < len(history.history['loss']):
                loss = history.history['loss'][i]
                acc = history.history['accuracy'][i]
                print(f"Epoch {i:4d}: Loss={loss:.4f}, Accuracy={acc:.4f}")
        
        # Final epoch
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        print(f"Epoch {epochs:4d}: Loss={final_loss:.4f}, Accuracy={final_acc:.4f}")
        
        return history
    
    def evaluate(self, X, y):
        """Test predictions"""
        print(f"\n📊 EVALUATION")
        print("=" * 50)
        
        predictions = self.model.predict(X, verbose=0)
        pred_binary = (predictions > 0.5).astype(int).flatten()
        
        print("\nPredictions vs Ground Truth:")
        print("Input1 | Input2 | True | Predicted | Confidence")
        print("-------|--------|------|-----------|------------")
        for i in range(len(X)):
            conf = predictions[i][0] * 100
            symbol = "✅" if pred_binary[i] == y[i] else "❌"
            print(f"  {int(X[i,0])}    |   {int(X[i,1])}    |  {int(y[i])}   |     {pred_binary[i]}     | {conf:5.1f}%  {symbol}")
        
        accuracy = np.mean(pred_binary == y)
        print(f"\nAccuracy: {accuracy*100:.1f}%")
        
        return predictions
    
    def visualize_decision_boundary(self, X, y):
        """Visualize how FCNN learned XOR"""
        print("\n📈 VISUALIZING DECISION BOUNDARY")
        print("=" * 50)
        
        # Create mesh grid
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Predict on mesh
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Decision boundary
        plt.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.8)
        plt.colorbar(label='Prediction')
        
        # Decision threshold line
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
        
        # Plot XOR points
        colors = ['red', 'green', 'green', 'red']
        for i in range(len(X)):
            plt.scatter(X[i, 0], X[i, 1], 
                       c=colors[i], 
                       s=500, 
                       edgecolor='black', 
                       linewidth=3,
                       marker='o' if y[i] == 0 else 's',
                       label=f'({int(X[i,0])},{int(X[i,1])})→{int(y[i])}')
        
        plt.xlabel('Input 1', fontsize=14)
        plt.ylabel('Input 2', fontsize=14)
        plt.title('FCNN Decision Boundary for XOR\n(Green=1, Red=0)', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_training(self):
        """Plot training curves"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], linewidth=2)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], linewidth=2, color='green')
        ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def inspect_weights(self):
        """Show what the network learned"""
        print("\n🔍 INSPECTING LEARNED WEIGHTS")
        print("=" * 50)
        
        # Hidden layer
        W1, b1 = self.model.get_layer('hidden').get_weights()
        print("\nHidden Layer Weights (Input → Hidden):")
        print("W1 =")
        print(W1)
        print(f"b1 = {b1}")
        
        # Output layer
        W2, b2 = self.model.get_layer('output').get_weights()
        print("\nOutput Layer Weights (Hidden → Output):")
        print("W2 =")
        print(W2)
        print(f"b2 = {b2}")
        
        print("\n💡 INTERPRETATION:")
        print("The network learned to:")
        print("1. Hidden layer creates non-linear features")
        print("2. Output layer combines features to solve XOR")


def main():
    """Run minimal FCNN on XOR"""
    print("🧠 MINIMAL FCNN ON XOR PROBLEM")
    print("=" * 60)
    print("Why XOR? It's NOT linearly separable!")
    print("Single perceptron CANNOT solve it.")
    print("FCNN with hidden layer CAN solve it!")
    print("=" * 60)
    
    # Initialize
    fcnn = MinimalFCNN()
    
    # Create data
    X, y = fcnn.create_xor_data()
    
    # Build FCNN
    model = fcnn.build_fcnn()
    
    # Train
    history = fcnn.train(X, y, epochs=1000)
    
    # Evaluate
    predictions = fcnn.evaluate(X, y)
    
    # Visualizations
    fcnn.visualize_training()
    fcnn.visualize_decision_boundary(X, y)
    fcnn.inspect_weights()
    
    print("\n" + "=" * 60)
    print("🎉 FCNN SUCCESSFULLY LEARNED XOR!")
    print("=" * 60)
    print("\n🎓 KEY LESSONS:")
    print("✅ FCNNs use only Dense layers (no Conv2D, MaxPool, Flatten)")
    print("✅ XOR proves you need hidden layers for non-linear problems")
    print("✅ 2 hidden neurons are enough for XOR")
    print("✅ This is the FOUNDATION before CNNs")
    print("\n🚀 NEXT STEPS:")
    print("→ Try MNIST with FCNN (flatten 28×28 → 784 inputs)")
    print("→ Then compare FCNN vs CNN on MNIST")
    print("→ See why CNNs are better for images!")


if __name__ == "__main__":
    main()
