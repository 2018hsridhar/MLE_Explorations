"""
Simple Neural Network Examples - From Scratch to Sklearn
=======================================================

This script demonstrates Multi-Layer Perceptrons (MLPs) with very simple examples,
perfect for understanding neural network fundamentals.

Starting from basic logic gates to small classification problems.

Author: ML Engineer
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, make_classification, make_circles
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleNeuralNetworkExamples:
    """
    A class to demonstrate simple neural network examples
    """
    
    def __init__(self):
        self.models = {}
        self.datasets = {}
        
    def example_1_logic_gates(self):
        """
        Example 1: Logic Gates (AND, OR, XOR)
        The simplest neural network examples
        """
        print("🔌 EXAMPLE 1: LOGIC GATES")
        print("=" * 40)
        
        # Define logic gate datasets
        gates = {
            'AND': {
                'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                'y': np.array([0, 0, 0, 1])
            },
            'OR': {
                'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                'y': np.array([0, 1, 1, 1])
            },
            'XOR': {
                'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                'y': np.array([0, 1, 1, 0])
            }
        }
        
        results = []
        
        # Test each gate
        for gate_name, data in gates.items():
            X, y = data['X'], data['y']
            
            print(f"\n🔍 Testing {gate_name} Gate:")
            print(f"   Input shape: {X.shape}")
            print(f"   Output shape: {y.shape}")
            
            # Try different network architectures
            architectures = [
                (1,),      # Single neuron (for AND/OR)
                (2,),      # 2 neurons in hidden layer
                (3,),      # 3 neurons in hidden layer  
                (2, 2),    # 2 hidden layers with 2 neurons each
            ]
            
            gate_results = []
            
            for arch in architectures:
                try:
                    # Create MLP
                    mlp = MLPClassifier(
                        hidden_layer_sizes=arch,
                        activation='relu',
                        solver='adam',
                        max_iter=1000,
                        random_state=42
                    )
                    
                    # Train (on all data since it's tiny)
                    mlp.fit(X, y)
                    
                    # Predict
                    y_pred = mlp.predict(X)
                    accuracy = accuracy_score(y, y_pred)
                    
                    gate_results.append({
                        'gate': gate_name,
                        'architecture': str(arch),
                        'accuracy': accuracy,
                        'converged': mlp.n_iter_ < 1000
                    })
                    
                    print(f"   {str(arch):15} -> Accuracy: {accuracy:.3f}")
                    
                except Exception as e:
                    print(f"   {str(arch):15} -> Failed: {e}")
            
            results.extend(gate_results)
            
            # Show truth table vs predictions for best model
            best_mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000, random_state=42)
            best_mlp.fit(X, y)
            y_pred = best_mlp.predict(X)
            
            print(f"\n   📋 {gate_name} Truth Table vs Predictions:")
            print("   Input1 | Input2 | Expected | Predicted")
            print("   -------|--------|----------|----------")
            for i in range(len(X)):
                print(f"      {X[i,0]}   |   {X[i,1]}    |    {y[i]}     |     {y_pred[i]}")
        
        # Summary results
        results_df = pd.DataFrame(results)
        print(f"\n📊 LOGIC GATES SUMMARY:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def example_2_simple_patterns(self):
        """
        Example 2: Simple Pattern Recognition
        Slightly larger but still very simple datasets
        """
        print("\n🎯 EXAMPLE 2: SIMPLE PATTERN RECOGNITION")
        print("=" * 45)
        
        # Create simple synthetic datasets
        np.random.seed(42)
        
        # Dataset 1: Linearly separable
        X_linear = np.random.randn(100, 2)
        y_linear = (X_linear[:, 0] + X_linear[:, 1] > 0).astype(int)
        
        # Dataset 2: Circular pattern
        X_circle, y_circle = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=42)
        
        # Dataset 3: Simple non-linear
        X_nonlinear = np.random.randn(100, 2)
        y_nonlinear = ((X_nonlinear[:, 0]**2 + X_nonlinear[:, 1]**2) > 1).astype(int)
        
        datasets = {
            'Linear Separable': (X_linear, y_linear),
            'Circular Pattern': (X_circle, y_circle),
            'Non-linear': (X_nonlinear, y_nonlinear)
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, (name, (X, y)) in enumerate(datasets.items()):
            # Plot original data
            axes[0, idx].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
            axes[0, idx].set_title(f'{name}\nOriginal Data')
            axes[0, idx].grid(True, alpha=0.3)
            
            # Train neural network
            mlp = MLPClassifier(
                hidden_layer_sizes=(5, 3),  # Small network
                activation='relu',
                max_iter=1000,
                random_state=42
            )
            
            mlp.fit(X, y)
            
            # Create decision boundary
            h = 0.1
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            axes[1, idx].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            axes[1, idx].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.8)
            axes[1, idx].set_title(f'{name}\nMLP Decision Boundary')
            axes[1, idx].grid(True, alpha=0.3)
            
            # Print accuracy
            accuracy = mlp.score(X, y)
            print(f"📊 {name}: Accuracy = {accuracy:.3f}")
        
        plt.tight_layout()
        plt.show()
    
    def example_3_iris_classification(self):
        """
        Example 3: Iris Classification
        Classic small dataset with multiple classes
        """
        print("\n🌸 EXAMPLE 3: IRIS CLASSIFICATION")
        print("=" * 40)
        
        # Load Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        print(f"📊 Dataset shape: {X.shape}")
        print(f"📊 Number of classes: {len(iris.target_names)}")
        print(f"📊 Classes: {iris.target_names}")
        print(f"📊 Features: {iris.feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try different architectures
        architectures = [
            (3,),           # Single hidden layer, 3 neurons
            (5,),           # Single hidden layer, 5 neurons
            (4, 3),         # Two hidden layers
            (10, 5),        # Larger network
            (5, 3, 2),      # Three hidden layers
        ]
        
        results = []
        
        print(f"\n🧠 TESTING DIFFERENT ARCHITECTURES:")
        print("Architecture    | Train Acc | Test Acc | Overfitting")
        print("----------------|-----------|----------|------------")
        
        for arch in architectures:
            # Create and train MLP
            mlp = MLPClassifier(
                hidden_layer_sizes=arch,
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
            
            mlp.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_acc = mlp.score(X_train_scaled, y_train)
            test_acc = mlp.score(X_test_scaled, y_test)
            overfitting = train_acc - test_acc
            
            results.append({
                'architecture': str(arch),
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'overfitting': overfitting,
                'n_iterations': mlp.n_iter_
            })
            
            print(f"{str(arch):15} | {train_acc:8.3f}  | {test_acc:7.3f}  | {overfitting:10.3f}")
        
        # Find best model
        results_df = pd.DataFrame(results)
        best_idx = results_df['test_accuracy'].idxmax()
        best_arch = architectures[best_idx]
        
        print(f"\n🏆 Best Architecture: {best_arch}")
        print(f"   Test Accuracy: {results_df.iloc[best_idx]['test_accuracy']:.3f}")
        
        # Train final model with best architecture
        final_mlp = MLPClassifier(
            hidden_layer_sizes=best_arch,
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        final_mlp.fit(X_train_scaled, y_train)
        y_pred = final_mlp.predict(X_test_scaled)
        
        # Detailed evaluation
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=iris.target_names,
                   yticklabels=iris.target_names)
        plt.title('Iris Classification - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Visualize training history (loss curve)
        plt.figure(figsize=(10, 6))
        plt.plot(final_mlp.loss_curve_)
        plt.title('Training Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        self.models['iris'] = final_mlp
        return final_mlp
    
    def example_4_simple_regression(self):
        """
        Example 4: Simple Regression Problem
        Neural network for regression instead of classification
        """
        print("\n📈 EXAMPLE 4: SIMPLE REGRESSION")
        print("=" * 35)
        
        # Generate simple regression data
        np.random.seed(42)
        X = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
        y = np.sin(X).ravel() + 0.1 * np.random.randn(100)
        
        print(f"📊 Dataset shape: {X.shape}")
        print(f"📊 Task: Learn sine function with noise")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Try different architectures
        architectures = [
            (3,),
            (5,),
            (10,),
            (5, 3),
            (10, 5),
        ]
        
        plt.figure(figsize=(15, 10))
        
        for idx, arch in enumerate(architectures):
            # Create and train MLP regressor
            mlp = MLPRegressor(
                hidden_layer_sizes=arch,
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
            
            mlp.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = mlp.predict(X_test)
            
            # Predict on full range for plotting
            X_full = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
            y_full_pred = mlp.predict(X_full)
            
            # Plot results
            plt.subplot(2, 3, idx + 1)
            plt.scatter(X_train, y_train, alpha=0.6, label='Train', s=20)
            plt.scatter(X_test, y_test, alpha=0.6, label='Test', s=20)
            plt.plot(X_full, y_full_pred, 'r-', label='MLP Prediction', linewidth=2)
            plt.plot(X_full, np.sin(X_full), 'g--', label='True Function', alpha=0.7)
            plt.title(f'Architecture: {arch}\nMSE: {mlp.score(X_test, y_test):.3f}')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def example_5_from_scratch_simple(self):
        """
        Example 5: Tiny Neural Network From Scratch
        Understanding the basics with minimal code
        """
        print("\n🔧 EXAMPLE 5: FROM SCRATCH - XOR PROBLEM")
        print("=" * 45)
        
        class SimpleNeuralNetwork:
            def __init__(self):
                # Initialize weights randomly
                np.random.seed(42)
                self.W1 = np.random.uniform(-1, 1, (2, 2))  # Input to hidden
                self.b1 = np.zeros((1, 2))                  # Hidden bias
                self.W2 = np.random.uniform(-1, 1, (2, 1))  # Hidden to output
                self.b2 = np.zeros((1, 1))                  # Output bias
                
            def sigmoid(self, x):
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            
            def forward(self, X):
                # Forward pass
                self.z1 = np.dot(X, self.W1) + self.b1
                self.a1 = self.sigmoid(self.z1)
                self.z2 = np.dot(self.a1, self.W2) + self.b2
                self.a2 = self.sigmoid(self.z2)
                return self.a2
            
            def backward(self, X, y, learning_rate=1.0):
                m = X.shape[0]
                
                # Backward pass
                dz2 = self.a2 - y.reshape(-1, 1)
                dW2 = (1/m) * np.dot(self.a1.T, dz2)
                db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
                
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self.a1 * (1 - self.a1)
                dW1 = (1/m) * np.dot(X.T, dz1)
                db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
                
                # Update weights
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
            
            def train(self, X, y, epochs=1000):
                losses = []
                for epoch in range(epochs):
                    # Forward pass
                    output = self.forward(X)
                    
                    # Calculate loss
                    loss = np.mean((output.flatten() - y)**2)
                    losses.append(loss)
                    
                    # Backward pass
                    self.backward(X, y)
                    
                    if epoch % 200 == 0:
                        print(f"Epoch {epoch:4d}, Loss: {loss:.6f}")
                
                return losses
            
            def predict(self, X):
                output = self.forward(X)
                return (output > 0.5).astype(int).flatten()
        
        # XOR data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        
        print("📊 Training Neural Network on XOR Problem...")
        print("   Input shape:", X.shape)
        print("   Output shape:", y.shape)
        print("   Architecture: 2 -> 2 -> 1 (2 hidden neurons)")
        
        # Create and train network
        nn = SimpleNeuralNetwork()
        losses = nn.train(X, y, epochs=1000)
        
        # Test the network
        predictions = nn.predict(X)
        
        print(f"\n📋 XOR RESULTS:")
        print("Input | Expected | Predicted | Probability")
        print("------|----------|-----------|------------")
        probabilities = nn.forward(X).flatten()
        for i in range(len(X)):
            prob = probabilities[i]
            print(f" {X[i]} |    {y[i]}     |     {predictions[i]}     |   {prob:.3f}")
        
        accuracy = np.mean(predictions == y)
        print(f"\n✅ Final Accuracy: {accuracy:.3f}")
        
        # Plot training curve
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot decision boundary
        plt.subplot(1, 2, 2)
        h = 0.1
        xx, yy = np.meshgrid(np.arange(-0.5, 1.5, h),
                           np.arange(-0.5, 1.5, h))
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = nn.forward(mesh_points).reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, edgecolors='black')
        plt.title('XOR Decision Boundary')
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        
        for i in range(len(X)):
            plt.annotate(f'({X[i,0]},{X[i,1]})', (X[i,0], X[i,1]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def compare_architectures_summary(self):
        """
        Summary comparison of different architectures
        """
        print("\n📚 NEURAL NETWORK ARCHITECTURE GUIDE")
        print("=" * 45)
        
        guide = """
🧠 ARCHITECTURE GUIDELINES:

📏 SIZE RECOMMENDATIONS:
• Logic Gates (XOR, AND, OR): (2,) or (3,)
• Small Classification: (5,) to (10,)
• Iris-like problems: (5, 3) to (10, 5)
• Simple Regression: (10,) to (20, 10)

🎯 LAYER DEPTH:
• 1 Hidden Layer: Most problems
• 2 Hidden Layers: Complex patterns
• 3+ Hidden Layers: Usually overkill for small data

⚙️ ACTIVATION FUNCTIONS:
• ReLU: Default choice (works well)
• Sigmoid: Binary classification output
• Tanh: Sometimes better than sigmoid
• Linear: Regression output layer

🔧 HYPERPARAMETERS:
• Learning Rate: 0.001 to 0.01 (Adam solver)
• Max Iterations: 1000 to 5000
• Solver: 'adam' (adaptive), 'lbfgs' (small data)

🚫 COMMON MISTAKES:
• Too many neurons (overfitting)
• Not scaling input features
• Wrong activation for output layer
• Too few iterations for convergence

✅ BEST PRACTICES:
• Start simple, add complexity gradually
• Always scale your features
• Monitor train vs test accuracy
• Use early stopping to prevent overfitting
"""
        print(guide)

def main():
    """
    Run all simple neural network examples
    """
    print("🧠 SIMPLE NEURAL NETWORK EXAMPLES")
    print("=" * 40)
    print("Learning neural networks with tiny datasets!")
    
    examples = SimpleNeuralNetworkExamples()
    
    # Run all examples
    examples.example_1_logic_gates()
    examples.example_2_simple_patterns()
    examples.example_3_iris_classification()
    examples.example_4_simple_regression()
    examples.example_5_from_scratch_simple()
    examples.compare_architectures_summary()
    
    print("\n✅ ALL EXAMPLES COMPLETED!")
    print("\n🎯 KEY TAKEAWAYS:")
    print("• Start with logic gates to understand basics")
    print("• XOR problem requires at least 1 hidden layer")
    print("• More neurons ≠ better performance (overfitting)")
    print("• Always scale your input features")
    print("• Monitor training vs test accuracy")
    print("• Relu activation works well for hidden layers")

if __name__ == "__main__":
    main()