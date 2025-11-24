"""
CNN on MNIST Dataset - The "Hello World" of Computer Vision
===========================================================

The Hello World of Learning CNNs
MNIST = hand-written digits dataset : 0-9

Humans can see the patterns:
- "0" is circular
- "1" is a vertical line
- "7" has a horizontal top
- "8" has two loops

Your CNN learns the SAME patterns you see!
This makes debugging intuitive.

Learning Objectives:
- Understand CNN architecture (Conv2D, MaxPooling, Dense layers)
- Learn data preprocessing for images
- Visualize filters and activation maps
- Achieve 99%+ accuracy on digit recognition
- Understand overfitting prevention (dropout, batch normalization)

Dataset: MNIST = Modified National Institute of Standards and Technology
Man-written digits
- 60,000 training images (28 by 28 grayscale)
- 10,000 test images
- 10 classes (digits 0-9)

Filters in Conv2D:
Conv2D(32, (3,3)): 32 filters of size 3x3
Each filter detects specific patterns:
- Edges (horizontal, vertical, diagonal)
- Corners
- Textures
As training progresses, filters learn to recognize more complex features.
Filters are initially random, but get optimized via backpropagation,
which adjusts filter weights to minimize classification error.

How do filters specialize and not remain random or collapse to same?
----------------------------------------------
Because each filter has its own set of weights and biases,
the optimization process (backpropagation) adjusts them differently
based on the gradients computed from the loss function.
This leads to diverse filters that specialize in detecting different features.

Do we have filters that are known to detect specific features?
------------------------------------------------
Not initially. Filters start with random weights.
During training, they learn to detect features that help minimize the loss.
For example, some filters may learn to detect edges, while others may learn to detect curves or textures.

How to determine best filter size and number of filters and other hyperparameters?
------------------------------------------------
Typically through experimentation and validation.
Common choices:
- Filter sizes: 3x3 or 5x5
- Number of filters: Start with 32 or 64, then increase in deeper layers
- Use validation set to monitor performance and adjust hyperparameters accordingly.
- Depth of network: More layers can capture complex patterns but risk overfitting.

Ohhh . Networks learn filters ( we avoid enginering via filters )


"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))

from UTILS.CentralizedLogger import get_logger
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MNISTCNN:
    """
    Comprehensive CNN implementation for MNIST digit classification
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.logger = get_logger()
        
    def load_and_preprocess_mnist_data(self):
        """Load MNIST dataset and preprocess for CNN"""
        self.logger.info("Loading MNIST dataset...")
        print("Load MNIST Dataset")
        print("=" * 50)
        
        # Load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Label range: {y_train.min()} to {y_train.max()}")
        
        # Visualize samples
        self._visualize_samples(X_train, y_train, num_samples=10)
        
        # Reshape: Add channel dimension (28, 28) → (28, 28, 1)
        # Add channel dimension for grayscale images
        # 1 = grayscale channel : single channel
        # 3 = RGB channels : color image
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Normalize: [0, 255] → [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        print(f"\n✅ Preprocessed shape: {X_train.shape}")
        print(f"✅ Value range: [{X_train.min():.2f}, {X_train.max():.2f}]")
        
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        
        return X_train, y_train, X_test, y_test
    
    def _visualize_samples(self, X, y, num_samples=10):
        """Visualize random samples"""
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.ravel()
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(X))
            axes[i].imshow(X[idx], cmap='gray')
            axes[i].set_title(f'Label: {y[idx]}')
            axes[i].axis('off')
        
        plt.suptitle('MNIST Sample Images', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def build_simple_cnn(self):
        """
        Build simple CNN architecture
        
        Architecture:
        Input(28×28×1) → Conv2D(32,3×3) → MaxPool → Conv2D(64,3×3) → MaxPool → 
        Flatten → Dense(128) → Dense(10)
        """
        self.logger.info("Building simple CNN...")
        print("\n🏗️  BUILDING SIMPLE CNN")
        print("=" * 50)
        
        '''
        Why 32 filters?
        - 32 is a common starting point for the number of filters in CNNs.
        - It provides a good balance between model capacity and computational efficiency.
        - As you go deeper in the network, the number of filters is often increased
            (e.g., 64, 128) to capture more complex features.
        - Is the number of filters a hyperparameter to tune?
        - Yes, the number of filters is a hyperparameter that can be tuned based on the
          complexity of the dataset and the model's performance on validation data.
        - Experimenting with different values (e.g., 16, 32, 64, 128) can help find the optimal configuration.
        - Too few filters may lead to underfitting, while too many can cause overfitting and increased computational cost.

        Why 3x3 filter size?
        - 3x3 filters are widely used in CNNs because they are small enough to
          capture local patterns while being computationally efficient.

        Why MaxPooling
        - MaxPooling reduces spatial dimensions, lowering computational load
          and helping to make the model invariant to small translations in the input.
        - It also helps to summarize the presence of features in patches of the feature map.

        So we're learing a feature map of 32 features for each 3x3 region
        And then downsampling via maxpooling to shape dimensions of 14x14x32

        Overall :
        (A) REduction of spatial dimensions while increasing depth
        (B) Learning hierarchical feature representations
        (C) Increasing the channel count to capture more complex features

        '''
        model = models.Sequential([
            # Conv Block 1
            # Filters -> output channels => # of feature maps
            # Oh our inputs are (28, 28, 1) grayscale images
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            
            # Conv Block 2
            # INput share is now (14, 14, 32) BUT we don't need to specify input shape again
            # Becaue Keras can infer it from previous layer :-O !
            # NumParameters = 3*3*32*64 + 64 biases = 18496
            layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            
            # Input shape = (7, 7, 64) numParameters = 7*7*64 = 3136
            # Dense Layers because after conv layers we lose spatial info
            # But dense layers need 1D input
            # Hence Flatten operation which converts 3D tensor to 1D vector
            layers.Flatten(),

            # 128 neurons to learn complex combinations of features
            # Huge number of parameters
            # given input of 3136, we have to learn 3136*128 + 128 biases = 401536 parameters
            layers.Dense(128, activation='relu'),
            # and the learn the final 10 classes with parameters = 128*10 + 10 biases = 1290 parameers
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        self.model = model
        return model
    
    def build_advanced_cnn(self):
        """
        Build advanced CNN with dropout and batch normalization
        Achieves 99%+ accuracy
        """
        self.logger.info("Building advanced CNN...")
        print("\n🏗️  BUILDING ADVANCED CNN (Regularized)")
        print("=" * 50)
        
        model = models.Sequential([
            # Conv Block 1
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        self.model = model
        return model
    
    def train(self, epochs=5, batch_size=128, validation_split=0.1):
        """Train the CNN model"""
        self.logger.info(f"Training model for {epochs} epochs...")
        print(f"\n🚀 TRAINING MODEL")
        print("=" * 50)
        
        history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )
        
        self.history = history
        return history
    
    def evaluate(self):
        """Evaluate on test set"""
        print(f"\n📊 EVALUATING ON TEST SET")
        print("=" * 50)
        
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        self.logger.info(f"Test accuracy: {test_acc:.4f}")
        return test_loss, test_acc
    
    def plot_training_history(self):
        """Plot training curves"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, num_samples=10):
        """Show predictions vs ground truth"""
        predictions = self.model.predict(self.X_test[:num_samples], verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(self.X_test[i].reshape(28, 28), cmap='gray')
            
            true_label = self.y_test[i]
            pred_label = pred_labels[i]
            confidence = predictions[i][pred_label] * 100
            
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label} | Pred: {pred_label}\n({confidence:.1f}%)', 
                            color=color, fontsize=9)
            axes[i].axis('off')
        
        plt.suptitle('Predictions (Green=Correct, Red=Wrong)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        predictions = self.model.predict(self.X_test, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        
        cm = confusion_matrix(self.y_test, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
        
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, pred_labels))
    
    def visualize_filters(self, layer_name='conv1'):
        """Visualize learned convolutional filters"""
        layer = self.model.get_layer(layer_name)
        filters, biases = layer.get_weights()
        
        # Normalize
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        
        n_filters = min(32, filters.shape[3])
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(n_filters):
            axes[i].imshow(filters[:, :, 0, i], cmap='gray')
            axes[i].set_title(f'F{i}', fontsize=8)
            axes[i].axis('off')
        
        plt.suptitle(f'Learned Filters - {layer_name}', fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    print(f"Execute CNN against loaded dataset")
    print("=" * 60)
    
    # Initialize
    cnn = MNISTCNN()
    
    # Load data
    X_train, y_train, X_test, y_test = cnn.load_and_preprocess_mnist_data()
    
    # Build model (choose one)
    model = cnn.build_simple_cnn()  # Fast, ~98% accuracy
    # model = cnn.build_advanced_cnn()  # Slower, ~99%+ accuracy
    
    # Train
    history = cnn.train(epochs=5, batch_size=128)
    
    # Evaluate
    test_loss, test_acc = cnn.evaluate()
    
    # Visualizations
    cnn.plot_training_history()
    cnn.visualize_predictions(num_samples=10)
    cnn.plot_confusion_matrix()
    cnn.visualize_filters(layer_name='conv1')
    
    print("\n" + "=" * 60)
    print(f"🎉 TRAINING COMPLETE! Final Accuracy: {test_acc*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
