'''
Principal Component Analysis (PCA) on the Iris dataset
This script demonstrates how to perform PCA using scikit-learn
and visualize the results.

PCA is unsupervised learning, so we 
do not use the target labels during PCA fitting.

# Dim reduction based on variance maximization
# Not on class separability or accuracy

Why Iris Species Dataset chosen?
- Well-known dataset for classification tasks.
- Contains 4 features and 3 classes, making it suitable for PCA.
- Helps illustrate dimensionality reduction and variance explanation.

Why focus on PCA?
- PCA is a fundamental technique for dimensionality reduction.
- It helps in visualizing high-dimensional data.
- PCA identifies the directions (principal components) that maximize variance.

#-of-target dimensions for PCA
- We reduce to 2 dimensions for easy visualization.

'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import seaborn as sns

# Make features look more "normal"
# Standardization is important before PCA


# Load Iris dataset with error handling and consistent variable types
# Ientified fallback mechanisms for robust loading
def load_iris_dataset():
    """
    Load Iris dataset with multiple fallback options
    Returns: pandas DataFrame with iris data
    """
    # Method 1: Try UCI repository
    # Why UCI repository? Because it's a classic source for datasets
    # and often used in academic settings.
    # But it can be down or unreachable sometimes - hence, the fallback to sklearn.
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target_column = 'species'  # Keep as string consistently
        columns = feature_columns + [target_column]  # Fix: use single string in list
        
        iris = pd.read_csv(url, names=columns)
        print("Successfully loaded Iris dataset from UCI repository")
        return iris
        
    except Exception as e:
        print(f"Failed to load from UCI: {e}")
        
        # Method 2: Fallback to sklearn dataset
        try:
            from sklearn.datasets import load_iris
            print("Loading Iris dataset from sklearn...")
            
            iris_sklearn = load_iris()
            iris = pd.DataFrame(iris_sklearn.data, columns=iris_sklearn.feature_names)
            iris['species'] = iris_sklearn.target
            
            # Map numeric targets to species names
            species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
            iris['species'] = iris['species'].map(species_mapping)
            
            # Standardize column names to match UCI format
            iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
            
            print("Successfully loaded Iris dataset from sklearn")
            return iris
            
        except Exception as e2:
            print(f"Failed to load from sklearn: {e2}")
            raise Exception("Could not load Iris dataset from either UCI or sklearn")

# Load the dataset
iris = load_iris_dataset()

# Verify data quality
print(f"Dataset shape: {iris.shape}")
print(f"Column names: {list(iris.columns)}")
print(f"Species counts:\n{iris['species'].value_counts()}")
print(f"Missing values: {iris.isnull().sum().sum()}")

ROW = 0
COL = 1

# Prepare data for PCA
# Separate features and target
TARGET_COLUMN = 'species'  # Use consistent naming convention
X = iris.drop(TARGET_COLUMN, axis=1)  # Features
numSamples, numFeatures = X.shape
print(f"For input data X: Number of samples: {numSamples}, Number of features: {numFeatures}")
y = iris[TARGET_COLUMN]  # Target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
# Linear dimensionality reduction leveraging SVD : Singular Value Decomposition
numComponents = 2  # Reduce to 2 dimensions for visualization
print(f"Applying PCA to reduce from {numFeatures} features to {numComponents} dimensions.")
pca = PCA(n_components=numComponents)
X_pca = pca.fit_transform(X_scaled)

# Visualize results
figure_shape = (10, 8)
alpha = 0.7  # Controls transparency for better visibility

plt.figure(figsize=figure_shape)
colors = ['red', 'blue', 'green']

# Get unique species and ensure they're sorted for consistency
# Why sort vs unsorted? To ensure consistent color mapping across runs
unique_species = sorted(iris['species'].unique())
print(f"Unique species found: {unique_species}")

for species_index, species in enumerate(unique_species):
    # Create boolean mask to filter points by species
    mask = y == species
    
    # Safety check
    if np.sum(mask) == 0:
        print(f"Warning: No data found for species {species}. Please check dataset integrity.")
        continue
        
    species_color = colors[species_index]
    
    # Plot points for this species
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=species_color, label=species, alpha=alpha)
    
    print(f"Plotted {np.sum(mask)} points for {species}")

# Add labels and title
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
    

# Explained_variance ratio shows variance captured by each PC
# Higher variance means more information retained
# We want to maximize variance in reduced dimensions
# This helps in understanding how well PCA has captured the data structure
# Since it corresponds to information content

# Explained variance ratio = eigenvalues / total variance
# Each principal component has an associated eigenvalue
# Eigenvalue indicates the amount of variance captured by that component
# Total variance is the sum of all eigenvalues

# Explained variance ratio = eigenvalue of PC / sum of all eigenvalues
# Mathematical formula:
# For principal component i:
# explained_variance_ratio[i] = λᵢ / Σ(λⱼ) where j = 1 to n
# 
# Where:
# λᵢ = eigenvalue of principal component i
# Σ(λⱼ) = sum of all eigenvalues = total variance in the dataset
# n = total number of features/components
#
# Example calculation:
# If eigenvalues = [2.5, 1.8, 0.4, 0.3]
# Total variance = 2.5 + 1.8 + 0.4 + 0.3 = 5.0
# explained_variance_ratio[0] = 2.5/5.0 = 0.50 (50%)
# explained_variance_ratio[1] = 1.8/5.0 = 0.36 (36%)
# explained_variance_ratio[2] = 0.4/5.0 = 0.08 (8%)
# explained_variance_ratio[3] = 0.3/5.0 = 0.06 (6%)

explained_variance_ratio = pca.explained_variance_ratio_
variance_explained = pca.explained_variance_ratio_
first_pc_variance = variance_explained[0]
second_pc_variance = variance_explained[1]
VARIANCE_THRESHOLD = 0.95  # 95% variance threshold
total_variance_explained = sum(explained_variance_ratio)
if total_variance_explained >= VARIANCE_THRESHOLD:
    print(f"PCA retained {total_variance_explained:.2%} variance, which meets the threshold of {VARIANCE_THRESHOLD:.2%}.")
else:
    print(f"PCA retained {total_variance_explained:.2%} variance, which is below the threshold of {VARIANCE_THRESHOLD:.2%}.")
    
plt.title('PCA of Iris Dataset')
plt.legend()
plt.show()

print(f"Variance explained by each principal component: {variance_explained}")
print(f"Variance explained by PC1: {first_pc_variance:.2%}")
print(f"Variance explained by PC2: {second_pc_variance:.2%}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")