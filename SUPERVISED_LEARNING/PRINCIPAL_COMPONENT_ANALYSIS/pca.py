'''
Principal Component Analysis (PCA) on the Iris dataset
This script demonstrates how to perform PCA using scikit-learn
and visualize the results.

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
import seaborn as sns

# Make features look more "normal"
# Standardization is important before PCA


# Load Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
featureColumns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
targetColumn = ['species']
columns = featureColumns + targetColumn
iris = pd.read_csv(url, names=columns)

ROW = 0
COL = 1

# Prepare data for PCA
# Separate features and target
targetColumn = 'species'
targetColumnAxis = iris.columns.get_loc(targetColumn)
X = iris.drop(targetColumn, axis=1) # Features
numSamples, numFeatures = X.shape
print(f" For input data X : Number of samples: {numSamples}, Number of features: {numFeatures}")
y = iris[targetColumn] # Target

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
figureShape = (10, 8)
# alpha parameters controls transparency for better visibility
alpha = 0.7 
plt.figure(figsize=figureShape)
colors = ['red', 'blue', 'green']
for speciesIndex, species in enumerate(iris['species'].unique()):
    # What is mask? A boolean array to filter points by species
    # Mask, Y, and species set to same value beause we want to 
    # plot points of same species together
    # y is not used here ( but could be useful for legends or further analysis )
    mask = y == species
    speciesColor = colors[speciesIndex]
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=speciesColor, label=species, alpha=alpha)
    

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")