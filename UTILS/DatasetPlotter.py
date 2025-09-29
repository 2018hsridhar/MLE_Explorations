from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DatasetPlotter:

    def __init__(self):
        self.SAMPLE_SIZE = 100
        self.RANDOM_STATE = 42

    def plot_logistic_regression_decision_boundary(self, X, y, model, feature_indices=[0, 1]):
        """
        Plots the decision boundary for a logistic regression model using two features.
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            model: Trained logistic regression model
            feature_indices: List of two feature indices to plot
        """
        # Select only the two features for plotting
        X_plot = X[:, feature_indices]
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

        # Create a mesh grid
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict probabilities for each point in the grid
        # Create a dummy array for unused features if needed
        if X.shape[1] > 2:
            grid_full = np.zeros((grid.shape[0], X.shape[1]))
            grid_full[:, feature_indices[0]] = grid[:, 0]
            grid_full[:, feature_indices[1]] = grid[:, 1]
        else:
            grid_full = grid

        probs = model.predict_proba(grid_full)[:, 1]
        probs = probs.reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=plt.cm.bwr, edgecolor='k', s=40)
        plt.xlabel(f'Feature {feature_indices[0]}')
        plt.ylabel(f'Feature {feature_indices[1]}')
        plt.title('Logistic Regression Decision Boundary')
        plt.show()

    def plotDataset(self, data, label_one=None, label_two=None):
        print(f"Entered method plotDataset()")
        # Use random sampling for better statistical analysis
        # RANDOM_STATE for reproducibility
        sampled_data = data.sample(n=self.SAMPLE_SIZE, random_state=self.RANDOM_STATE)


        # Perform any additional analysis or processing on the data here
        # For example, you could visualize the data or train a model

        # Visualize the data
        # Check if expected columns exist before plotting
        if (len(sampled_data.columns) > 1):
            print(f"We have {len(sampled_data.columns)} columns to work with.")
            # Use actual column names from the dataset
            x_col = sampled_data[label_one]  # Assuming first column is datetime
            y_col = sampled_data[label_two]     # Assuming second column is power consumption
            xLabel = label_one if label_one else x_col
            yLabel = label_two if label_two else y_col
            title = "Electric Power Consumption Over Time"

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(sampled_data[x_col], sampled_data[y_col], label=f'{y_col}')
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.title(title)
            plt.legend()
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.tight_layout()       # Adjust layout to prevent label cutoff
            plt.show()
        else:
            print("Dataset doesn't have enough columns for plotting")
        