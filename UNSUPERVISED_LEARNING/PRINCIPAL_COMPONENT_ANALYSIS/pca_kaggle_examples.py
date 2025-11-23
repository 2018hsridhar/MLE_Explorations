"""
Easy PCA Examples with Kaggle Datasets
======================================

This script demonstrates Principal Component Analysis (PCA) using popular Kaggle datasets.
Perfect for beginners to understand dimensionality reduction.

Author: ML Engineer
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PCAExplorer:
    """
    A class to explore PCA with different datasets
    """
    
    def __init__(self):
        self.datasets = {}
        self.pca_results = {}
    
    def load_iris_example(self):
        """
        Example 1: Iris Dataset - Perfect for PCA beginners
        4 features -> 2 components visualization
        """
        print("=" * 50)
        print("EXAMPLE 1: IRIS DATASET")
        print("=" * 50)
        
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        print(f"Original shape: {X.shape}")
        print(f"Features: {feature_names}")
        print(f"Classes: {target_names}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Store results
        self.datasets['iris'] = {
            'X_original': X, 'X_scaled': X_scaled, 'X_pca': X_pca,
            'y': y, 'target_names': target_names, 'feature_names': feature_names
        }
        self.pca_results['iris'] = pca
        
        # Print PCA results
        print(f"\nPCA Results:")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
        print(f"Reduced shape: {X_pca.shape}")
        
        # Visualize
        self._plot_pca_2d(X_pca, y, target_names, 'Iris Dataset PCA', pca.explained_variance_ratio_)
        
        # Show feature importance in components
        self._plot_feature_importance(pca, feature_names, 'Iris Features in Principal Components')
        
        return X_pca, y
    
    def load_breast_cancer_example(self):
        """
        Example 2: Breast Cancer Dataset - High dimensional example
        30 features -> 2 components for visualization
        """
        print("\n" + "=" * 50)
        print("EXAMPLE 2: BREAST CANCER DATASET")
        print("=" * 50)
        
        # Load data
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        feature_names = cancer.feature_names
        target_names = cancer.target_names
        
        print(f"Original shape: {X.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Classes: {target_names}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA with different numbers of components
        # First, let's see how much variance we can explain
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Plot cumulative explained variance
        self._plot_cumulative_variance(pca_full.explained_variance_ratio_, 'Breast Cancer Dataset')
        
        # Apply PCA for visualization (2 components)
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        # Apply PCA for 95% variance
        n_components_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
        pca_95 = PCA(n_components=n_components_95)
        X_pca_95 = pca_95.fit_transform(X_scaled)
        
        print(f"\nPCA Results:")
        print(f"Components for 95% variance: {n_components_95}")
        print(f"2D PCA explained variance: {sum(pca_2d.explained_variance_ratio_):.2%}")
        print(f"95% variance PCA shape: {X_pca_95.shape}")
        
        # Store results
        self.datasets['cancer'] = {
            'X_original': X, 'X_scaled': X_scaled, 'X_pca_2d': X_pca_2d, 'X_pca_95': X_pca_95,
            'y': y, 'target_names': target_names, 'feature_names': feature_names
        }
        self.pca_results['cancer'] = {'pca_2d': pca_2d, 'pca_95': pca_95}
        
        # Visualize 2D PCA
        self._plot_pca_2d(X_pca_2d, y, target_names, 'Breast Cancer Dataset PCA (2D)', 
                         pca_2d.explained_variance_ratio_)
        
        # Compare classification performance
        self._compare_classification_performance(X_scaled, X_pca_95, y, 'Breast Cancer')
        
        return X_pca_2d, X_pca_95, y
    
    def load_digits_example(self):
        """
        Example 3: Digits Dataset - Very high dimensional example
        64 features (8x8 images) -> various reductions
        """
        print("\n" + "=" * 50)
        print("EXAMPLE 3: DIGITS DATASET")
        print("=" * 50)
        
        # Load data
        digits = load_digits()
        X, y = digits.data, digits.target
        
        print(f"Original shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Image shape: {digits.images[0].shape}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA with different numbers of components
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Plot cumulative explained variance
        self._plot_cumulative_variance(pca_full.explained_variance_ratio_, 'Digits Dataset')
        
        # Different PCA reductions
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        pca_10 = PCA(n_components=10)
        X_pca_10 = pca_10.fit_transform(X_scaled)
        
        n_components_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
        pca_95 = PCA(n_components=n_components_95)
        X_pca_95 = pca_95.fit_transform(X_scaled)
        
        print(f"\nPCA Results:")
        print(f"Components for 95% variance: {n_components_95}")
        print(f"2D PCA explained variance: {sum(pca_2d.explained_variance_ratio_):.2%}")
        print(f"10D PCA explained variance: {sum(pca_10.explained_variance_ratio_):.2%}")
        
        # Visualize 2D PCA
        self._plot_pca_2d_digits(X_pca_2d, y, 'Digits Dataset PCA (2D)', pca_2d.explained_variance_ratio_)
        
        # Visualize principal components as images
        self._plot_principal_components_images(pca_full, 'Digits Principal Components')
        
        # Compare classification performance
        self._compare_classification_performance_digits(X_scaled, X_pca_10, X_pca_95, y)
        
        return X_pca_2d, X_pca_10, X_pca_95, y
    
    def _plot_pca_2d(self, X_pca, y, target_names, title, variance_ratios):
        """Plot 2D PCA results"""
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(y))))
        for i, name in enumerate(target_names):
            mask = y == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=name, alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({variance_ratios[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({variance_ratios[1]:.2%} variance)')
        plt.title(f'{title}\nTotal Variance Explained: {sum(variance_ratios):.2%}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_pca_2d_digits(self, X_pca, y, title, variance_ratios):
        """Plot 2D PCA results for digits dataset"""
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for digit in range(10):
            mask = y == digit
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[digit]], label=f'Digit {digit}', alpha=0.6, s=30)
        
        plt.xlabel(f'PC1 ({variance_ratios[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({variance_ratios[1]:.2%} variance)')
        plt.title(f'{title}\nTotal Variance Explained: {sum(variance_ratios):.2%}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_importance(self, pca, feature_names, title):
        """Plot feature importance in principal components"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, ax in enumerate(axes):
            # Get component loadings
            loadings = pca.components_[i]
            
            # Create feature importance plot
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(loadings)
            }).sort_values('importance', ascending=True)
            
            ax.barh(range(len(feature_importance)), feature_importance['importance'])
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['feature'])
            ax.set_xlabel('Absolute Loading')
            ax.set_title(f'PC{i+1} Feature Importance')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def _plot_cumulative_variance(self, variance_ratios, dataset_name):
        """Plot cumulative explained variance"""
        cumsum_variance = np.cumsum(variance_ratios)
        
        plt.figure(figsize=(12, 6))
        
        # Individual variance
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(variance_ratios) + 1), variance_ratios)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title(f'{dataset_name}\nIndividual Explained Variance')
        plt.grid(True, alpha=0.3)
        
        # Cumulative variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        plt.axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title(f'{dataset_name}\nCumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print key statistics
        n_95 = np.argmax(cumsum_variance >= 0.95) + 1
        n_90 = np.argmax(cumsum_variance >= 0.90) + 1
        print(f"Components needed for 90% variance: {n_90}")
        print(f"Components needed for 95% variance: {n_95}")
    
    def _plot_principal_components_images(self, pca, title, n_components=6):
        """Plot principal components as images for digits dataset"""
        components = pca.components_[:n_components]
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for i, (component, ax) in enumerate(zip(components, axes.flat)):
            # Reshape component to 8x8 image
            component_image = component.reshape(8, 8)
            
            # Plot component
            im = ax.imshow(component_image, cmap='RdBu_r')
            ax.set_title(f'PC{i+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def _compare_classification_performance(self, X_original, X_pca, y, dataset_name):
        """Compare classification performance: original vs PCA"""
        print(f"\n{'-'*30}")
        print(f"CLASSIFICATION COMPARISON: {dataset_name}")
        print(f"{'-'*30}")
        
        # Split data
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            X_original, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train_pca, X_test_pca, _, _ = train_test_split(
            X_pca, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train classifiers
        rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
        
        rf_orig.fit(X_train_orig, y_train)
        rf_pca.fit(X_train_pca, y_train)
        
        # Make predictions
        y_pred_orig = rf_orig.predict(X_test_orig)
        y_pred_pca = rf_pca.predict(X_test_pca)
        
        # Print results
        print(f"Original features ({X_original.shape[1]}D): {accuracy_score(y_test, y_pred_orig):.4f}")
        print(f"PCA features ({X_pca.shape[1]}D): {accuracy_score(y_test, y_pred_pca):.4f}")
        print(f"Dimensionality reduction: {X_original.shape[1]} → {X_pca.shape[1]} ({X_pca.shape[1]/X_original.shape[1]*100:.1f}%)")
    
    def _compare_classification_performance_digits(self, X_original, X_pca_10, X_pca_95, y):
        """Compare classification performance for digits with multiple PCA reductions"""
        print(f"\n{'-'*30}")
        print(f"CLASSIFICATION COMPARISON: DIGITS")
        print(f"{'-'*30}")
        
        # Split data
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            X_original, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train_10, X_test_10, _, _ = train_test_split(
            X_pca_10, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train_95, X_test_95, _, _ = train_test_split(
            X_pca_95, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train classifiers
        rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_10 = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_95 = RandomForestClassifier(n_estimators=100, random_state=42)
        
        rf_orig.fit(X_train_orig, y_train)
        rf_10.fit(X_train_10, y_train)
        rf_95.fit(X_train_95, y_train)
        
        # Make predictions
        y_pred_orig = rf_orig.predict(X_test_orig)
        y_pred_10 = rf_10.predict(X_test_10)
        y_pred_95 = rf_95.predict(X_test_95)
        
        # Print results
        print(f"Original features (64D): {accuracy_score(y_test, y_pred_orig):.4f}")
        print(f"PCA 10 components: {accuracy_score(y_test, y_pred_10):.4f}")
        print(f"PCA 95% variance ({X_pca_95.shape[1]}D): {accuracy_score(y_test, y_pred_95):.4f}")

def main():
    """
    Run all PCA examples
    """
    print("🎯 PCA EXAMPLES WITH KAGGLE DATASETS")
    print("====================================")
    
    # Create PCA explorer
    explorer = PCAExplorer()
    
    # Run examples
    explorer.load_iris_example()
    explorer.load_breast_cancer_example()
    explorer.load_digits_example()
    
    print("\n✅ All examples completed!")
    print("\n📚 KEY TAKEAWAYS:")
    print("- PCA reduces dimensionality while preserving most variance")
    print("- Always standardize features before PCA")
    print("- 2 components are often enough for visualization")
    print("- Choose components based on desired variance retention")
    print("- PCA can maintain classification performance with fewer features")

if __name__ == "__main__":
    main()