"""
Decision Tree Examples with Small Kaggle Datasets
================================================

This script demonstrates decision tree classification and regression
using popular small Kaggle datasets. Perfect for understanding node thresholds,
feature importance, and tree visualization.

Author: ML Engineer
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
import warnings
warnings.filterwarnings('ignore')

'''
Overarching Goals :
1. Set up a basic decision tree classifier : binary -> then multi-class.
2. Explore decision tree modeling : number of levels, questions asked at each step, thresholds to set.
3. Investing iteratively improving the decision tree structure.

#TODO : Implement a basic decision tree
'''


class DecisionTreeExplorer:
    """
    A class to explore decision trees with different datasets and parameters
    """
    
    def __init__(self):
        self.datasets = {}
        self.trees = {}
        
    def load_titanic_dataset(self):
        """
        Load Titanic dataset - Perfect for classification and understanding splits
        """
        print("🚢 TITANIC DATASET ANALYSIS")
        print("=" * 40)
        
        try:
            # Try loading from local file or create sample data
            # For demo purposes, let's create a simplified Titanic-like dataset
            np.random.seed(42)
            n_samples = 500
            
            # Create realistic Titanic-like features
            ages = np.random.normal(35, 12, n_samples)
            ages = np.clip(ages, 1, 80)  # Reasonable age range
            
            fares = np.random.lognormal(3, 1, n_samples)
            fares = np.clip(fares, 0, 500)  # Reasonable fare range
            
            pclass = np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.25, 0.5])
            sex = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])  # 0=female, 1=male
            sibsp = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03])
            parch = np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.15, 0.1, 0.05])
            
            # Create survival based on realistic patterns
            survival_prob = (
                0.1 +  # Base survival rate
                0.4 * (sex == 0) +  # Women more likely to survive
                0.3 * (pclass == 1) + 0.1 * (pclass == 2) +  # Higher class more likely
                0.2 * (ages < 15) +  # Children more likely
                -0.1 * (ages > 60) +  # Elderly less likely
                0.1 * (fares > 50)  # Expensive tickets more likely
            )
            survival_prob = np.clip(survival_prob, 0, 1)
            survived = np.random.binomial(1, survival_prob)
            
            # Create DataFrame
            titanic = pd.DataFrame({
                'age': ages,
                'fare': fares,
                'pclass': pclass,
                'sex': sex,  # 0=female, 1=male
                'sibsp': sibsp,
                'parch': parch,
                'survived': survived
            })
            
            print(f"📊 Dataset shape: {titanic.shape}")
            print(f"📊 Survival rate: {titanic['survived'].mean():.2%}")
            print(f"📊 Features: {list(titanic.columns[:-1])}")
            
            self.datasets['titanic'] = titanic
            return titanic
            
        except Exception as e:
            print(f"Error loading Titanic dataset: {e}")
            return None
    
    def load_iris_dataset(self):
        """
        Load Iris dataset - Perfect for multi-class classification
        """
        print("\n🌸 IRIS DATASET ANALYSIS")
        print("=" * 40)
        
        iris_sklearn = load_iris()
        iris = pd.DataFrame(iris_sklearn.data, columns=iris_sklearn.feature_names)
        iris['species'] = iris_sklearn.target
        
        print(f"📊 Dataset shape: {iris.shape}")
        print(f"📊 Classes: {iris_sklearn.target_names}")
        print(f"📊 Features: {list(iris.columns[:-1])}")
        
        self.datasets['iris'] = iris
        return iris
    
    def load_wine_dataset(self):
        """
        Load Wine dataset - Good for multi-class with many features
        """
        print("\n🍷 WINE DATASET ANALYSIS")
        print("=" * 40)
        
        wine_sklearn = load_wine()
        wine = pd.DataFrame(wine_sklearn.data, columns=wine_sklearn.feature_names)
        wine['wine_class'] = wine_sklearn.target
        
        # Select most important features for simplicity
        important_features = ['alcohol', 'flavanoids', 'color_intensity', 'proline']
        wine_simple = wine[important_features + ['wine_class']].copy()
        
        print(f"📊 Dataset shape: {wine_simple.shape}")
        print(f"📊 Classes: {wine_sklearn.target_names}")
        print(f"📊 Selected features: {important_features}")
        
        self.datasets['wine'] = wine_simple
        return wine_simple
    
    def explore_titanic_decision_tree(self):
        """
        Build and analyze decision tree for Titanic dataset
        """
        titanic = self.datasets['titanic']
        
        print("\n🌳 TITANIC DECISION TREE ANALYSIS")
        print("=" * 45)
        
        # Prepare features and target
        X = titanic.drop('survived', axis=1)
        y = titanic['survived']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Try different tree depths to understand node thresholds
        depths = [3, 5, 7, 10, None]
        results = []
        
        plt.figure(figsize=(15, 10))
        
        for i, depth in enumerate(depths):
            # Create decision tree
            dt = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=20,  # Minimum samples to split a node
                min_samples_leaf=10,   # Minimum samples in a leaf
                random_state=42
            )
            
            # Train the tree
            dt.fit(X_train, y_train)
            
            # Make predictions
            y_pred = dt.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results.append({
                'max_depth': depth,
                'accuracy': accuracy,
                'n_nodes': dt.tree_.node_count,
                'max_depth_actual': dt.tree_.max_depth
            })
            
            # Plot tree (first 3 only to save space)
            if i < 3:
                plt.subplot(2, 2, i + 1)
                plot_tree(dt, 
                         feature_names=X.columns,
                         class_names=['Died', 'Survived'],
                         filled=True,
                         rounded=True,
                         fontsize=8)
                plt.title(f'Decision Tree (max_depth={depth})\nAccuracy: {accuracy:.3f}')
        
        plt.tight_layout()
        plt.show()
        
        # Print results table
        results_df = pd.DataFrame(results)
        print("\n📊 DECISION TREE PERFORMANCE BY DEPTH:")
        print(results_df.to_string(index=False))
        
        # Feature importance analysis
        best_dt = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
        best_dt.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_dt.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 FEATURE IMPORTANCE (max_depth=5):")
        print(feature_importance.to_string(index=False))
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance in Titanic Survival Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        # Store the best tree
        self.trees['titanic'] = best_dt
        
        # Print tree rules
        print(f"\n📋 DECISION TREE RULES (first 20 lines):")
        tree_rules = export_text(best_dt, feature_names=list(X.columns))
        print('\n'.join(tree_rules.split('\n')[:20]))
        
        return best_dt
    
    def explore_iris_decision_tree(self):
        """
        Build and analyze decision tree for Iris dataset
        """
        iris = self.datasets['iris']
        
        print("\n🌸 IRIS DECISION TREE ANALYSIS")
        print("=" * 40)
        
        # Prepare features and target
        X = iris.drop('species', axis=1)
        y = iris['species']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Create decision tree
        dt = DecisionTreeClassifier(
            max_depth=3,  # Keep small for visualization
            min_samples_split=5,
            random_state=42
        )
        
        # Train the tree
        dt.fit(X_train, y_train)
        
        # Evaluate
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 Test Accuracy: {accuracy:.3f}")
        print(f"📊 Tree nodes: {dt.tree_.node_count}")
        print(f"📊 Tree depth: {dt.tree_.max_depth}")
        
        # Visualize tree
        plt.figure(figsize=(15, 10))
        plot_tree(dt,
                 feature_names=X.columns,
                 class_names=['Setosa', 'Versicolor', 'Virginica'],
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title(f'Iris Decision Tree (Accuracy: {accuracy:.3f})')
        plt.show()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': dt.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 FEATURE IMPORTANCE:")
        print(feature_importance.to_string(index=False))
        
        self.trees['iris'] = dt
        return dt
    
    def experiment_with_node_thresholds(self):
        """
        Experiment with different node threshold parameters
        """
        titanic = self.datasets['titanic']
        
        print("\n🔬 EXPERIMENTING WITH NODE THRESHOLDS")
        print("=" * 45)
        
        X = titanic.drop('survived', axis=1)
        y = titanic['survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Parameters to experiment with
        params = {
            'min_samples_split': [2, 10, 20, 50],
            'min_samples_leaf': [1, 5, 10, 20],
            'max_depth': [3, 5, 7, 10]
        }
        
        results = []
        
        for min_split in params['min_samples_split']:
            for min_leaf in params['min_samples_leaf']:
                for max_depth in params['max_depth']:
                    
                    dt = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=42
                    )
                    
                    dt.fit(X_train, y_train)
                    
                    train_acc = dt.score(X_train, y_train)
                    test_acc = dt.score(X_test, y_test)
                    
                    results.append({
                        'min_samples_split': min_split,
                        'min_samples_leaf': min_leaf,
                        'max_depth': max_depth,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'overfitting': train_acc - test_acc,
                        'n_nodes': dt.tree_.node_count
                    })
        
        # Convert to DataFrame and analyze
        results_df = pd.DataFrame(results)
        
        # Find best parameters
        best_idx = results_df['test_accuracy'].idxmax()
        best_params = results_df.iloc[best_idx]
        
        print(f"🏆 BEST PARAMETERS:")
        print(f"   max_depth: {best_params['max_depth']}")
        print(f"   min_samples_split: {best_params['min_samples_split']}")
        print(f"   min_samples_leaf: {best_params['min_samples_leaf']}")
        print(f"   Test Accuracy: {best_params['test_accuracy']:.3f}")
        print(f"   Overfitting: {best_params['overfitting']:.3f}")
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test accuracy by max_depth
        depth_acc = results_df.groupby('max_depth')['test_accuracy'].mean()
        axes[0,0].plot(depth_acc.index, depth_acc.values, 'bo-')
        axes[0,0].set_title('Test Accuracy vs Max Depth')
        axes[0,0].set_xlabel('Max Depth')
        axes[0,0].set_ylabel('Test Accuracy')
        axes[0,0].grid(True)
        
        # Overfitting by max_depth
        depth_overfit = results_df.groupby('max_depth')['overfitting'].mean()
        axes[0,1].plot(depth_overfit.index, depth_overfit.values, 'ro-')
        axes[0,1].set_title('Overfitting vs Max Depth')
        axes[0,1].set_xlabel('Max Depth')
        axes[0,1].set_ylabel('Overfitting (Train - Test)')
        axes[0,1].grid(True)
        
        # Test accuracy by min_samples_split
        split_acc = results_df.groupby('min_samples_split')['test_accuracy'].mean()
        axes[1,0].plot(split_acc.index, split_acc.values, 'go-')
        axes[1,0].set_title('Test Accuracy vs Min Samples Split')
        axes[1,0].set_xlabel('Min Samples Split')
        axes[1,0].set_ylabel('Test Accuracy')
        axes[1,0].grid(True)
        
        # Number of nodes by parameters
        node_counts = results_df.groupby('max_depth')['n_nodes'].mean()
        axes[1,1].plot(node_counts.index, node_counts.values, 'mo-')
        axes[1,1].set_title('Average Nodes vs Max Depth')
        axes[1,1].set_xlabel('Max Depth')
        axes[1,1].set_ylabel('Number of Nodes')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Show top 10 configurations
        top_configs = results_df.nlargest(10, 'test_accuracy')[
            ['max_depth', 'min_samples_split', 'min_samples_leaf', 'test_accuracy', 'overfitting']
        ]
        print(f"\n📋 TOP 10 CONFIGURATIONS:")
        print(top_configs.to_string(index=False))
        
        return results_df
    
    def visualize_decision_boundaries_2d(self):
        """
        Visualize decision boundaries for 2D case (Iris dataset)
        """
        iris = self.datasets['iris']
        
        print("\n🎨 VISUALIZING DECISION BOUNDARIES")
        print("=" * 40)
        
        # Use only 2 features for 2D visualization
        feature_pairs = [
            ('sepal length (cm)', 'sepal width (cm)'),
            ('petal length (cm)', 'petal width (cm)'),
            ('sepal length (cm)', 'petal length (cm)')
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            X_2d = iris[[feat1, feat2]].values
            y = iris['species'].values
            
            # Create decision tree
            dt = DecisionTreeClassifier(max_depth=3, random_state=42)
            dt.fit(X_2d, y)
            
            # Create mesh for decision boundary
            h = 0.02
            x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
            y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            # Predict on mesh
            Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot
            axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
            
            # Plot data points
            scatter = axes[idx].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
            axes[idx].set_xlabel(feat1)
            axes[idx].set_ylabel(feat2)
            axes[idx].set_title(f'Decision Tree\n{feat1[:15]} vs {feat2[:15]}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Run all decision tree examples
    """
    print("🌳 DECISION TREE ANALYSIS WITH KAGGLE DATASETS")
    print("=" * 55)
    
    explorer = DecisionTreeExplorer()
    
    # Load datasets
    explorer.load_titanic_dataset()
    explorer.load_iris_dataset()
    explorer.load_wine_dataset()
    
    # Analyze Titanic dataset
    explorer.explore_titanic_decision_tree()
    
    # Analyze Iris dataset
    explorer.explore_iris_decision_tree()
    
    # Experiment with node thresholds
    explorer.experiment_with_node_thresholds()
    
    # Visualize decision boundaries
    explorer.visualize_decision_boundaries_2d()
    
    print("\n✅ ALL ANALYSES COMPLETED!")
    print("\n📚 KEY LEARNINGS:")
    print("- Deeper trees can overfit (high train, lower test accuracy)")
    print("- min_samples_split controls when to stop splitting nodes")
    print("- min_samples_leaf ensures each leaf has enough samples")
    print("- Feature importance shows which variables matter most")
    print("- Decision boundaries show how the tree separates classes")

if __name__ == "__main__":
    main()