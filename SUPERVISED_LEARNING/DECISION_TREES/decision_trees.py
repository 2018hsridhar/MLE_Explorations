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
# What is seaborn? It's a data visualization library based on matplotlib 
# that provides a high-level interface for drawing attractive and informative statistical graphics.
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

Focus on Kaggle datasets that are small and easy to understand.
Three datasets to consider :
1. Titanic Dataset : Binary classification - survived or not.
2. Iris Dataset : Multi-class classification - species of iris flower.
3. Wine Quality Dataset : Multi-class classification - quality rating of wine.

Titanic Dataset Features :
- Pclass : Passenger class (1st, 2nd, 3rd)
- Sex : Male/F
- Age : Age in years
- SibSp : Number of siblings/spouses aboard
- Parch : Number of parents/children aboard
- Fare : Ticket fare
- Embarked : Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- Survived : Target variable (0 = No, 1 = Yes)

Decision tree training process :
1. Load and preprocess dataset : handle missing values, encode categorical variables.
2. Split data into training and testing sets.
3. Train decision tree classifier with varying max_depth and other hyperparameters 
such as min_samples_split, min_samples_leaf.
4. Visualize decision tree structure.
5. Evaluate model performance using accuracy, confusion matrix, classification report.

How are most important features determined?
- Feature importance in decision trees is determined by the reduction in impurity
    (e.g., Gini impurity or entropy) that each feature provides when it is used to split the data.
- Features that lead to larger reductions in impurity across the tree are considered more important.

Calculate impurity?
- Impurity measures how mixed the classes are in a node.
- Common impurity measures include Gini impurity and entropy.
- Gini impurity for a node is calculated as:
    Gini = 1 - Σ (p_i)^2
  where p_i is the proportion of class i in the node.

How are thresholds for splits chosen?
- Thresholds are chosen based on the feature values that best separate the classes.
- The algorithm evaluates all possible splits for each feature and selects the one that
    results in the greatest reduction in impurity.



'''


class DecisionTreeExplorer:
    """
    A class to explore decision trees with different datasets and parameters
    """
    
    def __init__(self):
        self.datasets = {}
        self.trees = {}
        
    '''
    Root Question: "What was the evacuation priority?"
    ├── Gender (primary factor - social policy)
    │   ├── Female → High survival probability
    │   └── Male → Check other factors
    ├── Age (secondary factor - "children first")
    │   ├── Child → High survival probability  
    │   └── Adult → Check class/wealth
    └── Passenger Class (tertiary factor - access to lifeboats)
        ├── 1st Class → Better lifeboat access
        ├── 2nd Class → Medium access
        └── 3rd Class → Poor access
    '''
    def load_titanic_dataset(self):
        """
        Load Titanic dataset - Perfect for classification and understanding splits
        """
        print(f"TITANIC DATASET ANALYSIS")
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
                'pclass': pclass, # Passenger class (1st, 2nd, 3rd)
                'sex': sex,  # 0=female, 1=male
                'sibsp': sibsp, # Number of siblings/spouses aboard
                'parch': parch, # Number of parents/children aboard
                'survived': survived # Target variable
            })
            
            print(f"Dataset shape: {titanic.shape}")
            print(f"Survival rate: {titanic['survived'].mean():.2%}")
            print(f"Features: {list(titanic.columns[:-1])}")
            print(f"First 5 rows:\n{titanic.head()}")
            
            self.datasets['titanic'] = titanic
            return titanic
            
        except Exception as e:
            print(f"Error loading Titanic dataset: {e}")
            return None
    
 
    def explore_titanic_decision_tree(self):
        """
        Build and analyze decision tree for Titanic dataset
        """
        titanic = self.datasets['titanic']
        
        print(f"TITANIC DECISION TREE ANALYSIS")
        print("=" * 45)
        
        # Prepare features and target
        targetColumnName = 'survived'
        X = titanic.drop(targetColumnName, axis=1)
        y = titanic[targetColumnName]
        
        # Split data
        '''
        stratify=y to maintain class distribution in train and test sets
        test_size=0.2 means 20% of data is used for testing
        maintain class distribution in train and test sets
        in order to avoid bias in model evaluation.
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Try different tree depths to understand node thresholds
        '''
        Expected depth trends :
        Depth 3 : Simple rules, e.g. "Is passenger female?"
        Depth 5 : More nuanced splits, e.g. "Is age < 16.5?"
        Depth 7+ : Complex interactions, e.g. "Is fare > $50 and pclass == 1?"
        '''
        depths = [3, 5, 7, 10, None]
        results = []
        
        plt.figure(figsize=(15, 10))
        
        for i, depth in enumerate(depths):
            # Create decision tree
            # min_samples_split=20 to avoid overfitting on small splits
            # min_samples_leaf=10 to ensure leaves have enough samples
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
            if i < 1:
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
        
        # # Feature importance analysis ( a todo ) 
        best_dt = DecisionTreeClassifier(max_depth=3, min_samples_split=20, random_state=42)
        best_dt.fit(X_train, y_train)
   
        # # Store the best tree
        self.trees['titanic'] = best_dt
        
        # # Print tree rules
        print(f"DECISION TREE RULES (first 20 lines):")
        tree_rules = export_text(best_dt, feature_names=list(X.columns))
        print('\n'.join(tree_rules.split('\n')[:20]))
        best_dt = None
        return best_dt
    
def main():
    """
    Run all decision tree examples across datasets
    """
    print(f"Exploring Decision Trees with Kaggle Datasets\n")
    print("=" * 55)
    
    explorer = DecisionTreeExplorer()
    
    # Load datasets
    explorer.load_titanic_dataset()
    
    # Analyze Titanic dataset
    explorer.explore_titanic_decision_tree()
    
    
if __name__ == "__main__":
    main()