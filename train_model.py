"""
Fetal Health Classification - Model Training Script
This script trains the Gradient Boosting model and saves it for deployment.
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(data_path):
    """Load and prepare the fetal health dataset."""
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Drop highly correlated features as identified in EDA
    features_to_drop = [
        'histogram_width', 
        'histogram_min', 
        'histogram_max', 
        'histogram_variance', 
        'histogram_median', 
        'histogram_tendency', 
        'histogram_mode'
    ]
    
    # Check which features exist in the dataset
    features_to_drop = [f for f in features_to_drop if f in df.columns]
    
    if features_to_drop:
        print(f"Dropping features: {features_to_drop}")
        df = df.drop(columns=features_to_drop)
    
    # Separate features and target
    X = df.drop('fetal_health', axis=1)
    y = df['fetal_health']
    
    # Convert target to integer (1: Normal, 2: Suspect, 3: Pathological)
    y = y.astype(int)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")
    
    return X, y, X.columns.tolist()


def train_model(X_train, y_train, X_test, y_test):
    """Train the Gradient Boosting Classifier with hyperparameter tuning."""
    print("\nTraining Gradient Boosting Classifier...")
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Initialize model
    gb_model = GradientBoostingClassifier(random_state=42)
    
    # Perform grid search with cross-validation
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        gb_model, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Suspect', 'Pathological']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return best_model, grid_search.best_params_, test_accuracy


def save_model_artifacts(model, scaler, feature_names, params, accuracy, save_dir='../model'):
    """Save the trained model and related artifacts."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'gradient_boosting_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(save_dir, 'feature_names.json')
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to: {features_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'GradientBoostingClassifier',
        'best_params': params,
        'test_accuracy': float(accuracy),
        'feature_count': len(feature_names),
        'classes': {
            1: 'Normal',
            2: 'Suspect',
            3: 'Pathological'
        }
    }
    
    metadata_path = os.path.join(save_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def main(data_path='fetal_health.csv'):
    """Main training pipeline."""
    print("="*60)
    print("FETAL HEALTH CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data(data_path)
    
    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model, best_params, test_accuracy = train_model(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Save artifacts
    save_model_artifacts(model, scaler, feature_names, best_params, test_accuracy)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'fetal_health.csv'
    
    main(data_path)
