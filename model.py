import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

class ModelPipeline:
    def __init__(self, model=None):
        # Default base model configuration
        self.model = model if model else LGBMClassifier(random_state=42, verbose=-1)
        self.is_fitted = False
    
    def train(self, X_train, y_train):
        """Fits the current model on training data."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
    
    def param_tuning(self, X_train, y_train):
        """Tunes hyperparameters and automatically replaces the internal model with the optimized, fitted version."""
        param_grid = {
            'num_leaves': [5, 20, 31],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 150]
        }
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=self.model, 
            param_grid=param_grid, 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Save best optimized and ALREADY fitted model
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return grid_search.best_params_

    def evaluate(self, X_test, y_test):
        """Evaluates metrics safely for binary target structures."""
        if not self.is_fitted:
            raise ValueError("Model must be trained or tuned before evaluation.")
            
        predictions = self.model.predict(X_test)
        prob_scores = self.model.predict_proba(X_test)
        
        # Safe handling for binary probabilities (works if shape is 1D or 2D)
        if prob_scores.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, prob_scores[:, 1])
        else:
            # Fallback handling for multi-class edge cases
            roc_auc = roc_auc_score(y_test, prob_scores, multi_class='ovr')

        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'recall': recall_score(y_test, predictions, average='binary' if prob_scores.shape[1] == 2 else 'macro'),
            'roc_auc': roc_auc,
            'report': classification_report(y_test, predictions)
        }

        return metrics

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
    
    def load_model(self, file_path):
        self.model = joblib.load(file_path)
        self.is_fitted = True