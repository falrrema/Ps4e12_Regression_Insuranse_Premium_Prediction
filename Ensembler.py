from collections import Counter
import numpy as np
import pandas as pd

class BaggedEnsembleSelection:
    def __init__(self, n_init=5, max_iter=30, decimals=5, corr_threshold=0.7, 
                 bag_fraction=0.25, epsilon=0.00001, warm_start=10):
        self.n_init = n_init
        self.max_iter = max_iter
        self.decimals = decimals
        self.corr_threshold = corr_threshold
        self.bag_fraction = bag_fraction
        self.epsilon = epsilon
        self.warm_start = warm_start
        self.best_models = None
        self.best_weights = None
        self.best_performance = None
    
    def get_diverse_init_models(self, model_performances, model_cols, corr_matrix):
        """Get initial diverse models based on correlation threshold"""
        # Start with best model
        init_models = [model_performances.idxmax()]
        available_models = list(model_cols)
        available_models.remove(init_models[0])
        
        while len(init_models) < self.n_init and available_models:
            # Remove highly correlated models using vectorized operations
            corr_with_selected = corr_matrix.loc[available_models, init_models].max(axis=1)
            available_models = [m for m, c in zip(available_models, corr_with_selected) 
                              if c <= self.corr_threshold]
            
            if not available_models:
                break
                
            # Add best remaining model
            best_model = model_performances[available_models].idxmax()
            init_models.append(best_model)
            available_models.remove(best_model)
            
        return init_models
    
    def get_ensemble_preds(self, X, ensemble):
        """Get predictions for current ensemble using vectorized operations"""
        if not ensemble:
            return np.zeros(len(X))
            
        ensemble_weights = pd.Series(ensemble) / sum(ensemble.values())
        return X[ensemble_weights.index].multiply(ensemble_weights).sum(axis=1)
    
    def fit(self, X, y, performance_func):
        """
        Fit ensemble using forward selection with bagging
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing model predictions as columns
        y : pd.Series
            True target values
        performance_func : callable
            Function that takes (y_true, y_pred) and returns a score to maximize
        """
        model_cols = X.columns
        n_samples = len(y)
        
        # Calculate initial model performances
        model_performances = pd.Series({
        col: performance_func(y, X[col]) for col in model_cols
        })
        
        best_single = model_performances.max()
        print(f"Best single model performance: {best_single:.5f} | Model: {model_performances.idxmax()}")

        # Get diverse initial models
        if self.n_init > 1:
            corr_matrix = X.corr().abs()
            init_models = self.get_diverse_init_models(model_performances, model_cols, corr_matrix)
        else:
            init_models = [model_performances.idxmax()]
            
        # Initialize ensemble
        ensemble = Counter(init_models)
        current_preds = self.get_ensemble_preds(X, ensemble)
        current_performance = performance_func(y, current_preds)
        print(f"Initial performance: {current_performance:.5f} | Models: {init_models}")
        
        # Track best ensemble
        best_ensemble = ensemble.copy()
        best_mean_performance = float('-inf')
        best_iteration = 0
        
        # Early stopping variables
        bag_scores = []
        consecutive_decreases = 0
        previous_mean_score = float('-inf')
        
        # Greedy forward selection with bagging
        bag_size = int(n_samples * self.bag_fraction)

        for i in range(self.max_iter):
            # Sample indices for this iteration
            bag_indices = np.random.choice(n_samples, size=bag_size, replace=False)
            
            # Get current ensemble size
            n_models = sum(ensemble.values())
            
            # Convert to numpy arrays before multi-dimensional indexing
            current_preds_np = current_preds.to_numpy() if isinstance(current_preds, pd.Series) else current_preds
            X_np = X.values
            
            # Vectorized calculation of all candidate predictions
            current_preds_expanded = current_preds_np[:, np.newaxis]
            candidate_preds = (n_models * current_preds_expanded + X_np) / (n_models + 1)
            
            # Calculate scores for all candidates at once
            candidate_scores = {
                model: performance_func(y.iloc[bag_indices], candidate_preds[bag_indices, j])
                for j, model in enumerate(model_cols)
            }
            
            best_model = max(candidate_scores.items(), key=lambda x: x[1])[0]
            best_score = candidate_scores[best_model]
            
            # Update ensemble
            ensemble.update({best_model: 1})
            current_preds = self.get_ensemble_preds(X, ensemble)
            full_performance = performance_func(y, current_preds)
            
            # Early stopping check
            bag_scores.append(best_score)
            current_mean_score = np.mean(bag_scores[-self.warm_start:])
            
            if i >= self.warm_start:
                if current_mean_score < previous_mean_score:
                    consecutive_decreases += 1
                    if consecutive_decreases >= 3:
                        print(f"\nEarly stopping triggered at iteration {i+1}")
                        break
                else:
                    consecutive_decreases = 0
                
                if current_mean_score > best_mean_performance:
                    best_mean_performance = current_mean_score
                    best_ensemble = ensemble.copy()
                    best_iteration = i + 1
                    
            previous_mean_score = current_mean_score
            
            print(f"Iteration {i+1}: Added {best_model}, Bag Score: {best_score:.5f}, "
                  f"Mean Bag Score: {current_mean_score:.5f}, Full Score: {full_performance:.5f}")
        
        # Convert counter to weights
        total_count = sum(best_ensemble.values())
        self.best_weights = pd.Series(best_ensemble) / total_count
        self.best_models = self.best_weights.index
        self.best_mean_performance = best_mean_performance
        
        print("\nFinal Ensemble Weights:")
        for model, weight in self.best_weights.sort_values(ascending=False).items():
            print(f"{model}: {weight:.4f}")
        
        return self
    
    def get_best_ensemble(self):
        """Return DataFrame with models and their weights"""
        return pd.DataFrame({
            'model': self.best_weights.index,
            'weight': self.best_weights.values
        }).sort_values('weight', ascending=False)
                
    def predict(self, X):
        """Generate ensemble predictions for new data"""
        if self.best_weights is None:
            raise ValueError("Model must be fitted before making predictions")
        return (X[self.best_models] * self.best_weights.values).sum(axis=1)