class Numerai_Bagged_Ensemble_Selection:
    def __init__(self, n_init=5, max_iter=30, decimals=5, corr_threshold=0.7, 
                 rank_normalize=True, bag_fraction=0.25, epsilon=0.00001, warm_start=10):
        self.n_init = n_init
        self.max_iter = max_iter
        self.decimals = decimals
        self.corr_threshold = corr_threshold
        self.rank_normalize = rank_normalize
        self.bag_fraction = bag_fraction
        self.epsilon = epsilon
        self.warm_start = warm_start  # Number of iterations before early stopping kicks in
        self.best_models = None
        self.best_weights = None
        self.best_performance = None
    
    def get_diverse_init_models(self, model_performances, model_cols, corr_matrix):
        """Get initial diverse models based on correlation threshold"""
        # Start with best model
        init_models = [model_performances.idxmax()]
        available_models = model_cols.copy()
        
        while len(init_models) < self.n_init and available_models:
            # Remove highly correlated models
            to_remove = []
            for model in available_models:
                if any(corr_matrix.loc[model, init_models] > self.corr_threshold):
                    to_remove.append(model)
            for model in to_remove:
                available_models.remove(model)
            
            if not available_models:
                break
                
            # Add best remaining model
            best_model = model_performances[available_models].idxmax()
            init_models.append(best_model)
            available_models.remove(best_model)
            
        return init_models
     
    def fit(self, data, model_cols, performance_func):
        """Fit ensemble using sorted initialization and forward selection"""
        self.data = data
        self._performance_func = performance_func
        
        # Rank normalize if requested
        if self.rank_normalize:
            data = data.copy()
            data[model_cols] = data.groupby('era')[model_cols].rank(pct=True)
        else:
            data = data.copy()
            print("Remember to rank your predictions if they are not already comparable!")
        
        # Get initial model(s)
        grouped = data.groupby('era')
        model_performances = performance_func(grouped, model_cols)
        best_single = model_performances.max()
        print(f"Best single model performance: {best_single:.5f} | Model: {model_performances.idxmax()}")

        # Get diverse initial models if requested
        if self.n_init > 1:
            corr_matrix = data[model_cols].corr().abs()
            init_models = self.get_diverse_init_models(model_performances, model_cols, corr_matrix)
        else:
            # Use best single model
            best_model = model_performances.idxmax()
            init_models = [best_model]
            
        # Initialize Counter with initial models
        ensemble = Counter(init_models)
        data['ensemble_preds'] = self.get_ensemble_preds(data, ensemble)
        current_preds = data['ensemble_preds']
        current_performance = performance_func(grouped, ['ensemble_preds']).iloc[0]
        print(f"Initial performance: {current_performance:.5f} | Models: {init_models}")
        
        # Track best ensemble
        best_ensemble = ensemble.copy()
        best_mean_performance = float('-inf')  
        best_iteration = 0
        
        # Early stopping variables
        bag_scores = []
        consecutive_decreases = 0
        previous_mean_score = float('-inf')     
                  
        # Greedy forward selection with era bagging
        for i in range(self.max_iter):
            # Sample eras for this iteration
            all_eras = data['era'].unique()
            bag_eras = np.random.choice(
                all_eras, 
                size=int(len(all_eras) * self.bag_fraction), 
                replace=False
            )
            bag_data = data[data['era'].isin(bag_eras)]
            
            # Get current ensemble size
            n_models = sum(ensemble.values())
            
            # Calculate all candidate predictions
            candidate_preds = pd.DataFrame(
                (n_models * current_preds.values[:, None] + data[model_cols].values) / (n_models + 1),
                index=data.index, columns=model_cols
            )
            
            # Calculate performances using only bagged eras
            candidate_bag = pd.concat([bag_data[['era', 'benchmark_preds', 'target']], 
                                    candidate_preds.loc[bag_data.index]], axis=1)
            candidate_performances = performance_func(
                candidate_bag.groupby('era'), 
                model_cols
            ).round(self.decimals)
            
            best_score = candidate_performances.max()
            tied_indices = np.argwhere(candidate_performances == best_score).flatten()
            tied_models = candidate_performances.index[tied_indices]
            
            if len(tied_indices) > 1:
                ensemble_candidates = [m for m in tied_models if m in ensemble]
                best_model = ensemble_candidates[0] if ensemble_candidates else tied_models[0]
            else:
                best_model = tied_models[0]
            
            # Update ensemble
            ensemble.update({best_model: 1})
            data['ensemble_preds'] = self.get_ensemble_preds(data, ensemble)
            current_preds = data['ensemble_preds']
            
            # Evaluate on full dataset
            full_performance = performance_func(grouped, ['ensemble_preds']).iloc[0]
            
            # Early stopping check - only after warm start period
            bag_scores.append(best_score)
            current_mean_score = np.mean(bag_scores)
            
            if i >= self.warm_start:  # Only check for early stopping and update best performance after warm start
                if current_mean_score < previous_mean_score:
                    consecutive_decreases += 1
                    if consecutive_decreases >= 3:
                        print(f"\nEarly stopping triggered at iteration {i+1}")
                        print(f"Mean bag score decreased for 3 consecutive iterations")
                        break
                else:
                    consecutive_decreases = 0
            
                if current_mean_score > best_mean_performance:
                    best_mean_performance = current_mean_score
                    best_performance = full_performance
                    best_ensemble = ensemble.copy()
                    best_iteration = i + 1
                    
            previous_mean_score = current_mean_score

            print(f"Iteration {i+1}: Added {best_model}, Bag Score: {best_score:.5f}, "
                  f"Mean Bag Score: {current_mean_score:.5f}, Full Score: {full_performance:.5f}")
            
        print(f"\nBest mean performance was {best_mean_performance:.5f} at iteration {best_iteration}")
        
        # Convert counter to weights
        total_count = sum(best_ensemble.values())
        self.best_weights = pd.Series(best_ensemble) / total_count
        self.best_models = self.best_weights.index
        self.best_mean_performance = best_mean_performance
        self.best_performance=best_performance
        
        print("\nFinal Ensemble Weights:")
        for model, weight in self.best_weights.sort_values(ascending=False).items():
            print(f"{model}: {weight:.4f}")
        
        return self
    
    def get_ensemble_preds(self, data, ensemble):
        """Get predictions for current ensemble"""
        if not ensemble:
            return pd.Series(0, index=data.index, name='ensemble_preds')
            
        ensemble_count = sum(ensemble.values())
        preds = pd.Series(0, index=data.index, name='ensemble_preds')
        
        for model, count in ensemble.items():
            preds += data[model] * count
            
        return preds / ensemble_count
    
    def get_best_ensemble(self):
        """Return DataFrame with models and their weights"""
        return pd.DataFrame({
            'model': self.best_weights.index,
            'weight': self.best_weights.values
        })
    
    def get_best_performance(self):
        """Return the best performance of the ensemble"""
        return self.best_performance
    
    def prune_ensemble(self):
        """Prune ensemble by removing models that don't decrease performance"""
        if not hasattr(self, 'best_models'):
            raise ValueError("Must call fit before pruning!")
        
        print(f"\nStarting pruning stage...")
        print(f"Initial performance: {self.best_performance:.5f} with {len(self.best_weights)} models")
        
        weights = self.best_weights.copy()
        current_performance = self.best_performance
        
        removed = True
        pruning_round = 1
        
        while removed:
            removed = False
            print(f"\nPruning Round {pruning_round}")
            
            # Sort models by ascending weights to try removing smallest weights first
            for model in weights.sort_values().index:
                # Create temporary weights without current model
                temp_weights = weights.drop(model)
                temp_weights = temp_weights / temp_weights.sum()
                
                # Calculate new ensemble performance
                temp_preds = (self.data[temp_weights.index] * temp_weights.values).sum(axis=1)
                self.data['temp_ensemble'] = temp_preds
                temp_performance = self._performance_func(self.data.groupby('era'), ['temp_ensemble']).iloc[0]
                
                print(f"Testing removal of {model}: {temp_performance:.5f} vs current {current_performance:.5f}")
                
                if temp_performance >= current_performance:
                    print(f"✓ Removing {model} - Performance improved to: {temp_performance:.5f}")
                    weights = temp_weights
                    current_performance = temp_performance
                    removed = True
                    break
            
            pruning_round += 1
        
        if current_performance > self.best_performance:
            self.best_weights = weights
            self.best_models = weights.index
            self.best_performance = current_performance
            
        print(f"\nFinal pruned ensemble:")
        print(f"Models: {len(self.best_weights)}")
        print(f"Performance: {self.best_performance:.5f}")
        for model, weight in self.best_weights.sort_values(ascending=False).items():
            print(f"{model}: {weight:.4f}")
        
        return self
    
    def predict(self, X):
        """Generate ensemble predictions for new data"""
        if self.rank_normalize:
            X = X.copy()
            X[self.best_models] = X.groupby('era')[self.best_models].rank(pct=True)
        return (X[self.best_models] * self.best_weights.values).sum(axis=1)