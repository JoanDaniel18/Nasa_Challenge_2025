"""
Machine learning model service for climate prediction
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

from config import MODEL_CONFIG, MODELS_DIR
from cache_service import cache
from data_service import data_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.models_dir = MODELS_DIR
        self.config = MODEL_CONFIG
        
    def _generate_model_key(self, model_type: str, target_variable: str) -> str:
        """Generate unique key for model caching"""
        return f"{model_type}_{target_variable}_model"
    
    def train_xgboost_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        target_variable: str = 'temperature'
    ) -> Dict[str, Any]:
        """
        Train XGBoost model for climate prediction
        """
        try:
            logger.info(f"Training XGBoost model for {target_variable}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=1-self.config['train_test_split'],
                random_state=self.config['random_state'],
                shuffle=False  # Preserve time order
            )
            
            # Create and train XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                learning_rate=self.config['learning_rate'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
            metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
            
            # Store model and metrics
            model_key = self._generate_model_key('xgboost', target_variable)
            self.models[model_key] = model
            self.model_metrics[model_key] = metrics
            
            # Cache model
            cache.set(model_key, {
                'model': model,
                'metrics': metrics,
                'target_variable': target_variable,
                'training_date': datetime.now().isoformat(),
                'feature_columns': list(X.columns)
            }, 'model')
            
            logger.info(f"XGBoost model trained successfully. Test RMSE: {metrics['test_rmse']:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'model_key': model_key
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            return {'error': str(e)}
    
    def train_random_forest_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        target_variable: str = 'temperature'
    ) -> Dict[str, Any]:
        """
        Train Random Forest model for climate prediction
        """
        try:
            logger.info(f"Training Random Forest model for {target_variable}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=1-self.config['train_test_split'],
                random_state=self.config['random_state'],
                shuffle=False
            )
            
            # Create and train Random Forest model
            model = RandomForestRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
            metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
            
            # Store model and metrics
            model_key = self._generate_model_key('random_forest', target_variable)
            self.models[model_key] = model
            self.model_metrics[model_key] = metrics
            
            # Cache model
            cache.set(model_key, {
                'model': model,
                'metrics': metrics,
                'target_variable': target_variable,
                'training_date': datetime.now().isoformat(),
                'feature_columns': list(X.columns)
            }, 'model')
            
            logger.info(f"Random Forest model trained successfully. Test RMSE: {metrics['test_rmse']:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'model_key': model_key
            }
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            return {'error': str(e)}
    
    def get_model(self, model_key: str) -> Optional[Any]:
        """Retrieve model from cache or memory"""
        if model_key in self.models:
            return self.models[model_key]
        
        # Try to load from cache
        cached_data = cache.get(model_key, 'model')
        if cached_data is not None:
            self.models[model_key] = cached_data['model']
            self.model_metrics[model_key] = cached_data['metrics']
            return cached_data['model']
        
        return None
    
    def predict(
        self, 
        model_key: str, 
        features: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Make predictions using trained model"""
        try:
            model = self.get_model(model_key)
            if model is None:
                logger.error(f"Model not found: {model_key}")
                return None
            
            predictions = model.predict(features)
            logger.info(f"Generated {len(predictions)} predictions using {model_key}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def predict_future(
        self, 
        model_key: str, 
        days_ahead: int = 7,
        target_variable: str = 'temperature'
    ) -> Optional[Dict[str, Any]]:
        """Predict future climate values"""
        try:
            model = self.get_model(model_key)
            if model is None:
                logger.error(f"Model not found: {model_key}")
                return None
            
            # Get recent data to use as features
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Get last 60 days for feature calculation
            
            X, y = data_service.prepare_ml_dataset(
                start_date=start_date,
                end_date=end_date,
                target_variable=target_variable
            )
            
            if X.empty:
                logger.error("No data available for future predictions")
                return None
            
            # Use the most recent data point as starting point
            last_features = X.iloc[-1:].copy()
            
            predictions = []
            dates = []
            
            current_date = end_date
            
            for day in range(days_ahead):
                current_date += timedelta(days=1)
                
                # Update time-based features
                last_features['day_of_year'] = current_date.day_of_year
                last_features['month'] = current_date.month
                last_features['season'] = ((current_date.month - 1) // 3) + 1
                
                # Make prediction
                pred = model.predict(last_features)[0]
                predictions.append(pred)
                dates.append(current_date)
                
                # Update features for next prediction (simplified approach)
                # In practice, you would update lagged features more carefully
                last_features[target_variable + '_lag_1'] = pred
            
            result = {
                'dates': [d.isoformat() for d in dates],
                'predictions': predictions,
                'target_variable': target_variable,
                'model_key': model_key,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {days_ahead} day forecast for {target_variable}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating future predictions: {str(e)}")
            return None
    
    def compare_models(self, target_variable: str = 'temperature') -> Dict[str, Any]:
        """Compare performance of different models"""
        try:
            results = {}
            
            # Get model keys for the target variable
            model_keys = [key for key in self.model_metrics.keys() 
                         if target_variable in key]
            
            if not model_keys:
                logger.warning(f"No trained models found for {target_variable}")
                return {}
            
            for model_key in model_keys:
                metrics = self.model_metrics[model_key]
                model_type = model_key.split('_')[0]
                
                results[model_type] = {
                    'test_rmse': metrics.get('test_rmse', float('inf')),
                    'test_r2': metrics.get('test_r2', 0),
                    'test_mae': metrics.get('test_mae', float('inf')),
                    'cv_rmse_mean': metrics.get('cv_rmse_mean', float('inf')),
                    'model_key': model_key
                }
            
            # Find best model
            best_model = min(results.keys(), key=lambda k: results[k]['test_rmse'])
            results['best_model'] = best_model
            
            logger.info(f"Model comparison completed. Best model: {best_model}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {}
    
    def retrain_models(self, target_variable: str = 'temperature') -> Dict[str, Any]:
        """Retrain all models with latest data"""
        try:
            logger.info(f"Retraining models for {target_variable}")
            
            # Prepare fresh dataset
            X, y = data_service.prepare_ml_dataset(target_variable=target_variable)
            
            if X.empty or y.empty:
                return {'error': 'No data available for retraining'}
            
            results = {}
            
            # Train XGBoost
            xgb_result = self.train_xgboost_model(X, y, target_variable)
            results['xgboost'] = xgb_result
            
            # Train Random Forest
            rf_result = self.train_random_forest_model(X, y, target_variable)
            results['random_forest'] = rf_result
            
            # Compare models
            comparison = self.compare_models(target_variable)
            results['comparison'] = comparison
            
            logger.info(f"Model retraining completed for {target_variable}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retraining models: {str(e)}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all trained models"""
        info = {
            'available_models': list(self.models.keys()),
            'model_metrics': self.model_metrics,
            'cache_stats': cache.get_cache_stats()
        }
        
        return info

# Global model service instance
model_service = ModelService()
