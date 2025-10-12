#!/usr/bin/env python3
"""
Unit tests for core components without DVC dependencies.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import joblib
from unittest.mock import patch, MagicMock

# Add project_name to path
import sys
sys.path.append('.')

from project_name.processing import DataPreprocessor, preprocess_data, postprocess_predictions
from project_name.configs import ModelConfig, load_config
from project_name.trainer import Trainer


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor.numeric_scaler is not None
        assert preprocessor.categorical_encoder is not None
        assert preprocessor.numeric_columns is None
        assert preprocessor.categorical_columns is None
        assert preprocessor.is_fitted == False
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from sklearn.preprocessing import RobustScaler, OneHotEncoder
        
        numeric_scaler = RobustScaler()
        categorical_encoder = OneHotEncoder()
        numeric_columns = ['col1', 'col2']
        categorical_columns = ['cat1', 'cat2']
        
        preprocessor = DataPreprocessor(
            numeric_scaler=numeric_scaler,
            categorical_encoder=categorical_encoder,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns
        )
        
        assert preprocessor.numeric_scaler == numeric_scaler
        assert preprocessor.categorical_encoder == categorical_encoder
        assert preprocessor.numeric_columns == numeric_columns
        assert preprocessor.categorical_columns == categorical_columns
    
    def test_detect_column_types(self):
        """Test automatic column type detection."""
        data = pd.DataFrame({
            'numeric_int': [1, 2, 3],
            'numeric_float': [1.1, 2.2, 3.3],
            'categorical_str': ['A', 'B', 'A'],
            'categorical_bool': [True, False, True]
        })
        
        preprocessor = DataPreprocessor()
        preprocessor._detect_column_types(data)
        
        assert 'numeric_int' in preprocessor.numeric_columns
        assert 'numeric_float' in preprocessor.numeric_columns
        assert 'categorical_str' in preprocessor.categorical_columns
        assert 'categorical_bool' in preprocessor.categorical_columns
    
    def test_fit_transform_numeric_only(self):
        """Test fit_transform with numeric data only."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10.1, 20.2, 30.3, 40.4, 50.5]
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 2)
        assert list(result.columns) == ['col1', 'col2']
        assert preprocessor.is_fitted == True
    
    def test_fit_transform_categorical_only(self):
        """Test fit_transform with categorical data only."""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'C'],
            'cat2': ['X', 'Y', 'X', 'Z']
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 4  # Same number of rows
        assert preprocessor.is_fitted == True
    
    def test_fit_transform_mixed_data(self):
        """Test fit_transform with mixed numeric and categorical data."""
        data = pd.DataFrame({
            'numeric': [1, 2, 3, 4],
            'categorical': ['A', 'B', 'A', 'C']
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(data)
        
        assert isinstance(result, pd.DataFrame)
        assert preprocessor.is_fitted == True
        # Should have more columns due to one-hot encoding
        assert result.shape[1] >= 2
    
    def test_save_and_load(self):
        """Test saving and loading preprocessor."""
        data = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['A', 'B', 'A']
        })
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(data)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save preprocessor
            preprocessor.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Load preprocessor
            loaded_preprocessor = DataPreprocessor.load(tmp_path)
            
            # Test that loaded preprocessor works
            result_original = preprocessor.transform(data)
            result_loaded = loaded_preprocessor.transform(data)
            
            assert np.allclose(result_original.values, result_loaded.values)
            assert loaded_preprocessor.is_fitted == True
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_error_handling(self):
        """Test error handling."""
        preprocessor = DataPreprocessor()
        
        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            preprocessor.fit(empty_data)
        
        # Test with wrong data type
        with pytest.raises(TypeError):
            preprocessor.fit([1, 2, 3])
        
        # Test transform without fitting
        data = pd.DataFrame({'col': [1, 2, 3]})
        with pytest.raises(Exception):  # Should be NotFittedError
            preprocessor.transform(data)


class TestPreprocessDataFunction:
    """Tests for preprocess_data function."""
    
    def test_preprocess_data_without_preprocessor(self):
        """Test preprocess_data without providing preprocessor."""
        data = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['A', 'B', 'A']
        })
        
        result = preprocess_data(data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3
    
    def test_preprocess_data_with_preprocessor(self):
        """Test preprocess_data with provided preprocessor."""
        data = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['A', 'B', 'A']
        })
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(data)
        
        result = preprocess_data(data, preprocessor)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3


class TestPostprocessPredictions:
    """Tests for postprocess_predictions function."""
    
    def test_postprocess_numpy_array(self):
        """Test postprocess_predictions with numpy array."""
        predictions = np.array([0.1, 0.5, 0.9])
        
        result = postprocess_predictions(predictions)
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, predictions)
    
    def test_postprocess_pandas_series(self):
        """Test postprocess_predictions with pandas Series."""
        predictions = pd.Series([1, 2, 3])
        
        result = postprocess_predictions(predictions)
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3]))
    
    def test_postprocess_with_inverse_transform(self):
        """Test postprocess_predictions with inverse transformation."""
        predictions = np.array([1, 2, 3])
        
        def inverse_func(x):
            return x * 2
        
        result = postprocess_predictions(predictions, inverse_func)
        
        expected = np.array([2, 4, 6])
        assert np.array_equal(result, expected)
    
    def test_postprocess_edge_cases(self):
        """Test postprocess_predictions with edge cases."""
        # Test with None
        with pytest.raises(ValueError):
            postprocess_predictions(None)
        
        # Test with empty array
        result = postprocess_predictions(np.array([]))
        assert len(result) == 0
        
        # Test with NaN values
        predictions = np.array([1.0, np.nan, 0.5])
        result = postprocess_predictions(predictions)
        assert np.any(np.isnan(result))


class TestModelConfig:
    """Tests for ModelConfig class."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = ModelConfig(model_type="logreg")
        
        assert config.model_type == "logreg"
        assert config.num_epoch == 3
        assert config.C == 1.0
        assert config.max_iter == 100
        assert config.random_state == 42
        assert config.test_size == 0.2
        assert config.target_column == 'label'
    
    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config_dict = {
            "model_type": "random_forest",
            "num_epoch": 5,
            "n_estimators": 200,
            "max_depth": 15,
            "random_state": 123,
            "test_size": 0.3,
            "target_column": "target"
        }
        
        config = ModelConfig(**config_dict)
        
        assert config.model_type == "random_forest"
        assert config.num_epoch == 5
        assert config.n_estimators == 200
        assert config.max_depth == 15
        assert config.random_state == 123
        assert config.test_size == 0.3
        assert config.target_column == "target"


class TestTrainer:
    """Tests for Trainer class."""
    
    def test_trainer_init_logreg(self):
        """Test Trainer initialization with logistic regression."""
        config = ModelConfig(model_type="logreg")
        
        trainer = Trainer(config)
        
        assert trainer.config == config
        assert trainer.model is not None
        assert hasattr(trainer.model, 'fit')
    
    def test_trainer_init_random_forest(self):
        """Test Trainer initialization with random forest."""
        config = ModelConfig(model_type="random_forest")
        
        trainer = Trainer(config)
        
        assert trainer.config == config
        assert trainer.model is not None
        assert hasattr(trainer.model, 'fit')
    
    def test_trainer_init_decision_tree(self):
        """Test Trainer initialization with decision tree."""
        config = ModelConfig(model_type="decision_tree")
        
        trainer = Trainer(config)
        
        assert trainer.config == config
        assert trainer.model is not None
        assert hasattr(trainer.model, 'fit')
    
    def test_trainer_init_invalid_model(self):
        """Test Trainer initialization with invalid model type."""
        with pytest.raises(Exception):
            config = ModelConfig(model_type="invalid_model")
        
        with pytest.raises(Exception):
            Trainer(config)
    
    def test_trainer_fit_and_evaluate(self):
        """Test Trainer fit and evaluate methods."""
        config = ModelConfig(model_type="logreg")
        trainer = Trainer(config)
        
        # Create simple synthetic data
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        X_test = np.array([[2, 3], [4, 5]])
        y_test = np.array([0, 1])
        
        # Test fit
        trainer.fit(X_train, y_train)
        
        # Test evaluate
        accuracy = trainer.evaluate(X_test, y_test)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1


class TestIntegration:
    """Integration tests for core components."""
    
    def test_complete_workflow(self):
        """Test complete workflow from data preprocessing to training."""
        # Create synthetic data
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'label': np.random.choice([0, 1], 100)
        })
        
        # Preprocess data
        X_processed = preprocess_data(data.drop('label', axis=1))
        y = data['label'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Train model
        config = ModelConfig(model_type="random_forest")
        trainer = Trainer(config)
        trainer.fit(X_train, y_train)
        
        # Evaluate model
        accuracy = trainer.evaluate(X_test, y_test)
        
        # Make predictions
        predictions = trainer.model.predict(X_test)
        
        # Postprocess predictions
        processed_predictions = postprocess_predictions(predictions)
        
        # Assertions
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert len(processed_predictions) == len(y_test)
        assert set(processed_predictions).issubset({0, 1})


if __name__ == "__main__":
    # Run tests
    test_preprocessor = TestDataPreprocessor()
    test_preprocessor.test_init_default_parameters()
    test_preprocessor.test_init_custom_parameters()
    test_preprocessor.test_detect_column_types()
    test_preprocessor.test_fit_transform_numeric_only()
    test_preprocessor.test_fit_transform_categorical_only()
    test_preprocessor.test_fit_transform_mixed_data()
    test_preprocessor.test_save_and_load()
    test_preprocessor.test_error_handling()
    
    test_preprocess_func = TestPreprocessDataFunction()
    test_preprocess_func.test_preprocess_data_without_preprocessor()
    test_preprocess_func.test_preprocess_data_with_preprocessor()
    
    test_postprocess = TestPostprocessPredictions()
    test_postprocess.test_postprocess_numpy_array()
    test_postprocess.test_postprocess_pandas_series()
    test_postprocess.test_postprocess_with_inverse_transform()
    test_postprocess.test_postprocess_edge_cases()
    
    test_config = TestModelConfig()
    test_config.test_model_config_defaults()
    test_config.test_model_config_custom_values()
    
    test_trainer = TestTrainer()
    test_trainer.test_trainer_init_logreg()
    test_trainer.test_trainer_init_random_forest()
    test_trainer.test_trainer_init_decision_tree()
    test_trainer.test_trainer_init_invalid_model()
    test_trainer.test_trainer_fit_and_evaluate()
    
    test_integration = TestIntegration()
    test_integration.test_complete_workflow()
    
    print("All core component tests completed successfully!")