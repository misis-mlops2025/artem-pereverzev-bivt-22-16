#!/usr/bin/env python3
"""
Tests for pipeline scripts without DVC dependencies.
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


class TestGenerateDataScript:
    """Tests for generate_data.py script."""
    
    def test_generate_data_functionality(self):
        """Test the main functionality of generate_data.py."""
        from project_name.scripts.generate_data import generate_synthetic_data, load_params
        
        # Mock parameters
        mock_params = {
            'data': {
                'seed': 42,
                'n_samples': 100,
                'noise_level': 0.1
            }
        }
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_params), \
             patch('os.makedirs'), \
             patch('pandas.DataFrame.to_csv'):
            
            data = generate_synthetic_data()
            
            # Verify data structure
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 100
            expected_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'label']
            for col in expected_columns:
                assert col in data.columns
    
    def test_load_params_function(self):
        """Test the load_params function."""
        from project_name.scripts.generate_data import load_params
        
        mock_params = {'data': {'seed': 42}}
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_params):
            
            params = load_params()
            assert params == mock_params


class TestPreprocessDataScript:
    """Tests for preprocess_data.py script."""
    
    def test_preprocess_data_functionality(self):
        """Test the main functionality of preprocess_data.py."""
        from project_name.scripts.preprocess_data import preprocess_data
        
        # Create mock data
        mock_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(5, 2, 50),
            'feature3': np.random.choice(['A', 'B', 'C'], 50),
            'feature4': np.random.choice(['X', 'Y'], 50),
            'label': np.random.choice([0, 1], 50)
        })
        
        mock_params = {'preprocess': {}}
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_params), \
             patch('pandas.read_csv', return_value=mock_data), \
             patch('joblib.dump'), \
             patch('os.makedirs'), \
             patch('sklearn.model_selection.train_test_split') as mock_split:
            
            train_df = mock_data.iloc[:40, :]
            test_df = mock_data.iloc[40:, :]
            
            mock_split.return_value = (train_df, test_df)
            
            # This should run without errors
            preprocess_data()


class TestTrainModelScript:
    """Tests for train_model.py script."""
    
    def test_train_model_functionality(self):
        """Test the main functionality of train_model.py."""
        from project_name.scripts.train_model import train_model
        
        # Mock data
        X_train = np.random.random((100, 5))
        y_train = np.random.choice([0, 1], 100)
        X_test = np.random.random((20, 5))
        y_test = np.random.choice([0, 1], 20)
        
        mock_params = {
            'train': {
                'model_type': 'random_forest',
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 10
            }
        }
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_params), \
             patch('joblib.load') as mock_load, \
             patch('joblib.dump'), \
             patch('os.makedirs'), \
             patch('sklearn.metrics.accuracy_score', return_value=0.85), \
             patch('sklearn.metrics.classification_report', return_value='report'):
            
            # Mock data loading
            mock_load.side_effect = [X_train, y_train, X_test, y_test]
            
            # This should run without errors
            train_model()


class TestEvaluateModelScript:
    """Tests for evaluate_model.py script."""
    
    def test_evaluate_model_functionality(self):
        """Test the main functionality of evaluate_model.py."""
        from project_name.scripts.evaluate_model import evaluate_model
        
        # Mock model and data
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([
            [0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9], [0.8, 0.2]
        ])
        
        X_test = np.random.random((5, 5))
        y_test = np.array([0, 1, 0, 1, 0])
        
        with patch('joblib.load') as mock_load, \
             patch('builtins.open'), \
             patch('json.dump'), \
             patch('os.makedirs'):
            
            # Mock data loading
            mock_load.side_effect = [mock_model, X_test, y_test]
            
            # This should run without errors
            evaluate_model()


class TestCreatePlotsScript:
    """Tests for create_plots.py script."""
    
    def test_create_plots_functionality(self):
        """Test the main functionality of create_plots.py."""
        from project_name.scripts.create_plots import create_plots
        
        # Mock metrics data
        mock_metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.87,
            'f1_score': 0.84,
            'roc_auc': 0.89
        }
        
        mock_cm_data = {
            'matrix': [[45, 5], [8, 42]],
            'labels': ['Class 0', 'Class 1']
        }
        
        mock_roc_data = {
            'fpr': [0.0, 0.1, 0.2, 0.3, 1.0],
            'tpr': [0.0, 0.8, 0.9, 0.95, 1.0],
            'thresholds': [1.5, 0.8, 0.6, 0.4, 0.0],
            'auc': 0.89
        }
        


class TestScriptImports:
    """Tests to verify script imports work correctly."""
    
    def test_generate_data_imports(self):
        """Test that generate_data.py imports work."""
        from project_name.scripts.generate_data import generate_synthetic_data, load_params
        assert callable(generate_synthetic_data)
        assert callable(load_params)
    
    def test_preprocess_data_imports(self):
        """Test that preprocess_data.py imports work."""
        from project_name.scripts.preprocess_data import preprocess_data
        assert callable(preprocess_data)
    
    def test_train_model_imports(self):
        """Test that train_model.py imports work."""
        from project_name.scripts.train_model import train_model
        assert callable(train_model)
    
    def test_evaluate_model_imports(self):
        """Test that evaluate_model.py imports work."""
        from project_name.scripts.evaluate_model import evaluate_model
        assert callable(evaluate_model)
    
    def test_create_plots_imports(self):
        """Test that create_plots.py imports work."""
        from project_name.scripts.create_plots import create_plots
        assert callable(create_plots)


class TestErrorHandling:
    """Tests for error handling in scripts."""
    
    def test_generate_data_missing_params(self):
        """Test generate_data with missing parameters."""
        from project_name.scripts.generate_data import generate_synthetic_data
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value={}), \
             patch('os.makedirs'), \
             patch('pandas.DataFrame.to_csv'):
            
            # Should use default values when params are missing
            data = generate_synthetic_data()
            assert isinstance(data, pd.DataFrame)
    
    def test_preprocess_data_missing_file(self):
        """Test preprocess_data with missing data file."""
        from project_name.scripts.preprocess_data import preprocess_data
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value={}), \
             patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            
            with pytest.raises(FileNotFoundError):
                preprocess_data()


if __name__ == "__main__":
    # Run tests
    test_generate = TestGenerateDataScript()
    test_generate.test_generate_data_functionality()
    test_generate.test_load_params_function()
    
    test_preprocess = TestPreprocessDataScript()
    test_preprocess.test_preprocess_data_functionality()
    
    test_train = TestTrainModelScript()
    test_train.test_train_model_functionality()
    
    test_evaluate = TestEvaluateModelScript()
    test_evaluate.test_evaluate_model_functionality()
    
    test_plots = TestCreatePlotsScript()
    test_plots.test_create_plots_functionality()
    
    test_imports = TestScriptImports()
    test_imports.test_generate_data_imports()
    test_imports.test_preprocess_data_imports()
    test_imports.test_train_model_imports()
    test_imports.test_evaluate_model_imports()
    test_imports.test_create_plots_imports()
    
    test_errors = TestErrorHandling()
    test_errors.test_generate_data_missing_params()
    test_errors.test_preprocess_data_missing_file()
    
    print("All pipeline script tests completed successfully!")