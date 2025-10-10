#!/usr/bin/env python3
"""
Test script for the enhanced processing module.
"""

import pandas as pd
import numpy as np
import tempfile
import os
from project_name.processing import DataPreprocessor, preprocess_data, postprocess_predictions


def test_data_preprocessor():
    """Test the DataPreprocessor class functionality."""
    print("Testing DataPreprocessor...")
    
    # Create sample data
    data = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10.5, 20.3, 30.1, 40.7, 50.9],
        'category1': ['A', 'B', 'A', 'C', 'B'],
        'category2': ['X', 'Y', 'X', 'Z', 'Y']
    })
    
    # Test 1: Basic fit and transform
    print("\n1. Testing basic fit and transform...")
    preprocessor = DataPreprocessor()
    transformed = preprocessor.fit_transform(data)
    print(f"Original shape: {data.shape}")
    print(f"Transformed shape: {transformed.shape}")
    print(f"Transformed columns: {list(transformed.columns)}")
    
    # Test 2: Transform new data
    print("\n2. Testing transform on new data...")
    new_data = pd.DataFrame({
        'numeric1': [6, 7],
        'numeric2': [60.1, 70.2],
        'category1': ['A', 'C'],
        'category2': ['X', 'Z']
    })
    new_transformed = preprocessor.transform(new_data)
    print(f"New data transformed shape: {new_transformed.shape}")
    
    # Test 3: Save and load
    print("\n3. Testing save and load...")
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        preprocessor.save(tmp_path)
        loaded_preprocessor = DataPreprocessor.load(tmp_path)
        
        # Test that loaded preprocessor works the same
        reloaded_transformed = loaded_preprocessor.transform(data)
        print(f"Loaded preprocessor works: {np.allclose(transformed.values, reloaded_transformed.values)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Test 4: Error handling
    print("\n4. Testing error handling...")
    try:
        empty_preprocessor = DataPreprocessor()
        empty_preprocessor.transform(data)  # Should raise NotFittedError
    except Exception as e:
        print(f"Correctly raised error: {type(e).__name__}")


def test_preprocess_data_function():
    """Test the preprocess_data function."""
    print("\n\nTesting preprocess_data function...")
    
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA']
    })
    
    # Test with new preprocessor
    print("\n1. Testing with new preprocessor...")
    result1 = preprocess_data(data)
    print(f"Result shape: {result1.shape}")
    
    # Test with existing preprocessor
    print("\n2. Testing with existing preprocessor...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(data)
    result2 = preprocess_data(data, preprocessor)
    print(f"Result shape: {result2.shape}")
    print(f"Results match: {np.allclose(result1.values, result2.values)}")


def test_postprocess_predictions():
    """Test the postprocess_predictions function."""
    print("\n\nTesting postprocess_predictions function...")
    
    # Test with numpy array
    print("\n1. Testing with numpy array...")
    predictions = np.array([0.1, 0.5, 0.9, 0.3])
    result = postprocess_predictions(predictions)
    print(f"Input type: {type(predictions)}")
    print(f"Output type: {type(result)}")
    
    # Test with pandas Series
    print("\n2. Testing with pandas Series...")
    series_predictions = pd.Series([1, 2, 3, 4])
    result = postprocess_predictions(series_predictions)
    print(f"Input type: {type(series_predictions)}")
    print(f"Output type: {type(result)}")
    
    # Test with inverse transformation
    print("\n3. Testing with inverse transformation...")
    def simple_inverse(x):
        return x * 2
    
    result = postprocess_predictions(predictions, simple_inverse)
    print(f"Original: {predictions}")
    print(f"Transformed: {result}")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n\nTesting edge cases...")
    
    # Test with only numeric data
    print("\n1. Testing with only numeric data...")
    numeric_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4.5, 5.5, 6.5]
    })
    preprocessor = DataPreprocessor()
    result = preprocessor.fit_transform(numeric_data)
    print(f"Numeric-only result shape: {result.shape}")
    
    # Test with only categorical data
    print("\n2. Testing with only categorical data...")
    categorical_data = pd.DataFrame({
        'cat1': ['A', 'B', 'C'],
        'cat2': ['X', 'Y', 'Z']
    })
    preprocessor = DataPreprocessor()
    result = preprocessor.fit_transform(categorical_data)
    print(f"Categorical-only result shape: {result.shape}")
    
    # Test with empty data
    print("\n3. Testing error with empty data...")
    try:
        empty_data = pd.DataFrame()
        preprocessor.fit(empty_data)
    except Exception as e:
        print(f"Correctly raised error: {type(e).__name__}")


if __name__ == "__main__":
    test_data_preprocessor()
    test_preprocess_data_function()
    test_postprocess_predictions()
    test_edge_cases()
    print("\nAll tests completed!")