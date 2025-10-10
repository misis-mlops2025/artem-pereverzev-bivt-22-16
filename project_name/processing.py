import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.exceptions import NotFittedError
import logging
from typing import Union, Optional, Dict, Any
import joblib
import os

# Set up logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A robust data preprocessing class for handling numeric and categorical features.
    
    This class provides methods for preprocessing data and saving/loading the fitted
    transformers for consistent preprocessing during training and inference.
    """
    
    def __init__(self, 
                 numeric_scaler: Optional[RobustScaler] = None,
                 categorical_encoder: Optional[OneHotEncoder] = None,
                 numeric_columns: Optional[list] = None,
                 categorical_columns: Optional[list] = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            numeric_scaler: Pre-initialized RobustScaler instance
            categorical_encoder: Pre-initialized OneHotEncoder instance
            numeric_columns: List of numeric column names
            categorical_columns: List of categorical column names
        """
        self.numeric_scaler = numeric_scaler or RobustScaler()
        self.categorical_encoder = categorical_encoder or OneHotEncoder(
            sparse_output=False, 
            handle_unknown='ignore',
            drop=None
        )
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.is_fitted = False
        
    def _detect_column_types(self, data: pd.DataFrame) -> None:
        """
        Detect numeric and categorical columns from the data.
        
        Args:
            data: Input DataFrame
        """
        if self.numeric_columns is None:
            self.numeric_columns = data.select_dtypes(
                include=["int64", "float64", "int32", "float32"]
            ).columns.tolist()
            
        if self.categorical_columns is None:
            self.categorical_columns = data.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
            
        logger.info(f"Detected {len(self.numeric_columns)} numeric columns: {self.numeric_columns}")
        logger.info(f"Detected {len(self.categorical_columns)} categorical columns: {self.categorical_columns}")
    
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on the training data.
        
        Args:
            data: Training data DataFrame
            
        Returns:
            self: Fitted preprocessor instance
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
            
        if data.empty:
            raise ValueError("Input data cannot be empty")
            
        self._detect_column_types(data)
        
        # Fit numeric scaler
        if self.numeric_columns:
            logger.info("Fitting numeric scaler...")
            self.numeric_scaler.fit(data[self.numeric_columns])
        
        # Fit categorical encoder
        if self.categorical_columns:
            logger.info("Fitting categorical encoder...")
            self.categorical_encoder.fit(data[self.categorical_columns])
        
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted preprocessor.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise NotFittedError("Preprocessor must be fitted before transforming data")
            
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
            
        processed_parts = []
        
        # Transform numeric columns
        if self.numeric_columns:
            # Check if all numeric columns are present
            missing_numeric = set(self.numeric_columns) - set(data.columns)
            if missing_numeric:
                raise ValueError(f"Missing numeric columns: {missing_numeric}")
                
            logger.info("Transforming numeric columns...")
            scaled = self.numeric_scaler.transform(data[self.numeric_columns])
            scaled_df = pd.DataFrame(scaled, columns=self.numeric_columns, index=data.index)
            processed_parts.append(scaled_df)
        
        # Transform categorical columns
        if self.categorical_columns:
            # Check if all categorical columns are present
            missing_categorical = set(self.categorical_columns) - set(data.columns)
            if missing_categorical:
                raise ValueError(f"Missing categorical columns: {missing_categorical}")
                
            logger.info("Transforming categorical columns...")
            encoded = self.categorical_encoder.transform(data[self.categorical_columns])
            encoded_cols = self.categorical_encoder.get_feature_names_out(self.categorical_columns)
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=data.index)
            processed_parts.append(encoded_df)
        
        # Handle case where no columns were processed
        if not processed_parts:
            logger.warning("No columns were processed, returning original data")
            return data.copy()
        
        result = pd.concat(processed_parts, axis=1)
        logger.info(f"Transformed data shape: {result.shape}")
        return result
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data in one step.
        
        Args:
            data: Data to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(data).transform(data)
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise NotFittedError("Cannot save unfitted preprocessor")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'numeric_scaler': self.numeric_scaler,
            'categorical_encoder': self.categorical_encoder,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded DataPreprocessor instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        preprocessor_data = joblib.load(filepath)
        
        preprocessor = cls(
            numeric_scaler=preprocessor_data['numeric_scaler'],
            categorical_encoder=preprocessor_data['categorical_encoder'],
            numeric_columns=preprocessor_data['numeric_columns'],
            categorical_columns=preprocessor_data['categorical_columns']
        )
        preprocessor.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def preprocess_data(data: pd.DataFrame, preprocessor: Optional[DataPreprocessor] = None) -> pd.DataFrame:
    """
    Preprocess data using the provided or newly created preprocessor.
    
    Args:
        data: Input DataFrame to preprocess
        preprocessor: Optional pre-fitted DataPreprocessor instance
        
    Returns:
        Preprocessed DataFrame
    """
    if preprocessor is None:
        preprocessor = DataPreprocessor()
        return preprocessor.fit_transform(data)
    else:
        return preprocessor.transform(data)


def postprocess_predictions(predictions: Union[np.ndarray, pd.Series, list], 
                          inverse_transform_func: Optional[callable] = None) -> Union[np.ndarray, pd.Series]:
    """
    Postprocess model predictions with optional inverse transformation.
    
    Args:
        predictions: Raw model predictions
        inverse_transform_func: Optional function to apply inverse transformation
        
    Returns:
        Postprocessed predictions
    """
    if predictions is None:
        raise ValueError("Predictions cannot be None")
    
    # Convert to numpy array for consistency
    if isinstance(predictions, (pd.Series, list)):
        predictions = np.array(predictions)
    
    # Apply inverse transformation if provided
    if inverse_transform_func is not None:
        try:
            predictions = inverse_transform_func(predictions)
            logger.info("Applied inverse transformation to predictions")
        except Exception as e:
            logger.warning(f"Failed to apply inverse transformation: {e}")
    
    # Handle edge cases
    if np.any(np.isnan(predictions)):
        logger.warning("Predictions contain NaN values")
    
    if np.any(np.isinf(predictions)):
        logger.warning("Predictions contain infinite values")
    
    return predictions