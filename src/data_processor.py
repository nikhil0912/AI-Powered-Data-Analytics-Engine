"""
Data Processor Module

Handles data cleaning, preprocessing, validation, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data processing, cleaning, and feature engineering."""
    
    def __init__(self, missing_value_strategy: str = 'mean'):
        """
        Initialize DataProcessor.
        
        Args:
            missing_value_strategy: Strategy for handling missing values ('mean', 'median', 'drop')
        """
        self.missing_value_strategy = missing_value_strategy
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy=missing_value_strategy)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data with shape {df.shape}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Remove rows with any remaining NaN
        df = df.dropna()
        
        logger.info(f"Data cleaned to shape {df.shape}")
        return df
    
    def detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and flag outliers using z-score method.
        
        Args:
            df: Input DataFrame
            threshold: Z-score threshold
            
        Returns:
            DataFrame with outlier flags
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = pd.DataFrame(False, index=df.index, columns=numeric_cols)
        
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = z_scores > threshold
        
        return outliers
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from existing data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Example: Create ratio features for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            df_features['numeric_mean'] = df[numeric_cols].mean(axis=1)
            df_features['numeric_std'] = df[numeric_cols].std(axis=1)
        
        logger.info(f"Features engineered. New shape: {df_features.shape}")
        return df_features
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numeric data to [0, 1] range.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_normalized = df.copy()
        df_normalized[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df_normalized
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate data quality report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.astype(str).to_dict()
        }
