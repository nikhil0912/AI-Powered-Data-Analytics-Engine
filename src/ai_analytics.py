"""
AI Analytics Module

Integrates Generative AI models for intelligent data analysis and insights.
"""

import pandas as pd
import json
from typing import Dict, List, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAnalytics:
    """Handles AI-powered analytics and insights generation."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize AIAnalytics.
        
        Args:
            model: LLM model to use for analysis
        """
        self.model = model
        self.analysis_history = []
        
    def analyze(self, df: pd.DataFrame, context: Optional[str] = None) -> Dict:
        """
        Perform AI-powered data analysis.
        
        Args:
            df: DataFrame to analyze
            context: Additional context for analysis
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing dataframe with shape {df.shape}")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'summary_statistics': df.describe().to_dict(),
            'correlations': df.corr().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            'missing_values': df.isnull().sum().to_dict(),
        }
        
        self.analysis_history.append(analysis)
        return analysis
    
    def generate_insights(self, df: pd.DataFrame) -> List[str]:
        """
        Generate natural language insights from data.
        
        Args:
            df: DataFrame to generate insights from
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Statistical insights
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            max_col = df[numeric_cols].max().idxmax()
            min_col = df[numeric_cols].min().idxmin()
            insights.append(f"Maximum value found in column '{max_col}'")
            insights.append(f"Minimum value found in column '{min_col}'")
        
        # Data quality insights
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        insights.append(f"Data completeness: {100 * (1 - missing_ratio):.2f}%")
        
        # Pattern detection
        if len(df) > 0:
            insights.append(f"Dataset contains {len(df)} records across {len(df.columns)} features")
        
        return insights
    
    def generate_summary(self, df: pd.DataFrame, insights: Dict) -> str:
        """
        Generate natural language summary of the data.
        
        Args:
            df: DataFrame being analyzed
            insights: Dictionary of insights
            
        Returns:
            Summary string
        """
        summary = f"""
        DATA SUMMARY REPORT
        ====================
        Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Dataset Overview:
        - Total Records: {len(df)}
        - Total Features: {len(df.columns)}
        - Data Completeness: {100 * (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))):.2f}%
        
        Columns: {', '.join(df.columns.tolist())}
        
        Key Findings:
        - Shape: {df.shape}
        - Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        """
        return summary
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect patterns in the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            'high_correlation_pairs': [],
            'outlier_columns': [],
            'skewed_distributions': [],
        }
        
        # Detect high correlations
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        patterns['high_correlation_pairs'].append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
        
        return patterns
    
    def export_report(self, analysis: Dict, filepath: str) -> None:
        """
        Export analysis report to JSON file.
        
        Args:
            analysis: Analysis dictionary
            filepath: Path to save report
        """
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Report exported to {filepath}")
