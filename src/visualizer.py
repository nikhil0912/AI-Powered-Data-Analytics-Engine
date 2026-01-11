"""
Visualizer Module

Handles data visualization and dashboard creation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
    """Handles data visualization and reporting."""
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize Visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.style = style
    
    def plot_distribution(self, df: pd.DataFrame, column: str, figsize: tuple = (10, 6)) -> None:
        """
        Plot distribution of a column.
        
        Args:
            df: DataFrame containing data
            column: Column name to plot
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        if df[column].dtype in ['int64', 'float64']:
            ax.hist(df[column], bins=30, edgecolor='black')
        else:
            df[column].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Distribution of {column}')
        plt.tight_layout()
        logger.info(f"Distribution plot created for {column}")
    
    def plot_correlation_matrix(self, df: pd.DataFrame, figsize: tuple = (10, 8)) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            df: DataFrame containing numeric data
            figsize: Figure size
        """
        numeric_df = df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        logger.info("Correlation matrix plot created")
    
    def plot_summary(self, df: pd.DataFrame, figsize: tuple = (14, 10)) -> None:
        """
        Create a summary dashboard of the dataset.
        
        Args:
            df: DataFrame to summarize
            figsize: Figure size
        """
        numeric_cols = df.select_dtypes(include=['number']).columns
        n_cols = min(4, len(numeric_cols))
        
        if n_cols == 0:
            logger.warning("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for idx, col in enumerate(numeric_cols[:4]):
            if idx < 4:
                axes[idx].hist(df[col], bins=20, edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
        
        plt.suptitle('Data Summary Dashboard')
        plt.tight_layout()
        logger.info("Summary dashboard created")
    
    def export_plot(self, filepath: str, dpi: int = 300) -> None:
        """
        Export current plot to file.
        
        Args:
            filepath: Path to save plot
            dpi: Resolution in DPI
        """
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot exported to {filepath}")
