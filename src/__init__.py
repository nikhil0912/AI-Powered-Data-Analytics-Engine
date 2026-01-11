"""
AI-Powered Data Analytics Engine

A comprehensive solution combining Generative AI with advanced data analytics.
This package provides tools for intelligent data processing, LLM-powered analysis,
and production-ready analytics APIs.
"""

__version__ = "1.0.0"
__author__ = "Nikhil Budhiraja"
__email__ = "nikhilbudhiraja002@gmail.com"

from .data_processor import DataProcessor
from .ai_analytics import AIAnalytics
from .visualizer import Visualizer

__all__ = [
    'DataProcessor',
    'AIAnalytics',
    'Visualizer',
]
