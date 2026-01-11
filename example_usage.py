"""
Example Usage: AI-Powered Data Analytics Engine

This script demonstrates how to use the AI-Powered Data Analytics Engine
for data processing, analysis, and visualization.
"""

import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.ai_analytics import AIAnalytics
from src.visualizer import Visualizer

def create_sample_data():
    """
    Create sample dataset for demonstration.
    """
    np.random.seed(42)
    data = {
        'revenue': np.random.randint(1000, 100000, 100),
        'customers': np.random.randint(10, 1000, 100),
        'product_cost': np.random.randint(100, 50000, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr'], 100)
    }
    return pd.DataFrame(data)

def main():
    print("="*60)
    print("AI-Powered Data Analytics Engine - Example Usage")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample dataset...")
    df = create_sample_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Initialize modules
    print("\n2. Initializing modules...")
    processor = DataProcessor(missing_value_strategy='mean')
    ai_engine = AIAnalytics(model='gpt-3.5-turbo')
    visualizer = Visualizer(style='seaborn')
    print("   ✓ DataProcessor initialized")
    print("   ✓ AIAnalytics initialized")
    print("   ✓ Visualizer initialized")
    
    # Data Processing
    print("\n3. Data Processing...")
    df_clean = processor.clean_data(df)
    print(f"   ✓ Data cleaned: {df_clean.shape}")
    
    df_features = processor.engineer_features(df_clean)
    print(f"   ✓ Features engineered: {df_features.shape}")
    
    # Data Quality Report
    print("\n4. Data Quality Report:")
    quality_report = processor.get_data_quality_report(df_clean)
    print(f"   - Total Records: {quality_report['total_rows']}")
    print(f"   - Total Features: {quality_report['total_columns']}")
    print(f"   - Duplicate Rows: {quality_report['duplicate_rows']}")
    print(f"   - Memory Usage: {quality_report['memory_usage_mb']:.2f} MB")
    
    # AI Analysis
    print("\n5. AI-Powered Analysis...")
    analysis = ai_engine.analyze(df_clean)
    print("   ✓ Analysis complete")
    
    # Generate Insights
    print("\n6. Generated Insights:")
    insights = ai_engine.generate_insights(df_clean)
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    # Pattern Detection
    print("\n7. Pattern Detection:")
    patterns = ai_engine.detect_patterns(df_clean)
    if patterns['high_correlation_pairs']:
        for pair in patterns['high_correlation_pairs']:
            print(f"   - High correlation: {pair['col1']} <-> {pair['col2']}: {pair['correlation']:.3f}")
    else:
        print("   - No high correlations detected")
    
    # Generate Summary
    print("\n8. Data Summary:")
    summary = ai_engine.generate_summary(df_clean, analysis)
    print(summary)
    
    # Outlier Detection
    print("\n9. Outlier Detection:")
    outliers = processor.detect_outliers(df_clean, threshold=2.0)
    print(f"   - Outliers detected: {outliers.sum().sum()} anomalies")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
