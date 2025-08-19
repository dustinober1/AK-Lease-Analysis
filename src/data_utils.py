"""
Data utility functions for Alaska OCS Lease Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_lease_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the lease dataset
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Convert date columns
    date_columns = ['LEASE_EXPIR_DATE', 'LEASE_EFF_DATE', 'LEASE_EXPT_EXPIR', 
                   'LEASE_STATUS_CHANGE_DT', 'LSE_STAT_EFF_DT', 'SALE_DATE']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for analysis
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional features
    """
    df_copy = df.copy()
    
    # Extract year from sale date
    df_copy['SALE_YEAR'] = df_copy['SALE_DATE'].dt.year
    
    # Calculate bid per hectare
    df_copy['BID_PER_HECTARE'] = df_copy['BID_AMOUNT'] / df_copy['CURRENT_AREA'].replace(0, np.nan)
    
    # Calculate lease duration
    df_copy['LEASE_DURATION'] = (df_copy['LEASE_EXPIR_DATE'] - df_copy['LEASE_EFF_DATE']).dt.days
    
    # Binary active indicator
    df_copy['IS_ACTIVE'] = (df_copy['LEASE_IS_ACTIVE'] == 'Y').astype(int)
    
    # Decade grouping
    df_copy['DECADE'] = (df_copy['SALE_YEAR'] // 10) * 10
    
    return df_copy

def get_summary_stats(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive summary statistics
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of summary statistics
    """
    stats = {
        'total_leases': len(df),
        'total_bid_value': df['BID_AMOUNT'].sum(),
        'avg_bid': df['BID_AMOUNT'].mean(),
        'median_bid': df['BID_AMOUNT'].median(),
        'total_area': df['CURRENT_AREA'].sum(),
        'avg_area': df['CURRENT_AREA'].mean(),
        'active_leases': (df['LEASE_IS_ACTIVE'] == 'Y').sum(),
        'inactive_leases': (df['LEASE_IS_ACTIVE'] == 'N').sum(),
        'unique_companies': df['BUS_ASC_NAME'].nunique(),
        'date_range': (df['SALE_DATE'].min(), df['SALE_DATE'].max()),
        'planning_areas': df['MMS_PLAN_AREA_CD'].nunique()
    }
    
    return stats

def filter_data_by_criteria(df: pd.DataFrame, 
                           active_only: bool = False,
                           min_bid: Optional[float] = None,
                           planning_areas: Optional[List[str]] = None,
                           year_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """
    Filter data based on various criteria
    
    Args:
        df: Input DataFrame
        active_only: Filter for active leases only
        min_bid: Minimum bid amount threshold
        planning_areas: List of planning areas to include
        year_range: Tuple of (start_year, end_year)
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if active_only:
        filtered_df = filtered_df[filtered_df['LEASE_IS_ACTIVE'] == 'Y']
    
    if min_bid is not None:
        filtered_df = filtered_df[filtered_df['BID_AMOUNT'] >= min_bid]
    
    if planning_areas is not None:
        filtered_df = filtered_df[filtered_df['MMS_PLAN_AREA_CD'].isin(planning_areas)]
    
    if year_range is not None:
        start_year, end_year = year_range
        filtered_df = filtered_df[
            (filtered_df['SALE_YEAR'] >= start_year) & 
            (filtered_df['SALE_YEAR'] <= end_year)
        ]
    
    return filtered_df

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """
    Detect outliers in a numerical column
    
    Args:
        df: Input DataFrame
        column: Column name to analyze
        method: Method to use ('iqr' or 'zscore')
        
    Returns:
        Boolean Series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = z_scores > 3
    
    return outliers

def prepare_ml_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for machine learning
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        
    Returns:
        Tuple of (features_df, target_series)
    """
    # Select numerical features
    numerical_features = ['CURRENT_AREA', 'ROYALTY_RATE', 'PRIMARY_TERM', 'SALE_YEAR']
    
    # Create feature matrix
    features_df = df[numerical_features].copy()
    
    # Handle missing values
    features_df = features_df.dropna()
    
    # Get corresponding target values
    target_series = df.loc[features_df.index, target_col]
    
    # Remove zero/negative targets if needed
    if target_col == 'BID_AMOUNT':
        valid_mask = target_series > 0
        features_df = features_df[valid_mask]
        target_series = target_series[valid_mask]
    
    return features_df, target_series

def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    Validate data quality and identify issues
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of data quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'duplicate_rows': df.duplicated().sum(),
        'missing_data': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numerical_ranges': {}
    }
    
    # Check numerical column ranges
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        quality_report['numerical_ranges'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'negative_values': (df[col] < 0).sum(),
            'zero_values': (df[col] == 0).sum()
        }
    
    return quality_report