"""
Data utility functions for Alaska OCS Lease Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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

def calculate_morans_i(x, y, values, distance_threshold=None):
    """Calculate Moran's I spatial autocorrelation statistic with significance testing"""
    n = len(values)
    
    # Calculate distance matrix
    coords = np.column_stack([x, y])
    distances = squareform(pdist(coords))
    
    # Create spatial weights matrix (inverse distance or binary)
    if distance_threshold is None:
        # Use inverse distance weights
        weights = 1 / (distances + 1e-10)  # Add small value to avoid division by zero
        np.fill_diagonal(weights, 0)  # No self-weights
    else:
        # Binary weights within threshold
        weights = (distances <= distance_threshold).astype(float)
        np.fill_diagonal(weights, 0)
    
    # Normalize weights
    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]
    weights[np.isnan(weights)] = 0
    
    # Calculate Moran's I
    values_centered = values - np.mean(values)
    numerator = np.sum(weights * np.outer(values_centered, values_centered))
    denominator = np.sum(values_centered**2)
    
    morans_i = (n / np.sum(weights)) * (numerator / denominator)
    
    # Expected value and variance under null hypothesis
    expected_i = -1 / (n - 1)
    
    # Simplified variance calculation
    S0 = np.sum(weights)
    S1 = 0.5 * np.sum((weights + weights.T)**2)
    S2 = np.sum(np.sum(weights + weights.T, axis=1)**2)
    
    b2 = n * np.sum(values_centered**4) / (np.sum(values_centered**2)**2)
    
    variance_i = ((n*((n**2 - 3*n + 3)*S1 - n*S2 + 3*S0**2) - 
                   b2*((n**2 - n)*S1 - 2*n*S2 + 6*S0**2)) / 
                  ((n-1)*(n-2)*(n-3)*S0**2)) - expected_i**2
    
    # Z-score and p-value
    if variance_i > 0:
        z_score = (morans_i - expected_i) / np.sqrt(variance_i)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = np.nan
        p_value = np.nan
    
    return {
        'morans_i': morans_i,
        'expected_i': expected_i,
        'variance_i': variance_i,
        'z_score': z_score,
        'p_value': p_value
    }

def getis_ord_gi_star(x, y, values, distance_threshold_km=50):
    """Calculate Getis-Ord Gi* hot spot statistic with significance testing"""
    n = len(values)
    coords = np.column_stack([x, y])
    
    # Convert distance threshold from km to degrees (approximate)
    distance_threshold = distance_threshold_km / 111.0  # Rough conversion
    
    # Calculate distances
    distances = squareform(pdist(coords))
    
    gi_stats = []
    p_values = []
    
    for i in range(n):
        # Create weights for neighbors within threshold
        weights = (distances[i] <= distance_threshold).astype(float)
        
        # Calculate Gi* statistic
        if np.sum(weights) > 1:  # Need at least one neighbor
            weighted_sum = np.sum(weights * values)
            sum_weights = np.sum(weights)
            
            # Mean and variance calculations
            mean_val = np.mean(values)
            var_val = np.var(values)
            
            # Expected value and variance of Gi*
            expected_gi = sum_weights * mean_val
            variance_gi = (sum_weights * (n - sum_weights) * var_val) / (n - 1)
            
            if variance_gi > 0:
                gi_star = (weighted_sum - expected_gi) / np.sqrt(variance_gi)
                p_val = 2 * (1 - stats.norm.cdf(abs(gi_star)))
            else:
                gi_star = 0
                p_val = 1.0
        else:
            gi_star = 0
            p_val = 1.0
        
        gi_stats.append(gi_star)
        p_values.append(p_val)
    
    return np.array(gi_stats), np.array(p_values)

def optimize_dbscan_parameters(coords, eps_range=None, min_samples_range=None):
    """Optimize DBSCAN parameters using silhouette score"""
    if eps_range is None:
        # Calculate reasonable eps range based on k-distance
        k = 4
        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        k_distances = np.sort(distances[:, k-1])
        
        eps_range = np.linspace(k_distances[len(k_distances)//4], 
                               k_distances[3*len(k_distances)//4], 10)
    
    if min_samples_range is None:
        min_samples_range = range(3, min(15, len(coords)//10))
    
    best_score = -1
    best_params = None
    results = []
    
    from sklearn.metrics import silhouette_score
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(coords)
            
            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1 and n_clusters < len(coords) - 1:
                score = silhouette_score(coords, labels)
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette_score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
    
    return best_params, results

def calculate_spatial_confidence_intervals(x, y, confidence=0.95):
    """Calculate confidence intervals for spatial center and dispersion"""
    n = len(x)
    
    # Mean center
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Standard errors
    se_x = np.std(x) / np.sqrt(n)
    se_y = np.std(y) / np.sqrt(n)
    
    # Confidence intervals for center
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, n - 1)
    
    ci_x = (mean_x - t_critical * se_x, mean_x + t_critical * se_x)
    ci_y = (mean_y - t_critical * se_y, mean_y + t_critical * se_y)
    
    # Standard distance (measure of dispersion)
    std_distance = np.sqrt(np.mean((x - mean_x)**2 + (y - mean_y)**2))
    
    # Confidence interval for standard distance
    distances = np.sqrt((x - mean_x)**2 + (y - mean_y)**2)
    se_std_dist = np.std(distances) / np.sqrt(n)
    ci_std_dist = (std_distance - t_critical * se_std_dist, 
                   std_distance + t_critical * se_std_dist)
    
    return {
        'mean_center': (mean_x, mean_y),
        'ci_center_x': ci_x,
        'ci_center_y': ci_y,
        'standard_distance': std_distance,
        'ci_standard_distance': ci_std_dist,
        'n': n
    }
