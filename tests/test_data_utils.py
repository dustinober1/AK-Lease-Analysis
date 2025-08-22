import pytest
import pandas as pd
import numpy as np
from src.data_utils import (
    load_lease_data, 
    create_derived_features, 
    get_summary_stats, 
    filter_data_by_criteria,
    detect_outliers,
    prepare_ml_features
)

@pytest.fixture
def sample_dataframe():
    data = {
        'LEASE_EXPIR_DATE': ['2025-01-01', '2026-01-01', '2027-01-01', '2028-01-01', '2029-01-01'],
        'LEASE_EFF_DATE': ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'],
        'SALE_DATE': ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'],
        'BID_AMOUNT': [1000, 2000, 1500, 2500, 100000],
        'CURRENT_AREA': [100, 200, 150, 250, 500],
        'LEASE_IS_ACTIVE': ['Y', 'N', 'Y', 'N', 'Y'],
        'MMS_PLAN_AREA_CD': ['BFT', 'CHU', 'BFT', 'CHU', 'BFT'],
        'BUS_ASC_NAME': ['COMPANY_A', 'COMPANY_B', 'COMPANY_A', 'COMPANY_B', 'COMPANY_A'],
        'ROYALTY_RATE': [12.5, 12.5, 18.75, 12.5, 18.75],
        'PRIMARY_TERM': [5, 10, 5, 10, 5]
    }
    df = pd.DataFrame(data)
    for col in ['LEASE_EXPIR_DATE', 'LEASE_EFF_DATE', 'SALE_DATE']:
        df[col] = pd.to_datetime(df[col])
    df = create_derived_features(df)
    return df

def test_load_lease_data(tmp_path, sample_dataframe):
    # Create a dummy csv file
    file_path = tmp_path / "test_leases.csv"
    sample_dataframe.to_csv(file_path, index=False)

    # Test loading the data
    df = load_lease_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert pd.api.types.is_datetime64_any_dtype(df['LEASE_EXPIR_DATE'])

def test_create_derived_features(sample_dataframe):
    # Test creating derived features
    df = create_derived_features(sample_dataframe)
    assert 'SALE_YEAR' in df.columns
    assert 'BID_PER_HECTARE' in df.columns
    assert 'LEASE_DURATION' in df.columns
    assert 'IS_ACTIVE' in df.columns
    assert 'DECADE' in df.columns
    assert df['SALE_YEAR'].iloc[0] == 2020
    assert df['BID_PER_HECTARE'].iloc[0] == 10.0
    assert df['IS_ACTIVE'].iloc[0] == 1

def test_get_summary_stats(sample_dataframe):
    stats = get_summary_stats(sample_dataframe)
    assert isinstance(stats, dict)
    assert stats['total_leases'] == 5
    assert stats['total_bid_value'] == 107000

def test_filter_data_by_criteria(sample_dataframe):
    # Test filtering by active only
    active_df = filter_data_by_criteria(sample_dataframe, active_only=True)
    assert len(active_df) == 3
    assert all(active_df['LEASE_IS_ACTIVE'] == 'Y')

    # Test filtering by min bid
    high_bid_df = filter_data_by_criteria(sample_dataframe, min_bid=2000)
    assert len(high_bid_df) == 3
    assert all(high_bid_df['BID_AMOUNT'] >= 2000)

    # Test filtering by planning area
    bft_df = filter_data_by_criteria(sample_dataframe, planning_areas=['BFT'])
    assert len(bft_df) == 3
    assert all(bft_df['MMS_PLAN_AREA_CD'] == 'BFT')

    # Test filtering by year range
    df_2020 = filter_data_by_criteria(sample_dataframe, year_range=(2020, 2021))
    assert len(df_2020) == 2
    assert all(df_2020['SALE_YEAR'].isin([2020, 2021]))

def test_detect_outliers(sample_dataframe):
    # Test detecting outliers with iqr method
    outliers_iqr = detect_outliers(sample_dataframe, 'BID_AMOUNT', method='iqr')
    assert isinstance(outliers_iqr, pd.Series)
    assert outliers_iqr.sum() == 1
    assert outliers_iqr.iloc[4] == True

def test_prepare_ml_features(sample_dataframe):
    features_df, target_series = prepare_ml_features(sample_dataframe, 'BID_AMOUNT')
    assert isinstance(features_df, pd.DataFrame)
    assert isinstance(target_series, pd.Series)
    assert not features_df.isnull().values.any()
    assert len(features_df) == len(target_series)
