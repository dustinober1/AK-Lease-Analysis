import pytest
import pandas as pd
import numpy as np
from src.data_utils import (
    calculate_morans_i, 
    getis_ord_gi_star, 
    optimize_dbscan_parameters, 
    calculate_spatial_confidence_intervals
)

@pytest.fixture
def geospatial_dataframe():
    data = {
        'longitude': [-148.7, -149.0, -148.9, -150.0, -150.1],
        'latitude': [70.2, 70.1, 70.15, 69.9, 69.95],
        'values': [10, 20, 15, 30, 25]
    }
    return pd.DataFrame(data)

def test_calculate_morans_i(geospatial_dataframe):
    result = calculate_morans_i(
        geospatial_dataframe['longitude'], 
        geospatial_dataframe['latitude'], 
        geospatial_dataframe['values']
    )
    assert isinstance(result, dict)
    assert 'morans_i' in result

def test_getis_ord_gi_star(geospatial_dataframe):
    gi_stats, p_values = getis_ord_gi_star(
        geospatial_dataframe['longitude'], 
        geospatial_dataframe['latitude'], 
        geospatial_dataframe['values']
    )
    assert isinstance(gi_stats, np.ndarray)
    assert isinstance(p_values, np.ndarray)
    assert len(gi_stats) == len(geospatial_dataframe)

def test_optimize_dbscan_parameters(geospatial_dataframe):
    coords = geospatial_dataframe[['longitude', 'latitude']].values
    best_params, results = optimize_dbscan_parameters(coords)
    assert isinstance(best_params, dict) or best_params is None
    assert isinstance(results, list)

def test_calculate_spatial_confidence_intervals(geospatial_dataframe):
    result = calculate_spatial_confidence_intervals(
        geospatial_dataframe['longitude'], 
        geospatial_dataframe['latitude']
    )
    assert isinstance(result, dict)
    assert 'mean_center' in result
    assert 'ci_center_x' in result
    assert 'ci_center_y' in result
    assert 'standard_distance' in result
    assert 'ci_standard_distance' in result
