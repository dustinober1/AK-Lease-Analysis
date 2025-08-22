import pytest
import pandas as pd
import numpy as np
from src.data_utils import validate_data_quality

@pytest.fixture
def sample_dataframe():
    data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
        'col3': ['a', 'b', 'c', 'd', 'e'],
        'col4': [1, 2, 3, 2, 1]
    }
    return pd.DataFrame(data)

def test_validate_data_quality(sample_dataframe):
    quality_report = validate_data_quality(sample_dataframe)
    assert isinstance(quality_report, dict)
    assert quality_report['total_rows'] == 5
    assert quality_report['duplicate_rows'] == 0
    assert quality_report['missing_data']['col1'] == 0
    assert quality_report['numerical_ranges']['col1']['min'] == 1
    assert quality_report['numerical_ranges']['col1']['max'] == 5
