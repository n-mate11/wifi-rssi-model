import pytest
import math
import pandas as pd

from model import load_data, add_directions, normalize_rssi, normalize_xyz
from model import F5_PATH, F6_PATH

TEST_NORMALIZE_DATA = pd.DataFrame({
    'AP1': [1, -60, -70, -80, -90, -100],
    'AP2': [-50, -60, -70, -80, -90, -100],
    'AP3': [-50, -60, 1, -80, -90, -100],
    'AP3_x': [0, 1, 2, 3, 0, 1],
    'AP4': [-50, -60, -70, -80, 1, -100],
    'x': [0, 1, 2, 3, 0, 1],
    'y': [0, 1, 2, 3, 0, 1],
    'z': [0, 1, 2, 3, 0, 1],
    'North': [1, 0, 0, 0, 1, 0],
})

class TestModel:
    def test_load_data(self):
        result = load_data(F5_PATH)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1800, 327)

        result = load_data(F6_PATH)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1860, 327)

    def test_add_directions(self):
        data = pd.DataFrame({
            'rssi_1': [-50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100],
            'rssi_2': [-50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100],
            'rssi_3': [-50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100],
            'rssi_4': [-50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100, -50, -60, -70, -80, -90, -100],
            'x': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3],
            'y': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3],
            'z': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3],
        })
        # Add directions to the DataFrame
        result = add_directions(data)

        # Check that the directions were added correctly
        assert 'North' in result.columns
        assert 'East' in result.columns
        assert 'South' in result.columns
        assert 'West' in result.columns
        assert result['North'].iloc[0] == 1
        assert result['East'].iloc[1] == 0
        assert result['South'].iloc[2] == 0
        assert result['West'].iloc[3] == 0
        assert result['North'].iloc[16] == 0
        assert result['East'].iloc[16] == 1
        assert result['South'].iloc[16] == 0
        assert result['West'].iloc[16] == 0

    def test_normalize_rssi(self):
        # Normalize the RSSI values
        result = normalize_rssi(TEST_NORMALIZE_DATA)

        # Check that the RSSI values were normalized correctly
        assert result['AP1'].iloc[0] == 0
        assert result['AP3'].iloc[3] == [0.2]
        assert all(result['x'] == [0, 1, 2, 3, 0, 1])
        assert all(result['AP3_x'] == [0, 1, 2, 3, 0, 1])
        assert all(result['North'] == [1, 0, 0, 0, 1, 0])

    def test_normalize_xyz(self):
        result = normalize_xyz(TEST_NORMALIZE_DATA)

        assert all(result['AP3_x'] == [0, 1, 2, 3, 0, 1])
        assert result['x'].iloc[0] == 0
        assert math.isclose(result['y'].iloc[2], 0.02857142857, rel_tol=1e-9) # type: ignore
        assert result['z'].iloc[3] == 0.25


if __name__ == '__main__':
    pytest.main([__file__])