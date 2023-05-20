import pytest
import math
import pandas as pd

from model import load_data, add_directions, normalize_rssi, normalize_xyz, normalize_ap_coords, enrich_with_ap_coords
from model import F5_PATH, F6_PATH

TEST_NORMALIZE_DATA = pd.DataFrame({
    'AP1': [1, -60, -70, -80, -90, -100],
    'AP2': [-50, -60, -70, -80, -90, -100],
    'AP3': [-50, -60, 1, -80, -90, -100],
    'AP3_x': [0, 1, 2, 3, 0, 1],
    'AP3_y': [0, 1, 2, 3, 0, 1],
    'AP3_z': [0, 1, 2, 3, 0, 1],
    'AP4': [-50, -60, -70, -80, 1, -100],
    'x': [0, 1, 2, 3, 0, 1],
    'y': [0, 1, 2, 3, 0, 1],
    'z': [0, 1, 2, 3, 0, 1],
    'North': [1, 0, 0, 0, 1, 0],
})

TEST_DATA = pd.DataFrame({
    'AP031': [1, -60, -70, -80, -90, -100],
    'AP052': [-50, -60, -70, -80, -90, -100],
    'AP053': [-50, -60, 1, -80, -90, -100],
    'AP064': [-50, -60, -70, -80, 1, -100],
    'AP075': [-50, -60, -70, -80, 1, -100],
    'x': [0, 1, 2, 3, 0, 1],
    'y': [0, 1, 2, 3, 0, 1],
    'z': [0, 1, 2, 3, 0, 1],
})

TEST_AP_COORDS = pd.DataFrame({
    'x': [54, 33, 21],
    'y': [23, 54, 12],
    'z': [5, 6, 6],
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

    def test_normalize_ap_coords(self):
        result = normalize_ap_coords(TEST_NORMALIZE_DATA)

        assert result['AP3_x'].iloc[0] == 0
        assert math.isclose(result['AP3_y'].iloc[1], 0.01428571429, rel_tol=1e-9) # type: ignore
        assert math.isclose(result['AP3_z'].iloc[2], 0.16666666666666666, rel_tol=1e-9) # type: ignore

    def test_enrich_with_ap_coords(self):
        result = enrich_with_ap_coords(TEST_DATA, TEST_AP_COORDS)

        print()
        print(result)

        assert 'AP031_x' not in result.columns
        assert 'AP031_y' not in result.columns
        assert 'AP031_z' not in result.columns
        assert 'AP075_x' not in result.columns
        assert 'AP075_y' not in result.columns
        assert 'AP075_z' not in result.columns

        assert list(result.columns) == ['AP031', 'AP052', 'AP052_x', 'AP052_y', 'AP052_z', 'AP053', 'AP053_x', 'AP053_y', 'AP053_z', 'AP064', 'AP064_x', 'AP064_y', 'AP064_z', 'AP075', 'x', 'y', 'z']

        assert result['AP052_x'].iloc[0] == 54
        assert result['AP053_x'].iloc[1] == 33
        assert result['AP064_x'].iloc[2] == 21
        assert result['AP052_y'].iloc[0] == 23
        assert result['AP053_y'].iloc[1] == 54
        assert result['AP064_y'].iloc[2] == 12
        assert result['AP052_z'].iloc[0] == 5
        assert result['AP053_z'].iloc[1] == 6
        assert result['AP064_z'].iloc[2] == 6

if __name__ == '__main__':
    pytest.main([__file__])