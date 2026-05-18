import pandas as pd

def test_data_file_exists():
    df = pd.read_csv("data/housing_v1.csv")
    assert df.shape[0] > 10
