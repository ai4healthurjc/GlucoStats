from glucostats.datasets import load_glucodata


def test_load_glucodata_shape():
    df = load_glucodata()
    assert not df.empty
    assert df.shape[0] > 0
    assert "glucose" in df.columns or len(df.columns) > 1

