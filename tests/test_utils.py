import pytest
import jax.numpy as jnp
import pandas as pd
from unittest.mock import patch
from ergodix.distributions import fetch_and_preprocess_uci, load_sonar, load_ionosphere

@patch("ergodix.distributions.utils.pd.read_csv")
def test_fetch_and_preprocess_uci(mock_read_csv):
    # Dummy data: 3 rows, 2 features, 1 target string
    df = pd.DataFrame({
        0: [10.0, 20.0, 30.0], 
        1: [1.0, 1.0, 1.0],  # Zero variance column edge-case
        2: ['pos', 'neg', 'pos']
    })
    mock_read_csv.return_value = df
    
    X, y = fetch_and_preprocess_uci("http://dummy_url", label_col=2, pos_class='pos')
    
    # Validation
    assert X.shape == (3, 2)
    assert y.shape == (3,)
    assert jnp.array_equal(y, jnp.array([1.0, 0.0, 1.0])), "Labels were not encoded correctly."
    
    # Z-score standardization check
    assert jnp.allclose(jnp.mean(X, axis=0), 0.0, atol=1e-6)
    
@patch("ergodix.distributions.utils.fetch_and_preprocess_uci")
def test_dataset_loaders(mock_fetch):
    mock_fetch.return_value = (jnp.zeros((10, 5)), jnp.ones(10))
    
    load_sonar()
    mock_fetch.assert_called_with(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
        label_col=60, 
        pos_class='M'
    )
    
    load_ionosphere()
    mock_fetch.assert_called_with(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        label_col=34, 
        pos_class='g'
    )