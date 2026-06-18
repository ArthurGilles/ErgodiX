import jax.numpy as jnp
from jaxtyping import Array, Float
import pandas as pd
import numpy as np

def fetch_and_preprocess_uci(url: str, label_col: int, pos_class: str) -> tuple[Float[Array, "N d"], Float[Array, "N"]]:
    """Fetches a dataset, standardizes features, and encodes labels to {0.0, 1.0}."""
    df = pd.read_csv(url, header=None)
    
    # Extract features and target
    X_raw = df.drop(columns=[label_col]).values.astype(float)
    y_raw = df[label_col].values
    
    # Standardize features (Z-score normalization)
    X_mean = np.mean(X_raw, axis=0)
    X_std = np.std(X_raw, axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)  # Prevent division by zero
    X_scaled = (X_raw - X_mean) / X_std
    
    # Encode targets as floats
    y_encoded = (y_raw == pos_class).astype(float)
    
    return jnp.array(X_scaled), jnp.array(y_encoded)

def load_sonar() -> tuple[Float[Array, "N 60"], Float[Array, "N"]]:
    """Load Sonar dataset (60 features)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    return fetch_and_preprocess_uci(url, label_col=60, pos_class='M')

def load_ionosphere() -> tuple[Float[Array, "N 34"], Float[Array, "N"]]:
    """Load Ionosphere dataset (34 features)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    return fetch_and_preprocess_uci(url, label_col=34, pos_class='g')