# Import libraries
import numpy as np

# Specify complex type hints
DATA_RETURN_TYPES = np.ndarray | tuple[np.ndarray, float]
GETITEM_RETURN_TYPE = tuple[DATA_RETURN_TYPES, str]

# Random Number Generator.
# Set seed to 42 for now for reproducibility
# There is no other way to set this number generator globally in main.py
RNG = np.random.default_rng(seed=42)

# File not found message
INVALID_FILE_ERROR = 'Directory not found: "{}"'

# Invalid param message
INVALID_S_T_MSG = "{} must be greater than {}"
