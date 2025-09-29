# Import libraries
import cv2
import librosa
import numpy as np

# Define complex type hints
DATA_RETURN_TYPES = np.ndarray | tuple[np.ndarray, float]


def _load_single_data(path: str, file_type: str) -> DATA_RETURN_TYPES | None:
    """
    Loads a single data item from the given path.

    Args:
        path (str): The path to the data item.
        file_type (str): The type of the data item.

    Returns:
        DATA_RETURN_TYPES: The loaded data item.
    """
    if file_type == "image":
        # File type of image

        # Read image
        image = cv2.imread(path)

        # Convert image to RGB
        converted_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        # Return numpy array of image
        return np.array(converted_image)

    if file_type == "audio":
        # File type of audio

        # Load audio
        audio, sr = librosa.load(path)

        # Return audio and sampling rate
        return audio, sr

    # Something went wrong, return None for now
    return None
