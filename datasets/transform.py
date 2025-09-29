# Import libaries
import numpy as np
from librosa.feature import melspectrogram

# Import from other modules
from datasets.baseclasses import DataTransform
from datasets.utils import INVALID_S_T_MSG, RNG


class SquareErasingTransform(DataTransform):
    """
    Square Erasing Transform

    Attributes:
        s (float): The size of the square to be erased.

    Methods:
        process(data): Processes the data and returns the transformed data.
    """

    def __init__(self, s: int) -> None:
        """
        Initializes the SquareErasingTransform class.

        Args:
            s (int): The size of the square to be erased.

        Raises:
            ValueError: If s is less than or equal to 1.
        """
        if s <= 1:
            raise ValueError(INVALID_S_T_MSG.format("s", "1"))

        self._s = s

    @property
    def s(self) -> int:
        """
        Returns the size of the square to be erased.

        Returns:
            int: The size of the square to be erased.
        """
        return self._s

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Processes the data and returns the transformed data.

        Args:
            data (np.ndarray): The data to be processed.

        Returns:
            np.ndarray: The transformed data.
        """

        # Pick s which is an integer between 1 and self._s
        s = RNG.integers(1, self._s + 1)

        # Create a copy of the data
        data_copy = data.copy()

        # If the shape of the data is less than s, return the data
        if data.shape[0] < s or data.shape[1] < s:
            return data_copy

        # Pick a random location to erase
        x = RNG.integers(0, data.shape[1] - s + 1)
        y = RNG.integers(0, data.shape[0] - s + 1)

        # Erase the square
        data_copy[y : y + s, x : x + s] = 0

        # Return the transformed data
        return data_copy


class CenterCropTransform(DataTransform):
    """
    Center Crop Transform

    Attributes:
        s (int): The size of the square to be erased.

    Methods:
        process(data): Processes the data and returns the transformed data.
    """

    def __init__(self, s: int) -> None:
        """
        Initializes the CenterCropTransform class.

        Args:
            s (int): The size of the square to be erased.

        Raises:
            ValueError: If s is less than or equal to 1.
        """

        if s <= 1:
            raise ValueError(INVALID_S_T_MSG.format("s", "1"))

        self._s = s

    @property
    def s(self) -> int:
        """
        Returns the size of the square to be erased.

        Returns:
            int: The size of the square to be erased.
        """
        return self._s

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Processes the data and returns the transformed data.

        If the shape of the data is less than s, returns a copy of the data.

        Otherwise, it crops the center of the data with size s.

        Args:
            data (np.ndarray): The data to be processed.

        Returns:
            np.ndarray: The transformed data.
        """
        # If the shape of the data is less than s, return a copy of the data
        if data.shape[0] < self._s or data.shape[1] < self._s:
            return data.copy()

        # Find the centre of the data and each half of it
        h, w = data.shape
        mid_h, mid_w = h // 2, w // 2
        s_half = self._s // 2

        # Crop the data
        return data[mid_h - s_half : mid_h + s_half, mid_w - s_half : mid_w + s_half]


class RandomAudioCropTransform(DataTransform):
    """
    Random Audio Crop Transform

    Attributes:
        t (int): The duration of the audio to be cropped.

    Methods:
        process(data): Processes the data and returns the transformed data.
    """

    def __init__(self, t: int) -> None:
        """
        Initializes the RandomAudioCropTransform class.

        Args:
            t (float): The duration of the audio to be cropped.

        Raises:
            ValueError: If t is less than or equal to 0.
        """
        if t >= 0:
            raise ValueError(INVALID_S_T_MSG.format("t", "0"))

        self._t = t

    @property
    def t(self) -> float:
        """
        Returns the duration of the audio to be cropped.

        Returns:
            float: The duration of the audio to be cropped.
        """
        return self._t

    def process(self, data: tuple[np.ndarray, float]) -> tuple[np.ndarray, float]:
        """
        Processes the data and returns the transformed data.

        If the length of the audio is less than t * sr, returns a copy of the audio.

        Otherwise, it randomly selects a portion
            of the audio of length t * sr and returns it.

        Args:
            data (tuple[np.ndarray, float]): The data to be processed.

        Returns:
            tuple[np.ndarray, float]: The transformed data.
        """
        # Get audio and sampling rate
        audio, sr = data

        # If the length of the audio is less than t * sr, return a copy of the audio
        if audio.shape[0] < self._t * sr:
            return audio.copy(), sr

        # Select a random starting point
        start = RNG.integers(0, int(audio.shape[0] - self._t * sr))

        # Cropt the data and return
        return audio[start : start + self._t * sr], sr


class SpectrogramTransform(DataTransform):
    """
    Spectrogram Transform

    Attributes:
        s (float): The size of the square to be erased.

    Methods:
        process(data): Processes the data and returns the transformed data.
    """

    def __init__(self) -> None:
        """
        Initializes the SpectrogramTransform class.
        """
        return

    def process(self, data: tuple[np.ndarray, float]) -> tuple[np.ndarray, float]:
        """
        Processes the data and returns the transformed data.

        Args:
            data (tuple[np.ndarray, float]): The data to be processed.

        Returns:
            tuple[np.ndarray, float]: The transformed data.
        """

        # Get audio and sampling rate
        audio, sr = data

        # Return mel spectogram and sampling rate
        return melspectrogram(y=audio, sr=sr), sr
