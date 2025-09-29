# Import libraries
import pathlib
from abc import abstractmethod
from copy import deepcopy

import cv2
import librosa
import numpy as np

# Import from other modules
from datasets.baseclasses import DataTransform
from datasets.exceptions import (
    AudioNotFoundError,
    ImageNotFoundError,
    InvalidTransformError,
)
from datasets.transform import (
    CenterCropTransform,
    RandomAudioCropTransform,
    SpectrogramTransform,
    SquareErasingTransform,
)
from datasets.utils import DATA_RETURN_TYPES, GETITEM_RETURN_TYPE


class EagerMixin:
    """
    Eager Mixin
    """

    # Define attributes
    _root: str
    _data: list[GETITEM_RETURN_TYPE]
    _transform: DataTransform | None

    @abstractmethod
    def _load_single_data(self, path: str) -> DATA_RETURN_TYPES:
        """
        Loads a single data item from the given path.

        Args:
            path (str): The path to the data item.

        Returns:
            DATA_RETURN_TYPES: The loaded data item.
        """

    def load(self) -> None:
        """
        Loads the data from the root directory into _data.

        Returns:
            None
        """
        # Convert to pathlib path
        root_path = pathlib.Path(self._root)

        # Iterate over labels (names of directories in root)
        for label in root_path.iterdir():
            # Iterate over files in label
            for file in label.iterdir():
                # If file is a file
                if file.is_file():
                    # Append to loaded data and label to _data
                    self._data.append((self._load_single_data(str(file)), label.name))

    def __getitem__(self, index: int) -> GETITEM_RETURN_TYPE:
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point to be retrieved.

        Returns:
            GETITEM_RETURN_TYPE: The data point at the given index.

        Raises:
            IndexError: If the index is out of range.
        """

        # If transform is not None
        if self._transform is not None:
            # Get data and label
            data, label = self._data[index]

            # Apply transform on data only and return transformed data and label
            return (self._transform.process(data), label)

        # Return a copy of the data
        return deepcopy(self._data[index])


class LazyMixin:
    """
    Lazy Mixin
    """

    # Define attributes
    _root: str
    _data: list[tuple[str, str]]
    _transform: DataTransform | None

    @abstractmethod
    def _load_single_data(self, path: str) -> DATA_RETURN_TYPES:
        """
        Loads a single data item from the given path.

        Args:
            path (str): The path to the data item.

        Returns:
            DATA_RETURN_TYPES: The loaded data item.
        """

    def load(self) -> None:
        """
        Loads the data from the root directory into _data.

        Returns:
            None
        """

        # Convert to pathlib path
        root_path = pathlib.Path(self._root)

        # Iterate over labels (names of directories in root)
        for label in root_path.iterdir():
            # Iterate over files in label
            for file in label.iterdir():
                # If file is a file
                if file.is_file():
                    # Append file path and label to _data
                    self._data.append((str(file), label.name))

    def __getitem__(self, index: int) -> GETITEM_RETURN_TYPE:
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point to be retrieved.

        Returns:
            GETITEM_RETURN_TYPE: The data point at the given index.

        Raises:
            IndexError: If the index is out of range.
        """

        # Get path and label
        path, label = self._data[index]

        # Load data
        data = self._load_single_data(path)

        # If transform is not None, apply transform
        if self._transform is not None:
            # Return transformed data and label
            return (self._transform.process(data), label)

        # Return data and label
        return data, label


class AudioMixin:
    """
    Audio Mixin
    """

    def _load_single_data(self, path: str) -> tuple[np.ndarray, float]:
        """
        Loads a single audio data item from the given path.

        Args:
            path (str): The path to the audio data item.

        Returns:
            DATA_RETURN_TYPES: The loaded audio data item.

        Raises:
            AudioNotFoundError: If the audio file is not found.
        """

        # Try to load audio, if not succesful then raise Exception
        try:
            audio, sr = librosa.load(path)
        except FileNotFoundError as exception:
            raise AudioNotFoundError(path) from exception

        # Return audio and sampling rate
        return audio, sr

    def _check_valid_transform(self, transform: DataTransform | None) -> None:
        """
        Checks if the given transform is valid for audio data.

        Args:
            transform (DataTransform | None): The transform to be checked.

        Raises:
            InvalidTransform: If the transform is not valid for audio data.
        """
        # Raise exception if transform is not valid for audio
        if transform is not None and not isinstance(
            transform, (RandomAudioCropTransform, SpectrogramTransform)
        ):
            raise InvalidTransformError(transform, "audio")


class ImageMixin:
    """
    Image Mixin
    """

    def _load_single_data(self, path: str) -> np.ndarray:
        """
        Loads a single image data item from the given path.

        Args:
            path (str): The path to the image data item.

        Returns:
            DATA_RETURN_TYPES: The loaded image data item.

        Raises:
            ImageNotFoundError: If the image file is not found.
        """

        # Read data
        image = cv2.imread(path)

        # If image is None, raise Exception
        if image is None:
            raise ImageNotFoundError(path)

        # Convert image to RGB
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Return numpy array of image
        return np.array(converted_image)

    def _check_valid_transform(self, transform: DataTransform | None) -> None:
        """
        Checks if the given transform is valid for image data.

        Args:
            transform (DataTransform | None): The transform to be checked.

        Raises:
            InvalidTransform: If the transform is not valid for image data.
        """
        # Raise exception if transform is not valid for image
        if transform is not None and not isinstance(
            transform, (SquareErasingTransform, CenterCropTransform)
        ):
            raise InvalidTransformError(transform, "image")
