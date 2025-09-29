# Import libaries
import pathlib
from abc import ABC, abstractmethod
from copy import deepcopy

# Import from other modules
from datasets.utils import DATA_RETURN_TYPES, GETITEM_RETURN_TYPE, INVALID_FILE_ERROR


class DataTransform(ABC):
    """
    Abstract Base Class for Data Transformations

    Methods:
        process(data): Processes the data and returns the transformed data.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Initializes the DataTransform.
        This method should be overridden in subclasses of DataTransform.
        """

    @abstractmethod
    def process(self, data: DATA_RETURN_TYPES) -> DATA_RETURN_TYPES:
        """
        Processes the data and returns the transformed data.
        This method should be overridden in subclasses of DataTransform.
        """


class BaseDataset(ABC):
    """
    Abstract Base Class for Datasets

    Attributes:
        _root (str): The root directory of the dataset.
        _data (list): The list of data points in the dataset.
        transform (DataTransform | None): The transformation
            to be applied to the data points.

    Methods:
        load(): Loads the dataset.
        __getitem__(index): Returns the data point at the given index.
        _load_single_data(path): Loads a single data point from the given path.
        _check_valid_transform(transform): Checks if the given transform is valid.
    """

    def __init__(self, root: str, transform: DataTransform | None = None) -> None:
        """
        Initializes the BaseDataset class.

        Args:
            root (str): The root directory of the dataset.
            transform (DataTransform | None): The transformation to be applied
                to the data points.

        Raises:
            DirectoryInvalidError: If the given root directory is invalid.
        """
        # Set root and initialise data
        self._root = root
        self._data = []

        # Set transform using setter
        self.transform = transform

        # Convert to pathlib path
        root_path = pathlib.Path(root)

        # Check for validity of path
        if root_path.exists() and root_path.is_dir():
            self.load()
        else:
            # Else raise an invalid directory error
            raise FileNotFoundError(INVALID_FILE_ERROR.format(self._root))

    @abstractmethod
    def load(self) -> None:
        """
        Loads the dataset.

        This method should be overridden in subclasses of BaseDataset.
        """

    @abstractmethod
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

    @abstractmethod
    def _load_single_data(self, path: str) -> DATA_RETURN_TYPES:
        """
        Loads a single data point from the given path.

        Args:
            path (str): The path to the data point to be loaded.

        Returns:
            DATA_RETURN_TYPES: The loaded data point.

        Raises:
            AudioNotFoundError: If the audio file is not found.
            ImageNotFoundError: If the image file is not found.
        """

    @abstractmethod
    def _check_valid_transform(self, transform: DataTransform | None) -> None:
        """
        Checks if the given transform is valid.

        Args:
            transform (DataTransform | None): The transform to be checked.

        Raises:
            InvalidTransform: If the transform is not valid.
        """

    @property
    def transform(self) -> DataTransform | None:
        """
        Returns a deep copy of the current transform.

        If the transform is None, it will return None.

        Returns:
            DataTransform | None: A deep copy of the current transform.
        """
        return deepcopy(self._transform)

    @transform.setter
    def transform(self, transform: DataTransform | None) -> None:
        """
        Sets the transform to be applied to the data points.

        Args:
            transform (DataTransform | None): The transform to be set.

        Raises:
            InvalidTransform: If the transform is not valid.
        """
        self._check_valid_transform(transform)
        self._transform = transform

    @property
    def root(self) -> str:
        """
        Returns the root directory of the dataset.

        Returns:
            str: The root directory of the dataset.
        """
        return self._root
