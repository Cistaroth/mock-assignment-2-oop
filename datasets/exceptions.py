from datasets.baseclasses import DataTransform


class ImageNotFoundError(Exception):
    """
    Exception raised when an image is not found.

    Attributes:
        path (str): The path to the image that was not found.
    """

    def __init__(self, path: str) -> None:
        """
        Initializes the ImageNotFoundError class.

        Args:
            path (str): The path to the image that was not found.
        """
        self._path = path
        super().__init__()

    def __str__(self) -> str:
        """
        Returns a string representation of the ImageNotFoundError.

        Returns:
            str: A string representation of the ImageNotFoundError.
        """
        return f'Image at "{self._path}" was not found.'


class AudioNotFoundError(Exception):
    """
    Exception raised when an audio file is not found.

    Attributes:
        path (str): The path to the audio file that was not found.
    """

    def __init__(self, path: str) -> None:
        """
        Initializes the AudioNotFoundError class.

        Args:
            path (str): The path to the audio file that was not found.
        """
        self._path = path
        super().__init__()

    def __str__(self) -> str:
        """
        Returns a string representation of the AudioNotFoundError.

        Returns:
            str: A string representation of the AudioNotFoundError.
        """
        return f'Audio at "{self._path}" was not found.'


class InvalidTransformError(Exception):
    """
    Exception raised when a transform is not valid for a given data type.

    Attributes:
        transform (DataTransform): The transform that is not valid.
        data_type (str): The data type that the transform is not valid for.
    """

    def __init__(self, transform: DataTransform, data_type: str) -> None:
        """
        Initializes the InvalidTransform class.

        Args:
            transform (DataTransform): The transform that is not valid.
            data_type (str): The data type that the transform is not valid for.
        """
        self._transform = transform.__class__.__name__
        self._data_type = data_type
        super().__init__()

    def __str__(self) -> str:
        """
        Returns a string representation of the InvalidTransform.

        Returns:
            str: A string representation of the InvalidTransform.
        """
        return f'"{self._transform}" is not a valid transform on data of type \
            {self._data_type}.'
