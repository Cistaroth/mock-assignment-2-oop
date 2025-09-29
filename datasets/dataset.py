# Import from other modules
from datasets.baseclasses import BaseDataset, DataTransform
from datasets.mixins import AudioMixin, EagerMixin, ImageMixin, LazyMixin


class EagerAudioDataset(AudioMixin, EagerMixin, BaseDataset):
    """
    EagerAudioDataset class

    Attributes:
        root (str): The root directory of the dataset.
        transform (DataTransform | None): The transformation to be applied
            to the data points.
        _data (list): The list of data points in the dataset.

    Methods:
        load(self)
        __getitem__(self, index: int)
        _load_single_data(self, path: str)
        _check_valid_transform(self, transform)
    """

    def __init__(self, root: str, transform: DataTransform | None = None) -> None:
        """
        Initializes the EagerAudioDataset class.

        Args:
            root (str): The root directory of the dataset.
            transform (DataTransform | None): The transformation to be applied
                to the data points.
        """
        super().__init__(root, transform)


class LazyAudioDataset(AudioMixin, LazyMixin, BaseDataset):
    """
    LazyAudioDataset class

    Attributes:
        root (str): The root directory of the dataset.
        transform (DataTransform | None): The transformation to be applied
            to the data points.
        _data (list): The list of data points in the dataset.

    Methods:
        load(self)
        __getitem__(self, index: int)
        _load_single_data(self, path: str)
        _check_valid_transform(self, transform)
    """

    def __init__(self, root: str, transform: DataTransform | None = None) -> None:
        """
        Initializes the LazyAudioDataset class.

        Args:
            root (str): The root directory of the dataset.
            transform (DataTransform | None): The transformation to be applied
                to the data points.
        """
        super().__init__(root, transform)


class EagerImageDataset(ImageMixin, EagerMixin, BaseDataset):
    """
    EagerImageDataset class

    Attributes:
        root (str): The root directory of the dataset.
        transform (DataTransform | None): The transformation to be applied
            to the data points.
        _data (list): The list of data points in the dataset.

    Methods:
        load(self)
        __getitem__(self, index: int)
        _load_single_data(self, path: str)
        _check_valid_transform(self, transform)
    """

    def __init__(self, root: str, transform: DataTransform | None = None) -> None:
        """
        Initializes the EagerImageDataset class.

        Args:
            root (str): The root directory of the dataset.
            transform (DataTransform | None): The transformation to be applied
                to the data points.
        """
        super().__init__(root, transform)


class LazyImageDataset(ImageMixin, LazyMixin, BaseDataset):
    """
    LazyImageDataset class

    Attributes:
        root (str): The root directory of the dataset.
        transform (DataTransform | None): The transformation to be applied
            to the data points.
        _data (list): The list of data points in the dataset.

    Methods:
        load(self)
        __getitem__(self, index: int)
        _load_single_data(self, path: str)
        _check_valid_transform(self, transform)
    """

    def __init__(self, root: str, transform: DataTransform | None = None) -> None:
        """
        Initializes the LazyImageDataset class.

        Args:
            root (str): The root directory of the dataset.
            transform (DataTransform | None): The transformation to be applied
                to the data points.
        """
        super().__init__(root, transform)
