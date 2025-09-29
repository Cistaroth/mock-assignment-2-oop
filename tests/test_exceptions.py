# Import libraries
import unittest

# Import from from other modules
from datasets.dataset import (
    EagerAudioDataset,
    EagerImageDataset,
    LazyAudioDataset,
    LazyImageDataset,
)
from datasets.exceptions import AudioNotFoundError, ImageNotFoundError


class TestSpecialExceptions(unittest.TestCase):
    """
    Tests that the correct exceptions are raised.
    """

    def setUp(self) -> None:
        """
        Set up the test
        """

        # Set root of exceptions dataset
        self.root = "tests/test_datasets/exceptions_dataset"

        # Get the classes of the errors
        self.audio_error = AudioNotFoundError
        self.image_error = ImageNotFoundError

        # Get the loaders
        self.eager_audio = EagerAudioDataset
        self.lazy_audio = LazyAudioDataset
        self.eager_image = EagerImageDataset
        self.lazy_image = LazyImageDataset

        # Define what is an audio loader and what is an image loader
        self.audio_loaders = {
            "eager": self.eager_audio,
            "lazy": self.lazy_audio,
        }

        self.image_loaders = {
            "eager": self.eager_image,
            "lazy": self.lazy_image,
        }

        # Set up the test
        super().setUp()

    def test_audio_exception(self) -> None:
        """
        Tests that an AudioNotFoundError is raised when an audio file is not found
        in the dataset.
        """

        # Loop through loaders
        for loader in self.audio_loaders.values():
            # Check for exception when it tries to access the first datapoint
            with self.assertRaises(self.audio_error):
                dataset_loader = loader(root=self.root)
                dataset_loader[0]

    def test_image_exception(self) -> None:
        """
        Tests that an ImageNotFoundError is raised when an image file is not found
        in the dataset.
        """

        # Loop through loaders
        for loader in self.image_loaders.values():
            # Check for exception when it tries to access the first datapoint
            with self.assertRaises(self.image_error):
                dataset_loader = loader(root=self.root)
                dataset_loader[0]


# Run tests
if __name__ == "__main__":
    unittest.main()
