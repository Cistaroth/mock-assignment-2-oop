# Import libraries
import unittest

import numpy as np

# Import from other modules
from datasets.dataset import (
    EagerAudioDataset,
    EagerImageDataset,
    LazyAudioDataset,
    LazyImageDataset,
)
from tests.utils import DATA_RETURN_TYPES, _load_single_data


class TestGetItem(unittest.TestCase):
    """
    Test the getitem method of the dataset
    """

    def setUp(self) -> None:
        """
        Set up the test
        """

        # Set root
        self.root = "tests/test_datasets/loading_dataset"

        # Define the loaders
        self.eager_audio = EagerAudioDataset
        self.lazy_audio = LazyAudioDataset
        self.eager_image = EagerImageDataset
        self.lazy_image = LazyImageDataset

        # Define what is an eager loader and what is a lazy loader
        self.eager_loader = {"audio": self.eager_audio, "image": self.eager_image}
        self.lazy_loader = {"audio": self.lazy_audio, "image": self.lazy_image}

        # Set up the test
        super().setUp()

    def _assert_data_equal(
        self,
        expected_data: tuple[DATA_RETURN_TYPES, str],
        actual_data: tuple[DATA_RETURN_TYPES, str],
        data_type: str,
    ) -> bool | None:
        """
        Method to check whether audio and image data are equal
        """
        if data_type == "audio":
            # If audio,
            # assert that:
            # 1. the audios are equal,
            # 2. the sample rates are equal,
            # 3. the labels are equal

            self.assertTrue(
                np.array_equal(expected_data[0][0], actual_data[0][0])
                and expected_data[0][1] == actual_data[0][1]
                and expected_data[1] == actual_data[1],
                f"Expected data to be {expected_data} but got {actual_data}",
            )
        elif data_type == "image":
            # If image,
            # assert that:
            # 1. the images are equal
            # 2. the labels are equal

            self.assertTrue(
                np.array_equal(expected_data[0], actual_data[0])
                and expected_data[1] == actual_data[1],
                f"Expected data to be {expected_data} but got {actual_data}",
            )
        else:
            # Not valid data
            return None

    def test_eager_getitem(self) -> None:
        """
        Test the getitem method of the eager dataset
        """

        # Loop through all loaders
        for loader_type, loader in self.eager_loader.items():
            # Get correct dataset
            dataset_path = f"{self.root}/{loader_type}_dataset"

            # Instantiate the loader
            dataset_loader = loader(root=dataset_path)

            # Loop through all data and assert that is equal
            for i in range(len(dataset_loader._data)):
                self._assert_data_equal(
                    dataset_loader._data[i], dataset_loader[i], loader_type
                )

    def test_lazy_getitem(self) -> None:
        """
        Test the getitem method of the lazy dataset
        """

        # Loop through all loaders
        for loader_type, loader in self.lazy_loader.items():
            # Get correct dataset
            dataset_path = f"{self.root}/{loader_type}_dataset"

            # Instantiate the loader
            dataset_loader = loader(root=dataset_path)

            # Loop through all data and assert that is equal, make sure to load the data
            for i in range(len(dataset_loader._data)):
                loaded_data = _load_single_data(dataset_loader._data[i][0], loader_type)
                exp_data = (loaded_data, dataset_loader._data[i][1])
                self._assert_data_equal(exp_data, dataset_loader[i], loader_type)


# Run the tests
if __name__ == "__main__":
    unittest.main()
