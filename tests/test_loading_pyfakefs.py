# Import libraries
import os
import pathlib
import unittest

import librosa
import numpy as np
from pyfakefs.fake_filesystem_unittest import TestCase

# Import from other modules
from datasets.dataset import (
    EagerAudioDataset,
    EagerImageDataset,
    LazyAudioDataset,
    LazyImageDataset,
)
from tests.utils import DATA_RETURN_TYPES, _load_single_data


class TestLoading(TestCase):
    """
    Tests the loading of data from the dataset
    """

    def _setup_real_directories_pyfakefs(self) -> None:
        """
        Set up real directories for pyfakefs
        """
        self.fs.add_real_directory(self.root, target_path=self.root)

        # This is commented out because it is not necessarily needed
        # import os
        # import librosa
        # self.fs.add_real_directory(os.path.dirname(librosa.__file__))

    def setUp(self) -> None:
        """
        Set up the test
        """
        # Set root
        self.root = "tests/test_datasets/loading_dataset"

        # Set up PyFakeFs
        self.setUpClassPyfakefs()

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

    def _assert_eager_data_equals(
        self,
        expected_data: list[tuple[DATA_RETURN_TYPES, str]],
        actual_data: list[tuple[DATA_RETURN_TYPES, str]],
        data_type: str,
    ) -> None:
        """
        Asserts that the eager data is equal to the expected data

        Args:
            expected_data (list[tuple[DATA_RETURN_TYPES, str]]): The expected data
            actual_data (list[tuple[DATA_RETURN_TYPES, str]]): The actual data
            data_type (str): The type of data (audio or image)

        """

        # Assert that their length is equal
        self.assertEqual(
            len(expected_data),
            len(actual_data),
            f"Expected data to have length {len(expected_data)} but got length {len(actual_data)}",
        )

        # If data type is audio
        if data_type == "audio":
            # Match every single element from exp_data to actual_data
            matched = [False] * len(actual_data)

            # Loop through expected data
            for exp_datapoint, exp_label in expected_data:
                found = False
                exp_audio, exp_sr = exp_datapoint

                # Loop through actual data
                for i, (act_datapoint, act_label) in enumerate(actual_data):
                    act_audio, act_sr = act_datapoint

                    # If a match is found, then set matched and break
                    # Match is found when:
                    # 1. Audios are equal
                    # 2. Sample rates are equal
                    # 3. Labels are equal
                    if (
                        not matched[i]
                        and np.array_equal(exp_audio, act_audio)
                        and exp_sr == act_sr
                        and exp_label == act_label
                    ):
                        matched[i] = True
                        found = True
                        break

                # If no match found, then fail assert
                self.assertTrue(
                    found,
                    f"Expected data to contain datapoint {exp_datapoint} with label {exp_label}",
                )

        # If it is an image
        elif data_type == "image":
            # Match every single element from exp_data to actual_data
            matched = [False] * len(actual_data)

            # Loop through expected data
            for exp_datapoint, exp_label in expected_data:
                found = False

                # Loop through actual data
                for i, (act_datapoint, act_label) in enumerate(actual_data):
                    # If a match is found, then set matched and break
                    # Match is found when:
                    # 1. Images are equal
                    # 2. Labels are equal
                    if (
                        not matched[i]
                        and np.array_equal(exp_datapoint, act_datapoint)
                        and exp_label == act_label
                    ):
                        matched[i] = True
                        found = True
                        break

                # If no match found, then fail assert
                self.assertTrue(
                    found,
                    f"Expected data to contain datapoint {exp_datapoint} with label {exp_label}",
                )

    def _assert_lazy_data_equals(
        self, expected_data: list[tuple[str, str]], actual_data: list[tuple[str, str]]
    ) -> None:
        """
        Asserts that the lazy data is equal to the expected data
        """

        # Assert that their length is equal
        self.assertEqual(
            len(expected_data),
            len(actual_data),
            f"Expected data to have length {len(expected_data)} but got length \
                {len(actual_data)}",
        )

        # Match every single element from exp_data to actual_data
        matched = [False] * len(actual_data)

        # Loop through expected data
        for exp_datapoint in expected_data:
            found = False

            # Loop through actual data
            for i, act_datapoint in enumerate(actual_data):
                # If a match is found, then set matched and break
                # Match is found when:
                # 1. Datapoints are equal
                if not matched[i] and exp_datapoint == act_datapoint:
                    matched[i] = True
                    found = True
                    break

            # If no match found, then fail assert
            self.assertTrue(
                found, f"Expected data to contain datapoint {exp_datapoint}"
            )

    def test_loading_eager(self) -> None:
        """
        Tests that the eager data is equal to the expected data
        """

        # Loop through loaders
        for loader_type, loader in self.eager_loader.items():
            # Set up real directories with pyfakefs
            self.fs.reset()
            self._setup_real_directories_pyfakefs()

            # Set dataset path
            dataset_path = f"{self.root}/{loader_type}_dataset"

            # Load dataset
            dataset_loader = loader(root=dataset_path)

            # Collect expected data by looping through all files in dataset_path
            # Also load the data immediately
            exp_data = []
            dataset_path = pathlib.Path(dataset_path)
            for label in dataset_path.iterdir():
                for file in label.iterdir():
                    datapoint = (_load_single_data(str(file), loader_type), label.name)
                    exp_data.append(datapoint)

            # Assert that the eager data is equal to the expected data
            self._assert_eager_data_equals(exp_data, dataset_loader._data, loader_type)

    def test_loading_lazy(self) -> None:
        """
        Tests that the lazy data is equal to the expected data
        """
        # Loop through loaders
        for loader_type, loader in self.lazy_loader.items():
            # Set up real directories with pyfakefs
            self.fs.reset()
            self._setup_real_directories_pyfakefs()

            # Set dataset path
            dataset_path = f"{self.root}/{loader_type}_dataset"

            # Load dataset
            dataset_loader = loader(root=dataset_path)

            # Load the data, store the filepath and label name
            exp_data = []
            dataset_path = pathlib.Path(dataset_path)
            for label in dataset_path.iterdir():
                for file in label.iterdir():
                    datapoint = (str(file), label.name)
                    exp_data.append(datapoint)

            # Assert that the lazy data is equal to the expected data
            self._assert_lazy_data_equals(exp_data, dataset_loader._data)


# Run the tests
if __name__ == "__main__":
    unittest.main()
