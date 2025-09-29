# Import libraries
import inspect
import unittest

# Import from other modules
from datasets.baseclasses import BaseDataset


class TestABC(unittest.TestCase):
    """
    Tests the Abstract Baseclass BaseDataset class

    """

    def setUp(self) -> None:
        """
        Set up the test
        """

        self.baseclass = BaseDataset
        self.attributes = ["root"]
        self.methods = {"load": 1, "__getitem__": 2, "_load_single_data": 2}

        super().__init__()

    def test_attributes(self) -> None:
        """
        Tests that the BaseDataset class has the required attributes and
            that they are properties
        """

        # Loop through all required attributes
        for attr in self.attributes:
            # Check that the attribute exists
            self.assertTrue(
                hasattr(self.baseclass, attr),
                f"Class {self.baseclass.__name__} does not have attribute {attr}",
            )

            # Check that the attribute is a property
            self.assertTrue(
                isinstance(getattr(self.baseclass, attr), property),
                f"Class {self.baseclass.__name__} does not have attribute {attr} \
                    as property",
            )

    def test_methods(self) -> None:
        """
        Tests that the BaseDataset class has the required methods and
            that they are abstract and callable, and
            that they have the required number of parameters.
        """

        # Loop through all required methods
        for method, exp_param_len in self.methods.items():
            # Assert that the method exists
            self.assertTrue(
                hasattr(self.baseclass, method),
                f"Class {self.baseclass.__name__} does not have method {method}",
            )

            # Assert that the method is abstract
            self.assertTrue(
                callable(getattr(self.baseclass, method)),
                f"Class {self.baseclass.__name__} does not have method {method} \
                as callable",
            )

            # Assert that the method has the required number of parameters
            act_param = inspect.signature(getattr(self.baseclass, method)).parameters
            self.assertEqual(
                len(act_param),
                exp_param_len,
                f"Class {self.baseclass.__name__} does not have method {method} \
                with {exp_param_len} parameters",
            )


# Run tests
if __name__ == "__main__":
    unittest.main()
