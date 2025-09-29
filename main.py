# Import libraries to display image and audio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from librosa.display import specshow

# Import Datasets
from datasets.dataset import (
    EagerAudioDataset,
    EagerImageDataset,
    LazyAudioDataset,
    LazyImageDataset,
)

# Import Transforms
from datasets.transform import SpectrogramTransform, SquareErasingTransform

try:
    import sounddevice as sd

    SOUND_AVAILABLE = True
except OSError:
    SOUND_AVAILABLE = False


# Paths to audio and image datasets
AUDIO_DATASET_PATH = "data/audio_dataset"
IMAGE_DATASET_PATH = "data/image_dataset"


def main() -> None:
    """
    Main function created by UV
    """

    print("Hello from assignment-2-group-33-1!")


def play_audio(audio: tuple[np.ndarray, float], label: str, loader_type: str) -> None:
    """
    Plays an audio using sounddevice or saves it to a file
    """

    # Play the audio if possible, else save to wav file
    if SOUND_AVAILABLE:
        try:
            sd.play(audio[0], samplerate=audio[1])
        except sd.PortAudioError:
            sf.write(
                f"demonstration/{loader_type}_audio_{label}.wav",
                audio[0],
                audio[1],
            )
    else:
        sf.write(f"demonstration/{loader_type}_audio_{label}.wav", audio[0], audio[1])


def display_image(image: np.ndarray, label: str, loader_type: str) -> None:
    """
    Displays an image using matplotlib.pyplot or saves it to a file
    """

    # Plot the figure
    plt.clf()
    plt.figure(figsize=(image.shape[0] / 100, image.shape[1] / 100))
    plt.imshow(image)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # If matplotlib is interactive, display figure
    if mpl.is_interactive():
        plt.show()

    # Else safe figure to file
    else:
        plt.savefig(f"demonstration/{loader_type}_image_{label}.png")

    # Close plot
    plt.close()


def eager_audio_dataset() -> None:
    """
    Demonstration of the EagerAudioDataset.

    This function loads an audio dataset using the EagerAudioDataset class and
    prints the first data point from the loaded dataset.

    """
    print("EAGER AUDIO DATASET LOADER")

    # Instantiate the loader
    loader = EagerAudioDataset(root=AUDIO_DATASET_PATH)

    # Load the dataset, this is not necessary as it is loaded on instantiation
    # However, we showcase this to show the user how to reload the data
    loader.load()

    # Obtain the first datapoint using __getitem__
    first_datapoint = loader[0]
    print("Eager Audio Loader First Data Point: \n", first_datapoint)

    # Play audio if possible, else save to file
    play_audio(first_datapoint[0], first_datapoint[1], "eager")


def lazy_audio_dataset() -> None:
    """
    Demonstration of the LazyAudioDataset.

    This function loads an audio dataset using the LazyAudioDataset class and
    prints the first data point from the loaded dataset.

    """
    print("LAZY AUDIO DATASET LOADER")

    # Instantiate the loader
    loader = LazyAudioDataset(root=AUDIO_DATASET_PATH)

    # Load the dataset, this is not necessary as it is loaded on instantiation
    # However, we showcase this to show the user how to reload the data
    loader.load()

    # Obtain the first datapoint using __getitem__
    first_datapoint = loader[0]
    print("Lazy Audio Loader First Data Point: \n", first_datapoint)

    # Play audio if possible, else save to file
    play_audio(first_datapoint[0], first_datapoint[1], "lazy")


def eager_image_dataset() -> None:
    """
    Demonstration of the EagerImageDataset.

    This function loads an image dataset using the EagerImageDataset class and
    prints the first data point from the loaded dataset.
    """
    print("EAGER IMAGE DATASET LOADER")

    # Instantiate the loader
    loader = EagerImageDataset(root=IMAGE_DATASET_PATH)

    # Load the dataset, this is not necessary as it is loaded on instantiation
    # However, we showcase this to show the user how to reload the data
    loader.load()

    # Obtain the first datapoint using __getitem__
    first_datapoint = loader[0]
    print("Eager Image Loader First Data Point: \n", first_datapoint)

    # Display the first datapoint or save to file
    display_image(first_datapoint[0], first_datapoint[1], "eager")


def lazy_image_dataset() -> None:
    """
    Demonstration of the LazyImageDataset.

    This function loads an image dataset using the LazyImageDataset class and
    prints the first data point from the loaded dataset.
    """
    print("LAZY IMAGE DATASET LOADER")

    # Instantiate the loader
    loader = LazyImageDataset(root=IMAGE_DATASET_PATH)

    # Load the dataset, this is not necessary as it is loaded on instantiation
    # However, we showcase this to show the user how to reload the data
    loader.load()

    # Obtain the first datapoint using __getitem__
    first_datapoint = loader[0]
    print("Lazy Image Loader First Data Point: \n", first_datapoint)

    # Display the first datapoint or save to file
    display_image(first_datapoint[0], first_datapoint[1], "lazy")


def transform_on_image() -> None:
    """
    Demonstration of the EagerImageDataset with a transformation.

    This function loads an image dataset using the EagerImageDataset class with a
    transformation and prints a data point from the loaded dataset.
    """
    print("EAGER IMAGE DATASET LOADER WITH TRANSFORM")

    # Load the transform
    transform = SquareErasingTransform(s=100)

    # Instantiate the loader
    loader = EagerImageDataset(root=IMAGE_DATASET_PATH, transform=transform)

    # Load the dataset, this is not necessary as it is loaded on instantiation
    # However, we showcase this to show the user how to reload the data
    loader.load()

    # Obtain the a datapoint using __getitem__
    # We use the 21th datapoint as the erasure is easier to see
    datapoint = loader[20]
    print("Eager Image Loader First Data Point: \n", datapoint)

    # Display the first datapoint or save to file
    display_image(datapoint[0], datapoint[1], "eager_image_transform")


def transform_on_audio() -> None:
    """
    Demonstration of the EagerAudioDataset with a transformation.

    This function loads an audio dataset using the EagerAudioDataset class with a
    transformation and prints the first data point from the loaded dataset.

    """
    print("EAGER AUDIO DATASET LOADER WITH TRANSFORM")

    # Load the transform
    transform = SpectrogramTransform()

    # Instantiate the loader
    loader = EagerAudioDataset(root=AUDIO_DATASET_PATH, transform=transform)

    # Load the dataset, this is not necessary as it is loaded on instantiation
    # However, we showcase this to show the user how to reload the data
    loader.load()

    # Obtain the first datapoint using __getitem__
    first_datapoint = loader[0]
    print("Eager Audio Loader First Data Point: \n", first_datapoint)

    # Plot the spectogram
    plt.clf()
    plt.figure(figsize=(10, 4))
    specshow(first_datapoint[0][0], sr=first_datapoint[0][1])
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-frequency spectrogram")
    plt.tight_layout()

    # If matplotlib is interactive, display figure
    if mpl.is_interactive():
        plt.show()

    # Else safe figure to file
    else:
        plt.savefig(f"demonstration/spectogram_transform_{first_datapoint[1]}.png")

    # Close plot
    plt.close()


if __name__ == "__main__":
    # Run Eager Audio Loader
    eager_audio_dataset()

    # Run Lazy Audio Loader
    lazy_audio_dataset()

    # Run Eager Image Loader
    eager_image_dataset()

    # Run Lazy Image Loader
    lazy_image_dataset()

    # Run Eager Image Loader with a transformation
    transform_on_image()

    # Run Eager Audio Loader with a transformation
    transform_on_audio()
