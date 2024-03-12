import tensorflow as tf
import librosa
from pathlib import Path
import numpy as np
from scipy import signal
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display


class SpokenDataLoader:
    def __init__(self, data_split: pd.DataFrame, batch_size: int):
        """
        Initialize the SpokenDataLoader.

        Args:
            data_split (pd.DataFrame): Dataframe containing the data split.
            batch_size (int): Batch size for the dataset.
        """
        self.data_split = data_split
        self.batch_size = batch_size
        self.label_type = None

        self.language_classes = {"English": 0, "Mandarin": 1, "German": 2, "French": 3}


    def _get_label(self, df_row: pd.Series):
        """
        Get the label for a given dataframe row.

        Args:
            df_row (pd.Series): The dataframe row.

        Returns:
            The label for the given row.
        """
        if self.label_type == "language":
            label = self.language_classes.get(df_row["language"])
            return to_categorical(label, num_classes=len(self.language_classes))
        elif self.label_type == "number":
            return to_categorical(df_row["number"], num_classes=100)

    @staticmethod
    def _resample_func(x, samples):
        """
        Resample the given signal to the given number of samples.

        Args:
            x: The signal to be resampled.
            samples: The number of samples to resample to.

        Returns:
            The resampled signal.
        """
        x = signal.resample(x, samples, axis=0)
        return x

    def _load_data(self, df: pd.DataFrame, spectrogram=False, channel_dim=False):
        """
        Load the data from the dataframe.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            spectrogram (bool, optional): Whether to compute spectrogram. Defaults to False.
            channel_dim (int, optional): The index of the channel dimension in the spectrogram data. Set to 0 to not set any channel dimension, and set to 1 or 3 to set channel dimension to 1 or 3. Defaults to 0. (Only used if spectrogram=True)
        Returns:
            The loaded data and labels.
        """
        X, y = [], []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Loading data"):
            file_path = row["file_path"]
            label = self._get_label(row)
            data, _ = librosa.load(file_path, sr=44100)

            data = self._resample_func(data, 8000)

            if spectrogram:
                # Create spectrogram
                data = librosa.feature.melspectrogram(y=data, sr=8000, n_fft=1024, hop_length=256, n_mels=128)
                data = librosa.power_to_db(data, ref=np.max)
                if channel_dim == 1:
                    data = data[..., np.newaxis]
                elif channel_dim == 3:
                    data = np.repeat(data[..., np.newaxis], 3, -1)
                elif channel_dim == "transformer":
                    # switch frequency and time axes
                    data = data.transpose()

            X.append(data)
            y.append(label)

        # Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]

        return np.array(X), np.array(y)

    def load_waveform(self, label_type="language"):
            """
            Loads raw waveform data from the dataset.

            Args:
                label_type (str): The type of labels to load. Default is "language".

            Returns:
                dict: A dictionary containing the loaded datasets for training, validation, and testing.
                      The keys are "train", "val", and "test", respectively.
                      The values are TensorFlow datasets.

            """
            self.label_type = label_type

            ds_train = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "train"]))).batch(self.batch_size)
            ds_val = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "validation"]))).batch(self.batch_size)
            ds_test = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "test"]))).batch(self.batch_size)

            return {"train": ds_train, "val": ds_val, "test": ds_test}

    def load_waveforms_as_numpy(self, speakers, label_type="language"):
        """
        Loads raw waveform data from the dataset.

        Args:
            speakers (list): The list of speakers to load.
            label_type (str): The type of labels to load. Default is "language".

        Returns:
            dict: A dictionary containing the loaded datasets for training, validation, and testing.
                  The keys are "train", "val", and "test", respectively.
                  The values are numpy arrays.

        """
        self.label_type = label_type

        data, _ = self._load_data(speakers)

        return data
    
    def load_spectrogram(self, label_type: str = "language", channel_dim: int = 0):
        """
        Loads spectrogram data from the dataset.

        Args:
            label_type (str, optional): Type of labels to use. Defaults to "language".
            channel_dim (int, optional): The index of the channel dimension in the spectrogram data. Set to 0 to not set any channel dimension, and set to 1 or 3 to set channel dimension to 1 or 3. Defaults to 0.

        Returns:
            dict: A dictionary containing the loaded spectrogram datasets for training, validation, and testing.
                  The keys are "train", "val", and "test", respectively.
        """
        self.label_type = label_type

        ds_train = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "train"], spectrogram=True, channel_dim=channel_dim))).batch(self.batch_size)
        try:
            ds_val = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "validation"], spectrogram=True, channel_dim=channel_dim))).batch(self.batch_size)
        except:
            ds_val = None
        ds_test = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "test"], spectrogram=True, channel_dim=channel_dim))).batch(self.batch_size)

        return {"train": ds_train, "val": ds_val, "test": ds_test}

    