"""Code for the dataset_generator for task1."""
import itertools
import os

import numpy as np
import tensorflow as tf



@tf.function
def default_batch_equalizer_fn(*args):
    """Batch equalizer.
    Prepares the inputs for a model to be trained in
    match-mismatch task. It makes sure that match_env
    and mismatch_env are equally presented as a first
    envelope in match-mismatch task.

    Parameters
    ----------
    args : Sequence[tf.Tensor]
        List of tensors representing feature data

    Returns
    -------
    Tuple[Tuple[tf.Tensor], tf.Tensor]
        Tuple of the EEG/speech features serving as the input to the model and
        the labels for the match/mismatch task

    Notes
    -----
    This function will also double the batch size. E.g. if the batch size of
    the elements in each of the args was 32, the output features will have
    a batch size of 64.
    """
    eeg = args[0]
    new_eeg = tf.concat([eeg, eeg], axis=0)
    all_features = [new_eeg]
    for match, mismatch in zip(args[1::2], args[2::2]):
        stimulus_feature1 = tf.concat([match, mismatch], axis=0)
        stimulus_feature2 = tf.concat([mismatch, match], axis=0)
        all_features += [stimulus_feature1, stimulus_feature2]
    labels = tf.concat(
        [
            tf.tile(tf.constant([[0]]), [tf.shape(eeg)[0], 1]),
            tf.tile(tf.constant([[1]]), [tf.shape(eeg)[0], 1]),
        ],
        axis=0,
    )

    # print(new_eeg.shape, env1.shape, env2.shape, labels.shape)
    return tuple(all_features), labels


def create_tf_dataset(
    data_generator,
    window_length,
    batch_equalizer_fn=None,
    hop_length=64,
    batch_size=64,
    data_types=(tf.float32, tf.float32, tf.float32),
    feature_dims=(64, 28, 28)
):
    """Creates a tf.data.Dataset.

    This will be used to create a dataset generator that will
    pass windowed data to a model in both tasks.

    Parameters
    ---------
    data_generator: DataGenerator
        A data generator.
    window_length: int
        Length of the decision window in samples.
    batch_equalizer_fn: Callable
        Function that will be applied on the data after batching (using
        the `map` method from tf.data.Dataset). In the match/mismatch task,
        this function creates the imposter segments and labels.
    hop_length: int
        Hop length between two consecutive decision windows.
    batch_size: Optional[int]
        If not None, specifies the batch size. In the match/mismatch task,
        this amount will be doubled by the default_batch_equalizer_fn
    data_types: Union[Sequence[tf.dtype], tf.dtype]
        The data types that the individual features of data_generator should
        be cast to. If you only specify a single datatype, it will be chosen
        for all EEG/speech features.

    Returns
    -------
    tf.data.Dataset
        A Dataset object that generates data to train/evaluate models
        efficiently
    """
    # create tf dataset from generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tuple(
            tf.TensorSpec(shape=(None, x), dtype=data_types[index])
            for index, x in enumerate(feature_dims)
        ),
    )
    # window dataset
    dataset = dataset.map(
        lambda *args: [
            tf.signal.frame(arg, window_length, hop_length, axis=0)
            for arg in args
        ]
    )

    # batch data
    dataset = dataset.interleave(
        lambda *args: tf.data.Dataset.from_tensor_slices(args),
        cycle_length=4,
        block_length=16,
    )
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    if batch_equalizer_fn is not None:
        # Create the labels and make sure classes are balanced
        dataset = dataset.map(batch_equalizer_fn)

    return dataset



class MatchMismatchDataGenerator:
    """Generate data for the Match/Mismatch task."""

    def __init__(
        self,
        files,
        window_length,
        spacing
    ):
        """Initialize the DataGenerator.

        Parameters
        ----------
        files: Sequence[Union[str, pathlib.Path]]
            Files to load.
        window_length: int
            Length of the decision window.
        spacing: int
            Spacing between matched and mismatched samples
        """
        self.window_length = window_length
        self.files = self.group_recordings(files)
        self.spacing = spacing

    def group_recordings(self, files):
        """Group recordings and corresponding stimuli.

        Parameters
        ----------
        files : Sequence[Union[str, pathlib.Path]]
            List of filepaths to preprocessed and split EEG and speech features

        Returns
        -------
        list
            Files grouped by the self.group_key_fn and subsequently sorted
            by the self.feature_sort_fn.
        """
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, recording_index):
        """Get data for a certain recording.

        Parameters
        ----------
        recording_index: int
            Index of the recording in this dataset

        Returns
        -------
        Union[Tuple[tf.Tensor,...], Tuple[np.ndarray,...]]
            The features corresponding to the recording_index recording
        """
        data = []
        for feature in self.files[recording_index]:
            data += [np.load(feature).astype(np.float32)]
        data = self.prepare_data(data)
        return tuple(tf.constant(x) for x in data)


    def __call__(self):
        """Load data for the next recording.

        Yields
        -------
        Union[Tuple[tf.Tensor,...], Tuple[np.ndarray,...]]
            The features corresponding to the recording_index recording
        """
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

            if idx == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        """Change state at the end of an epoch."""
        np.random.shuffle(self.files)

    def prepare_data(self, data):
        """Creates mismatch (imposter) envelope.

        Parameters
        ----------
        data: Sequence[numpy.ndarray]
            Data to create an imposter for.

        Returns
        -------
        tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray, ...)
            (EEG, matched stimulus feature, mismatched stimulus feature, ...).
        """
        eeg = data[0]
        new_length = eeg.shape[0] - self.window_length - self.spacing
        resulting_data = [eeg[:new_length, ...]]
        for stimulus_feature in data[1:]:
            match_feature = stimulus_feature[:new_length, ...]
            mismatch_feature = stimulus_feature[
                self.spacing + self.window_length:, ...
            ]
            resulting_data += [match_feature, mismatch_feature]
        return resulting_data

