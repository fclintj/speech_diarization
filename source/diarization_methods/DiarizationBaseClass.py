from abc import ABC, abstractmethod
import tensorflow as tf


class DiarizationBaseClass(ABC):
    """
    The DiarizationBaseClass provides the interfaces between custom diarization
    methods and the wrapper code. It provides a tensorflow session available as
    `self.sess` that is usable within the `train_on_data`, `get_train_error`, and
    `run_on_data` methods.

    You may add other variables needed within your own implementation. Not here
    please. You can pass parameters into your custom diarization method via the
    `params` init parameter. This can be an array, dictionary, or object. In
    other words, that is the only parameter you'll ever need. You can pack
    everything into it.
    """

    def __init__(self, params=None):
        """
        Initialization for the base diarization class.
        :param params: the tuple, dictonary, or object used to initialize the
                        diarization method.
        """
        self.tf_inititializer = tf.global_variables_initializer()
        self.sess = None
        self.init_diarization_method(params)
        pass

    @abstractmethod
    def init_diarization_method(self, params):
        """
        This method is called once to initialize the speech diarization algorithm.
        :param params: the tuple, dictonary, or object used to initialize the
                        diarization method.
        """
        pass

    @abstractmethod
    def train_on_data(self, data, label):
        """
        This method is meant to be called once per iteration during the training cycle.

        The tensorflow session is available as `self.sess`. tensorflow session initialization
        has already been taken care of with the `tf.global_variables_initializer()`

        :param data: This is the single set of data to be used by a single training
                    operation. This will change with each call of `train_on_data`.
        :param label: This is the textgrid label for this data sample.
        """
        pass

    @abstractmethod
    def run_on_data(self, data):
        """
        This method will be used to run the diarization method on a set of data and
        returns the TextGrid result.
        :param data: the audiofile to run the diarization on.
        :return: the TextGrid object.
        """
        pass

    @abstractmethod
    def get_train_error(self, test_data, test_label):
        """
        This method will be called at the end of each epoch to get the error of the
        diarization method.
        :param test_data: the data to use to evaluate the accuracy of the model.
        :param test_label: the label to use to evaluate the accuracy of the model.
        :return: The error of the diarization method.
        """

    @abstractmethod
    def load(self, path):
        """
        This method will be called when we want to load the diarization method from disk
        :param path: the path to the storage file that is to be loaded.
        :return: True on success, False on failure
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        This method will be called when we want to save the diarization method to disk to
        be recalled later.
        :param path: the path to where the storage file should be put.
        :return: True on success, False on failure
        """
