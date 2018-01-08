import tensorflow as tf

from scipy.io import wavfile
from textgrid import TextGrid
from diarization_methods.ExampleDiarizationMethod import ExampleDiarizationMethod
from os import walk
from os.path import splitext
from numpy import array, vstack
from numpy.random import shuffle, randint

audio_sample_rate = 44100

# Parameters to be set in some more elegant way later on.
audio_files_path = '../media/Sound Files/'
text_grids_path = '../media/TextGrids/'
load_all_to_memory = True
window_size = int(audio_sample_rate * 0.2)
epochs = 1000
batch_size = 10
percent_train = .8
params = {
    'layers': [window_size, 500, 500, 4],
    'learning_rate': 0.01
}
diarization_method = ExampleDiarizationMethod(params)
training_flag = True
# load_path = '../media/example_method.save'
# save_path = '../media/example_method.save'
load_path = None
save_path = None


# End parameters section


def get_file_list(directory):
    """
    Used to get all the file names in the directory specified.
    :param directory: the path to the directory to get the file names from (ending in '/'
    :return: a list of all files names in directory starting with directory
    """
    f = []
    for (dirpath, dirnames, filenames) in walk(directory):
        f = filenames
        break
    f.sort()
    for i in range(len(f)):
        f[i] = directory + f[i]
    return f


def get_label(path):
    """
    Used to load a TextGrid from disk and return the array representation of it.
    :param path: Path to the TextGrid
    :return: Array representation of the TextGrid.
    """
    grid = TextGrid(name=path)
    grid.read(path, Fs=int(audio_sample_rate / window_size))
    return grid.FsArrayCombined


class Trainer:
    def __init__(self):
        self.data_paths = None
        self.label_paths = None
        self.training_data = None
        self.training_labels = None
        self.testing_data = None
        self.testing_labels = None
        self.imax = -1

        self.data_paths = array([get_file_list(audio_files_path)])
        self.label_paths = array([get_file_list(text_grids_path)])
        if len(self.data_paths) != len(self.label_paths):
            print('Error! Data and labels do not match up!')
            ValueError()

        self.split_train_test()

        if load_all_to_memory:
            print('Loading Data...', end='', flush=True)
            self.load_all_data_to_memory()
            print('Done!')
        else:
            print("Training is going to take a bit longer without loading all data first..")

    def get_chunk(self, is_train=True):
        data = self.training_data if is_train else self.testing_data
        labels = self.training_labels if is_train else self.testing_labels

        i = randint(0, len(data))
        x = data[i]  # should be a list at this point
        label = labels[i]

        if not load_all_to_memory:  # means that data is a path instead
            fs, x = wavfile.read(x[0])
            label = get_label(label[0])
        start = randint(0, len(label) - 1)
        return x.T[:, start * window_size:start * window_size + window_size], label[start]

    def load_all_data_to_memory(self):
        paths = array(self.training_data, copy=True)
        self.training_data = []
        for path in paths:
            fs, x = wavfile.read(path[0])
            self.training_data.append(x)

        paths = array(self.testing_data, copy=True)
        self.testing_data = []
        for path in paths:
            fs, x = wavfile.read(path[0])
            self.testing_data.append(x)

        paths = array(self.training_labels, copy=True)
        self.training_labels = []
        for path in paths:
            self.training_labels.append(get_label(path[0]))

        paths = array(self.testing_labels, copy=True)
        self.testing_labels = []
        for path in paths:
            self.testing_labels.append(get_label(path[0]))

    def split_train_test(self):
        shuf = vstack((self.data_paths, self.label_paths)).T
        shuffle(shuf)

        self.imax = i_max = int(len(self.data_paths[0]) * percent_train) + 1
        self.training_data = shuf[0:i_max, [0]]
        self.training_labels = shuf[0:i_max, [1]]
        self.testing_data = shuf[i_max - 1:-1, [0]]
        self.testing_labels = shuf[i_max - 1:-1, [1]]
        # At this point the training and testing sets contain a list of paths to data and labels.

    def train(self):

        try:
            diarization_method.tf_inititializer = tf.global_variables_initializer()
            diarization_method.sess = tf.Session()
            diarization_method.sess.run(diarization_method.tf_inititializer)

            if load_path is not None:
                print('Loading Diarization')
                diarization_method.load(load_path)
            print('Progress:')
            print('[0%' + '-' * 94 + '100%]')
            for i in range(epochs):
                for j in range(batch_size):
                    x, label = self.get_chunk()
                    diarization_method.train_on_data(x.T, label)
                    per = int(j/float(batch_size) * 100)
                    print('\r[' + '.' * per + ' ' * (100-per) + ']', end="", flush=True)

                x, label = self.get_chunk(is_train=True)
                train_error = diarization_method.get_train_error(x.T, label)
                print('\rEpoch: %6d, Train Error: %8.8f ' % (i, train_error))

            print('Finished Training...')
            print('Testing...')

            # Testing section
            error = 0
            l = len(self.testing_data)
            for i in range(l):
                x, label = self.get_chunk(is_train=False)
                error += diarization_method.get_train_error(x.T, label)

            print('Average testing Error: ', error / l)

        except Exception as ex:
            print("ERROR! and exception Occurred!")
            print(ex)
        except KeyboardInterrupt:
            print("\n Interrupted..")
        finally:
            if save_path is not None:
                print('Saving Diarization...')
                diarization_method.save(save_path)
            diarization_method.sess.close()


if __name__ == '__main__':
    # TODO: command line argument parsing. Use argparse
    trainer = Trainer()
    trainer.train()
