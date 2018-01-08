from diarization_methods.DiarizationBaseClass import DiarizationBaseClass


class ExampleDiarizationMethod(DiarizationBaseClass):
    def run_on_data(self, data):
        print('run on data method')

    def init_diarization_method(self, params):
        print('diarization method instantiated with parameters:', params)

    def train_on_data(self, data, label):
        print('training on data.')

    def get_train_error(self, test_data, test_label):
        print('getting training error')
        return 0.1

    def load(self, path):
        print('loaded method')

    def save(self, path):
        print('saved method')
