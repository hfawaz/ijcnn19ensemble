import keras
import numpy as np
from utils.utils import calculate_metrics
from utils.constants import UNIVARIATE_DATASET_NAMES as datasets_names
from utils.utils import create_directory
import gc
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES
from utils.utils import check_if_file_exits


class Classifier_ENSEMBLETRANSFER:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):

        self.archive_name = ARCHIVE_NAMES[0]

        self.dataset_name = output_directory.split('/')[-2]

        # this should contain the transferred datasets
        self.transfer_directory = '/b/home/uha/hfawaz-datas/dl-tsc/transfer-learning-results/fcn_2000_train_all/'

        self.verbose = verbose

        self.output_directory = output_directory

        create_directory(self.output_directory)

    def fit(self, x_train, y_train, x_test, y_test, y_true):

        y_pred = np.zeros(shape=y_test.shape)

        l = 0

        for dataset in datasets_names:

            if dataset == self.dataset_name:
                continue

            curr_dir = self.transfer_directory+dataset+'/'+self.dataset_name+'/'

            predictions_file_name = curr_dir + 'y_pred.npy'

            if check_if_file_exits(predictions_file_name):
                # then load only the predictions from the file
                curr_y_pred = np.load(predictions_file_name)
            else:
                # predict from models saved
                model = keras.models.load_model(curr_dir+'best_model.hdf5')
                curr_y_pred = model.predict(x_test)
                keras.backend.clear_session()
                np.save(predictions_file_name, curr_y_pred)

            y_pred = y_pred + curr_y_pred

            l += 1

            keras.backend.clear_session()

        y_pred = y_pred / l

        # save predictions
        np.save(self.output_directory+'y_pred.npy',y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = calculate_metrics(y_true, y_pred, 0.0)

        df_metrics.to_csv(self.output_directory + 'df_metrics.csv', index=False)

        print(self.dataset_name,df_metrics['accuracy'][0])

        gc.collect()