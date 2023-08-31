import tensorflow as tf
import numpy as np
import gc
import time

from tensorflow import keras
from utils.utils import calculate_metrics
from utils.utils import create_directory
from utils.utils import check_if_file_exits
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES

class Classifier_NNE:

    def create_classifier(self, model_name, input_shape, nb_classes, output_directory, verbose=False,
                          build=True, load_weights=False):
        if model_name == 'fcn':
            from classifiers import fcn
            return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose, build=build)
        if model_name == 'mlp':
            from classifiers import mlp
            return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose, build=build)
        if model_name == 'resnet':
            from classifiers import resnet
            return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose,
                                            build=build, load_weights=load_weights)
        if model_name == 'encoder':
            from classifiers import encoder
            return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose, build=build)
        if model_name == 'mcdcnn':
            from classifiers import mcdcnn
            return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose, build=build)
        if model_name == 'cnn':
            from classifiers import cnn
            return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose, build=build)

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        # self.classifiers = ['mlp','fcn','resnet','encoder','mcdcnn','cnn']
        # self.classifiers = ['fcn','resnet','encoder'] # this represents NNE in the paper
        self.classifiers = ['fcn','resnet','inception']
        # self.classifiers = ['fcn','resnet','encoder','inception']
        out_add = ''
        for cc in self.classifiers:
            out_add = out_add + cc + '-'
        self.archive_name = ARCHIVE_NAMES[0]
        self.output_directory = output_directory.replace('nne',
                                                         'nne'+'/'+out_add)
        create_directory(self.output_directory)
        self.dataset_name = output_directory.split('/')[-2]
        self.verbose = verbose
        self.models_dir = output_directory.replace('nne','classifier')
        self.iterations = 10

    def fit(self, x_train, y_train, x_test, y_test, y_true):
        # no training since models are pre-trained
        start_time = time.time()

        y_pred = np.zeros(shape=y_test.shape)

        l = 0

        # loop through all classifiers
        for model_name in self.classifiers:
            # loop through different initialization of classifiers
            for itr in range(self.iterations):
                if itr == 0:
                    itr_str = ''
                else:
                    itr_str = '_itr_' + str(itr)

                curr_archive_name = self.archive_name+itr_str

                curr_dir = self.models_dir.replace('classifier',model_name).replace(
                    self.archive_name,curr_archive_name)

                model = self.create_classifier(model_name, None, None,
                                               curr_dir, build=False)

                predictions_file_name = curr_dir+'y_pred.npy'
                # check if predictions already made
                if check_if_file_exits(predictions_file_name):
                    # then load only the predictions from the file
                    curr_y_pred = np.load(predictions_file_name)
                else:
                    # then compute the predictions
                    curr_y_pred = model.predict(x_test,y_true,x_train,y_train,y_test,
                                                     return_df_metrics = False)
                    keras.backend.clear_session()

                    np.save(predictions_file_name,curr_y_pred)

                y_pred = y_pred+curr_y_pred

                l+=1

        # average predictions
        y_pred = y_pred / l

        # save predictiosn
        np.save(self.output_directory+'y_pred.npy',y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        duration = time.time() - start_time

        df_metrics = calculate_metrics(y_true, y_pred, duration)

        df_metrics.to_csv(self.output_directory + 'df_metrics.csv', index=False)

        # the creation of this directory means
        create_directory(self.output_directory + '/DONE')

        gc.collect()