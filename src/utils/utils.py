import numpy as np
import pandas as pd 
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import os
import operator

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split


def check_if_file_exits(file_name):
    return os.path.exists(file_name)

def readucr(filename,delimiter='\t'):
    data = np.loadtxt(filename, delimiter = delimiter)
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return None 
        return directory_path

def create_path(root_dir,classifier_name, archive_name):
    output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+'/'
    if os.path.exists(output_directory): 
        return None
    else: 
        os.makedirs(output_directory)
        return output_directory

def read_all_datasets(root_dir,archive_name, split_val = False): 
    datasets_dict = {}

    dataset_names_to_sort = []


    for dataset_name in DATASET_NAMES:
        root_dir_dataset =root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'
        file_name = root_dir_dataset+dataset_name
        x_train, y_train = readucr(file_name+'_TRAIN.tsv')
        x_test, y_test = readucr(file_name+'_TEST.tsv')

        if split_val == True:
            # check if dataset has already been splitted
            temp_dir =root_dir_dataset+'TRAIN_VAL/'
            # print(temp_dir)
            train_test_dir = create_directory(temp_dir)
            # print(train_test_dir)
            if train_test_split is None:
                # then do no re-split because already splitted
                # read train set
                x_train,y_train = readucr(temp_dir+dataset_name+'_TRAIN')
                # read val set
                x_val,y_val = readucr(temp_dir+dataset_name+'_VAL')
            else:
                # split for cross validation set
                x_train,x_val,y_train,y_val  = train_test_split(x_train,y_train,
                    test_size=0.25)
                # concat train set
                train_set = np.zeros((y_train.shape[0],x_train.shape[1]+1),dtype=np.float64)
                train_set[:,0] = y_train
                train_set[:,1:] = x_train
                # concat val set
                val_set = np.zeros((y_val.shape[0],x_val.shape[1]+1),dtype=np.float64)
                val_set[:,0] = y_val
                val_set[:,1:] = x_val
                # save the train set
                np.savetxt(temp_dir+dataset_name+'_TRAIN',train_set,delimiter=',')
                # save the val set
                np.savetxt(temp_dir+dataset_name+'_VAL',val_set,delimiter=',')


            datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_val.copy(),
                y_val.copy(),x_test.copy(),y_test.copy())

        else:
            datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
                y_test.copy())

        dataset_names_to_sort.append((dataset_name,len(x_train)))

    dataset_names_to_sort.sort(key=operator.itemgetter(1))

    for i in range(len(DATASET_NAMES)):
        DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict

def calculate_metrics(y_true, y_pred,duration,y_true_val=None,y_pred_val=None): 
    res = pd.DataFrame(data = np.zeros((1,4),dtype=float), index=[0], 
        columns=['precision','accuracy','recall','duration'])
    res['precision'] = precision_score(y_true,y_pred,average='macro')
    res['accuracy'] = accuracy_score(y_true,y_pred)
    
    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val,y_pred_val)

    res['recall'] = recall_score(y_true,y_pred,average='macro')
    res['duration'] = duration
    return res

def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)

def transform_labels(y_train,y_test,y_val=None):
    """
    Transform label to min equal zero and continuous 
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None : 
        # index for when resplitting the concatenation 
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit 
        y_train_val_test = np.concatenate((y_train,y_val,y_test),axis =0)
        # fit the encoder 
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels 
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val,new_y_test 
    else: 
        # no validation split 
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit 
        y_train_test = np.concatenate((y_train,y_test),axis =0)
        # fit the encoder 
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels 
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test

def plot_epochs_metric(hist, file_name, metric=None):
    if metric is None:
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss',fontsize='large')
        plt.xlabel('epoch',fontsize='large')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(file_name+'epochs_loss.png',bbox_inches='tight')
        plt.close()
        plt.figure()
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy',fontsize='large')
        plt.xlabel('epoch',fontsize='large')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(file_name+'epochs_accuracy.png',bbox_inches='tight')
        plt.close()
    else:
        plt.figure()
        plt.plot(hist.history[metric])
        plt.plot(hist.history['val_'+metric])
        plt.title('model '+metric)
        plt.ylabel(metric,fontsize='large')
        plt.xlabel('epoch',fontsize='large')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(file_name+'epochs_loss.png',bbox_inches='tight')
        plt.close()

def save_logs(output_directory, hist, y_pred, y_true,duration,lr=True,y_true_val=None,y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory+'history.csv', index=False)

    df_metrics = calculate_metrics(y_true,y_pred, duration,y_true_val,y_pred_val)
    df_metrics.to_csv(output_directory+'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin() 
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=float) , index = [0], 
        columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc', 
        'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])
    
    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code 

    # plot losses 
    plot_epochs_metric(hist, output_directory)

    return df_metrics