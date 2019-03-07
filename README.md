# Deep Neural Network Ensembles for Time Series Classification
This is the companion repository for our paper also available on ArXiv titled "Deep Neural Network Ensembles for Time Series Classification". This paper has been accepted at the [International Joint Conference on Neural Networks (IJCNN) 2019](https://www.ijcnn.org/). 

# Approach
![ensemble](https://github.com/hfawaz/ijcnn19ensemble/blob/master/png/ensemble.png)

## Data 
The data used in this project comes from the [UCR/UEA archive](http://timeseriesclassification.com/TSC.zip), which contains the 85 univariate time series datasets. 

## Code 
The code is divided as follows: 
* The [main.py](https://github.com/hfawaz/ijcnn19ensemble/blob/master/src/main.py) python file contains the necessary code to run all experiements. 
* The [utils](https://github.com/hfawaz/ijcnn19ensemble/blob/master/src/utils/) folder contains the necessary functions to read the datasets and manipulate the data.
* The [classifiers](https://github.com/hfawaz/ijcnn19ensemble/tree/master/src/classifiers) folder contains eight python files one for each deep individual/ensemble classifier presented in our paper. 

To run a model on all datasets you should issue the following command: 
```
python3 main.py
```
To control which datasets and which individual/ensemble classifiers to run see the options in [constants.py](https://github.com/hfawaz/ijcnn19ensemble/blob/master/src/utils/constants.py).  
