# Deep Neural Network Ensembles for Time Series Classification
This is the companion repository for our paper also available on [ArXiv](https://arxiv.org/abs/1903.06602) titled "Deep Neural Network Ensembles for Time Series Classification". This paper has been accepted at the [IEEE International Joint Conference on Neural Networks (IJCNN) 2019](https://www.ijcnn.org/). 

## Approach
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

You can control which algorithms to include in the ensemble by changing [this line of code](https://github.com/hfawaz/ijcnn19ensemble/blob/cb822a0783ea6bd10359348f727b8fd81ae2c131/src/classifiers/nne.py#L35). 

## Prerequisites
All python packages needed are listed in [pip-requirements.txt](https://github.com/hfawaz/ijcnn19ensemble/blob/master/src/utils/pip-requirements.txt) file and can be installed simply using the pip command.

* [numpy](http://www.numpy.org/)  
* [pandas](https://pandas.pydata.org/)  
* [sklearn](http://scikit-learn.org/stable/)  
* [scipy](https://www.scipy.org/)  
* [matplotlib](https://matplotlib.org/)  
* [tensorflow-gpu](https://www.tensorflow.org/)  
* [keras](https://keras.io/)  
* [h5py](http://docs.h5py.org/en/latest/build.html)
* [keras_contrib](https://www.github.com/keras-team/keras-contrib.git)

## Results
The following table shows the results of four ensembles, the raw results can be found [here](https://github.com/hfawaz/ijcnn19ensemble/blob/master/results/results.csv). 

|                                | Fine-tuned FCNs | NNE    | ALL    | ResNets | 
|--------------------------------|-----------------|--------|--------|---------| 
| 50words                        | 66.81           | **80.00**  | **80.00**  | 77.14   | 
| Adiac                          | **85.17**           | **85.17**  | 83.38  | 83.63   | 
| ArrowHead                      | 84.00           | 86.29  | 86.29  | **86.86**   | 
| Beef                           | 76.67           | 76.67  | **80.00**  | 76.67   | 
| BeetleFly                      | **90.00**           | 85.00  | 85.00  | 85.00   | 
| BirdChicken                    | 90.00           | **95.00**  | 85.00  | 90.00   | 
| CBF                            | **99.78**           | 99.44  | 98.56  | **99.78**   | 
| Car                            | 91.67           | **95.00**  | 86.67  | 93.33   | 
| ChlorineConcentration          | 82.42           | 85.05  | 83.98  | **85.49**   | 
| CinC_ECG_torso                 | 85.87           | 89.71  | **92.90**  | 83.55   | 
| Coffee                         | **100.00**          | **100.00** | **100.00** | **100.00**  | 
| Computers                      | 83.20           | **83.60**  | 71.60  | **83.60**   | 
| Cricket_X                      | 78.97           | **82.05**  | 77.95  | 81.54   | 
| Cricket_Y                      | 79.23           | **84.36**  | 78.72  | 82.05   | 
| Cricket_Z                      | 82.05           | **83.85**  | 79.49  | 82.05   | 
| DiatomSizeReduction            | 30.07           | 30.07  | **88.56**  | 30.07   | 
| DistalPhalanxOutlineAgeGroup   | 71.94           | 72.66  | **76.26**  | 73.38   | 
| DistalPhalanxOutlineCorrect    | 77.54           | 77.90  | 77.90  | **78.99**   | 
| DistalPhalanxTW                | **71.22**           | 65.47  | 67.63  | 66.19   | 
| ECG200                         | 89.00           | 89.00  | **92.00**  | 88.00   | 
| ECG5000                        | 94.16           | 94.42  | **94.51**  | 93.67   | 
| ECGFiveDays                    | 99.54           | **99.88**  | 99.65  | 98.61   | 
| Earthquakes                    | 71.94           | **74.82**  | **74.82**  | 72.66   | 
| ElectricDevices                | 71.74           | **74.39**  | 73.03  | 74.22   | 
| FISH                           | 96.00           | 97.71  | 93.71  | **98.29**   | 
| FaceAll                        | **92.84**           | 86.39  | 83.91  | 84.02   | 
| FaceFour                       | 93.18           | **95.45**  | 92.05  | **95.45**   | 
| FacesUCR                       | 93.95           | 95.76  | 95.51  | **95.90**   | 
| FordA                          | 90.67           | 93.70  | **94.22**  | 92.56   | 
| FordB                          | 88.04           | **92.90**  | 92.33  | 92.16   | 
| Gun_Point                      | **100.00**          | **100.00** | 99.33  | 99.33   | 
| Ham                            | 74.29           | 75.24  | 74.29  | **78.10**   | 
| HandOutlines                   | 92.70           | **95.14**  | 93.78  | 93.78   | 
| Haptics                        | 50.65           | 52.60  | 50.97  | **53.25**   | 
| Herring                        | **65.62**           | 60.94  | 62.50  | 60.94   | 
| InlineSkate                    | **40.55**           | 38.36  | 38.00  | 38.55   | 
| InsectWingbeatSound            | 39.49           | 59.75  | **65.91**  | 52.73   | 
| ItalyPowerDemand               | 96.11           | 96.50  | **96.89**  | 96.40   | 
| LargeKitchenAppliances         | 89.60           | **90.93**  | 83.20  | 89.60   | 
| Lighting2                      | **80.33**           | **80.33**  | 77.05  | 78.69   | 
| Lighting7                      | 89.04           | **90.41**  | 83.56  | 83.56   | 
| MALLAT                         | 96.93           | 96.93  | 95.44  | **97.40**   | 
| Meat                           | 91.67           | 95.00  | 93.33  | **96.67**   | 
| MedicalImages                  | 78.29           | 79.74  | **80.13**  | 78.42   | 
| MiddlePhalanxOutlineAgeGroup   | 53.90           | 59.09  | **60.39**  | 59.09   | 
| MiddlePhalanxOutlineCorrect    | 81.10           | 83.51  | **83.85**  | 83.51   | 
| MiddlePhalanxTW                | 51.95           | 51.95  | **55.19**  | 49.35   | 
| MoteStrain                     | 93.37           | **93.93**  | 93.45  | 93.05   | 
| NonInvasiveFatalECG_Thorax1    | **96.44**           | 96.39  | 95.88  | 95.01   | 
| NonInvasiveFatalECG_Thorax2    | 95.73           | 96.18  | 96.54  | 95.01   | 
| OSULeaf                        | 97.52           | **98.76**  | 78.51  | 98.35   | 
| OliveOil                       | **86.67**           | **86.67**  | **86.67**  | **86.67**   | 
| PhalangesOutlinesCorrect       | 83.57           | 84.27  | 83.57  | **84.97**   | 
| Phoneme                        | 32.65           | **35.13**  | 30.91  | 34.81   | 
| Plane                          | **100.00**          | **100.00** | 99.05  | **100.00**  | 
| ProximalPhalanxOutlineAgeGroup | 84.39           | 84.88  | **85.85**  | 85.37   | 
| ProximalPhalanxOutlineCorrect  | **92.10**           | 91.75  | 90.38  | **92.10**   | 
| ProximalPhalanxTW              | 79.51           | 77.56  | **80.98**  | 78.54   | 
| RefrigerationDevices           | 50.40           | 53.07  | **53.33**  | 52.80   | 
| ScreenType                     | **65.07**           | 62.13  | 52.27  | 62.13   | 
| ShapeletSim                    | 86.11           | 81.11  | 70.56  | **93.89**   | 
| ShapesAll                      | 90.00           | **92.83**  | 89.17  | 92.00   | 
| SmallKitchenAppliances         | 79.47           | **82.13**  | 77.60  | 78.93   | 
| SonyAIBORobotSurface           | 95.84           | 94.68  | 78.04  | **96.17**   | 
| SonyAIBORobotSurfaceII         | **98.22**           | 97.69  | 88.88  | 98.11   | 
| StarLightCurves                | 96.78           | **97.92**  | 97.79  | 97.38   | 
| Strawberry                     | 97.84           | **98.11**  | 97.57  | **98.11**   | 
| SwedishLeaf                    | **97.28**           | **97.28**  | 96.16  | 96.48   | 
| Symbols                        | 95.68           | **95.88**  | 91.06  | 91.56   | 
| ToeSegmentation1               | 96.49           | **98.25**  | 81.58  | 96.05   | 
| ToeSegmentation2               | 90.77           | 92.31  | **93.08**  | 91.54   | 
| Trace                          | **100.00**          | **100.00** | 98.00  | **100.00**  | 
| TwoLeadECG                     | 99.91           | **100.00** | 97.72  | **100.00**  | 
| Two_Patterns                   | 87.62           | **100.00** | **100.00** | **100.00**  | 
| UWaveGestureLibraryAll         | 82.86           | 92.27  | **96.26**  | 87.16   | 
| Wine                           | 77.78           | 87.04  | **90.74**  | 83.33   | 
| WordsSynonyms                  | 55.96           | 66.93  | **68.97**  | 62.85   | 
| Worms                          | 76.62           | 81.82  | 62.34  | **83.12**   | 
| WormsTwoClass                  | 74.03           | **77.92**  | 63.64  | **77.92**   | 
| synthetic_control              | 98.67           | **100.00** | **100.00** | **100.00**  | 
| uWaveGestureLibrary_X          | 76.13           | 82.10  | **83.28**  | 79.51   | 
| uWaveGestureLibrary_Y          | 64.82           | 73.20  | **75.38**  | 68.68   | 
| uWaveGestureLibrary_Z          | 73.12           | **78.03**  | 77.41  | 76.19   | 
| wafer                          | 99.61           | 99.84  | 99.81  | **99.90**   | 
| yoga                           | 87.10           | **89.33**  | 88.57  | 88.17   | 
| **Wins**                           | 18           | **38**  | 29 | 27 | 


## Critical difference diagrams
If you would like to generate these diagrams, take a look at [this code](https://github.com/hfawaz/cd-diagram)!

![cd-diagram-resnets](https://github.com/hfawaz/ijcnn19ensemble/blob/master/png/cd-diagram-resnets.png)
![cd-diagram-all](https://github.com/hfawaz/ijcnn19ensemble/blob/master/png/cd-diagram-all.png)
![cd-diagram-nne](https://github.com/hfawaz/ijcnn19ensemble/blob/master/png/cd-diagram-nne.png)
 

## Reference

If you re-use this work, please cite:

```
@InProceedings{IsmailFawaz2019deep,
  Title                    = {Deep Neural Network Ensembles for Time Series Classification},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  booktitle                = {IEEE International Joint Conference on Neural Networks},
  Year                     = {2019}
}
```

## Acknowledgement

We would like to thank NVIDIA Corporation for the Quadro P6000 grant and the Mésocentre of Strasbourg for providing access to the cluster.
