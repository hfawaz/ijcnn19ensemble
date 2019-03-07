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
|                                | Fine-Tuned FCNs     | NNE                | ALL                | ResNets             | 
|--------------------------------|---------------------|--------------------|--------------------|---------------------| 
| 50words                        | 0.6681318681318681  | 0.8                | 0.8                | 0.7714285714285715  | 
| Adiac                          | 0.8516624040920716  | 0.8516624040920716 | 0.8337595907928389 | 0.8363171355498721  | 
| ArrowHead                      | 0.84                | 0.8628571428571429 | 0.8628571428571429 | 0.8685714285714285  | 
| Beef                           | 0.7666666666666667  | 0.7666666666666667 | 0.8                | 0.7666666666666667  | 
| BeetleFly                      | 0.9                 | 0.85               | 0.85               | 0.85                | 
| BirdChicken                    | 0.9                 | 0.95               | 0.85               | 0.9                 | 
| CBF                            | 0.9977777777777778  | 0.9944444444444444 | 0.9855555555555556 | 0.9977777777777778  | 
| Car                            | 0.9166666666666666  | 0.95               | 0.8666666666666667 | 0.9333333333333332  | 
| ChlorineConcentration          | 0.82421875          | 0.8505208333333333 | 0.83984375         | 0.8549479166666667  | 
| CinC_ECG_torso                 | 0.8586956521739131  | 0.8971014492753623 | 0.9289855072463769 | 0.8355072463768116  | 
| Coffee                         | 1.0                 | 1.0                | 1.0                | 1.0                 | 
| Computers                      | 0.8320000000000001  | 0.836              | 0.716              | 0.836               | 
| Cricket_X                      | 0.7897435897435897  | 0.8205128205128205 | 0.7794871794871795 | 0.8153846153846154  | 
| Cricket_Y                      | 0.7923076923076923  | 0.8435897435897436 | 0.7871794871794872 | 0.8205128205128205  | 
| Cricket_Z                      | 0.8205128205128205  | 0.8384615384615385 | 0.7948717948717948 | 0.8205128205128205  | 
| DiatomSizeReduction            | 0.3006535947712418  | 0.3006535947712418 | 0.8856209150326797 | 0.3006535947712418  | 
| DistalPhalanxOutlineAgeGroup   | 0.7194244604316546  | 0.7266187050359713 | 0.7625899280575541 | 0.7338129496402878  | 
| DistalPhalanxOutlineCorrect    | 0.7753623188405797  | 0.7789855072463768 | 0.7789855072463768 | 0.7898550724637681  | 
| DistalPhalanxTW                | 0.7122302158273381  | 0.6546762589928058 | 0.6762589928057554 | 0.6618705035971223  | 
| ECG200                         | 0.89                | 0.89               | 0.92               | 0.88                | 
| ECG5000                        | 0.9415555555555556  | 0.9442222222222222 | 0.9451111111111112 | 0.9366666666666666  | 
| ECGFiveDays                    | 0.9953542392566784  | 0.9988385598141696 | 0.9965156794425089 | 0.9860627177700348  | 
| Earthquakes                    | 0.7194244604316546  | 0.7482014388489209 | 0.7482014388489209 | 0.7266187050359713  | 
| ElectricDevices                | 0.7174166774737388  | 0.7438723900920763 | 0.7302554791855791 | 0.7421864868369861  | 
| FISH                           | 0.96                | 0.9771428571428572 | 0.9371428571428572 | 0.9828571428571428  | 
| FaceAll                        | 0.9284023668639052  | 0.863905325443787  | 0.8390532544378698 | 0.8402366863905325  | 
| FaceFour                       | 0.9318181818181818  | 0.9545454545454546 | 0.9204545454545454 | 0.9545454545454546  | 
| FacesUCR                       | 0.9395121951219512  | 0.957560975609756  | 0.9551219512195122 | 0.9590243902439024  | 
| FordA                          | 0.9066925853929464  | 0.9369619550124966 | 0.9422382671480144 | 0.9255762288253264  | 
| FordB                          | 0.8803630363036303  | 0.929042904290429  | 0.9232673267326732 | 0.9216171617161716  | 
| Gun_Point                      | 1.0                 | 1.0                | 0.9933333333333332 | 0.9933333333333332  | 
| Ham                            | 0.7428571428571429  | 0.7523809523809524 | 0.7428571428571429 | 0.7809523809523811  | 
| HandOutlines                   | 0.927027027027027   | 0.9513513513513514 | 0.9378378378378378 | 0.9378378378378378  | 
| Haptics                        | 0.5064935064935064  | 0.525974025974026  | 0.5097402597402597 | 0.5324675324675324  | 
| Herring                        | 0.65625             | 0.609375           | 0.625              | 0.609375            | 
| InlineSkate                    | 0.4054545454545455  | 0.3836363636363636 | 0.38               | 0.38545454545454544 | 
| InsectWingbeatSound            | 0.3949494949494949  | 0.5974747474747475 | 0.6590909090909091 | 0.5272727272727272  | 
| ItalyPowerDemand               | 0.9611273080660836  | 0.9650145772594751 | 0.9689018464528668 | 0.9640427599611272  | 
| LargeKitchenAppliances         | 0.8959999999999999  | 0.9093333333333332 | 0.8320000000000001 | 0.8959999999999999  | 
| Lighting2                      | 0.8032786885245902  | 0.8032786885245902 | 0.7704918032786885 | 0.7868852459016393  | 
| Lighting7                      | 0.8904109589041096  | 0.9041095890410958 | 0.8356164383561644 | 0.8356164383561644  | 
| MALLAT                         | 0.9692963752665246  | 0.9692963752665246 | 0.9543710021321962 | 0.9739872068230278  | 
| Meat                           | 0.9166666666666666  | 0.95               | 0.9333333333333332 | 0.9666666666666668  | 
| MedicalImages                  | 0.7828947368421053  | 0.7973684210526316 | 0.8013157894736842 | 0.7842105263157895  | 
| MiddlePhalanxOutlineAgeGroup   | 0.5389610389610391  | 0.5909090909090909 | 0.6038961038961039 | 0.5909090909090909  | 
| MiddlePhalanxOutlineCorrect    | 0.8109965635738832  | 0.8350515463917526 | 0.8384879725085911 | 0.8350515463917526  | 
| MiddlePhalanxTW                | 0.5194805194805194  | 0.5194805194805194 | 0.551948051948052  | 0.4935064935064935  | 
| MoteStrain                     | 0.9337060702875402  | 0.9392971246006392 | 0.9345047923322684 | 0.9305111821086262  | 
| NonInvasiveFatalECG_Thorax1    | 0.9643765903307888  | 0.9638676844783716 | 0.9587786259541984 | 0.9501272264631044  | 
| NonInvasiveFatalECG_Thorax2    | 0.9572519083969464  | 0.9618320610687024 | 0.9653944020356234 | 0.9501272264631044  | 
| OSULeaf                        | 0.9752066115702479  | 0.987603305785124  | 0.7851239669421488 | 0.9834710743801652  | 
| OliveOil                       | 0.8666666666666667  | 0.8666666666666667 | 0.8666666666666667 | 0.8666666666666667  | 
| PhalangesOutlinesCorrect       | 0.8356643356643356  | 0.8426573426573427 | 0.8356643356643356 | 0.8496503496503497  | 
| Phoneme                        | 0.32647679324894513 | 0.3512658227848101 | 0.3090717299578059 | 0.34810126582278483 | 
| Plane                          | 1.0                 | 1.0                | 0.9904761904761904 | 1.0                 | 
| ProximalPhalanxOutlineAgeGroup | 0.8439024390243902  | 0.8487804878048779 | 0.8585365853658536 | 0.8536585365853658  | 
| ProximalPhalanxOutlineCorrect  | 0.9209621993127148  | 0.9175257731958762 | 0.9037800687285223 | 0.9209621993127148  | 
| ProximalPhalanxTW              | 0.7951219512195122  | 0.775609756097561  | 0.8097560975609757 | 0.7853658536585366  | 
| RefrigerationDevices           | 0.504               | 0.5306666666666666 | 0.5333333333333333 | 0.528               | 
| ScreenType                     | 0.6506666666666666  | 0.6213333333333333 | 0.5226666666666666 | 0.6213333333333333  | 
| ShapeletSim                    | 0.8611111111111112  | 0.8111111111111111 | 0.7055555555555556 | 0.9388888888888888  | 
| ShapesAll                      | 0.9                 | 0.9283333333333332 | 0.8916666666666667 | 0.92                | 
| SmallKitchenAppliances         | 0.7946666666666666  | 0.8213333333333334 | 0.7759999999999999 | 0.7893333333333333  | 
| SonyAIBORobotSurface           | 0.9584026622296172  | 0.9467554076539102 | 0.7803660565723793 | 0.961730449251248   | 
| SonyAIBORobotSurfaceII         | 0.9821615949632738  | 0.9769150052465896 | 0.8887722980062959 | 0.9811122770199372  | 
| StarLightCurves                | 0.9678241864983002  | 0.9792374939290918 | 0.9779018941233608 | 0.9737736765420106  | 
| Strawberry                     | 0.9783783783783784  | 0.9810810810810808 | 0.9756756756756756 | 0.9810810810810808  | 
| SwedishLeaf                    | 0.9728              | 0.9728             | 0.9616             | 0.9648              | 
| Symbols                        | 0.9567839195979899  | 0.9587939698492464 | 0.9105527638190954 | 0.9155778894472362  | 
| ToeSegmentation1               | 0.9649122807017544  | 0.9824561403508772 | 0.8157894736842105 | 0.9605263157894736  | 
| ToeSegmentation2               | 0.9076923076923076  | 0.9230769230769232 | 0.9307692307692308 | 0.9153846153846154  | 
| Trace                          | 1.0                 | 1.0                | 0.98               | 1.0                 | 
| TwoLeadECG                     | 0.9991220368744512  | 1.0                | 0.9771729587357332 | 1.0                 | 
| Two_Patterns                   | 0.87625             | 1.0                | 1.0                | 1.0                 | 
| UWaveGestureLibraryAll         | 0.8285873813512005  | 0.9226689000558348 | 0.9625907314349526 | 0.8715801228364043  | 
| Wine                           | 0.7777777777777778  | 0.8703703703703703 | 0.9074074074074074 | 0.8333333333333334  | 
| WordsSynonyms                  | 0.5595611285266457  | 0.6692789968652038 | 0.6896551724137931 | 0.6285266457680251  | 
| Worms                          | 0.7662337662337663  | 0.8181818181818182 | 0.6233766233766234 | 0.8311688311688312  | 
| WormsTwoClass                  | 0.7402597402597403  | 0.7792207792207793 | 0.6363636363636364 | 0.7792207792207793  | 
| synthetic_control              | 0.9866666666666668  | 1.0                | 1.0                | 1.0                 | 
| uWaveGestureLibrary_X          | 0.7613065326633166  | 0.8210496929089894 | 0.8327749860413177 | 0.7950865438302624  | 
| uWaveGestureLibrary_Y          | 0.6482412060301508  | 0.7319932998324958 | 0.7537688442211056 | 0.6867671691792295  | 
| uWaveGestureLibrary_Z          | 0.7311557788944724  | 0.7802903405918481 | 0.7741485203796762 | 0.7618648799553323  | 
| wafer                          | 0.9961064243997404  | 0.9983776768332252 | 0.9980532121998702 | 0.9990266060999352  | 
| yoga                           | 0.871               | 0.8933333333333333 | 0.8856666666666667 | 0.8816666666666667  | 

