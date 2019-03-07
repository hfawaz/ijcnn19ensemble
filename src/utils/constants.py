UNIVARIATE_DATASET_NAMES = ['50words','Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','Car','CBF','ChlorineConcentration','CinC_ECG_torso','Coffee',
'Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW',
'Earthquakes','ECG200','ECG5000','ECGFiveDays','ElectricDevices','FaceAll','FaceFour','FacesUCR','FISH','FordA','FordB','Gun_Point','Ham','HandOutlines',
'Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lighting2','Lighting7','MALLAT','Meat','MedicalImages',
'MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','NonInvasiveFatalECG_Thorax1','NonInvasiveFatalECG_Thorax2','OliveOil',
'OSULeaf','PhalangesOutlinesCorrect','Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices',
'ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface','SonyAIBORobotSurfaceII','StarLightCurves','Strawberry','SwedishLeaf','Symbols',
'synthetic_control','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','Two_Patterns','UWaveGestureLibraryAll','uWaveGestureLibrary_X','uWaveGestureLibrary_Y',
'uWaveGestureLibrary_Z','wafer','Wine','WordsSynonyms','Worms','WormsTwoClass','yoga']

UNIVARIATE_DATASET_NAMES = ['Adiac']

ITERATIONS = 1 # nb of random runs for random initializations

UNIVARIATE_ARCHIVE_NAMES = ['TSC']

CLASSIFIERS = ['fcn','mlp','resnet','encoder','mcdcnn','cnn','nne','ensembletransfer']

# CLASSIFIERS = ['nne/mlp-','nne/fcn-','nne/resnet-','nne/encoder-','nne/mcdcnn-','nne/cnn-',
			   # 'nne/fcn-resnet-encoder-','ensembletransfer/NN(10)-weight(True)','ensembletransfer/NN(84)-weight(True)',
			   # 'ensembletransfer/NN(10)-weight(False)','ensembletransfer/NN(84)-weight(False)']
# CLASSIFIERS = ['ensembletransfer']
