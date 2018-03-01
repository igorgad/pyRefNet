
import os
import numpy as np

import pyRef_train

import importlib
importlib.reload(pyRef_train)

class trainParams:
    pass

# Fill pyRef_train.trainParams class with training parameters.
trainParams.numEpochs   = 200
trainParams.combSets    = [3, 4, 5]

trainParams.datasetfile  = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/bitrate_medleydb_blocksize512.tfrecord'
trainParams.log_root     = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/tensorlogs/'

trainParams.runName      = "{}_N{}_NW{}".format(pyRef_train.model.name, pyRef_train.model.N, pyRef_train.model.nwin)
trainParams.n = sum(1 for f in os.listdir(trainParams.log_root) if os.path.isdir(os.path.join(trainParams.log_root, f)))
trainParams.log_dir = "{}{}_run_{}".format(trainParams.log_root, trainParams.runName, trainParams.n+1)
trainParams.sumPerEpoch = 4

trainParams.hptext = {key:value for key, value in trainParams.__dict__.items() if not key.startswith('__') and not callable(key)}
print ('logdir ' + trainParams.log_dir)

trainParams.trainIds = np.random.randint(0,1000,[10000])
trainParams.evalIds  = np.random.randint(0,1000,[10000])

# Run experiments
stats = pyRef_train.runExperiment(trainParams)
