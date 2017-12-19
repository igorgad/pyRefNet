
import numpy as np

import pyRef
import pyRef_train

class trainParams:
    pass

# Fill pyRef_train.trainParams class with training parameters.
trainParams.lr          = 0.1
trainParams.momentum    = 0.8
trainParams.weigthDecay = 0.0

trainParams.numEpochs   = 200
trainParams.batch_size  = 10

trainParams.combSets    = [3, 4, 5]

trainParams.prefix      = 'REFTEST_BNORM'
trainParams.datasetfile = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/REFTEST_N' + str(pyRef.N) + '_NW' + str(pyRef.nwin) + '_XPAN10_medleyVBRdataset.mat'
trainParams.log_dir     = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/tensorlogs/' + str(trainParams.prefix) + 'N' + str(pyRef.N) + '_NW' + str(pyRef.nwin)


# Prepare DATASET Access and choose random examples from the combSets specified classess to fill train and eval indexes
trainParams.dtf = open(trainParams.datasetfile, 'r')

ncomb = np.fromfile(trainParams.dtf, dtype=np.int32, count=1)[0] - 1
combClass = np.fromfile(trainParams.dtf, dtype=np.int32, count=ncomb)

bind = {}
for i, elt in enumerate(trainParams.combSets):
    if elt not in bind:
        bind[elt] = 1

cid = np.nonzero(np.array([bind.get(itm,0)  for itm in combClass]))[0][:]
trainParams.trainIds = np.random.choice(cid, int(cid.size*0.8))
trainParams.evalIds  = np.random.choice(np.setdiff1d(cid, trainParams.trainIds), int(cid.size*0.2))

# Run experiments
pyRef_train.runExperiment(trainParams)