
import os
import numpy as np

import pyRef
import pyRef_train

import importlib
importlib.reload(pyRef)
importlib.reload(pyRef_train)

class trainParams:
    pass

# Fill pyRef_train.trainParams class with training parameters.
trainParams.lr          = 0.001
trainParams.momentum    = 0.8 # Not used
trainParams.weigthDecay = 0.0 # Not used

trainParams.numEpochs   = 200
trainParams.batch_size  = 5

trainParams.combSets    = [3, 4, 5]

trainParams.prefix      = 'REFTEST_BNORM'
trainParams.datasetfile = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/REFTEST_N' + str(pyRef.N) + '_NW' + str(pyRef.nwin) + '_XPAN10_medleyVBRdataset.mat'
trainParams.log_dir     = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/tensorlogs/' + str(trainParams.prefix) + 'N' + str(pyRef.N) + '_NW' + str(pyRef.nwin)


# Prepare DATASET Access and choose random examples from the specified classess to fill train and eval indexes
trainParams.dtf = open(trainParams.datasetfile, 'r')
ncomb = np.int32(np.fromfile(trainParams.dtf, dtype=np.int32, count=1)[0] - 1)
combClass = np.int32(np.fromfile(trainParams.dtf, dtype=np.int32, count=ncomb))

cid = np.array(np.nonzero(np.in1d(combClass, trainParams.combSets) * 1))[0][:]
trainParams.trainIds = np.random.choice(cid, int(cid.size*0.8))
trainParams.evalIds  = np.random.choice(np.setdiff1d(cid, trainParams.trainIds), int(cid.size*0.2))

trainParams.mmap = np.memmap(trainParams.datasetfile, dtype=np.dtype([('ins', (np.float32, (pyRef.N, pyRef.nwin, pyRef.nsigs))), ('lbls', np.int32)]), mode='r', offset=ncomb.nbytes+combClass.nbytes)

# Start tensorboard on logdir
os.system('python -m tensorflow.tensorboard --logdir=' + trainParams.log_dir)

# Run experiments
pyRef_train.runExperiment(trainParams)

