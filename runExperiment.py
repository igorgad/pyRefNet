
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
trainParams.lr          = 0.0001
trainParams.momentum    = 0.6 # Not used
trainParams.weigthDecay = 0.0 # Not used

trainParams.numEpochs   = 200
trainParams.batch_size  = 52

trainParams.combSets    = [4, 5]

trainParams.datasetfile = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/REFTEST_N' + str(pyRef.N) + '_NW' + str(pyRef.nwin) + '_XPAN40_medleyVBRdataset.mat'
trainParams.log_root     = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/tensorlogs/'

trainParams.runName      = "{}_N{}_NW{}".format('initTest' , pyRef.N, pyRef.nwin)
trainParams.n = sum(1 for f in os.listdir(trainParams.log_root) if os.path.isdir(os.path.join(trainParams.log_root, f)))
trainParams.log_dir = "{}{}_run_{}".format(trainParams.log_root, trainParams.runName, trainParams.n+1)
trainParams.sumPerEpoch = 4

print ('logdir ' + trainParams.log_dir)

# Prepare DATASET Access and choose random examples from the specified classess to fill train and eval indexes
trainParams.dtf = open(trainParams.datasetfile, 'r')
ncomb = np.int32(np.fromfile(trainParams.dtf, dtype=np.int32, count=1)[0] - 1)
combClass = np.int32(np.fromfile(trainParams.dtf, dtype=np.int32, count=ncomb))

cid = np.array(np.nonzero(np.in1d(combClass, trainParams.combSets) * 1))[0][:]
trainParams.trainIds = np.random.choice(cid, int(cid.size * 0.8))
trainParams.evalIds  = np.random.choice(np.setdiff1d(cid, trainParams.trainIds), int(cid.size * 0.2))

trainParams.mmap = np.memmap(trainParams.datasetfile, dtype=np.dtype([('ins', (np.float32, (pyRef.nsigs, pyRef.nwin, pyRef.N))), ('lbls', np.int32)]),
                             mode='r', offset=ncomb.nbytes+combClass.nbytes)

# Run experiments
dbg = pyRef_train.runExperiment(trainParams)

