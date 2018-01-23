
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

trainParams.datasetfile = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/REFTEST_N' + str(pyRef_train.model.N) + '_NW' + str(pyRef_train.model.nwin) + '_XPAN40_medleyVBRdataset.mmap'
trainParams.log_root     = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/tensorlogs/'

trainParams.runName      = "{}_N{}_NW{}".format(pyRef_train.model.name, pyRef_train.model.N, pyRef_train.model.nwin)
trainParams.n = sum(1 for f in os.listdir(trainParams.log_root) if os.path.isdir(os.path.join(trainParams.log_root, f)))
trainParams.log_dir = "{}{}_run_{}".format(trainParams.log_root, trainParams.runName, trainParams.n+1)
trainParams.sumPerEpoch = 4

trainParams.hptext = {key:value for key, value in trainParams.__dict__.items() if not key.startswith('__') and not callable(key)}
print ('logdir ' + trainParams.log_dir)

# Prepare DATASET Access and choose random examples from the specified classess to fill train and eval indexes
trainParams.dtf = open(trainParams.datasetfile, 'r')
ncomb = np.int32(np.fromfile(trainParams.dtf, dtype=np.int32, count=1)[0])
combClass = np.int32(np.fromfile(trainParams.dtf, dtype=np.int32, count=ncomb))

cid = np.array(np.nonzero(np.in1d(combClass, trainParams.combSets) * 1))[0][:]
trainParams.trainIds = np.random.choice(cid, int(cid.size * 0.8))
trainParams.evalIds  = np.random.choice(np.setdiff1d(cid, trainParams.trainIds), int(cid.size * 0.2))

print ('runExperiment: found %d total combinations... %d will be used for training and %d for evaluating' % (ncomb, trainParams.trainIds.size, trainParams.evalIds.size))

pyRef_train.mmap = np.memmap(trainParams.datasetfile, dtype=np.dtype([('ins', (np.float32, (pyRef_train.model.nsigs, pyRef_train.model.nwin, pyRef_train.model.N))),
                                                                      ('lbls', np.int32), ('instcomb', 'S64'), ('typecomb', 'S64')]), mode='r', offset=ncomb.nbytes+combClass.nbytes)


# Run experiments
stats = pyRef_train.runExperiment(trainParams)
