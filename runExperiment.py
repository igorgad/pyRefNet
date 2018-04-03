
import os
import numpy as np

import pyRef_train

import importlib
importlib.reload(pyRef_train)

class trainParams:
    pass

# Fill pyRef_train.trainParams class with training parameters.
trainParams.num_epochs   = 200
trainParams.selected_class    = [3, 4, 5]

trainParams.datasetfile  = '/home/pepeu/workspace/DOC/Dataset/bitrate_medleydb_blocksize1152.tfrecord'
trainParams.log_root     = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/tensorlogs/'

trainParams.runName      = "datasetapi-{}_N{}_NW{}".format(pyRef_train.model.name, pyRef_train.model.N, pyRef_train.model.nwin)
trainParams.n = sum(1 for f in os.listdir(trainParams.log_root) if os.path.isdir(os.path.join(trainParams.log_root, f)))
trainParams.log_dir = "{}{}_run_{}".format(trainParams.log_root, trainParams.runName, trainParams.n+1)
trainParams.sum_interval = 400
trainParams.blocksize = trainParams.datasetfile.split('blocksize')[1].split('.')[0]

trainParams.hptext = {key:value for key, value in trainParams.__dict__.items() if not key.startswith('__') and not callable(key)}
print ('logdir ' + trainParams.log_dir)

np.random.seed(0)
trainParams.ncombs = 192401
trainParams.eval_ids = np.random.randint(0, trainParams.ncombs, [np.int32(np.floor(trainParams.ncombs * 0.22))])
trainParams.train_ids = np.setdiff1d(np.array(range(0,trainParams.ncombs)), trainParams.eval_ids)
trainParams.encode_blocksize = int(trainParams.datasetfile.split('blocksize')[1].split('.')[0])

# Run experiments
stats = pyRef_train.runExperiment(trainParams)
