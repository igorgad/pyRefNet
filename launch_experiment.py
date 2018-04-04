
import os
import argparse
import numpy as np
import pyRef_train
import importlib
importlib.reload(pyRef_train)


# Default Parameters of Argparse
num_steps   = 100000
selected_class    = [3, 4, 5]
dataset_file  = '/home/pepeu/workspace/DOC/Dataset/bitrate_medleydb_blocksize1152.tfrecord'
log_dir     = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/tensorlogs/'

run_name      = "sr-datasetapi-{}_N{}_NW{}".format(pyRef_train.model.name, pyRef_train.model.N, pyRef_train.model.nwin)
sum_interval = 800

# Parse arguments
parser = argparse.ArgumentParser(description='Launch training session of pyrefnet.')

parser.add_argument('restore_from_dir', nargs='*', type=str, default=[], help='Specify logging dir to restore training - optional')
parser.add_argument('--num_steps', type=int, default=num_steps, help='Number of steps to run experiment (default: %s)' % str(num_steps))
parser.add_argument('--dataset_file', type=str, default=dataset_file, help='Dataset file in tfrecord format (default: %s)' % str(dataset_file))
parser.add_argument('--log_dir', type=str, default=log_dir, help='The directory to store the experiments logs (default: %s)' % str(log_dir))
parser.add_argument('--run_name', type=str, default=run_name, help='Specify a run name to use in log directory (default: %s)' % str(run_name))
parser.add_argument('--summary_interval', type=int, default=sum_interval, help='Interval in steps to log results (default: %s)' % str(sum_interval))

# Fill trainParams
trainParams = parser.parse_args()

trainParams.n = sum(1 for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f)))
trainParams.log_path_dir = "{}{}_run_{}".format(log_dir, trainParams.run_name, trainParams.n+1)
trainParams.hptext = {key:value for key, value in trainParams.__dict__.items() if not key.startswith('__') and not callable(key)}

np.random.seed(0)
trainParams.ncombs = 192401
trainParams.eval_ids = np.random.randint(0, trainParams.ncombs, [np.int32(np.floor(trainParams.ncombs * 0.22))])
trainParams.train_ids = np.setdiff1d(np.array(range(0,trainParams.ncombs)), trainParams.eval_ids)
trainParams.encode_blocksize = int(trainParams.dataset_file.split('blocksize')[1].split('.')[0])

# Run experiments
stats = pyRef_train.start_training(trainParams)
