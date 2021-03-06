
import os
import argparse
import numpy as np
import pyRef_train
import importlib
import matplotlib.pyplot as plt
importlib.reload(pyRef_train)


# Default Parameters of Argparse
num_steps   = 1000000
selected_class    = [3, 4, 5]
dataset_file  = '/home/pepeu/workspace/Dataset/BACH10/SME_bitrate_BACH10_xpan100_split8_blocksize1024.tfrecord'
log_dir     = '/home/pepeu/DATA_DRIVE/DATASETS/Bach10/tensorlogs/'
train_test_rate = 0.7

run_name      = "BACH10_{}_N{}_NW{}_bs1024".format(pyRef_train.model.name, pyRef_train.model.N, pyRef_train.model.nwin)
sum_interval = 40

# Parse arguments
parser = argparse.ArgumentParser(description='Launch training session of pyrefnet.')

parser.add_argument('restore_from_dir', nargs='*', type=str, default=[], help='Specify logging dir to restore training - optional')
parser.add_argument('--num_steps', type=int, default=num_steps, help='Number of steps to run experiment (default: %s)' % str(num_steps))
parser.add_argument('--dataset_file', type=str, default=dataset_file, help='Dataset file in tfrecord format (default: %s)' % str(dataset_file))
parser.add_argument('--log_dir', type=str, default=log_dir, help='The directory to store the experiments logs (default: %s)' % str(log_dir))
parser.add_argument('--run_name', type=str, default=run_name, help='Specify a run name to use in log directory (default: %s)' % str(run_name))
parser.add_argument('--summary_interval', type=int, default=sum_interval, help='Interval in steps to log results (default: %s)' % str(sum_interval))
parser.add_argument('--train_test_rate', type=int, default=train_test_rate, help='Interval in steps to log results (default: %s)' % str(train_test_rate))
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--include_trace', dest='trace', default=False, action='store_true')

# Fill trainParams
trainParams = parser.parse_args()

n = sum(1 for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f)))
trainParams.run_name = "{}_run_{}".format(trainParams.run_name, n+1)
trainParams.log_path_dir = log_dir + trainParams.run_name
trainParams.hptext = {key:value for key, value in trainParams.__dict__.items() if not key.startswith('__') and not callable(key)}

# Run experiments
stats = pyRef_train.start_training(trainParams)
