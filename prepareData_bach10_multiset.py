
import time
import multiprocessing
import tensorflow as tf
import numpy as np
import yaml
import os
from subprocess import call

train_rate = 0.8
maxSamplesDelay = 88200 #88200_1024 44100_512 22050_256
#### n of dataset augmentation
nexpan = 1
#### ENCODE PARAMS
blocksize = 1024
maxBlockDelay = 1 + maxSamplesDelay // blocksize
#### PATHs
dataroot = '/home/pepeu/workspace/Dataset/BACH10/'
# dataroot = '/home/pepeu/DATA_DRIVE/DATASETS/MedleyDB'
audio_dir = dataroot + '/Audio/'
tfrecordfile = '/home/pepeu/workspace/Dataset/BACH10/BACH10_MC_xpan' + str(nexpan) + '_split' + str(int(train_rate * 10)) + '_blocksize' + str(blocksize) + '.tfrecord'
#### Dataset type classification


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def insert_delay_and_gather_bitratesignal(audiofile, delay, blocksize):
    path, filename = os.path.split(audiofile)
    basename = os.path.splitext(filename)[0]
    dlyfilename = basename + '_SME_blocksize' + str(blocksize) + '_dly' + str(delay)
    itsoffset = (1 / 44100) * delay

    if not os.path.isfile(path + '/' + dlyfilename + '.npy'):

        cmd = 'ffmpeg -hide_banner -nostats -loglevel 0 -y -ss ' + str(itsoffset) + '  -i ' + audiofile \
              + ' -acodec flac -frame_size ' + str(blocksize) + ' -f flac - | ffprobe - -hide_banner -loglevel 0 -show_frames > ' + path + '/' + dlyfilename + '.ana'

        os.system(cmd)

        try:
            tmpf = open(path + '/' + dlyfilename + '.ana', 'r')
            fstr = tmpf.read(-1)
            tmpf.close()
        except IOError:
            print('################# AUDIO IO ERROR on file ' + dlyfilename + '.ana' + ' ###################')
            return -1, -1

        kval = np.array([l.split('=') for l in fstr.replace('[/FRAME]', 'FRAME=0').replace('[FRAME]', 'FRAME=0').split()])

        if kval.ndim < 2:
            return -1, -1

        idx = np.nonzero(kval[:, 0] == 'pkt_size')
        bitratesignal = np.squeeze(kval[idx, 1])

        bitratesignal = np.int32(bitratesignal)[maxBlockDelay:]
        bitratesignal = np.float32((bitratesignal - np.mean(bitratesignal)) / np.std(bitratesignal))  ## standardization

        np.save(path + '/' + dlyfilename, bitratesignal)
        # bitratesignal.tofile(path + '/' + dlyfilename + '.bin')

        call(('rm -f ' + path + '/' + dlyfilename + '.ana').split())

    else:
        # print ('recovering from file ' + dlyfilename + '.bin')
        bitratesignal = np.load(path + '/' + dlyfilename + '.npy')
        # bitratesignal = np.fromfile(path + '/' + dlyfilename + '.bin', np.float32)

    return bitratesignal, delay // blocksize


def compute_vbr(params):
    audiofile = params[0]
    samples_delay = params[1]
    blocksize = params[2]

    sig, delay = insert_delay_and_gather_bitratesignal(audiofile, samples_delay, blocksize)

    return sig, delay


def create_tf_example(yml, id, istrain):
    features = {
        'comb/id': int64_feature([id]),
        'comb/genre': bytes_feature([os.fsencode(yml['genre'])]),
        'comb/istrain': int64_feature([int(istrain)]),
        'comb/inst': bytes_feature([os.fsencode(s['instrument']) for s in yml['stems'].values()]),
        'comb/type': bytes_feature([os.fsencode(s['type']) for s in yml['stems'].values()]),
        'comb/file': bytes_feature([os.fsencode(s['filename']) for s in yml['stems'].values()]),
        'comb/signal': bytes_feature([s['VBR']['signal'].tostring() for s in yml['stems'].values()]),
        'comb/labvec': bytes_feature([s['VBR']['labvec'].tostring() for s in yml['stems'].values()]),
        'comb/sample_delay': int64_feature([s['sample_delay'] for s in yml['stems'].values()]),
        'comb/vbr_delay': int64_feature([s['VBR']['vbr_delay'] for s in yml['stems'].values()]),
        'comb/vbr_ref': [],
        'comb/label': [],
    }

    for s1 in range(len(yml['stems'])):
        for s2 in range(s1 + 1, len(yml['stems'])):
            features['comb/vbr_ref'].append(yml['stems']['S%02d' % (s2 + 1)]['VBR']['vbr_delay'] - yml['stems']['S%02d' % (s1 + 1)]['VBR']['vbr_delay'])
            features['comb/label'].append(yml['stems']['S%02d' % (s2 + 1)]['VBR']['vbr_delay'] - yml['stems']['S%02d' % (s1 + 1)]['VBR']['vbr_delay'] + maxBlockDelay + 1)

    features['comb/vbr_ref'] = int64_feature(features['comb/vbr_ref'])
    features['comb/label'] = int64_feature(features['comb/label'])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example


np.random.seed(0)
options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
writer = tf.python_io.TFRecordWriter(tfrecordfile, options=options)

audiodir = os.fsencode(audio_dir)

pool = multiprocessing.Pool(processes=4)

id = 1
lost = 0

rng1 = np.random.RandomState(0)
rng2 = np.random.RandomState(1)

print('starting for bs ' + str(blocksize))

try:
    for xpan in range(nexpan):
        for dir in sorted(os.listdir(audiodir)):
            yml = {}
            yml['mix_filename'] = os.fsdecode(dir)
            yml['genre'] = os.fsdecode(dir)
            yml['stem_dir'] = os.fsdecode(audiodir + dir)

            tl = []
            [tl.append(f) if f.split(b'.')[-1] == b'wav' else 0 for f in sorted(os.listdir(audiodir + dir))]

            yml['stems'] = {}
            for s, filename in enumerate(tl):
                yml['stems']['S%02d' % (s + 1)] = {}
                yml['stems']['S%02d' % (s + 1)]['filename'] = os.fsdecode(filename)
                yml['stems']['S%02d' % (s + 1)]['instrument'] = os.fsdecode(filename.split(b'-')[-1].split(b'.')[0])
                yml['stems']['S%02d' % (s + 1)]['type'] = os.fsdecode(filename.split(b'-')[-1].split(b'.')[0])

            stems = yml['stems']
            nstems = len(stems)

            combparams = list()

            st = time.time()

            pool_params = []
            for s, stem in enumerate(stems):
                samples_delay = rng1.randint(0, maxSamplesDelay)
                stems[stem]['sample_delay'] = samples_delay

                pool_params.append([yml['stem_dir'] + '/' + stems[stem]['filename'], samples_delay, blocksize])

            sig_delay_lab = pool.map(compute_vbr, pool_params)
            for s, stem in enumerate(stems):
                stems[stem]['VBR'] = {}
                stems[stem]['VBR']['signal'] = sig_delay_lab[s][0]
                stems[stem]['VBR']['vbr_delay'] = sig_delay_lab[s][1]
                stems[stem]['VBR']['labvec'] = np.ones_like(sig_delay_lab[s][0])
                stems[stem]['type'] = stems[stem]['filename'].split('-')[-1].split('.')[0]

            yml['stems'] = stems
            istrain = rng2.randint(0, 100) < train_rate * 100
            tf_example = create_tf_example(yml, id, istrain)
            writer.write(tf_example.SerializeToString())
            id += 1

            print('################################ processed data from ' + yml['mix_filename'] + ' from xpan ' + str(xpan) + ' in ' + str(time.time() - st))

finally:
    pool.terminate()
    writer.close()
    print('*********************** Total combinations written to tfrecorf file is ' + str(id))
    print('*********************** Total combinations lost ' + str(lost))
