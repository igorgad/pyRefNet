
import time
import multiprocessing
import tensorflow as tf
import numpy as np
import yaml
import os
from subprocess import call

train_rate = 0.8
maxSamplesDelay = 44100 #88200_1024 44100_512 22050_256
#### n of dataset augmentation
nexpan = 10
#### ENCODE PARAMS
blocksize = 512
maxBlockDelay = 1 + maxSamplesDelay // blocksize
#### PATHs
dataroot = '/home/pepeu/workspace/Dataset/'
# dataroot = '/home/pepeu/DATA_DRIVE/DATASETS/MedleyDB'
active_dir = dataroot + '/ACTIVATION_CONF'
metadata_dir = dataroot + '/METADATA/'
audio_dir = dataroot + '/Audio/'
tfrecordfile = '/home/pepeu/workspace/Dataset/SME_bitrate_medleydb_xpan' + str(nexpan) + '_split' + str(int(train_rate * 10)) + '_blocksize' + str(blocksize) + '.tfrecord'
#### Dataset type classification
rythm = ['gong', 'auxiliary percussion', 'bass drum', 'bongo', 'chimes', 'claps', 'cymbal', 'drum machine', 'darbuka', 'glockenspiel', 'doumbek', 'drum set', 'kick drum', 'shaker', 'snare drum',
         'tabla', 'tambourine', 'timpani', 'toms', 'vibraphone']
eletronic = ['Main System', 'fx/processed sound', 'sampler', 'scratches']
strings = ['gu', 'zhongruan', 'liuqin', 'guzheng', 'erhu', 'harp', 'electric bass', 'acoustic guitar', 'banjo', 'cello', 'cello section', 'clean electric guitar', 'distorted electric guitar',
           'double bass', 'lap steel guitar', 'mandolin', 'string section', 'viola', 'viola section', 'violin', 'violin section', 'yangqin', 'zhongruan']
brass = ['piccolo', 'soprano saxophone', 'horn section', 'alto saxophone', 'bamboo flute', 'baritone saxophone', 'bass clarinet', 'bassoon', 'brass section', 'clarinet', 'clarinet section', 'dizi',
         'flute', 'flute section', 'french horn', 'french horn section', 'oboe', 'oud', 'tenor saxophone', 'trombone', 'trombone section', 'trumpet', 'trumpet section', 'tuba']
voice = ['female singer', 'male rapper', 'male singer', 'male speaker', 'vocalists']
melody = ['electric piano', 'accordion', 'piano', 'synthesizer', 'tack piano', 'harmonica', 'melodica']
tps = {'rythm': rythm, 'electronic': eletronic, 'strings': strings, 'brass': brass, 'voice': voice, 'melody': melody}


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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


def resample_labvec(reftime, labvec, dly):
    dtime = np.diff(reftime)
    labt = np.concatenate((np.zeros(dly, np.float32), np.hstack([np.ones(int(dtime[i] / (1 / 44100)), np.float32) * labvec[i] for i in range(dtime.size)])), axis=0)
    labmat = np.resize(labt, [np.ceil(labt.size / blocksize).astype(np.int32), blocksize])
    labmean = np.mean(labmat, axis=1)[maxBlockDelay:]
    return labmean


def compute_vbr(params):
    audiofile = params[0]
    samples_delay = params[1]
    blocksize = params[2]
    labvec = params[3]
    reftime = params[4]

    sig, delay = insert_delay_and_gather_bitratesignal(audiofile, samples_delay, blocksize)
    labm = resample_labvec(reftime, labvec, samples_delay)

    return sig, delay, labm


def get_class(inst1, inst2, type1, type2):
    if inst1 == inst2:
        combClass = 5
    elif type1 == type2:
        combClass = 4
    elif type1 != 'voice' and type2 != 'voice':
        combClass = 3
    elif type1 == 'voice' or type2 == 'voice':
        combClass = 2
    else:
        combClass = 1

    return combClass


def create_tf_example(yml, st1, st2, id, cclass, istrain):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'comb/id': int64_feature(id),
        'comb/class': int64_feature(cclass),
        'comb/genre': bytes_feature(os.fsencode(yml['genre'])),
        'comb/inst1': bytes_feature(os.fsencode(st1['instrument'])),
        'comb/inst2': bytes_feature(os.fsencode(st2['instrument'])),
        'comb/type1': bytes_feature(os.fsencode(st1['type'])),
        'comb/type2': bytes_feature(os.fsencode(st2['type'])),
        'comb/file1': bytes_feature(os.fsencode(yml['stem_dir'] + '/' + st1['filename'])),
        'comb/file2': bytes_feature(os.fsencode(yml['stem_dir'] + '/' + st2['filename'])),
        'comb/sig1': bytes_feature(st1['VBR']['signal'].tostring()),
        'comb/sig2': bytes_feature(st2['VBR']['signal'].tostring()),
        'comb/lab1': bytes_feature(st1['VBR']['labvec'].tostring()),
        'comb/lab2': bytes_feature(st2['VBR']['labvec'].tostring()),
        'comb/sig1_sample_delay': int64_feature(st1['sample_delay']),
        'comb/sig2_sample_delay': int64_feature(st2['sample_delay']),
        'comb/ref': int64_feature(st2['VBR']['vbr_delay'] - st1['VBR']['vbr_delay']),
        'comb/label': int64_feature(st2['VBR']['vbr_delay'] - st1['VBR']['vbr_delay'] + maxBlockDelay + 1),
        'comb/istrain': int64_feature(int(istrain))
    }))

    return tf_example


np.random.seed(0)
options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
writer = tf.python_io.TFRecordWriter(tfrecordfile, options=options)

audiodir = os.fsencode(audio_dir)
metdir = os.fsencode(metadata_dir)
activedir = os.fsencode(active_dir)

pool = multiprocessing.Pool(processes=12)

id = 1
lost = 0

rng1 = np.random.RandomState(0)
rng2 = np.random.RandomState(1)

print('starting for bs ' + str(blocksize))

try:

    for xpan in range(nexpan):
        for file in sorted(os.listdir(metdir)):
            yml = yaml.load(open(os.path.join(metdir, file), 'r').read(-1))

            lab = open(os.path.join(activedir, file.split(b'_METADATA.yaml')[0] + b'_ACTIVATION_CONF.lab'), 'r').read(-1)
            labmat = np.stack([np.fromstring(lb, dtype=np.float32, sep=',') for lb in lab.split('\n')[1:-1]])

            sdir = yml['stem_dir']
            stems = yml['stems']
            nstems = len(stems)

            combparams = list()

            st = time.time()

            pool_params = []
            for s, stem in enumerate(stems):
                samples_delay = rng1.randint(0, maxSamplesDelay)
                stems[stem]['sample_delay'] = samples_delay
                reftime = labmat[:, 0]
                if s + 1 > labmat.shape[1] - 1:
                    labvec = np.ones(labmat.shape[0], np.float32)
                else:
                    labvec = labmat[:, s + 1]

                pool_params.append([audio_dir + '/' + yml['stem_dir'] + '/' + stems[stem]['filename'], samples_delay, blocksize, labvec, reftime])

            sig_delay_lab = pool.map(compute_vbr, pool_params)
            for s, stem in enumerate(stems):
                stems[stem]['VBR'] = {}
                stems[stem]['VBR']['signal'] = sig_delay_lab[s][0]
                stems[stem]['VBR']['vbr_delay'] = sig_delay_lab[s][1]
                stems[stem]['VBR']['labvec'] = sig_delay_lab[s][2]
                stems[stem]['type'] = list(tps.keys())[np.nonzero([s.count(stems[stem]['instrument']) for s in tps.values()])[0][0]]

            for s1 in range(nstems):
                for s2 in range(s1 + 1, nstems):
                    istrain = rng2.randint(0, 100) < train_rate * 100

                    st1 = stems['S%02d' % (s1 + 1)]
                    st2 = stems['S%02d' % (s2 + 1)]

                    if type(st1['VBR']['signal']) == int or type(st2['VBR']['signal']) == int:
                        continue

                    cclass = get_class(st1['instrument'], st2['instrument'], st1['type'], st2['type'])

                    tf_example = create_tf_example(yml, st1, st2, id, cclass, istrain)
                    writer.write(tf_example.SerializeToString())

                    id += 1

            print('################################ processed data from ' + yml['mix_filename'] + ' from xpan ' + str(xpan) + ' in ' + str(time.time() - st))

finally:
    pool.terminate()
    writer.close()
    print('*********************** Total combinations written to tfrecorf file is ' + str(id))
    print('*********************** Total combinations lost ' + str(lost))
