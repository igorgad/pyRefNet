
import time
import multiprocessing
import tensorflow as tf
import numpy as np
import yaml
import os
from subprocess import call
import scipy.io.wavfile as wf

maxSamplesDelay = 39424 #88200_1152 19712_256
#### n of dataset augmentation
nexpan = 50
#### ENCODE PARAMS
blocksize = 512
maxBlockDelay = 1 + maxSamplesDelay // blocksize
#### PATHs
dataroot = '/home/pepeu/workspace/Dataset/'
# dataroot = '/home/pepeu/DATA_DRIVE/DATASETS/MedleyDB'
active_dir = dataroot + '/ACTIVATION_CONF'
metadata_dir = dataroot + '/METADATA/'
audio_dir = dataroot + '/Audio/'
tfrecordfile = '/home/pepeu/workspace/Dataset/stereo_wgenre_bitrate_medleydb_blocksize' + str(blocksize) + '.tfrecord'
#### Dataset type classification
rythm = ['gong', 'auxiliary percussion', 'bass drum', 'bongo', 'chimes','claps', 'cymbal', 'drum machine', 'darbuka', 'glockenspiel','doumbek', 'drum set', 'kick drum', 'shaker', 'snare drum', 'tabla', 'tambourine', 'timpani', 'toms', 'vibraphone']
eletronic = ['Main System', 'fx/processed sound', 'sampler','scratches' ]
strings = ['gu', 'zhongruan', 'liuqin', 'guzheng', 'erhu', 'harp', 'electric bass', 'acoustic guitar', 'banjo', 'cello', 'cello section', 'clean electric guitar', 'distorted electric guitar', 'double bass','lap steel guitar','mandolin','string section','viola','viola section','violin','violin section','yangqin', 'zhongruan']
brass = ['piccolo', 'soprano saxophone', 'horn section', 'alto saxophone', 'bamboo flute', 'baritone saxophone', 'bass clarinet', 'bassoon','brass section', 'clarinet', 'clarinet section','dizi', 'flute','flute section', 'french horn', 'french horn section','oboe','oud','tenor saxophone','trombone', 'trombone section','trumpet','trumpet section' ,'tuba' ]
voice = ['female singer', 'male rapper','male singer','male speaker', 'vocalists']
melody = ['electric piano', 'accordion','piano', 'synthesizer','tack piano','harmonica', 'melodica']
tps = {'rythm': rythm, 'electronic': eletronic, 'strings': strings, 'brass': brass, 'voice': voice, 'melody': melody}


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def insert_delay_and_gather_bitratesignal (audiofile, delay, blocksize):
    path, filename = os.path.split(audiofile)
    basename = os.path.splitext(filename)[0]
    dlyfilename = basename + '_blocksize' + str(blocksize) + '_dly' + str(delay)

    if not os.path.isfile(path + '/' + dlyfilename + '.bin'):

        # Open audio and insert delay
        try:
            rate, samples = wf.read(audiofile)
        except ValueError:
            print('################# AUDIO READ ERROR on file ' + audiofile + ' ###################')
            print('################# recovering... ###################')
            call(('ffmpeg -nostats -loglevel quiet -y -i ' + audiofile + ' ' + path + '/recovered_' + filename).split())

            try:
                rate, samples = wf.read(path + '/recovered_' + filename)
            except ValueError:
                print('################# not recovered ###################')
                return -1, -1

        # samples = np.sum(samples, axis=1).astype(np.int16) # Convert stereo to mono
        samplesdly = np.concatenate((np.zeros((delay,2), np.int16), samples), axis=0)

        # save delayed audio in wav format
        wf.write(path + '/' + dlyfilename + '.wav', rate, samplesdly)

        # Generate analyzer file with system flac
        call(('flac --totally-silent -f -b ' + str(blocksize) + ' ' + path + '/' + dlyfilename + '.wav').split())
        call(('flac --totally-silent -a ' + path + '/' + dlyfilename + '.flac').split())

        try:
            tmpf = open(path + '/' + dlyfilename + '.ana', 'r')
            fstr = tmpf.read(-1)
            tmpf.close()
        except IOError:
            print('################# AUDIO IO ERROR on file ' + dlyfilename + '.ana' + ' ###################')
            return -1, -1

        kval = np.array([k.split('=') for k in fstr.split()])

        if kval.ndim < 2:
            return -1, -1

        idx = np.nonzero(kval[:,0] == 'bits')
        bitratesignal = np.squeeze(kval[idx, 1])

        bitratesignal = np.int32(bitratesignal)[maxBlockDelay:]
        bitratesignal = np.float32((bitratesignal - np.mean(bitratesignal)) / np.std(bitratesignal)) ## standardization

        bitratesignal.tofile(path + '/' + dlyfilename + '.bin')

        call(('rm -f ' + path + '/' + dlyfilename + '.wav').split())
        call(('rm -f ' + path + '/' + dlyfilename + '.flac').split())
        call(('rm -f ' + path + '/' + dlyfilename + '.ana').split())

    else:
        # print ('recovering from file ' + dlyfilename + '.bin')
        bitratesignal = np.fromfile(path + '/' + dlyfilename + '.bin')

    return bitratesignal, delay // blocksize


def resample_labvecs(reftime, labvec1, labvec2, dlys1, dlys2):
    dtime = np.diff(reftime)

    labt1 = np.concatenate((np.zeros(dlys1, np.float32), np.hstack([np.ones(int(dtime[i] / (1 / 44100)), np.float32) * labvec1[i] for i in range(dtime.size)])), axis=0)
    labt2 = np.concatenate((np.zeros(dlys2, np.float32), np.hstack([np.ones(int(dtime[i] / (1 / 44100)), np.float32) * labvec2[i] for i in range(dtime.size)])), axis=0)

    labmat1 = np.resize(labt1, [np.ceil(labt1.size/blocksize).astype(np.int32), blocksize])
    labmat2 = np.resize(labt2, [np.ceil(labt2.size/blocksize).astype(np.int32), blocksize])

    labmean1 = np.mean(labmat1, axis=1)[maxBlockDelay:]
    labmean2 = np.mean(labmat2, axis=1)[maxBlockDelay:]

    return labmean1, labmean2


def get_class (inst1, inst2, type1, type2):

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


def create_tf_example(yml, st1, st2, id, cclass, sig1, sig2, dly1, dly2, labm1, labm2):

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'comb/id'   : int64_feature(id),
        'comb/class': int64_feature(cclass),
        'comb/genre': bytes_feature(os.fsencode(yml['genre'])),
        'comb/inst1': bytes_feature(os.fsencode(st1['instrument'])),
        'comb/inst2': bytes_feature(os.fsencode(st2['instrument'])),
        'comb/type1': bytes_feature(os.fsencode(st1['type'])),
        'comb/type2': bytes_feature(os.fsencode(st2['type'])),
        'comb/sig1' : bytes_feature(sig1.tostring()),
        'comb/sig2' : bytes_feature(sig2.tostring()),
        'comb/lab1' : bytes_feature(labm1.tostring()),
        'comb/lab2' : bytes_feature(labm2.tostring()),
        'comb/ref'  : int64_feature(dly2 - dly1),
        'comb/label': int64_feature(dly2 - dly1 + maxBlockDelay + 1),
    }))

    return tf_example


def combFunc(params):
    yml = params[0]
    st1 = params[1]
    st2 = params[2]
    delay_samples1 = params[3]
    delay_samples2 = params[4]
    id = params[5]
    labvec1 = params[6]
    labvec2 = params[7]
    reftime = params[8]

    st1['type'] = list(tps.keys())[np.nonzero([s.count(st1['instrument']) for s in tps.values()])[0][0]]
    st2['type'] = list(tps.keys())[np.nonzero([s.count(st2['instrument']) for s in tps.values()])[0][0]]

    cclass = get_class(st1['instrument'], st2['instrument'], st1['type'], st2['type'])

    fn1 = st1['filename']
    fn2 = st2['filename']
    sdir = yml['stem_dir']

    sig1, dly1 = insert_delay_and_gather_bitratesignal(audio_dir + '/' + sdir + '/' + fn1, delay_samples1, blocksize)
    sig2, dly2 = insert_delay_and_gather_bitratesignal(audio_dir + '/' + sdir + '/' + fn2, delay_samples2, blocksize)

    if dly1 == -1 or dly2 == -1:
        print('################################# skipping comb' + st1['instrument'] + ' x ' + st2['instrument'])
        return -1

    labm1, labm2 = resample_labvecs(reftime, labvec1, labvec2, delay_samples1, delay_samples2)

    tf_example = create_tf_example(yml, st1, st2, id, cclass, sig1, sig2, dly1, dly2, labm1, labm2)

    return tf_example


np.random.seed(0)
writer = tf.python_io.TFRecordWriter(tfrecordfile)

audiodir = os.fsencode(audio_dir)
metdir = os.fsencode(metadata_dir)
activedir = os.fsencode(active_dir)

pool = multiprocessing.Pool(processes=4)

id = 1
lost = 0

try:

    for xpan in range(nexpan):
        for file in sorted(os.listdir(metdir)):
            yml = yaml.load(open(os.path.join(metdir, file), 'r').read(-1))


            lab = open(os.path.join(activedir, file.split(b'_METADATA.yaml')[0] + b'_ACTIVATION_CONF.lab'), 'r').read(-1)
            labmat = np.stack([np.fromstring(lb, dtype=np.float32, sep=',') for lb in lab.split('\n')[1:-1]])

            stems = yml['stems']
            nstems = len(stems)

            combparams = list()

            st = time.time()

            for s1 in range(nstems):
                for s2 in range(s1+1,nstems):

                    st1 = stems['S%02d' % (s1+1)]
                    st2 = stems['S%02d' % (s2+1)]

                    dly1 = np.random.randint(0, maxSamplesDelay)
                    dly2 = np.random.randint(0, maxSamplesDelay)

                    labvec1 = labmat[:,s1 + 1]
                    reftime = labmat[:,0]

                    if s2 + 1 > labmat.shape[1] - 1:
                        labvec2 = np.ones(labmat.shape[0], np.float32)
                    else:
                        labvec2 = labmat[:,s2 + 1]

                    combparams.append([yml, st1, st2, dly1, dly2, id, labvec1, labvec2, reftime])

                    id += 1

            tf_examples = pool.map(combFunc, combparams)

            for tf_example in tf_examples:
                if tf_example != -1:
                    writer.write(tf_example.SerializeToString())
                else:
                    lost += 1

            print('################################ processed data from ' + yml['mix_filename'] + ' from xpan ' + str(xpan) + ' in ' + str(time.time() - st))

finally:
    pool.terminate()
    writer.close()
    print ('*********************** Total combinations written to tfrecorf file is ' + str(id))
    print ('*********************** Total combinations lost ' + str(lost))
