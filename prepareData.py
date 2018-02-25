
import time
import multiprocessing
import tensorflow as tf
import numpy as np
import yaml
import os
import scipy.io.wavfile as wf

#### n of dataset augmentation
nexpan = 60
#### ENCODE PARAMS
blocksize = 512
#### PATHs
metadata_dir = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/METADATA/'
audio_dir = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/Audio/'
tfrecordfile = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/bitrate_medleydb_blocksize' + str(blocksize) + '.tfrecord'
#### Dataset type classification
rythm = ['auxiliary percussion', 'bass drum', 'bongo', 'chimes','claps', 'cymbal', 'drum machine', 'darbuka', 'glockenspiel','doumbek', 'drum set', 'kick drum', 'shaker', 'snare drum', 'tabla', 'tambourine', 'timpani', 'toms', 'vibraphone']
eletronic = ['Main System', 'fx/processed sound', 'sampler','scratches' ]
strings = ['electric bass', 'acoustic guitar', 'banjo', 'cello', 'cello section', 'clean electric guitar', 'distorted electric guitar', 'double bass','lap steel guitar','mandolin','string section','viola','viola section','violin','violin section','yangqin', 'zhongruan']
brass = ['horn section', 'alto saxophone', 'bamboo flute', 'baritone saxophone', 'bass clarinet', 'bassoon','brass section', 'clarinet', 'clarinet section','dizi', 'flute','flute section', 'french horn', 'french horn section','oboe','oud','tenor saxophone','trombone', 'trombone section','trumpet','trumpet section' ,'tuba' ]
voice = ['female singer', 'male rapper','male singer','male speaker', 'vocalists']
melody = ['accordion','piano', 'synthesizer','tack piano','harmonica', 'melodica']
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
    dlyfilename = basename + '_dly' + str(delay)

    if not os.path.isfile(path + '/' + dlyfilename + '.ana'):

        # Open audio and insert delay
        try:
            rate, samples = wf.read(audiofile)
        except ValueError:
            print ("################# AUDIO READ ERROR ###################")
            return -1, -1

        samples = np.sum(samples, axis=1).astype(np.int16) # Convert stereo to mono
        samplesdly = np.concatenate((np.zeros(delay, np.int16), samples[:-delay]))

        # save delayed audio in wav format
        wf.write(path + '/' + dlyfilename + '.wav', rate, samplesdly)

        # Generate analyzer file with system flac
        os.system('flac -b ' + str(blocksize) + ' ' + path + '/' + dlyfilename + '.wav')
        os.system('flac -a ' + path + '/' + dlyfilename + '.flac')

        os.remove(path + '/' + dlyfilename + '.wav')
        os.remove(path + '/' + dlyfilename + '.flac')

    tmpf = open(path + '/' + dlyfilename + '.ana', 'r')
    fstr = tmpf.read(-1)

    kval = fstr.split()
    bitratesignal = np.empty(0)
    for k in kval:
        key, val = k.split('=')
        if key == 'bits':
            bitratesignal = np.append(bitratesignal, val)

    bitratesignal = np.int32(bitratesignal)[1:]
    bitratesignal = np.float32((bitratesignal - np.mean(bitratesignal)) / np.std(bitratesignal))

    return bitratesignal, delay // blocksize


def cleancomb(bitsig1, bitsig2):
    wsize = 24

    bitmat1 = np.resize(bitsig1, [np.ceil(bitsig1.size/wsize).astype(np.int32), wsize])
    bitmat2 = np.resize(bitsig2, [np.ceil(bitsig2.size/wsize).astype(np.int32), wsize])

    bitmean1 = np.mean(bitmat1, axis=1)
    bitmean2 = np.mean(bitmat2, axis=1)

    good_windows = np.int32(bitmean1 > 0) & np.int32(bitmean2 > 0)
    gwindex = np.nonzero(good_windows)

    fbitsig1 = bitmat1[gwindex, :].reshape(-1)
    fbitsig2 = bitmat2[gwindex, :].reshape(-1)

    return fbitsig1, fbitsig2

def get_class (inst1, inst2, type1, type2):

    if inst1 == inst2:
        combClass = 5
    elif type1 == type2:
            combClass = 4
    elif type1 == 'voice' and type2 == 'voice':
            combClass = 3
    elif type1 == 'voice' or type2 == 'voice':
            combClass = 2
    else:
            combClass = 1

    return combClass


def create_tf_example(st1, st2, cclass, sig1, sig2, dly1, dly2):

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'comb/class': int64_feature(cclass),
        'comb/inst1': bytes_feature(os.fsencode(st1['instrument'])),
        'comb/inst2': bytes_feature(os.fsencode(st2['instrument'])),
        'comb/type1': bytes_feature(os.fsencode(st1['type'])),
        'comb/type2': bytes_feature(os.fsencode(st2['type'])),
        'comb/sig1' : bytes_feature(sig1.tostring()),
        'comb/sig2' : bytes_feature(sig2.tostring()),
        'comb/ref'  : int64_feature(dly2 - dly1),
    }))

    return tf_example


def combFunc(params):
    yml = params[0]
    st1 = params[1]
    st2 = params[2]
    delay_samples1 = params[3]
    delay_samples2 = params[4]

    st = time.time()
    #print('############################## start ' + st1['instrument'] + ' x ' + st2['instrument'])

    st1['type'] = list(tps.keys())[np.squeeze(np.nonzero([s.count(st1['instrument']) for s in tps.values()]))]
    st2['type'] = list(tps.keys())[np.squeeze(np.nonzero([s.count(st2['instrument']) for s in tps.values()]))]

    cclass = get_class(st1['instrument'], st2['instrument'], st1['type'], st2['type'])

    fn1 = st1['filename']
    fn2 = st2['filename']
    sdir = yml['stem_dir']

    sig1, dly1 = insert_delay_and_gather_bitratesignal(audio_dir + '/' + sdir + '/' + fn1, delay_samples1, blocksize)
    sig2, dly2 = insert_delay_and_gather_bitratesignal(audio_dir + '/' + sdir + '/' + fn2, delay_samples2, blocksize)

    if dly1 == -1 or dly2 == -1:
        print('################################# skipping dir')
        return -1

    sig1, sig2 = cleancomb(sig1, sig2)

    tf_example = create_tf_example(st1, st2, cclass, sig1, sig2, dly1, dly2)

    print('############################################# finished ' + st1['instrument'] + ' x ' + st2['instrument'] + ' in ' + str(time.time() - st))

    return tf_example



np.random.seed(0)
writer = tf.python_io.TFRecordWriter(tfrecordfile)

audiodir = os.fsencode(audio_dir)
metdir = os.fsencode(metadata_dir)

pool = multiprocessing.Pool(processes=2)

try:

    for xpan in range(nexpan):
        for file in os.listdir(metdir):
            print (file)
            yml = yaml.load(open(os.path.join(metdir, file), 'r').read(-1))

            stems = yml['stems']
            nstems = len(stems)

            print ('################################ gathering data from ' + yml['mix_filename'] + ' from xpan ' + str(xpan))

            combparams = list()

            for s1 in range(nstems):
                for s2 in range(s1+1,nstems):

                    st1 = stems['S%02d' % (s1+1)]
                    st2 = stems['S%02d' % (s2+1)]

                    dly1 = np.random.randint(0, 88200)
                    dly2 = np.random.randint(0, 88200)

                    combparams.append([yml, st1, st2, dly1, dly2])

            tf_examples = pool.map(combFunc, combparams)

            print('############################ writing tf examples')
            for tf_example in tf_examples:
                if tf_example != -1:
                    writer.write(tf_example.SerializeToString())

finally:
    print ('ENDING: comb ' + st1['instrument'] + ' x ' + st2['instrument'])