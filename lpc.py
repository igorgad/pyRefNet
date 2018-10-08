
import numpy as np
import wave
import math
from audiolazy.lazy_lpc import lpc

import matplotlib.pyplot as plt

def golomb_cod(x,m):
    c = int(math.ceil(math.log(m,2)))
    remin = x % m
    quo =int(math.floor(x / m))
    #print "quo is",quo
    #print "reminder",remin
    #print "c",c
    div = int(math.pow(2,c) - m)
    #print "div",div
    first = ""
    for i in range(quo):
        first = first + "1"
    #print first

    if (remin < div):
        b = c - 1
        a = "{0:0" + str(b) + "b}"
        #print "1",a.format(remin)
        bi = a.format(remin)
    else:
        b = c
        a = "{0:0" + str(b) + "b}"
        #print "2",a.format(remin+div)
        bi = a.format(remin+div)

    final = first + "0" +str(bi)
    #print "final",final
    return final


filename = '/home/pepeu/DATA_DRIVE/DATASETS/MedleyDB/Audio/AClassicEducation_NightOwl_STEMS/AClassicEducation_NightOwl_STEM_02.wav'

# Read from file.
spf = wave.open(filename, 'r')

# Get file as numpy array.
x = spf.readframes(-1)
x = np.fromstring(x, 'Int16')

wi = 10.9
bsize = 512
wb = np.int(np.floor(wi * bsize))
we = np.int(np.floor(wb + bsize))

xw = x[wb:we]
# xw = (x[wb:we] - np.mean(x[wb:we])) / np.std(x[wb:we])
ncoef = 8
afilt = lpc(xw, ncoef)
residual = list(afilt(xw))

sfilt = 1 / afilt
yw = list(sfilt(residual))

plt.figure(figsize=(16,4))
plt.plot(xw[4:], label='Reference')
plt.plot(xw[4:] - np.array(residual)[4:], label='Linear Approximation')
plt.ylabel('Amplitude')
plt.xlabel('Audio Samples')
plt.legend()
# plt.title('ncoeff ' + str(ncoef))

plt.figure(figsize=(16,4))
plt.plot(np.array(residual)[4:], label='Residual')
plt.ylabel('Amplitude')
plt.xlabel('Audio Samples')
# plt.legend()

m = 16
wsize = 8
wcodesize = []

xmat = np.resize(np.array(residual).astype(np.int32), [-1, wsize])
for w in range(xmat.shape[0]):
    bitcount = 0
    for s in range(xmat.shape[1]):
        bitcount += len(golomb_cod(xmat[w,s],m))

    wcodesize.append(bitcount)

bitaxis = []
for e in wcodesize:
    bitaxis.append(e*np.ones(wsize))

bitaxis = np.array(bitaxis).reshape(-1)

plt.plot(bitaxis, label='Bit Rate')
plt.legend()