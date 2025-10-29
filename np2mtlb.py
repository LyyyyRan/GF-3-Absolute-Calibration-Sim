import numpy as np
from numpy.fft import fft, ifft, fftshift


# (1) fft(axis=-1) default. So, fft(x, axis=0) == fft(x) in matlab; fft(x, axis=1) == fft(x.').' in mtlb
# (2) fftshift(x, axes=None) == fftshift(x) in mtlb
# (3) x.T == x.' in matlab ------ always
# (4) x.conj().T == x' in matlab ------ if iscomplexobj(x)
# (5) x.conj() == x == x in matlab ------ if not iscomplexobj(x)
# (6) x.T == x' in matlab ------ if not iscomplexobj(x)

def FFT_Range(x, shift=False):
    return fft(x, axis=1) if not shift else fftshift(fft(fftshift(x), axis=1))


def FFT_Azimuth(x, shift=False):
    return fft(x, axis=0) if not shift else fftshift(fft(fftshift(x), axis=0))


def IFFT_Range(x, shift=False):
    return ifft(x, axis=1) if not shift else fftshift(ifft(fftshift(x), axis=1))


def IFFT_Azimuth(x, shift=False):
    return ifft(x, axis=0) if not shift else fftshift(ifft(fftshift(x), axis=0))


def FFTShift(x):
    return fftshift(x, axes=None)


# mtlb.pie(x) == x' in matlab
def apostrophe(x):
    return x.conj().T


# mtlb.pointwise_apostrophe(x) == x.' in matlab
def pointwise_apostrophe(x):
    return x.T


# other functions:
def nextpow2(x):
    return int(np.ceil(np.log2(x)))

