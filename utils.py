import numpy as np


# complex img -> Modulus && Enhanced:
def img2View(img, enhance=False, Gama=1):
    if enhance:
        return Gama * np.log(1 + np.abs(img.copy()))
    else:
        return np.abs(img.copy())


def ZeroPadding(signal, new_shape=None):
    Na, Nr = signal.shape

    if new_shape is None:
        new_Na, new_Nr = int(2 ** np.ceil(np.log2(Na))), int(2 ** np.ceil(np.log2(Nr)))
    else:
        new_Na, new_Nr = new_shape

    new_signal = np.pad(signal, (new_Na, new_Nr))
    return new_signal


def UpSampling(signal):
    Na, Nr = signal.shape
    new_Na, new_Nr = int(2 ** np.ceil(np.log2(Na))), int(2 ** np.ceil(np.log2(Nr)))
    new_signal = np.kron(signal, np.ones((new_Na, new_Nr)))
    return new_signal

def rad2deg(rad):
    return rad * 180 / np.pi

def mag2db(x):
    return 20 * np.log10(x)
