import numpy as np

def Calibration(RealRCS, Energy):
    K = 0.488772919
    RCS = Energy / K
    RCS_dB = 10 * np.log10(RCS)
    RealRCS_dB = 10 * np.log10(RealRCS)
    print('delta RCS(dB): ', RealRCS_dB - RCS_dB)

if __name__ == '__main__':
    Calibration(RealRCS=3000, Energy=1435.463)
