# -*- coding: utf-8 -*-
"""
Created on Sept 12 2025

@author: https://github.com/LyyyyRan
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from np2mtlb import nextpow2, FFT_Range, FFT_Azimuth, FFTShift, apostrophe, pointwise_apostrophe, IFFT_Range, \
    IFFT_Azimuth
from utils import img2View, rad2deg, mag2db

# load Focused Image:
RCS = 90
mat_path = 'Origin_Focused_IMG_RCS_{}_{}_{}.mat'.format(RCS, RCS, RCS)[:-4]
Focused_IMG = loadmat(mat_path + '.mat')['Origin_Focused_IMG']

# show Image:
plt.figure('Focused Image')
plt.title('Focused Image')
plt.imshow(img2View(Focused_IMG, enhance=True))
plt.xlabel('Range')
plt.ylabel('Azimuth')

# Region of Interest:
CutResolution = 32  # in fact, it'll be 33 * 33
Profile_Position = [800 - 1, 1250 - 1]  # matlab from 1 on, python from 0 on
ROI = Focused_IMG[int(Profile_Position[0] - CutResolution / 2) - 1: int(Profile_Position[0] + CutResolution / 2), int(
    Profile_Position[1] - CutResolution / 2 - 1): int(Profile_Position[1] + CutResolution / 2)]

# show ROI:
plt.figure('Region of Interest')
plt.title('Region of Interest')
plt.imshow(img2View(ROI))
plt.xlabel('Range')
plt.ylabel('Azimuth')

# Get Frequency Map:
ROI_f = FFT_Azimuth(FFT_Range(ROI, shift=False), shift=False)  # Get Frequency Map

# show Map:
plt.figure('Frequency Map')
plt.imshow(img2View(ROI_f))
plt.title('Frequency Map')
plt.xlabel('Range')
plt.ylabel('Azimuth')

# Zero Padding in Frequency Domain:
# ROI_f_Upsampling = ZeroPadding(ROI_f, new_shape=(5 * CutResolution, 5 * CutResolution))
Y_Insert = np.zeros([10 * CutResolution, 33])
ROI_f = np.concatenate((ROI_f[0:17], Y_Insert, ROI_f[17:33]), axis=0)
X_Insert = np.zeros([ROI_f.shape[0], 10 * CutResolution])
ROI_f_Padded = np.concatenate((ROI_f[:, 0:22], X_Insert, ROI_f[:, 22:33]), axis=1)

# show Padded Map:
plt.figure('Padded Frequency Map')
plt.imshow(img2View(ROI_f_Padded))
plt.title('Padded Frequency Map')
plt.xlabel('Range')
plt.ylabel('Azimuth')

# Back to Range-Azimuth Space Domain:
ROI_Upsampled = IFFT_Range(IFFT_Azimuth(ROI_f_Padded, shift=False), shift=False)

# show Upsampled ROI:
plt.figure('Upsampled ROI')
plt.imshow(img2View(ROI_Upsampled))
plt.title('Upsampled ROI')
plt.xlabel('Range')
plt.ylabel('Azimuth')

# Find Max
UP_Profile_Position_Ran, UP_Profile_Position_Azi = np.where(ROI_Upsampled == ROI_Upsampled.max())
UP_Profile_Position_Ran, UP_Profile_Position_Azi = int(UP_Profile_Position_Ran[0]), int(UP_Profile_Position_Azi[0])
print('Max Value at [{}, {}]'.format(UP_Profile_Position_Ran, UP_Profile_Position_Azi))

# 姿态参数
H = 755e3  # 卫星轨道高度
phi = 20 * np.pi / 180  # 俯仰角+20°
incidence = 20.5 * np.pi / 180  # 入射角
R_eta_c = H / np.cos(incidence)  # 景中心斜距
R0 = H / np.cos(phi)
theta_r_c = np.acos(R0 / R_eta_c)  # 斜视角, 单位为 弧度, 斜视角为 4.6°

# Rotate:
Upsampled_ROI_Mod = np.abs(ROI_Upsampled)
rotate_matrix = cv.getRotationMatrix2D(center=(UP_Profile_Position_Ran, UP_Profile_Position_Azi),
                                       angle=rad2deg(theta_r_c), scale=1)
Rotated_ROI_Modulus = cv.warpAffine(Upsampled_ROI_Mod, rotate_matrix, Upsampled_ROI_Mod.shape[1::-1])

# show Rotated ROI:
plt.figure('Rotated ROI Modulus')
plt.imshow(img2View(Rotated_ROI_Modulus))
plt.title('Rotated ROI Modulus')
plt.xlabel('Range')
plt.ylabel('Azimuth')

# Find Max
UP_Profile_Position_Ran, UP_Profile_Position_Azi = np.where(Rotated_ROI_Modulus == Rotated_ROI_Modulus.max())
UP_Profile_Position_Ran, UP_Profile_Position_Azi = UP_Profile_Position_Ran[0], UP_Profile_Position_Azi[0]
print('New Max Value at [{}, {}]'.format(UP_Profile_Position_Ran, UP_Profile_Position_Azi))

# 幅度db化 + 搬移峰值至 0 dB
# 基于上面的升采样插值结果 -> 获得剖面
Abs_S5_Azi = np.abs(Rotated_ROI_Modulus[:, UP_Profile_Position_Azi])  # 方位向剖面
Abs_S5_Azi = Abs_S5_Azi / Abs_S5_Azi.max()  # 移动峰值点
Abs_S5_Ran = np.abs(Rotated_ROI_Modulus[UP_Profile_Position_Ran, :])  # 距离向剖面
Abs_S5_Ran = Abs_S5_Ran / Abs_S5_Ran.max()

########################### whether <= -13 dB ###########################
# plot Range:
plt.figure('ROI[Range]')
plt.plot(mag2db(Abs_S5_Ran))
plt.title('ROI[Range]')
plt.xlabel('Azimuth')
plt.ylabel('Modulus')

# plot -13 dB line:
tmp = np.zeros([600]) + -13
plt.plot(tmp)

# plot:
plt.figure('ROI[Azimuth]')
plt.plot(mag2db(Abs_S5_Azi))
plt.title('ROI[Azimuth]')
plt.xlabel('Range')
plt.ylabel('Modulus')

# plot -13 dB line:
tmp = np.zeros([600]) + -13
plt.plot(tmp)
########################### whether <= -13 dB ###########################

############################ Extract Energy ############################
# Get shape of ROI:
ROI_Modulus = Rotated_ROI_Modulus.copy()
Na, Nr = int(ROI_Modulus.shape[0]), int(ROI_Modulus.shape[1])

########### integration B ###########
SideLength_B = 100
NumB = 4 * (SideLength_B ** 2)

# [a, b) is used in numpy index:
ROI_B1 = ROI_Modulus[: SideLength_B, : SideLength_B]
ROI_B2 = ROI_Modulus[: SideLength_B, Nr - SideLength_B:]
ROI_B3 = ROI_Modulus[Na - SideLength_B:, : SideLength_B]
ROI_B4 = ROI_Modulus[Na - SideLength_B:, Nr - SideLength_B:]

# DN_B ** 2:
IntegrationB = (ROI_B1 ** 2).sum() + (ROI_B2 ** 2).sum() + (ROI_B3 ** 2).sum() + (ROI_B4 ** 2).sum()

########### integration A ###########
A_Mask = np.zeros_like(ROI_Modulus, dtype=int)  # [a, b)
A_Mask[:, 175: 210 + 1] = 1
A_Mask[145: 217 + 1, :] = 1
NumA = A_Mask.sum()

# Segmentation:
ROI_A = ROI_Modulus * A_Mask

# DN_A ** 2:
IntegrationA = np.sum((ROI_A ** 2))

# 卫星轨道速度Vr计算
EarthMass = 6e24  # 地球质量(kg)
EarthRadius = 6.37e6  # 地球半径6371km
Gravitational = 6.67e-11  # 万有引力常量

# 计算等效雷达速度(卫星做圆周运动的线速度)
Vr = np.sqrt(Gravitational * EarthMass / (EarthRadius + H))  # 第一宇宙速度

##信号参数设置
# 电磁波参数
c = 3e+8  # 光速
Vs = Vr  # 卫星平台速度
Vg = Vr  # 波束扫描速度
La = 15  # 方位向天线长度->椭圆的长轴
Lr = 1.5  # 距离向天线尺寸—— > 椭圆的短轴
f0 = 5.4e+9  # 雷达工作频率
lamda = c / f0  # 电磁波波长

# 距离向信号参数
Tr = 40e-6  # 发射脉冲时宽
Br = 2.8 * 6e6  # 距离向信号带宽
Kr = Br / Tr  # 距离向调频率
alpha_os_r = 1.2  # 距离过采样率
Nrg = 2500  # 距离线采样点数
Fr = alpha_os_r * Br  # 距离向采样率

# 方位向信号参数
alpha_os_a = 1.23  # 方位过采样率(高过采样率避免鬼影目标)
Naz = 1600  # 距离线数
delta_f_dop = 2 * 0.886 * Vr * (np.cos(theta_r_c)) / La  # 多普勒带宽
Fa = alpha_os_a * delta_f_dop  # 方位向采样率

delta_Azimuth = Vs / Fa  # Azimuth
delta_Range = Vr / Fr  # Range (斜距向)
print('delta_Azimuth = {}'.format(delta_Azimuth))
print('delta_Range = {}'.format(delta_Range))

# Extract Energy:
Energy = IntegrationA - (NumA * IntegrationB / NumB)
print('IntegrationA: {}'.format(IntegrationA))
print('IntegrationB: {}'.format(IntegrationB))
print('NumA: {}'.format(NumA))
print('NumB: {}'.format(NumB))

Energy = Energy * delta_Range * delta_Azimuth / np.sin(incidence)
print('本地入射角 θ:', incidence)
print('sin(θ):', np.sin(incidence))
############################ Extract Energy ############################

#################### Visualize the Energy Extraction ####################
ROI_BBox_show = Rotated_ROI_Modulus.copy()  # Copy
ROI_BBox_show = ROI_BBox_show / ROI_BBox_show.max()  # squeeze val domain
ROI_BBox_show = ROI_BBox_show * 255  # put val domain to be 0~255
ROI_BBox_show = np.uint8(ROI_BBox_show)
ROI_BBox_show = cv.cvtColor(ROI_BBox_show, cv.COLOR_GRAY2BGR)  # 1-Channel to 3-Channels

# Drew B BBoxs: (Range, Azimuth) is used in OpenCV
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, 0), (SideLength_B - 1, SideLength_B - 1), color=(0, 0, 255),
                             thickness=1)
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (Nr - SideLength_B, 0), (Nr - 1, SideLength_B - 1), color=(0, 0, 255),
                             thickness=1)
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, Na - SideLength_B), (SideLength_B - 1, Na - 1), color=(0, 0, 255),
                             thickness=1)
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (Nr - SideLength_B, Na - SideLength_B), (Nr, Na - 1), color=(0, 0, 255),
                             thickness=1)

# Drew A BBoxs: (Range, Azimuth) is used in OpenCV
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (175, 0), (210, Na - 1), color=(255, 0, 0), thickness=1)  # Azimuth longer
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, 145), (Nr - 1, 217), color=(255, 0, 0), thickness=1)  # Range longer

# Show Mask_A
plt.figure('Mask A')
plt.imshow(A_Mask, cmap='gray')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('"Mask A"')

# Show Extraction:
plt.figure('Energy Extraction')
plt.imshow(ROI_BBox_show)
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('"Energy Extraction"')
#################### Visualize the Energy Extraction ####################

######################### Absolute Calibration #########################
# the Setting RCS:
RCS = eval(mat_path.split('RCS_')[-1].split('_')[0])

# N factor of Upsample:
N_factor = ROI_Modulus.shape[0] / ROI.shape[0]
print('N_factor:', N_factor)

# The Consequent Energy:
Energy = Energy / N_factor

print('Energy = {}'.format(Energy))  # print the consequent Energy

# Calibration:
Calibration_K = Energy / RCS
Calibration_K_dB = 10 * np.log10(Calibration_K)

print('The Setting RCS σ = {}'.format(RCS))
print('Calibration Constant K:', Calibration_K)
print('Calibration Constant K/dB:', Calibration_K_dB)
######################### Absolute Calibration #########################

# Finally show all:
plt.show()

print('ROI_Modulus.max(): ', ROI_Modulus.max())

if __name__ == '__main__':
    pass
