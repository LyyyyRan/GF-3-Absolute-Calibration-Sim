import sys
import cv2 as cv
import numpy as np
from numpy.ma.core import angle
from scipy.io import loadmat

from np2mtlb import nextpow2, FFT_Range, FFT_Azimuth, FFTShift, apostrophe, pointwise_apostrophe, IFFT_Range, \
    IFFT_Azimuth
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


# complex img -> Modulus && Enhanced:
def img2View(img, Gama=1):
    return Gama * np.log(1 + np.abs(img.copy()))


# azimuth idx to view:
view_azimuth = 800

H = 755e3  # 卫星轨道高度

# 卫星轨道速度Vr计算
EarthMass = 6e24  # 地球质量(kg)
EarthRadius = 6.37e6  # 地球半径6371km
Gravitational = 6.67e-11  # 万有引力常量

# 姿态参数
phi = 20 * np.pi / 180  # 俯仰角+20°
incidence = 20.5 * np.pi / 180  # 入射角

# 计算等效雷达速度(卫星做圆周运动的线速度)
Vr = np.sqrt(Gravitational * EarthMass / (EarthRadius + H))  # 第一宇宙速度

# 景中心斜距R_eta_c和最近斜距R0，斜视角theta_rc由轨道高度、俯仰角、入射角计算得出
R_eta_c = H / np.cos(incidence)  # 景中心斜距
R0 = H / np.cos(phi)
theta_r_c = np.acos(R0 / R_eta_c)  # 斜视角, 单位为 弧度, 斜视角为 4.6°

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
alpha_os_a = 1.7  # 方位过采样率(高过采样率避免鬼影目标)
Naz = 1600  # 距离线数
delta_f_dop = 2 * 0.886 * Vr * (np.cos(theta_r_c)) / La  # 多普勒带宽
Fa = alpha_os_a * delta_f_dop  # 方位向采样率
Ta = 0.886 * lamda * R_eta_c / (La * Vg * np.cos(theta_r_c))  # 目标照射时间

# 景中心点(原点)的参数
time_eta_c = -R_eta_c * np.sin(theta_r_c) / Vr  # 波束中心穿越时刻
f_eta_c = 2 * Vr * np.sin(theta_r_c) / lamda  # 多普勒中心频率

# 合成孔径参数
rho_r = c / (2 * Fr)  # 距离向分辨率
rho_a = Vr / Fa  # 距离向分辨率 | La / 2
theta_bw = 0.886 * lamda / Lr  # 方位向3dB波束宽度
theta_syn = Vs / Vg * theta_bw  # 合成角宽度(斜面上的合成角)
Ls = R_eta_c * theta_syn  # 合成孔径长度

##时间轴参数
Trg = Nrg / Fr
Taz = Naz / Fa

# 距离向 / 方位向采样时间间隔
Gap_t_tau = 1 / Fr
Gap_t_eta = 1 / Fa

# 距离向 / 方位向采样频率间隔
Gap_f_tau = Fr / Nrg
Gap_f_eta = Fa / Naz

# 时间轴变量
time_tau_r = 2 * R_eta_c / c + np.arange(-Trg / 2, Gap_t_tau, Trg / 2 - Gap_t_tau)  # 距离时间变量
time_eta_a = time_eta_c + np.arange(-Taz / 2, Gap_t_eta, Taz / 2 - Gap_t_eta)  # 方位时间变量 η(eta)
time_tau_r = np.expand_dims(time_tau_r, axis=0)  # to vector
time_eta_a = np.expand_dims(time_eta_a, axis=0)  # to vector

# 随着距离向时间变化的最近斜距, c / 2 是因为距离向上一个时间包含两次电磁波来回
R0_tau_r = (time_tau_r * c / 2) * np.cos(theta_r_c)
Ext_R0_tau_r = loadmat('Ext_R0_tau_r')['Ext_R0_tau_r']

# 频率变量
f_tau = np.arange(-Fr / 2, Gap_f_tau, Fr / 2 - Gap_f_tau)  # 距离频率变量
f_tau = np.expand_dims(f_tau, axis=0)  # to vector
f_tau = f_tau - (np.around(f_tau / Fr)) / Fr  # 混叠方程
f_eta = f_eta_c + np.arange(-Fa / 2, Gap_f_eta, Fa / 2 - Gap_f_eta)  # 方位频率变量
f_eta = f_eta - (np.around(f_eta / Fa)) / Fa

# 时间轴
# [Ext_time_tau_r, Ext_time_eta_a] = meshgrid(time_tau_r, time_eta_a)  # 设置距离时域-方位时域二维网络坐标
Ext_time_tau_r = loadmat('Ext_time_tau_r')['Ext_time_tau_r']
Ext_time_eta_a = loadmat('Ext_time_eta_a')['Ext_time_eta_a']

# 频率轴
# [Ext_f_tau, Ext_f_eta] = meshgrid(f_tau, f_eta)  # 设置频率时域-方位频域二维网络坐标
Ext_f_tau = loadmat('Ext_f_tau')['Ext_f_tau']
Ext_f_eta = loadmat('Ext_f_eta')['Ext_f_eta']

## 点目标(三个)坐标设置
#  设置目标点相对于景中心之间的距离

xA = 0
yA = 0  # A=(0,0)
xB = xA + 500
yB = xA + 500  # （500,500）

xC = H * np.tan(phi + theta_bw / 2) - R0 * np.sin(phi)  # 计算的C点距离向坐标
yC = xA + 500
Position_x_r = np.array([xA, xB, xC])
Position_y_a = np.array([yA, yB, yC])  # 点目标的坐标矩阵表示

# sigma = 25.136:
sigma = eval(sys.argv[1]) if len(sys.argv) > 1 else 0

Positions = np.array([[xA, yA, sigma],  # 点目标位置，这里设置了5个点目标，构成一个矩形以及矩形的中心
                      [xB, yB, sigma],
                      [xC, yC, sigma]])

Target_num = 3
S_echo = np.random.normal(0, 1, size=[Naz, Nrg]) + 1j * np.random.normal(0, 1, size=[Naz, Nrg])  # 生成白噪声矩阵存储回波信号

sn = time_eta_a

################################################################################################################################

# load Echo of GF-3 from matlab:
mat_path = 'GF3_Echo_90.mat'
OriginData_dict = loadmat(mat_path)

# load keys of dict:
dictkeys = list(OriginData_dict.keys())

# Get Origin Data:
OriginData = OriginData_dict[dictkeys[-1]]  # get np.array by keyword
S_echo += OriginData  # Noise + Origin Data

# show echo:
plt.figure()
plt.imshow(img2View(S_echo))
plt.title('GF-3 Echo')

# show OriginData[idx, :]:
plt.figure()
plt.plot(np.abs(OriginData[view_azimuth, :]))
plt.title('GF-3 OriginData[idx, :]')

# show Echo[idx, :]:
plt.figure()
plt.plot(np.abs(S_echo[view_azimuth, :]))
plt.title('GF-3 Echo[idx, :]')

# 距离压缩 (小斜视角)
# tmp = Br / 2.
# condition = np.abs(Ext_f_tau) <= tmp
# condition = condition.astype(int)
# Hf = condition * np.exp(+1j * np.pi * (Ext_f_tau ** 2) / Kr)  # 滤波器

# 距离压缩 (大斜视角)
D_feta_Vr = np.sqrt(1 - ((lamda * Ext_f_eta) ** 2) / (4 * (Vr ** 2)))  # 徙动因子
K_src = (2 * (Vr ** 2) * (f0 ** 3) * (D_feta_Vr ** 3)) / (c * R0 * (Ext_f_eta ** 2))
Km = Kr / (1 - Kr / K_src)  # 改进的调频率
Hf = (np.abs(Ext_f_tau) <= Br / 2) * np.exp(+1j * np.pi * (Ext_f_tau ** 2) / Km)  # 匹配滤波器

# 匹配滤波
S1_ftau_eta = FFT_Range(S_echo, shift=True)
S1_ftau_eta = S1_ftau_eta * Hf
S1_tau_eta = IFFT_Range(S1_ftau_eta, shift=True)

# show:
plt.figure()
plt.imshow(img2View(S1_tau_eta))
plt.title('Pulse Compress over Range')

# 方位向傅里叶变换
S2_tau_feta = FFTShift(FFT_Azimuth(FFTShift(S1_tau_eta)))

# 距离徙动校正RCMC: 采用相位补偿法
# 虽然Ka是随着R0变化的，但是在相位补偿时需要假设R0是不变的

## RCMC (小斜视角)
# delta_R = (((lamda * Ext_f_eta) ** 2) * R0) / (8 * (Vr ** 2))  # 距离徙动表达式
# G_rcmc = np.exp((+4j * np.pi * Ext_f_tau * delta_R) / c)  # 补偿相位

# RCMC (大斜视角)
delta_R = R0 * ((1 - D_feta_Vr) / D_feta_Vr)  # 距离徙动表达式
G_rcmc = np.exp((+4j * np.pi * Ext_f_tau * delta_R) / c)  # 补偿相位

S3_ftau_feta = FFT_Range(S2_tau_feta, shift=True)  # 在方位向傅里叶变换的基础上进行距离向傅里叶变换
S3_ftau_feta = S3_ftau_feta * G_rcmc  # 与补偿相位相乘

# 距离向傅里叶逆变换
S3_tau_feta_RCMC = IFFT_Range(S3_ftau_feta, shift=True)

# show:
plt.figure()
plt.imshow(img2View(S3_tau_feta_RCMC))
plt.title('RCMC')

##方位压缩
# 根据变化的R0计算出相应的Ka矩阵(距离向变化，方位向不变)
Ka = 2 * Vr ** 2 * np.cos(theta_r_c) ** 2. / (lamda * Ext_R0_tau_r)

# 方位向匹配滤波器
Haz = np.exp(-1j * np.pi * Ext_f_eta ** 2. / Ka)
# Offset = np.exp(-1j * 2 * np.pi * Ext_f_eta * time_eta_c)  # 偏移滤波器，将原点搬移到Naz / 2的位置，校准坐标

# 匹配滤波
S4_tau_feta = S3_tau_feta_RCMC * Haz  # * Offset
S4_tau_eta = IFFT_Azimuth(S4_tau_feta, shift=True)

# show:
plt.figure()
plt.imshow(img2View(S4_tau_eta))
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('Pulse Compress over Azimuth')

# Peaky Value of Target C (DN):
# ROI_Echo = S_echo[1200:, 1800:]
C_DN_idx = np.abs(S_echo).argmax()
C_DN_Aidx = int(C_DN_idx / S_echo.shape[1])
C_DN_Ridx = C_DN_idx - C_DN_Aidx * S_echo.shape[1]
PeakyVal = S_echo[C_DN_Aidx, C_DN_Ridx]
PeakyVal_Modulus = np.abs(PeakyVal)
print('Peaky Value of TargetC (DN):', PeakyVal, PeakyVal_Modulus)

########################################## find A B C ##########################################
Results = np.abs(S4_tau_eta)
A_y, A_x = np.where(Results == Results[:, :500].max())
B_y, B_x = np.where(Results == Results[:, :1000].max())
C_y, C_x = np.where(Results == Results.max())

A_y, A_x = A_y[0], A_x[0]
B_y, B_x = B_y[0], B_x[0]
C_y, C_x = C_y[0], C_x[0]

print('A: [{}, {}]'.format(A_y, A_x))
print('B: [{}, {}]'.format(B_y, B_x))
print('C: [{}, {}]'.format(C_y, C_x))

# delta_Azimuth = abs(((-10000) - (-2000)) / (C_y - A_y))
# delta_Range = abs(((-10000) - (-2000)) * 0.15 / (C_x - A_x))
# print('delta_Azimuth:', delta_Azimuth)
# print('delta_Range:', delta_Range)
delta_Azimuth = Vs / Fa  # Azimuth
delta_Range = Vr / Fr  # Range (斜距向)
print('V/Fa:', Vs / Fa)
print('Vr/Fr:', Vr / Fr)
print()

########################################## Extract Energy by Calculation(Summary) ##########################################
# ROI_Modulus:
ROI = S4_tau_eta[1000 + 40:1300 + 40, 1200 - 28:1500 - 28]
ROI_Modulus = np.abs(ROI)

Center_y, Center_x = np.where(ROI_Modulus == ROI_Modulus.max())
Center_y, Center_x = int(Center_y[0]), int(Center_x[0])
print('Center Y:', Center_y, 'Center X:', Center_x)
rotate_matrix = cv.getRotationMatrix2D(center=(Center_x, Center_y), angle=theta_r_c / np.pi * 180, scale=1)
ROI_Modulus = cv.warpAffine(ROI_Modulus, rotate_matrix, ROI_Modulus.shape[1::-1])

plt.figure()
plt.imshow(img2View(ROI_Modulus))
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('ROI after Rotation')

# Get Peak Point:
peak_y, peak_x = np.where(ROI_Modulus == ROI_Modulus.max())

# Get shape of ROI:
Na, Nr = ROI_Modulus.shape
Na, Nr = int(Na), int(Nr)

########### integration B ###########
SideLength_B = 125
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
A_Mask[:, 135: 165 + 1] = 1
A_Mask[144: 170 + 1, :] = 1
NumA = A_Mask.sum()

# Segmentation:
ROI_A = ROI_Modulus * A_Mask

# DN_A ** 2:
IntegrationA = (ROI_A ** 2).sum()

########### Calculate Energy ###########
# Num_UpSampling =
# Energy = (IntegrationA - (NumA * IntegrationB / NumB)) * 0.4 * 0.2 * D_Azimuth * D_Range  # Unknow delta_a and delta_r, so that use part of their Spatial Resolution
Energy = (IntegrationA - (NumA * IntegrationB / NumB)) * delta_Range * delta_Azimuth / np.sin(incidence)
print('本地入射角 θ:', incidence)
print('sin(θ):', np.sin(incidence))

# Energy /= 9.  # cuz by Upsampling, whether need to perform?
Energy_dB = 10 * np.log10(Energy)
print('Extracted Energy:', Energy)
print('Extracted Energy/dB:', Energy_dB)
print()

# RCS:
RCS = eval(sys.argv[1]) if len(sys.argv) > 1 else int(mat_path.split('.')[0].split('_')[-1])
RCS_dB = 10 * np.log10(RCS)

print('Setting RCS:', RCS)
print('Setting RCS/dB:', RCS_dB)
print()

# Calibration:
Calibration_K = Energy / RCS
Calibration_K_dB = 10 * np.log10(Calibration_K)

print('Calibration Constant K:', Calibration_K)
print('Calibration Constant K/dB:', Calibration_K_dB)

###########################################################################
# To show:
ROI_BBox_show = ROI_Modulus.copy()  # Copy
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
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (135, 0), (165, Na - 1), color=(255, 0, 0), thickness=1)  # Azimuth longer
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, 144), (Nr - 1, 170), color=(255, 0, 0), thickness=1)  # Range longer

# Show Mask_A
plt.figure('Mask A')
plt.imshow(A_Mask, cmap='gray')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('"Mask A"')

# Show image:
plt.figure('Region of Interest')
plt.imshow(ROI_BBox_show)
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('"Region of Interest"')

# whether < 13 dB:
interest_Azimuth = 10 * np.log10(ROI_Modulus[150, :])
interest_Azimuth = interest_Azimuth / interest_Azimuth.max()
plt.figure()
plt.plot(interest_Azimuth)

tmp = np.zeros([400]) + -13
plt.plot(tmp)
plt.title('ROI_Modulus[150, :] dB')

print('斜视角: {}°'.format(theta_r_c / np.pi * 180))

# show all img:
plt.show()

if __name__ == '__main__':
    pass
