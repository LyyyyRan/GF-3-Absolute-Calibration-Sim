import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from np2mtlb import *
from scipy.io import loadmat as loadfile

def loadmat(filepath):
    return loadfile(filepath)['Focused_Data']

def findpeak(mat_file):
    # find peak position:
    axis_Azimath, axis_Range = mat_file.shape
    peak_val_arg = mat_file.argmax()
    peak_val_Azimuth = int(peak_val_arg / axis_Range)
    peak_val_Range = peak_val_arg - axis_Range * peak_val_Azimuth

    return peak_val_Azimuth, peak_val_Range

def GetROI(mat_file):
    # fine max position:
    peak_val_Azimuth, peak_val_Range = findpeak(mat_file)

    # Get ROI:
    y_min, y_max = peak_val_Azimuth - 16, peak_val_Azimuth + 16
    x_min, x_max = peak_val_Range - 16, peak_val_Range + 16
    ROI = mat_file[y_min:y_max, x_min:x_max]

    return ROI

def ExtractEnergy(ROI, show_=True):
    # Get shape of ROI:
    ROI_Modulus = ROI.copy()
    Na, Nr = int(ROI_Modulus.shape[0]), int(ROI_Modulus.shape[1])

    # init upsampling factor:
    upsam_factor = Na * Nr

    # upsampling:
    ROI_Modulus = UpSampling(ROI_Modulus)
    Na, Nr = int(ROI_Modulus.shape[0]), int(ROI_Modulus.shape[1])

    # get upsampling factor:
    upsam_factor = upsam_factor / Na / Nr

    # 姿态参数
    H = 755e3  # 卫星轨道高度
    phi = 20 * np.pi / 180  # 俯仰角+20°
    incidence = 20.5 * np.pi / 180  # 入射角
    R_eta_c = H / np.cos(incidence)  # 景中心斜距
    R0 = H / np.cos(phi)
    theta_r_c = np.acos(R0 / R_eta_c)  # 斜视角, 单位为 弧度, 斜视角为 4.6°

    ########### integration B ###########
    SideLength_B = 200
    NumB = 4 * (SideLength_B ** 2)

    # [a, b) is used in numpy index:
    ROI_B1 = ROI_Modulus[: SideLength_B, : SideLength_B]
    ROI_B2 = ROI_Modulus[: SideLength_B, Nr - SideLength_B:]
    ROI_B3 = ROI_Modulus[Na - SideLength_B:, : SideLength_B]
    ROI_B4 = ROI_Modulus[Na - SideLength_B:, Nr - SideLength_B:]

    # DN_B ** 2:
    IntegrationB = (ROI_B1 ** 2).sum() + (ROI_B2 ** 2).sum() + (ROI_B3 ** 2).sum() + (ROI_B4 ** 2).sum()

    # Find Max
    UP_Profile_Position_Ran, UP_Profile_Position_Azi = np.where(ROI_Modulus == ROI_Modulus.max())
    UP_Profile_Position_Ran, UP_Profile_Position_Azi = int(UP_Profile_Position_Ran[0]), int(UP_Profile_Position_Azi[0])

    # Rotate:
    Upsampled_ROI_Mod = np.abs(ROI_Modulus)
    rotate_matrix = cv.getRotationMatrix2D(center=(UP_Profile_Position_Ran, UP_Profile_Position_Azi),
                                           angle=rad2deg(theta_r_c), scale=1)
    Rotated_ROI_Modulus = cv.warpAffine(Upsampled_ROI_Mod, rotate_matrix, Upsampled_ROI_Mod.shape[1::-1])

    # update fine max position:
    peak_val_Azimuth, peak_val_Range = findpeak(Rotated_ROI_Modulus)

    ########### integration A ###########
    A_Mask = np.zeros_like(ROI_Modulus, dtype=int)  # [a, b)
    A_Mask[:, peak_val_Range - 100: peak_val_Range + 100] = 1
    A_Mask[peak_val_Azimuth - 150: peak_val_Azimuth + 150, :] = 1
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
    # print('delta_Azimuth = {}'.format(delta_Azimuth))
    # print('delta_Range = {}'.format(delta_Range))

    # Extract Energy:
    Energy = IntegrationA - (NumA * IntegrationB / NumB)
    # print('IntegrationA: {}'.format(IntegrationA))
    # print('IntegrationB: {}'.format(IntegrationB))
    # print('NumA: {}'.format(NumA))
    # print('NumB: {}'.format(NumB))

    Energy = Energy * delta_Range * delta_Azimuth / np.sin(incidence)
    # print('本地入射角 θ:', incidence)
    # print('sin(θ):', np.sin(incidence))

    # why upsam_factor? cuz of upsampling
    # why upsam_factor ** 2? cuz Energy == DN ** 2
    Energy = Energy * (upsam_factor ** 2)

    if show_:
        ROI_BBox_show = Rotated_ROI_Modulus.copy()  # Copy
        ROI_BBox_show = ROI_BBox_show / ROI_BBox_show.max()  # squeeze val domain
        ROI_BBox_show = ROI_BBox_show * 255  # put val domain to be 0~255
        ROI_BBox_show = np.uint8(ROI_BBox_show)
        ROI_BBox_show = cv.cvtColor(ROI_BBox_show, cv.COLOR_GRAY2BGR)  # 1-Channel to 3-Channels

        # Drew B BBoxs: (Range, Azimuth) is used in OpenCV
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, 0), (SideLength_B - 1, SideLength_B - 1), color=(0, 0, 255),
                                     thickness=1)
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (Nr - SideLength_B, 0), (Nr - 1, SideLength_B - 1),
                                     color=(0, 0, 255),
                                     thickness=1)
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, Na - SideLength_B), (SideLength_B - 1, Na - 1),
                                     color=(0, 0, 255),
                                     thickness=1)
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (Nr - SideLength_B, Na - SideLength_B), (Nr, Na - 1),
                                     color=(0, 0, 255),
                                     thickness=1)

        # Drew A BBoxs: (Range, Azimuth) is used in OpenCV
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (peak_val_Range - 100, 0), (peak_val_Range + 100, Na - 1), color=(255, 0, 0),
                                     thickness=1)  # Azimuth longer
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, peak_val_Azimuth - 150), (Nr - 1, peak_val_Azimuth + 150), color=(255, 0, 0),
                                     thickness=1)  # Range longer

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

    return Energy

# Rotate:
def Rotate(ROI_Modulus, theta):
    # Find peak:
    peak_Azimuth, peak_Range = findpeak(ROI_Modulus)
    peak_Azimuth, peak_Range = int(peak_Azimuth), int(peak_Range)

    # Get Rotate Matrix:
    rotate_matrix = cv.getRotationMatrix2D(center=(peak_Azimuth, peak_Range),
                                           angle=rad2deg(theta), scale=1)

    # Perform Rotation:
    Rotated_ROI_Modulus = cv.warpAffine(ROI_Modulus, rotate_matrix, ROI_Modulus.shape[1::-1])

    return Rotated_ROI_Modulus