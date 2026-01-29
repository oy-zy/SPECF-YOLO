# filename: camera_configs.py
import cv2
import numpy as np

left_camera_matrix = np.array([[526.776883, 2.629362754, 312.3941476],
                               [0., 524.6867917, 252.7268285],
                               [0., 0., 1.]])
left_distortion = np.array([[-0.150318731, 1.126730625, -0.010997061, -0.007421492, 0.]])

right_camera_matrix = np.array([[522.8812723, -0.215372341, 316.4214689],
                                [0., 520.4973901, 254.0936398],
                                [0., 0., 1.]])
right_distortion = np.array([[0.065107685, -0.185810704, -0.012538538, 0.003887504, 0.]])

R = np.array([[0.999722367, 0.005702325, -0.022862031],
              [-0.005682477, 0.999983419, 0.000933048],
              [0.022866973, -0.000802876, 0.999738194]])

T = np.array([-57.47398743, 0.288556235, -0.904731802])  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
