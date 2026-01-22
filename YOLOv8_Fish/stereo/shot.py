import os
import cv2

id_image = 0  # 图片的ID
cameraCapture = cv2.VideoCapture(1)
# cameraCapture = cv2.VideoCapture(6)

# #迭代停止模式选择（type, max_iter, epsilon）
# #cv2.TERM_CRITERIA_EPS ：精确度（误差）满足epsilon，则停止。
# #cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter，则停止。
# #两者结合，满足任意一个结束。
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 设置高宽
cameraCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cameraCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 创建左右摄像头对应的目录
dirsL = 'calibration/left/chessboard-L'
dirsR = 'calibration/right/chessboard-R'

if not os.path.exists(dirsL):
    os.makedirs(dirsL)
if not os.path.exists(dirsR):
    os.makedirs(dirsR)

while True:
    ret, frame = cameraCapture.read()
    # 这里的左右两个摄像头的图像是连在一起的，所以进行一下分割
    # 如果你的摄像头本来就是双设备号，直接获取两个摄像头做为frame1，frame2就好

    frame1 = frame[0:720, 0:1280]
    frame2 = frame[0:720, 1280:2560]

    # 转换成灰度图 棋盘格识别需要时灰度图
    grayR = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格这个 注意到我们用的棋盘格 内角数是8*6的 用其他的要修改一下
    retR, cornersR = cv2.findChessboardCorners(grayR, (8, 6), None)
    retL, cornersL = cv2.findChessboardCorners(grayL, (8, 6), None)
    # cv2.imshow('imgR', frame1)
    # cv2.imshow('imgL', frame2)
    cv2.imshow('imgR', grayR)
    cv2.imshow('imgL', grayL)
    # 如果找到了棋盘格就显示内角点
    if (retR == True) & (retL == True):
        corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)

        # 画出角点
        cv2.drawChessboardCorners(grayR, (8, 6), corners2R, retR)
        cv2.drawChessboardCorners(grayL, (8, 6), corners2L, retL)
        cv2.imshow('VideoR', grayR)
        cv2.imshow('VideoL', grayL)

        if cv2.waitKey(0) & 0xFF == ord('s'):  # S 存储 C取消存储
            print('S PRESSESED')
            str_id_image = str(id_image)
            print('Images ' + str_id_image + ' saved for right and left cameras')
            #  需要提前创建好路径
            cv2.imwrite('calibration/right/chessboard-R' + str_id_image + '.png', frame1)
            cv2.imwrite('calibration/left/chessboard-L' + str_id_image + '.png', frame2)
            id_image = id_image + 1
        else:
            print('Images not saved')

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 结束当前程序
        break

# Release the Cameras
cameraCapture.release()
cv2.destroyAllWindows()
