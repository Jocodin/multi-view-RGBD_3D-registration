import pykinect_azure as pykinect
import cv2
import numpy as np
import open3d as o3d
import random
import math


# Azure kinect RGBD 카메라 초기값설정
pykinect.initialize_libraries()
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
device = pykinect.start_device(config = device_config)


# 캡쳐 프로세스 (color image, depth map, point cloud, trns color image)
capture_num = 0
dios = []
while 1:
    capture = device.update()
    ret1, color_img = capture.get_color_image()
    dio = capture.get_depth_image_object()
    ret, trns_color = capture.get_transformed_color_image()

    if not ret1:
        continue

    color_img1 = cv2.resize(color_img, (800,600))
    cv2.imshow('img capture', color_img1)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) == ord('c'):
        capture_num = capture_num + 1

        cv2.imwrite('color/'+str(capture_num)+'.png', color_img)
        dios.append(dio)
        pc = capture.get_pointcloud()
        np.save('pc/'+str(capture_num)+'.npy', pc[1])
        cv2.imwrite('trns_color/'+str(capture_num)+'.png', trns_color)

        print("capture success (" + str(capture_num) + ")")
print()


# 시점 변환행렬 구하기
if capture_num >= 2:
    calib = device.get_calibration(pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED,
                                   pykinect.K4A_COLOR_RESOLUTION_720P)
    trans = pykinect.Transformation(calib.handle())
    source_camera = pykinect.K4A_CALIBRATION_TYPE_COLOR
    target_camera = pykinect.K4A_CALIBRATION_TYPE_DEPTH

    # 특징점 매칭 (+정제)
    detector = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
    for i1 in range(capture_num-1):

        img1 = cv2.imread('color/'+str(i1+1)+'.png')
        img2 = cv2.imread('color/'+str(i1+2)+'.png')
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        matches = matcher.knnMatch(desc1, desc2, 2)

        ratio = 0.75
        good_matches = [first for first, second in matches \
                        if first.distance < second.distance * ratio]

        img1_pts = []
        img2_pts = []
        for i in good_matches:
            img1_pts.append(kp1[i.queryIdx].pt)
            img2_pts.append(kp2[i.trainIdx].pt)


        # 매칭점들 3차원으로 매핑
        all_3d_pts1 = []
        for i in range(len(good_matches)):
            dio = dios[i1]

            x = img1_pts[i][0]
            y = img1_pts[i][1]

            transformed_depth_image = trans.depth_image_to_color_camera(dio)
            depth_map = transformed_depth_image.to_numpy()[1]
            depth_mm = depth_map[round(y)][round(x)]

            xy = pykinect.k4a._k4atypes._xy(float(x), float(y))
            source_point2d = pykinect.k4a_float2_t(xy)
            source_depth = depth_mm
            pc_point3d = calib.convert_2d_to_3d(source_point2d, source_depth, source_camera, target_camera)
            pc_xyz = []
            pc_xyz.append(pc_point3d.__iter__()['x'])
            pc_xyz.append(pc_point3d.__iter__()['y'])
            pc_xyz.append(pc_point3d.__iter__()['z'])

            all_3d_pts1.append(pc_xyz)

        all_3d_pts2 = []
        for i in range(len(good_matches)):
            dio = dios[i1 + 1]

            x = img2_pts[i][0]
            y = img2_pts[i][1]

            transformed_depth_image = trans.depth_image_to_color_camera(dio)
            depth_map = transformed_depth_image.to_numpy()[1]
            depth_mm = depth_map[round(y)][round(x)]

            xy = pykinect.k4a._k4atypes._xy(float(x), float(y))
            source_point2d = pykinect.k4a_float2_t(xy)
            source_depth = depth_mm
            pc_point3d = calib.convert_2d_to_3d(source_point2d, source_depth, source_camera, target_camera)
            pc_xyz = []
            pc_xyz.append(pc_point3d.__iter__()['x'])
            pc_xyz.append(pc_point3d.__iter__()['y'])
            pc_xyz.append(pc_point3d.__iter__()['z'])

            all_3d_pts2.append(pc_xyz)


        # 최소자승법으로 시점변환행렬 구하기
        a_pts = []
        b_pts = []
        for i in range(len(good_matches)):
            if all_3d_pts1[i][2] > 0 and all_3d_pts2[i][2] > 0:
                a_pts.append(all_3d_pts1[i])
                b_pts.append(all_3d_pts2[i])

        A = np.array(b_pts)
        B = np.array(a_pts)

        assert A.shape == B.shape

        # 차원 얻기
        m = A.shape[1]

        # 각 점군 중심점 및 중심 편차 계산
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A  # A행렬 편차
        BB = B - centroid_B  # B행렬 편차

        # SVD이용한 회전 행렬 계산
        H = np.dot(AA.T, BB)  # 분산
        U, S, Vt = np.linalg.svd(H)  # SVD 계산
        R = np.dot(Vt.T, U.T)  # 회전 행렬

        # 반사된 경우 행렬 계산
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # 이동 행렬 계산
        t = centroid_B.T - np.dot(R, centroid_A.T)  #

        # 동차 변환행렬 계산
        T = np.identity(m + 1)
        T[:m, :m] = R  # 회전 행렬
        T[:m, m] = t  # 이동 행렬

        np.save('rt/'+str(i1+2)+'_'+str(i1+1)+'_R.npy', R)
        np.save('rt/'+str(i1+2)+'_'+str(i1+1)+'_t.npy', t)
        print(str(i1+1)+" RT saved!")
