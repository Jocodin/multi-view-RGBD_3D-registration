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

        # 일정거리 떨어진 세점 선택
        print("selecting three points...")
        while 1:
            img1_ranpts = []
            img2_ranpts = []

            idx = random.randint(0, (len(img1_pts) - 1))
            img1_ranpts.append(img1_pts[idx])
            img2_ranpts.append(img2_pts[idx])
            tmpi = 0
            for i in range(len(img1_pts)):
                if ((img1_pts[i][0]-img1_ranpts[0][0])**2 + (img1_pts[i][1]-img1_ranpts[0][1])**2)**(0.5) > 200:
                    tmpi = i
                    img1_ranpts.append(img1_pts[i])
                    img2_ranpts.append(img2_pts[i])
                    break
            for i in range(len(img1_pts)):
                if i > tmpi:
                    if ((img1_pts[i][0]-img1_ranpts[0][0])**2 + (img1_pts[i][1]-img1_ranpts[0][1])**2)**(0.5) > 200 and ((img1_pts[i][0]-img1_ranpts[1][0])**2 + (img1_pts[i][1]-img1_ranpts[1][1])**2)**(0.5) > 200:
                        img1_ranpts.append(img1_pts[i])
                        img2_ranpts.append(img2_pts[i])
                        break

            if len(img1_ranpts) < 3:
                continue

            all_2d_pts = []
            for i in img1_ranpts:
                all_2d_pts.append(i)
            for i in img2_ranpts:
                all_2d_pts.append(i)

            all_3d_pts = []
            for i in range(6):
                if i < 3:
                    dio = dios[i1]
                else:
                    dio = dios[i1+1]

                x = all_2d_pts[i][0]
                y = all_2d_pts[i][1]

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

                all_3d_pts.append(pc_xyz)

            re = 0
            for i in all_3d_pts:
                if i[2] < 0:
                    re = 1
            if re == 0:
                break
        print("selected three points!")


        # 축회전방식으로 시점변환행렬 계산후 저장
        a_pts = []
        a_pts.append(all_3d_pts[0])
        a_pts.append(all_3d_pts[1])
        a_pts.append(all_3d_pts[2])
        b_pts = []
        b_pts.append(all_3d_pts[3])
        b_pts.append(all_3d_pts[4])
        b_pts.append(all_3d_pts[5])

        a_v1 = np.array([a_pts[1][0] - a_pts[0][0], a_pts[1][1] - a_pts[0][1], a_pts[1][2] - a_pts[0][2]])
        a_v2 = np.array([a_pts[2][0] - a_pts[0][0], a_pts[2][1] - a_pts[0][1], a_pts[2][2] - a_pts[0][2]])
        a_nv = np.cross(a_v1, a_v2)
        b_v1 = np.array([b_pts[1][0] - b_pts[0][0], b_pts[1][1] - b_pts[0][1], b_pts[1][2] - b_pts[0][2]])
        b_v2 = np.array([b_pts[2][0] - b_pts[0][0], b_pts[2][1] - b_pts[0][1], b_pts[2][2] - b_pts[0][2]])
        b_nv = np.cross(b_v1, b_v2)

        nv = np.cross(b_nv, a_nv)
        tmp = ((nv[0] ** 2 + nv[1] ** 2 + nv[2] ** 2) ** (0.5))
        nv[0] = nv[0] / tmp
        nv[1] = nv[1] / tmp
        nv[2] = nv[2] / tmp
        theta = math.acos(np.inner(b_nv, a_nv) / (a_nv[0] ** 2 + a_nv[1] ** 2 + a_nv[2] ** 2) ** (0.5) / (
                b_nv[0] ** 2 + b_nv[1] ** 2 + b_nv[2] ** 2) ** (0.5))

        r11 = nv[0] * nv[0] * (1 - math.cos(theta)) + math.cos(theta)
        r12 = nv[1] * nv[0] * (1 - math.cos(theta)) - nv[2] * math.sin(theta)
        r13 = nv[2] * nv[0] * (1 - math.cos(theta)) + nv[1] * math.sin(theta)
        r21 = nv[0] * nv[1] * (1 - math.cos(theta)) + nv[2] * math.sin(theta)
        r22 = nv[1] * nv[1] * (1 - math.cos(theta)) + math.cos(theta)
        r23 = nv[2] * nv[1] * (1 - math.cos(theta)) - nv[0] * math.sin(theta)
        r31 = nv[0] * nv[2] * (1 - math.cos(theta)) - nv[1] * math.sin(theta)
        r32 = nv[1] * nv[2] * (1 - math.cos(theta)) + nv[0] * math.sin(theta)
        r33 = nv[2] * nv[2] * (1 - math.cos(theta)) + math.cos(theta)

        R1 = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        t1 = np.array([[b_v1[0]], [b_v1[1]], [b_v1[2]]])
        t2 = R1 @ t1
        b_v1_r = np.array([t2[0][0], t2[1][0], t2[2][0]])

        nv2 = np.cross(b_v1_r, a_v1)
        tmp2 = ((nv2[0] ** 2 + nv2[1] ** 2 + nv2[2] ** 2) ** (0.5))
        nv2[0] = nv2[0] / tmp2
        nv2[1] = nv2[1] / tmp2
        nv2[2] = nv2[2] / tmp2
        theta2 = math.acos(np.inner(b_v1_r, a_v1) / (b_v1_r[0] ** 2 + b_v1_r[1] ** 2 + b_v1_r[2] ** 2) ** (0.5) / (
                a_v1[0] ** 2 + a_v1[1] ** 2 + a_v1[2] ** 2) ** (0.5))

        r11_2 = nv2[0] * nv2[0] * (1 - math.cos(theta2)) + math.cos(theta2)
        r12_2 = nv2[1] * nv2[0] * (1 - math.cos(theta2)) - nv2[2] * math.sin(theta2)
        r13_2 = nv2[2] * nv2[0] * (1 - math.cos(theta2)) + nv2[1] * math.sin(theta2)
        r21_2 = nv2[0] * nv2[1] * (1 - math.cos(theta2)) + nv2[2] * math.sin(theta2)
        r22_2 = nv2[1] * nv2[1] * (1 - math.cos(theta2)) + math.cos(theta2)
        r23_2 = nv2[2] * nv2[1] * (1 - math.cos(theta2)) - nv2[0] * math.sin(theta2)
        r31_2 = nv2[0] * nv2[2] * (1 - math.cos(theta2)) - nv2[1] * math.sin(theta2)
        r32_2 = nv2[1] * nv2[2] * (1 - math.cos(theta2)) + nv2[0] * math.sin(theta2)
        r33_2 = nv2[2] * nv2[2] * (1 - math.cos(theta2)) + math.cos(theta2)

        R2 = np.array([[r11_2, r12_2, r13_2], [r21_2, r22_2, r23_2], [r31_2, r32_2, r33_2]])
        t1 = np.array([[b_v1_r[0]], [b_v1_r[1]], [b_v1_r[2]]])
        t2 = R2 @ t1
        b_v1_rr = np.array([t2[0][0], t2[1][0], t2[2][0]])

        t1 = np.array([[b_pts[1][0]], [b_pts[1][1]], [b_pts[1][2]]])
        t2 = R2 @ R1 @ t1
        translation = [(a_pts[1][0] - t2[0][0]), (a_pts[1][1] - t2[1][0]), (a_pts[1][2] - t2[2][0])]

        np.save('rt/'+str(i1+2)+'_'+str(i1+1)+'_R1.npy', R1)
        np.save('rt/'+str(i1+2)+'_'+str(i1+1)+'_R2.npy', R2)
        np.save('rt/'+str(i1+2)+'_'+str(i1+1)+'_trns.npy', translation)
        print(str(i1+1)+" RT saved!")

