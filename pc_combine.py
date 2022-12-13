import numpy as np
import open3d as o3d
import cv2

# 시점변환행렬 계산방식, 촬영횟수
cal_type = (input("Calculation type (ls or 3pr) : "))
cap_num = int(input("Captured numbers : "))

# 포인트클라우드 리스트
pc_list = []
for i in range(cap_num):
    pc = np.load('pc/'+str(i+1)+'.npy')
    pc_list.append(pc)

# 포인트클라우드 시점변환 적용과정
if cal_type == "3pr":
    for i in range(cap_num):
        print("pc registration"+str(i+1)+" in progress...")
        if i != 0:
            pc = pc_list[i]
            for j in range(i):
                R1 = np.load('rt/' + str(i+1-j) + '_' + str(i-j) + '_R1.npy')
                R2 = np.load('rt/' + str(i+1-j) + '_' + str(i-j) + '_R2.npy')
                trns = np.load('rt/' + str(i+1-j) + '_' + str(i-j) + '_trns.npy')
                trns = trns.tolist()
                for k in range(len(pc)):
                    t1 = np.array([[pc[k][0]], [pc[k][1]], [pc[k][2]]])
                    t2 = R2 @ R1 @ t1
                    t3 = [(trns[0] + t2[0][0]), (trns[1] + t2[1][0]), (trns[2] + t2[2][0])]
                    pc[k] = t3

            pc_list[i] = pc
    print("pc registration complete!")

elif cal_type == 'ls':
    for i in range(cap_num):
        print("pc registration" + str(i + 1) + " in progress...")
        if i != 0:
            pc = pc_list[i]
            for j in range(i):
                R = np.load('rt/' + str(i + 1 - j) + '_' + str(i - j) + '_R.npy')
                t = np.load('rt/' + str(i + 1 - j) + '_' + str(i - j) + '_t.npy')
                t = t.tolist()
                for k in range(len(pc)):
                    t1 = np.array([[pc[k][0]], [pc[k][1]], [pc[k][2]]])
                    t2 = R @ t1
                    t3 = [(t[0] + t2[0][0]), (t[1] + t2[1][0]), (t[2] + t2[2][0])]
                    pc[k] = t3

            pc_list[i] = pc
    print("pc registration complete!")

# 포인트클라우드 RGB이미지 리스트
tc_img_list = []
for i in range(cap_num):
    tc_img = cv2.imread('trns_color/' + str(i+1) + '.png')
    tc_img = cv2.cvtColor(tc_img, cv2.COLOR_BGRA2RGB).reshape(-1, 3) / 255
    tc_img_list.append(tc_img)

# 무색좌표 제거
pc_list_r = []
tc_img_list_r = []
for i in range(cap_num):
    pc_list_r.append([])
    tc_img_list_r.append([])

for i in range(cap_num):
    for j in range(len(tc_img_list[i])):
        if not np.array_equal(tc_img_list[i][j], [0,0,0]):
            tc_img_list_r[i].append(tc_img_list[i][j].tolist())
            pc_list_r[i].append(pc_list[i][j].tolist())

for i in range(cap_num):
    pc_list_r[i] = np.array(pc_list_r[i])
for i in range(cap_num):
    tc_img_list_r[i] = np.array(tc_img_list_r[i])

# 포인트클라우드 저장 및 시각화
pcd_list = []
for i in range(cap_num):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_list_r[i])
    pcd.colors = o3d.utility.Vector3dVector(tc_img_list_r[i])
    pcd_list.append(pcd)
    o3d.io.write_point_cloud("fin_pc/" + str(i+1) + ".pcd", pcd)

o3d.visualization.draw_geometries(pcd_list)
