# multi-view-RGBD_3D-restoration (swcon capstone design)

- 주제 : 다시점 RGBD 영상을 활용한 3D 복원
- 장비 : MS Azure kinect
- 라이브러리 : opencv, open3d, numpy, pykinect
- 코드\
  cal_RT_leastSquares.py : 최소자승법으로 시점변환행렬 계산\
  cal_RT_3PointAxisRot.py : 세점축회전방식으로 시점변환행렬 계산\
  pc_combine.py : 변환행렬로 포인트클라우드 정합수행\
  pc_play.py : 변환된 포인트클라우드 시각화
- 폴더\
  pykinect_azure : pykinect 라이브러리\
  color : 촬영된 컬러이미지\
  trns_color : 깊이맵으로 매핑된 컬러이미지\
  pc : 변환 전 포인트클라우드\
  rt : 계산된 시점변환행렬\
  fin_pc : 최종 변환된 포인트클라우드
- 테스트방법
  1. 파이썬 프로젝트 생성
  2. 코드 및 폴더 전체 복사+붙여넣기
  3. opencv, open3d, numpy 라이브러리 추가
  4. MS Azure kinect 장비 연결
  5. main 함수에서 cal_RT_leastSquares.py 또는 cal_RT_3PointAxisRot.py 실행
    (최소자승법 또는 세점축회전방식)
  6. pc_combine.py 실행
  7. pc_play.py 실행
  
