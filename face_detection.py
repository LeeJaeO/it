import cv2
import numpy as np
import mediapipe as mp  # 가볍고 빠른 얼굴 검출 라이브러리
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS

# ─── [설정: 거리 필터링 범위] ───
# 테스트하시던 30cm ~ 50cm 설정 유지
MIN_DISTANCE = 0.3
MAX_DISTANCE = 0.5

# 1. 미디어파이프(얼굴 검출 AI) 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

def main():
    # 2. Azure Kinect 카메라 설정
    k4a = PyK4A(
        Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True,
        )
    )
    
    try:
        k4a.start()
        print(f"✅ 시스템 초기화 완료 (롤백 버전)")
        print(f"🎯 유효 거리 설정: {MIN_DISTANCE}m ~ {MAX_DISTANCE}m")
    except Exception as e:
        print(f"❌ 카메라 연결 실패: {e}")
        return

    # 좌표 계산용 파라미터
    intrinsics = k4a.calibration.get_camera_matrix(1)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    while True:
        capture = k4a.get_capture()
        
        if capture.color is not None and capture.depth is not None:
            # 1. 이미지 처리 (MediaPipe는 RGB를 사용)
            img_bgr = capture.color[:, :, :3].copy() # 쓰기 가능하도록 복사
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 2. Depth 맵 정렬
            transformed_depth = capture.transformed_depth

            # 3. 얼굴 검출 수행
            results = face_detection.process(img_rgb)

            if results.detections:
                for detection in results.detections:
                    # [A] Bounding Box 좌표 계산
                    h, w, _ = img_bgr.shape
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    box_w = int(bboxC.width * w)
                    box_h = int(bboxC.height * h)

                    # [B] 중심점 계산
                    center_x = x + box_w // 2
                    center_y = y + box_h // 2
                    
                    # 좌표 안전장치
                    center_x = max(0, min(center_x, w - 1))
                    center_y = max(0, min(center_y, h - 1))

                    # [C] 거리 측정 (mm -> m)
                    z_mm = transformed_depth[center_y, center_x]
                    z_meter = z_mm / 1000.0

                    if z_meter == 0: continue

                    # [D] 3D 좌표 계산
                    real_x = (center_x - cx) * z_mm / fx / 1000.0
                    real_y = (center_y - cy) * z_mm / fy / 1000.0
                    
                    # ──────── [핵심: 거리 기반 필터링] ────────
                    # 조건: 설정된 거리(0.3~0.5m) 사이에 있는가?
                    if MIN_DISTANCE <= z_meter <= MAX_DISTANCE:
                        # [조건 만족] -> 초록색 (PASS)
                        color = (0, 255, 0) 
                        status = "PASS"
                        thickness = 3
                    else:
                        # [조건 불만족] -> 빨간색 (FAIL)
                        color = (0, 0, 255) 
                        status = "FAIL"
                        thickness = 2
                    
                    # ──────── [화면 그리기] ────────
                    cv2.rectangle(img_bgr, (x, y), (x + box_w, y + box_h), color, thickness)
                    
                    # 텍스트 정보 표시
                    info_status = f"[{status}] Dist: {z_meter:.2f}m"
                    cv2.putText(img_bgr, info_status, (x, y - 25), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                    
                    # 좌표 정보
                    info_coord = f"X:{real_x:.2f} Y:{real_y:.2f}"
                    cv2.putText(img_bgr, info_coord, (x, y - 5), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
                    
                    # 중심점 표시
                    cv2.circle(img_bgr, (center_x, center_y), 5, color, -1)

            cv2.imshow("School Access System (Distance Filter)", img_bgr)

        if cv2.waitKey(1) == 27: # ESC 키
            break

    k4a.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
