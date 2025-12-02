import cv2
import numpy as np
import mediapipe as mp  # 가볍고 빠른 얼굴 검출 라이브러리
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS

# ─── [설정: 거리 필터링 범위] ───
MIN_DISTANCE = 0.3   # m
MAX_DISTANCE = 0.5   # m

# ─── [설정: 안티 스푸핑(실물 vs 사진) 임계값] ───
# 얼굴 깊이 패치의 표준편차가 이 값보다 작으면 "평면(사진/모니터)"일 가능성이 높다고 간주
DEPTH_STD_THRESHOLD_MM = 15.0    # 15mm (환경에 따라 튜닝 필요)
MIN_VALID_DEPTH_PIXELS = 80      # 최소 유효 깊이 픽셀 수 (너무 적으면 신뢰도 낮음)

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
        print(f"✅ 시스템 초기화 완료 (멀티 얼굴 + 안티스푸핑 버전)")
        print(f"🎯 유효 거리 설정: {MIN_DISTANCE}m ~ {MAX_DISTANCE}m")
        print(f"🛡 안티스푸핑 깊이 표준편차 임계값: {DEPTH_STD_THRESHOLD_MM} mm")
    except Exception as e:
        print(f"❌ 카메라 연결 실패: {e}")
        return

    # 좌표 계산용 파라미터 (color 카메라 내참수)
    intrinsics = k4a.calibration.get_camera_matrix(1)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    while True:
        capture = k4a.get_capture()
        
        if capture.color is not None and capture.depth is not None:
            # 1. 이미지 처리 (MediaPipe는 RGB를 사용)
            img_bgr = capture.color[:, :, :3].copy()  # 쓰기 가능하도록 복사
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 2. Depth 맵 (이미 color에 정렬된 깊이)
            transformed_depth = capture.transformed_depth

            # 3. 얼굴 검출 수행
            results = face_detection.process(img_rgb)

            best_face = None  # 가장 가까운 얼굴 정보 저장용

            if results.detections:
                h, w, _ = img_bgr.shape

                for detection in results.detections:
                    # ─── [A] Bounding Box 좌표 계산 ───
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    box_w = int(bboxC.width * w)
                    box_h = int(bboxC.height * h)

                    # 박스 영역 클램핑 (화면 밖으로 나가는 경우 대비)
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w - 1, x + box_w)
                    y2 = min(h - 1, y + box_h)

                    if x2 <= x1 or y2 <= y1:
                        continue  # 잘못된 박스

                    # ─── [B] 중심점 계산 ───
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # ─── [C] 깊이 패치 추출 (얼굴 박스 중앙 부분만 사용) ───
                    # 얼굴 박스의 중앙 40% 영역 사용
                    patch_w = int((x2 - x1) * 0.4)
                    patch_h = int((y2 - y1) * 0.4)

                    patch_cx = center_x
                    patch_cy = center_y

                    px1 = max(0, patch_cx - patch_w // 2)
                    px2 = min(w - 1, patch_cx + patch_w // 2)
                    py1 = max(0, patch_cy - patch_h // 2)
                    py2 = min(h - 1, patch_cy + patch_h // 2)

                    if px2 <= px1 or py2 <= py1:
                        continue

                    depth_patch = transformed_depth[py1:py2+1, px1:px2+1]
                    valid_depth = depth_patch[depth_patch > 0]  # 0은 미측정값이므로 제외

                    if valid_depth.size < MIN_VALID_DEPTH_PIXELS:
                        # 유효 깊이 픽셀 너무 적으면 신뢰도 떨어지니 스킵
                        continue

                    # ─── [D] 깊이 기반 거리 계산 ───
                    z_mm_med = np.median(valid_depth)        # 중앙값 (노이즈에 강함)
                    z_meter = z_mm_med / 1000.0              # m 단위로 변환

                    if z_meter <= 0:
                        continue

                    # ─── [E] 깊이 변동성(입체감) 계산 ───
                    # 중앙값에서 ±150mm 이내 값만 사용하여 배경 영향 줄이기
                    depth_window = valid_depth[np.abs(valid_depth - z_mm_med) < 150]
                    if depth_window.size < MIN_VALID_DEPTH_PIXELS // 2:
                        depth_window = valid_depth  # 너무 줄어들면 다시 전체 사용

                    depth_std_mm = float(np.std(depth_window))  # mm 단위 표준편차

                    # ─── [F] 3D 좌표 계산 (카메라 좌표계 기준) ───
                    real_x = (center_x - cx) * z_mm_med / fx / 1000.0
                    real_y = (center_y - cy) * z_mm_med / fy / 1000.0

                    face_info = {
                        "bbox": (x1, y1, x2, y2),
                        "center": (center_x, center_y),
                        "z_meter": z_meter,
                        "z_mm": z_mm_med,
                        "depth_std_mm": depth_std_mm,
                        "real_x": real_x,
                        "real_y": real_y,
                    }

                    # ─── [G] 가장 가까운 얼굴만 선택 ───
                    if best_face is None or z_meter < best_face["z_meter"]:
                        best_face = face_info

            # ─── [가장 가까운 얼굴만 화면에 표시 + 필터링/스푸핑 판정] ───
            if best_face is not None:
                x1, y1, x2, y2 = best_face["bbox"]
                center_x, center_y = best_face["center"]
                z_meter = best_face["z_meter"]
                depth_std_mm = best_face["depth_std_mm"]
                real_x = best_face["real_x"]
                real_y = best_face["real_y"]

                # 1단계: 거리 필터링
                in_range = (MIN_DISTANCE <= z_meter <= MAX_DISTANCE)

                # 2단계: 안티 스푸핑 (실물/사진 구분)
                # 거리 범위 안에 있는 경우에만 스푸핑 판정 의미가 있음
                if in_range:
                    if depth_std_mm < DEPTH_STD_THRESHOLD_MM:
                        # 평면에 가까움 -> 사진/모니터일 가능성 큼
                        status = "FAIL-FAKE"   # 거리 OK지만 사진일 가능성
                        color = (0, 0, 255)
                        thickness = 2
                        spoof_label = "PHOTO/FLAT"
                    else:
                        status = "PASS-REAL"   # 거리 OK + 얼굴이 입체적
                        color = (0, 255, 0)
                        thickness = 3
                        spoof_label = "REAL"
                else:
                    # 거리 범위 밖이면 스푸핑 여부와 상관없이 불합격
                    status = "FAIL-DIST"
                    color = (0, 0, 255)
                    thickness = 2
                    spoof_label = "OUT_RANGE"

                # ─── [화면 그리기] ───
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

                # 상태 + 거리 정보
                info_status = f"[{status}] Dist: {z_meter:.2f}m"
                cv2.putText(
                    img_bgr,
                    info_status,
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    color,
                    2
                )

                # 실물/사진 판정 + 깊이 표준편차
                info_spoof = f"{spoof_label} | DepthStd: {depth_std_mm:.1f}mm"
                cv2.putText(
                    img_bgr,
                    info_spoof,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.55,
                    color,
                    1
                )

                # 좌표 정보
                info_coord = f"X:{real_x:.2f} Y:{real_y:.2f}"
                cv2.putText(
                    img_bgr,
                    info_coord,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.55,
                    color,
                    1
                )

                # 중심점 표시
                cv2.circle(img_bgr, (center_x, center_y), 5, color, -1)

            # 최종 화면 출력
            cv2.imshow("School Access System (Nearest + Anti-Spoof)", img_bgr)

        if cv2.waitKey(1) == 27:  # ESC 키
            break

    k4a.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
