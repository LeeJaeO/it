import os
import glob

import cv2
import numpy as np
import mediapipe as mp  # 가볍고 빠른 얼굴 검출 라이브러리
import face_recognition  # 얼굴 임베딩/비교 라이브러리
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS

# ─── [설정: 거리 필터링 범위] ───
MIN_DISTANCE = 0.3   # m
MAX_DISTANCE = 0.5   # m

# ─── [설정: 안티 스푸핑(실물 vs 사진) 임계값] ───
DEPTH_STD_THRESHOLD_MM = 15.0    # 15mm (환경에 맞게 튜닝)
MIN_VALID_DEPTH_PIXELS = 80      # 깊이 패치 내 최소 유효 픽셀 수

# ─── [설정: 한양대 학생 얼굴 DB 폴더] ───
HYU_STUDENTS_DIR = "./hyu_students"   # ★★ 여기를 본인 폴더 경로로 변경
FACE_MATCH_THRESHOLD = 0.5            # ★★ 얼굴 매칭 임계값 (작을수록 엄격)

# ─── [전역: 한양대 학생 임베딩 DB] ───
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_LABELS = []

# 1. 미디어파이프(얼굴 검출 AI) 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)


# ─────────────────────────────────────────────
# 한양대학교 학생 얼굴 DB 로딩
# ─────────────────────────────────────────────
def load_hyu_students():
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_LABELS

    KNOWN_FACE_ENCODINGS = []
    KNOWN_FACE_LABELS = []

    if not os.path.isdir(HYU_STUDENTS_DIR):
        print(f"⚠ HYU_STUDENTS_DIR가 존재하지 않습니다: {HYU_STUDENTS_DIR}")
        return

    img_paths = glob.glob(os.path.join(HYU_STUDENTS_DIR, "*.*"))

    print(f"📂 한양대 학생 사진 로딩 중... (폴더: {HYU_STUDENTS_DIR})")

    for img_path in img_paths:
        try:
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if not encodings:
                print(f"  ➤ 얼굴을 찾지 못해 스킵: {img_path}")
                continue

            encoding = encodings[0]
            label = os.path.splitext(os.path.basename(img_path))[0]  # 파일명(확장자 제외)을 라벨로 사용
            KNOWN_FACE_ENCODINGS.append(encoding)
            KNOWN_FACE_LABELS.append(label)
            print(f"  ✅ 등록: {label}")
        except Exception as e:
            print(f"  ❌ 로딩 실패: {img_path}, 에러: {e}")

    print(f"✅ 한양대 학생 DB 로딩 완료: {len(KNOWN_FACE_LABELS)}명 등록")


# ─────────────────────────────────────────────
# 현재 얼굴이 한양대 학생 DB에 있는지 판단
# ─────────────────────────────────────────────
def recognize_hyu_student(face_bgr):
    """
    face_bgr: 얼굴 부분이 잘린 BGR 이미지 (numpy array)
    return: (label or None, best_distance or None)
    """
    if len(KNOWN_FACE_ENCODINGS) == 0:
        return None, None

    # face_recognition은 RGB를 사용
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(face_rgb)

    if not encodings:
        return None, None

    face_encoding = encodings[0]

    # DB와 모든 거리 계산
    distances = face_recognition.face_distance(KNOWN_FACE_ENCODINGS, face_encoding)
    if len(distances) == 0:
        return None, None

    best_idx = np.argmin(distances)
    best_distance = float(distances[best_idx])

    if best_distance < FACE_MATCH_THRESHOLD:
        return KNOWN_FACE_LABELS[best_idx], best_distance
    else:
        return None, best_distance


# ─────────────────────────────────────────────
# 메인 로직
# ─────────────────────────────────────────────
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
        print(f"✅ 시스템 초기화 완료 (멀티 얼굴 + 안티스푸핑 + HYU 인증)")
        print(f"🎯 유효 거리 설정: {MIN_DISTANCE}m ~ {MAX_DISTANCE}m")
        print(f"🛡 안티스푸핑 깊이 표준편차 임계값: {DEPTH_STD_THRESHOLD_MM} mm")
        print(f"🎓 HYU 얼굴 매칭 임계값: {FACE_MATCH_THRESHOLD}")
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

                    # 박스 영역 클램핑
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w - 1, x + box_w)
                    y2 = min(h - 1, y + box_h)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # ─── [B] 중심점 계산 ───
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # ─── [C] 깊이 패치 추출 (얼굴 박스 중앙 40%) ───
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
                        continue

                    # ─── [D] 깊이 기반 거리 계산 ───
                    z_mm_med = np.median(valid_depth)
                    z_meter = z_mm_med / 1000.0

                    if z_meter <= 0:
                        continue

                    # ─── [E] 깊이 변동성(입체감) 계산 ───
                    depth_window = valid_depth[np.abs(valid_depth - z_mm_med) < 150]
                    if depth_window.size < MIN_VALID_DEPTH_PIXELS // 2:
                        depth_window = valid_depth

                    depth_std_mm = float(np.std(depth_window))

                    # ─── [F] 3D 좌표 계산 ───
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

            # ─── [가장 가까운 얼굴만 화면에 표시 + 거리/스푸핑/HYU 판정] ───
            if best_face is not None:
                x1, y1, x2, y2 = best_face["bbox"]
                center_x, center_y = best_face["center"]
                z_meter = best_face["z_meter"]
                depth_std_mm = best_face["depth_std_mm"]
                real_x = best_face["real_x"]
                real_y = best_face["real_y"]

                # 1단계: 거리 필터링
                in_range = (MIN_DISTANCE <= z_meter <= MAX_DISTANCE)

                # 기본값
                status = "FAIL"
                spoof_label = "UNKNOWN"
                hyu_label = None
                face_dist = None

                color = (0, 0, 255)
                thickness = 2

                if in_range:
                    # 2단계: 안티 스푸핑 (실물/사진)
                    if depth_std_mm < DEPTH_STD_THRESHOLD_MM:
                        status = "FAIL-FAKE"
                        spoof_label = "PHOTO/FLAT"
                    else:
                        # 3단계: HYU 얼굴 매칭
                        face_roi = img_bgr[y1:y2, x1:x2]
                        hyu_label, face_dist = recognize_hyu_student(face_roi)

                        if hyu_label is not None:
                            status = "PASS-HYU"
                            spoof_label = f"REAL-HYU ({hyu_label})"
                            color = (0, 255, 0)
                            thickness = 3
                        else:
                            status = "FAIL-NOT-HYU"
                            spoof_label = "REAL-NOT_IN_DB"
                else:
                    status = "FAIL-DIST"
                    spoof_label = "OUT_RANGE"

                # ─── [화면 그리기] ───
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

                # 상태 + 거리
                info_status = f"[{status}] Dist: {z_meter:.2f}m"
                cv2.putText(
                    img_bgr,
                    info_status,
                    (x1, y1 - 35),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    color,
                    2
                )

                # 스푸핑/HYU 정보 + 깊이 표준편차
                if face_dist is not None:
                    spoof_text = f"{spoof_label} | DepthStd: {depth_std_mm:.1f}mm | FaceDist: {face_dist:.2f}"
                else:
                    spoof_text = f"{spoof_label} | DepthStd: {depth_std_mm:.1f}mm"

                cv2.putText(
                    img_bgr,
                    spoof_text,
                    (x1, y1 - 12),
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
            cv2.imshow("School Access System (HYU + Anti-Spoof)", img_bgr)

        if cv2.waitKey(1) == 27:  # ESC 키
            break

    k4a.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 한양대 학생 DB 먼저 로딩
    load_hyu_students()
    main()
