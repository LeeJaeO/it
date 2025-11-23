from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
import cv2
import numpy as np

# ----------------------------
# Azure Kinect 설정
# ----------------------------
k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,  
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_30        # ★ 숫자가 아닌 ENUM
    )
)
k4a.start()

# ----------------------------
# YOLO 모델 로드
# ----------------------------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

print("YOLO loaded successfully")
print("Kinect started")

# ----------------------------
# 메인 루프
# ----------------------------
while True:
    # Kinect 프레임 받아오기
    capture = k4a.get_capture()
    img = capture.color

    if img is None:
        continue

    # Kinect RGB → OpenCV BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # YOLO 입력 전처리
    blob = cv2.dnn.blobFromImage(img_bgr, 1/255, (416, 416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    h, w = img_bgr.shape[:2]

    # 바운딩박스
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)

            # 사람(class_id=0)만 탐지
            if class_id == 0:
                confidence = scores[class_id]

                if confidence > 0.5:
                    cx, cy, bw, bh = det[:4] * np.array([w, h, w, h])

                    x = int(cx - bw/2)
                    y = int(cy - bh/2)
                    bw = int(bw)
                    bh = int(bh)

                    cv2.rectangle(img_bgr, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(img_bgr, "person", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("People Detection", img_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

k4a.stop()
cv2.destroyAllWindows()
