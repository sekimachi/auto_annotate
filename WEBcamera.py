import cv2
import numpy as np
from ultralytics import YOLO

# ===============================
# YOLO 設定
# ===============================
MODEL_PATH = "best_yellow.pt"
CONF_TH = 0.5
ALLOWED_CLASS_IDS = [0, 1, 2, 3]   # 学習済みクラス
MIN_AREA_RATIO = 0.01
NMS_IOU_TH = 0.5

model = YOLO(MODEL_PATH)

# ===============================
# Webカメラ設定
# ===============================
CAM_ID = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480

cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

img_area = CAM_WIDTH * CAM_HEIGHT

# 画面中心
center_x = CAM_WIDTH // 2
center_y = CAM_HEIGHT // 2

print("Start Webcam YOLO (minimal GUI)")

# ===============================
# IoU & NMS
# ===============================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(areaA + areaB - interArea)


def suppress_boxes(detections, iou_th=0.5):
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [
            d for d in detections
            if iou(best[:4], d[:4]) < iou_th
        ]
    return keep

# ===============================
# メインループ
# ===============================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=CONF_TH,
            classes=ALLOWED_CLASS_IDS,
            verbose=False
        )

        display = frame.copy()
        detections = []

        # 検出収集
        for result in results:
            if result.boxes is None:
                continue

            for box, cls_id, score in zip(
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf
            ):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1

                if (w * h) / img_area < MIN_AREA_RATIO:
                    continue

                detections.append(
                    (x1, y1, x2, y2, float(score))
                )

        detections = suppress_boxes(detections, NMS_IOU_TH)

        # 描画（枠線のみ）
        for x1, y1, x2, y2, score in detections:
            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        cv2.imshow("Webcam YOLO Ball Detection (Minimal)", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped")
