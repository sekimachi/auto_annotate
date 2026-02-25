import os
import glob
from ultralytics import YOLO
import cv2

# ===== 設定 =====
IMAGE_DIR = "images"
LABEL_DIR = "labels"
CONF_THRES = 0.4

models = ["best_blue.pt", "best_red.pt", "best_yellow.pt"]

print("利用可能なモデルを選択してください:")
for i, model in enumerate(models, 1):
    print(f"{i}. {model}")

choice = input("番号を入力してください (1-3): ")
try:
    model_index = int(choice) - 1
    if 0 <= model_index < len(models):
        MODEL_PATH = models[model_index]
    else:
        MODEL_PATH = "best.pt"
except ValueError:
    MODEL_PATH = "best.pt"

os.makedirs(LABEL_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

def add_instructions(img):
    instructions = [
        "Keyboard shortcuts:",
        "Left Drag  : Add box",
        "Right Drag : Delete box",
        "g: Auto annotate (YOLO)",
        "s: Save annotations",
        "c: Clear all boxes",
        "n: Next image",
        "z: Previous image",
        "ESC: Exit"
    ]
    y_offset = 20
    for line in instructions:
        cv2.putText(img, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 20
    return img

# ===== グローバル変数 =====
drawing = False
right_drawing = False
start_x, start_y = -1, -1
right_start_x, right_start_y = -1, -1
boxes = []
current_img = None

def redraw():
    img_copy = current_img.copy()
    for box in boxes:
        cv2.rectangle(img_copy,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0, 255, 0), 2)
    add_instructions(img_copy)
    cv2.imshow("Manual Annotation", img_copy)

def boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    return not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2)

def draw_rectangle(event, x, y, flags, param):
    global drawing, right_drawing
    global start_x, start_y, right_start_x, right_start_y
    global boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_RBUTTONDOWN:
        right_drawing = True
        right_start_x, right_start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        img_copy = current_img.copy()
        h, w = img_copy.shape[:2]

        cv2.line(img_copy, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(img_copy, (0, y), (w, y), (255, 255, 255), 1)

        if drawing:
            cv2.rectangle(img_copy,
                          (start_x, start_y),
                          (x, y),
                          (0, 255, 0), 2)

        if right_drawing:
            cv2.rectangle(img_copy,
                          (right_start_x, right_start_y),
                          (x, y),
                          (0, 0, 255), 2)

        for box in boxes:
            cv2.rectangle(img_copy,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 255, 0), 2)

        add_instructions(img_copy)
        cv2.imshow("Manual Annotation", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

        if start_x != end_x and start_y != end_y:
            boxes.append((
                min(start_x, end_x),
                min(start_y, end_y),
                max(start_x, end_x),
                max(start_y, end_y)
            ))
        redraw()

    elif event == cv2.EVENT_RBUTTONUP:
        right_drawing = False
        end_x, end_y = x, y

        if right_start_x != end_x and right_start_y != end_y:
            delete_rect = (
                min(right_start_x, end_x),
                min(right_start_y, end_y),
                max(right_start_x, end_x),
                max(right_start_y, end_y)
            )

            boxes[:] = [b for b in boxes if not boxes_overlap(b, delete_rect)]

        redraw()

# ===== 画像取得 =====
img_list = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "**", "*.jpg"), recursive=True) +
    glob.glob(os.path.join(IMAGE_DIR, "**", "*.png"), recursive=True) +
    glob.glob(os.path.join(IMAGE_DIR, "**", "*.jpeg"), recursive=True)
)

index = 0

while index < len(img_list):

    img_path = img_list[index]
    img_name = os.path.basename(img_path)
    current_img = cv2.imread(img_path)
    boxes = []

    h, w, _ = current_img.shape

    print(f"Annotating: {img_name} ({index+1}/{len(img_list)})")

    cv2.namedWindow("Manual Annotation")
    cv2.setMouseCallback("Manual Annotation", draw_rectangle)

    redraw()

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('g'):
            print("Running YOLO auto annotation...")
            results = model(current_img, conf=CONF_THRES)
            boxes.clear()

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((int(x1), int(y1),
                                  int(x2), int(y2)))

            print(f"Auto annotations: {len(boxes)}")
            redraw()

        elif key == ord('s'):
            label_path = os.path.join(
                LABEL_DIR,
                os.path.splitext(img_name)[0] + ".txt"
            )

            with open(label_path, "w") as f:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"0 {x_center} {y_center} {bw} {bh}\n")

            print(f"Saved {len(boxes)} annotations")
            index += 1
            break

        elif key == ord('c'):
            boxes.clear()
            redraw()

        elif key == ord('n'):
            index += 1
            break

        elif key == ord('z'):
            if index > 0:
                index -= 1
            break

        elif key == 27:
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()