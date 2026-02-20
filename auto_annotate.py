import os
import glob
from ultralytics import YOLO
import cv2

# ===== 設定 =====
IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"
CONF_THRES = 0.1   # 信頼度しきい値（重要）

# 利用可能なモデルリスト
models = ["best_blue.pt", "best_red.pt", "best_yellow.pt"]

# モデル選択
print("利用可能なモデルを選択してください:")
for i, model in enumerate(models, 1):
    print(f"{i}. {model}")
choice = input("番号を入力してください (1-3): ")
try:
    model_index = int(choice) - 1
    if 0 <= model_index < len(models):
        MODEL_PATH = models[model_index]
    else:
        print("無効な選択です。デフォルトのbest.ptを使用します。")
        MODEL_PATH = "best.pt"
except ValueError:
    print("無効な入力です。デフォルトのbest.ptを使用します。")
    MODEL_PATH = "best.pt"

os.makedirs(LABEL_DIR, exist_ok=True)

# ===== モデル読み込み =====
model = YOLO(MODEL_PATH)

def add_instructions(img):
    """画像に操作説明を追加"""
    instructions = [
        "Keyboard shortcuts:",
        "s: Save annotations",
        "c: Clear all boxes",
        "n: Next image",
        "z: Previous image",
        "ESC: Exit"
    ]
    y_offset = 20
    for line in instructions:
        cv2.putText(img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 20
    return img

# グローバル変数
drawing = False
start_x, start_y = -1, -1
boxes = []
current_img = None
current_img_name = None

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_x, start_y, current_img, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        img_copy = current_img.copy()
        # 十字線を描画
        h, w = img_copy.shape[:2]
        cv2.line(img_copy, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(img_copy, (0, y), (w, y), (255, 255, 255), 1)
        if drawing:
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
        for box in boxes:
            cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        add_instructions(img_copy)
        cv2.imshow("Manual Annotation", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        if start_x != end_x and start_y != end_y:
            boxes.append((min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)))
            img_copy = current_img.copy()
            for box in boxes:
                cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            add_instructions(img_copy)
            cv2.imshow("Manual Annotation", img_copy)

# ===== 画像ループ =====
img_list = sorted(glob.glob(os.path.join(IMAGE_DIR, "**", "*.jpg"), recursive=True) + 
                  glob.glob(os.path.join(IMAGE_DIR, "**", "*.png"), recursive=True) + 
                  glob.glob(os.path.join(IMAGE_DIR, "**", "*.jpeg"), recursive=True))
index = 0

while index < len(img_list):
    img_path = img_list[index]
    img_name = os.path.basename(img_path)
    current_img = cv2.imread(img_path)
    current_img_name = img_name
    boxes = []

    # 自動アノテーション実行
    h, w, _ = current_img.shape
    results = model(current_img, conf=CONF_THRES)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    
    # アノテーション結果を表示
    auto_annotation_count = len(boxes)
    print(f"Annotating: {img_name} ({index+1}/{len(img_list)}). Auto annotations: {auto_annotation_count}")

    cv2.namedWindow("Manual Annotation")
    cv2.setMouseCallback("Manual Annotation", draw_rectangle)

    # 操作説明を画像に追加
    img_with_text = current_img.copy()
    for box in boxes:
        cv2.rectangle(img_with_text, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    add_instructions(img_with_text)

    cv2.imshow("Manual Annotation", img_with_text)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 保存
            label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
            with open(label_path, "w") as f:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"0 {x_center} {y_center} {bw} {bh}\n")
            print(f"Saved {auto_annotation_count} annotations for {img_name}")
            index += 1
            break
        elif key == ord('c'):  # クリア
            boxes = []
            img_with_text = current_img.copy()
            add_instructions(img_with_text)
            cv2.imshow("Manual Annotation", img_with_text)
        elif key == ord('n'):  # 次へ（保存せずに）
            index += 1
            break
        elif key == ord('z'):  # 前の画像へ
            if index > 0:
                index -= 1
            else:
                print("Already at first image.")
            break

cv2.destroyAllWindows()
